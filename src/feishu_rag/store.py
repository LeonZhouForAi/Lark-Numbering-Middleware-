"""SQLite 文档索引和中文友好的轻量检索。"""

from __future__ import annotations

import re
import sqlite3
import time
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Collection, Iterable

from .models import Chunk, RetrievalScope, SearchResult


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9_]+")
_QUESTION_SHELLS = (
    "需要什么",
    "是什么",
    "有什么",
    "有哪些",
    "请问",
    "有何",
    "哪些",
    "怎么",
    "如何",
)
_GENERIC_TERMS = ("流程", "制度", "规定", "办法")
_GENERIC_RESIDUALS = frozenset({"这", "那", "呢", "都", "是", "有", "要", "吗", "么", "的", "了"})
_QUERY_STOP_WORDS = frozenset({"and", "or", "not", "near"})
_FEISHU_SOURCE_RE = re.compile(r"^feishu:([^:]+):")
_SQLITE_INT_MAX = 2**63 - 1
_LOCK_RETRY_ATTEMPTS = 20


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _tokens(text: str, *, include_single_chinese: bool = True) -> list[str]:
    terms: list[str] = []
    for part in _TOKEN_RE.findall(_normalize(text)):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 1:
                if include_single_chinese:
                    terms.append(part)
            else:
                terms.extend(part[index : index + 2] for index in range(len(part) - 1))
                if include_single_chinese:
                    terms.extend(part)
        else:
            terms.append(part)
    return list(dict.fromkeys(terms))


def _pretokenize(text: str) -> str:
    """生成供 FTS5 unicode61 tokenizer 使用的 NFKC 预分词文本。"""

    return " ".join(_tokens(text))


def _meaningful_query(query: str) -> str:
    cleaned = _normalize(query)
    for shell in _QUESTION_SHELLS:
        cleaned = cleaned.replace(shell, " ")
    for generic in _GENERIC_TERMS:
        cleaned = cleaned.replace(generic, " ")
    searchable = "".join(_TOKEN_RE.findall(cleaned))
    if searchable and all(character in _GENERIC_RESIDUALS for character in searchable):
        return ""
    return _normalize(cleaned)


def _core_terms(query: str) -> list[str]:
    return [
        term
        for term in _tokens(_meaningful_query(query), include_single_chinese=False)
        if term not in _QUERY_STOP_WORDS
    ]


def _execute_with_lock_retry(
    connection: sqlite3.Connection,
    statement: str,
) -> sqlite3.Cursor:
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        try:
            return connection.execute(statement)
        except sqlite3.OperationalError as exc:
            message = str(exc).casefold()
            if not ("locked" in message or "busy" in message):
                raise
            if attempt == _LOCK_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(min(0.01 * (attempt + 1), 0.1))
    raise AssertionError("unreachable")


class IndexStore:
    """保存文档元数据和片段，并提供本地检索。"""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA busy_timeout = 30000")
            _execute_with_lock_retry(self.connection, "PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self.connection.execute("PRAGMA foreign_keys = ON")
            self._fts_available = True
            self._initialize()
        except Exception:
            self.connection.close()
            raise

    def _initialize(self) -> None:
        _execute_with_lock_retry(self.connection, "BEGIN IMMEDIATE")
        try:
            schema_statements = (
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id)",
                """
                CREATE TABLE IF NOT EXISTS processed_messages (
                    message_id TEXT PRIMARY KEY,
                    processed_at REAL NOT NULL,
                    state TEXT NOT NULL DEFAULT 'completed',
                    claim_token TEXT NOT NULL DEFAULT ''
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS llm_usage_daily (
                    day TEXT NOT NULL,
                    model TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    requests INTEGER NOT NULL,
                    prompt_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL,
                    total_tokens INTEGER NOT NULL,
                    PRIMARY KEY(day, model, purpose)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS rate_limit_buckets (
                    user_hash TEXT NOT NULL,
                    window_seconds INTEGER NOT NULL,
                    bucket_start INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    PRIMARY KEY(user_hash, window_seconds, bucket_start)
                )
                """,
            )
            for statement in schema_statements:
                self.connection.execute(statement)

            document_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "space_id" not in document_columns:
                self.connection.execute(
                    "ALTER TABLE documents ADD COLUMN space_id TEXT NOT NULL DEFAULT ''"
                )
            legacy_space_updates = []
            for row in self.connection.execute(
                "SELECT source_id,space_id FROM documents WHERE space_id = ''"
            ).fetchall():
                match = _FEISHU_SOURCE_RE.match(row["source_id"])
                if match is not None:
                    legacy_space_updates.append((match.group(1), row["source_id"]))
            self.connection.executemany(
                "UPDATE documents SET space_id = ? WHERE source_id = ?",
                legacy_space_updates,
            )

            chunk_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(chunks)").fetchall()
            }
            if "search_text" not in chunk_columns:
                self.connection.execute(
                    "ALTER TABLE chunks ADD COLUMN search_text TEXT NOT NULL DEFAULT ''"
                )

            processed_message_columns = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info(processed_messages)"
                ).fetchall()
            }
            if "state" not in processed_message_columns:
                self.connection.execute(
                    "ALTER TABLE processed_messages "
                    "ADD COLUMN state TEXT NOT NULL DEFAULT 'completed'"
                )
            if "claim_token" not in processed_message_columns:
                self.connection.execute(
                    "ALTER TABLE processed_messages "
                    "ADD COLUMN claim_token TEXT NOT NULL DEFAULT ''"
                )

            fts_savepoint = "initialize_fts"
            self.connection.execute(f"SAVEPOINT {fts_savepoint}")
            try:
                expected_columns = [
                    "chunk_id",
                    "title_terms",
                    "content_terms",
                    "search_terms",
                ]
                existing = self.connection.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'chunks_fts'"
                ).fetchone()
                rebuild = existing is None
                if existing is not None:
                    actual_columns = [
                        row[1]
                        for row in self.connection.execute(
                            "PRAGMA table_info(chunks_fts)"
                        ).fetchall()
                    ]
                    rebuild = actual_columns != expected_columns
                    if not rebuild:
                        chunk_ids = [
                            row[0]
                            for row in self.connection.execute(
                                "SELECT id FROM chunks ORDER BY id"
                            ).fetchall()
                        ]
                        fts_ids = [
                            row[0]
                            for row in self.connection.execute(
                                "SELECT chunk_id FROM chunks_fts ORDER BY chunk_id"
                            ).fetchall()
                        ]
                        rebuild = chunk_ids != fts_ids
                if rebuild and existing is not None:
                    self.connection.execute("DROP TABLE chunks_fts")
                self.connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING "
                    "fts5(chunk_id UNINDEXED, title_terms, content_terms, search_terms)"
                )
                if rebuild:
                    rows = self.connection.execute(
                        "SELECT id,title,content,search_text FROM chunks"
                    ).fetchall()
                    self.connection.executemany(
                        "INSERT INTO chunks_fts("
                        "chunk_id,title_terms,content_terms,search_terms"
                        ") VALUES(?,?,?,?)",
                        (
                            (
                                row["id"],
                                _pretokenize(row["title"]),
                                _pretokenize(row["content"]),
                                _pretokenize(row["search_text"] or ""),
                            )
                            for row in rows
                        ),
                    )
            except sqlite3.OperationalError:
                self.connection.execute(f"ROLLBACK TO SAVEPOINT {fts_savepoint}")
                self.connection.execute(f"RELEASE SAVEPOINT {fts_savepoint}")
                self._fts_available = False
            else:
                self.connection.execute(f"RELEASE SAVEPOINT {fts_savepoint}")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def upsert_document(
        self,
        source_id: str,
        title: str,
        path: str,
        checksum: str,
        chunks: Iterable[Chunk],
        *,
        space_id: str = "",
    ) -> None:
        chunk_list = list(chunks)
        with self.connection:
            old_ids = [
                row[0]
                for row in self.connection.execute(
                    "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
                ).fetchall()
            ]
            if self._fts_available and old_ids:
                self.connection.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", ((cid,) for cid in old_ids))
            self.connection.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
            self.connection.execute(
                "INSERT INTO documents(source_id,title,path,checksum,updated_at,space_id) "
                "VALUES(?,?,?,?,?,?)",
                (source_id, title, path, checksum, time.time(), space_id),
            )
            self.connection.executemany(
                "INSERT INTO chunks(id,source_id,title,content,page,section,search_text) VALUES(?,?,?,?,?,?,?)",
                ((c.id, c.source_id, c.title, c.content, c.page, c.section, c.search_text) for c in chunk_list),
            )
            if self._fts_available:
                self.connection.executemany(
                    "INSERT INTO chunks_fts(chunk_id,title_terms,content_terms,search_terms) "
                    "VALUES(?,?,?,?)",
                    (
                        (
                            c.id,
                            _pretokenize(c.title),
                            _pretokenize(c.content),
                            _pretokenize(c.search_text),
                        )
                        for c in chunk_list
                    ),
                )

    @staticmethod
    def _terms(query: str) -> list[str]:
        return _tokens(query)

    def search(
        self,
        query: str,
        top_k: int = 6,
        min_relevance: float = 0.42,
        scope: RetrievalScope | None = None,
    ) -> list[SearchResult]:
        if top_k < 1:
            return []
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be between 0 and 1")
        allowed_space_ids = None if scope is None else scope.allowed_space_ids
        if allowed_space_ids is not None and not allowed_space_ids:
            return []
        core_terms = _core_terms(query)
        if not core_terms:
            return []
        terms = self._terms(query)
        if not terms:
            return []
        row_sql = (
            "SELECT chunks.id,chunks.source_id,chunks.title,chunks.content,"
            "chunks.page,chunks.section,chunks.search_text FROM chunks "
            "JOIN documents ON documents.source_id = chunks.source_id"
        )
        scope_parameters: tuple[str, ...] = ()
        if allowed_space_ids is not None:
            scope_parameters = tuple(sorted(allowed_space_ids))
            placeholders = ",".join("?" for _ in scope_parameters)
            row_sql += f" WHERE documents.space_id IN ({placeholders})"
        rows = self.connection.execute(row_sql, scope_parameters).fetchall()

        candidate_limit = top_k * 4
        literal_scores: list[tuple[str, float]] = []
        for row in rows:
            title = _normalize(row["title"])
            content = _normalize(row["content"])
            search_text = _normalize(row["search_text"] or "")
            score = 0.0
            for term in terms:
                score += title.count(term) * 3.0
                score += content.count(term)
                score += search_text.count(term) * 0.75
            if score:
                literal_scores.append((row["id"], score))
        literal_scores.sort(key=lambda item: (-item[1], item[0]))
        literal_ids = [chunk_id for chunk_id, _ in literal_scores[:candidate_limit]]

        fts_ids: list[str] = []
        if self._fts_available:
            match_query = " OR ".join(f'"{term}"' for term in terms)
            try:
                fts_sql = (
                    "SELECT chunks_fts.chunk_id FROM chunks_fts "
                    "JOIN chunks ON chunks.id = chunks_fts.chunk_id "
                    "JOIN documents ON documents.source_id = chunks.source_id "
                    "WHERE chunks_fts MATCH ?"
                )
                fts_parameters: tuple[object, ...] = (match_query,)
                if allowed_space_ids is not None:
                    placeholders = ",".join("?" for _ in scope_parameters)
                    fts_sql += f" AND documents.space_id IN ({placeholders})"
                    fts_parameters += scope_parameters
                fts_sql += (
                    " ORDER BY bm25(chunks_fts, 0.0, 3.0, 1.0, 0.75), "
                    "chunks_fts.chunk_id LIMIT ?"
                )
                fts_parameters += (candidate_limit,)
                fts_ids = [
                    row[0]
                    for row in self.connection.execute(
                        fts_sql, fts_parameters
                    ).fetchall()
                ]
            except sqlite3.OperationalError:
                self._fts_available = False

        rrf_scores: Counter[str] = Counter()
        for rank, chunk_id in enumerate(fts_ids, start=1):
            rrf_scores[chunk_id] += 1.0 / (60 + rank)
        for rank, chunk_id in enumerate(literal_ids, start=1):
            rrf_scores[chunk_id] += 0.8 / (60 + rank)
        if not rrf_scores:
            return []

        normalized_documents = [
            set(_tokens(f'{row["title"]} {row["content"]} {row["search_text"] or ""}'))
            for row in rows
        ]
        document_frequency = {
            term: sum(term in document_terms for document_terms in normalized_documents)
            for term in core_terms
        }
        rarity_weights = {
            term: 1.0 / max(document_frequency[term], 1) for term in core_terms
        }
        rarity_total = sum(rarity_weights.values())
        phrase = "".join(_TOKEN_RE.findall(_meaningful_query(query)))
        row_by_id = {row["id"]: row for row in rows}
        results: list[tuple[float, SearchResult]] = []
        for chunk_id, rrf_score in rrf_scores.items():
            row = row_by_id[chunk_id]
            title_terms = set(_tokens(row["title"]))
            all_terms = set(
                _tokens(f'{row["title"]} {row["content"]} {row["search_text"] or ""}')
            )
            matched = [term for term in core_terms if term in all_terms]
            coverage = len(matched) / len(core_terms)
            title_coverage = sum(term in title_terms for term in core_terms) / len(core_terms)
            rarity_coverage = sum(rarity_weights[term] for term in matched) / rarity_total
            searchable_phrase = "".join(
                _TOKEN_RE.findall(
                    _normalize(f'{row["title"]} {row["content"]} {row["search_text"] or ""}')
                )
            )
            phrase_match = bool(phrase and phrase in searchable_phrase)
            confidence = min(
                1.0,
                0.50 * coverage
                + 0.15 * title_coverage
                + 0.20 * float(phrase_match)
                + 0.15 * rarity_coverage,
            )
            if confidence >= min_relevance:
                chunk = Chunk(
                    row["id"],
                    row["source_id"],
                    row["title"],
                    row["content"],
                    row["page"],
                    row["section"],
                    row["search_text"] or "",
                )
                results.append((rrf_score, SearchResult(chunk, confidence)))
        results.sort(key=lambda item: (-item[0], -item[1].score, item[1].chunk.id))
        return [result for _, result in results[:top_k]]

    def count_documents(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def claim_rate_limit(
        self,
        actor_id: str,
        per_minute: int,
        per_day: int,
        now: float | datetime | None = None,
    ) -> bool:
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise ValueError("actor_id must not be empty")
        if any(
            type(limit) is not int or not 0 <= limit <= _SQLITE_INT_MAX
            for limit in (per_minute, per_day)
        ):
            raise ValueError("rate limits must be non-negative integers")
        timestamp = time.time() if now is None else (
            now.timestamp() if isinstance(now, datetime) else float(now)
        )
        if timestamp != timestamp or timestamp in {float("inf"), float("-inf")}:
            raise ValueError("now must be finite")

        actor_hash = sha256(actor_id.strip().encode("utf-8")).hexdigest()
        active_limits = tuple(
            (window_seconds, limit)
            for window_seconds, limit in ((60, per_minute), (86400, per_day))
            if limit > 0
        )
        bucket_starts = {
            window_seconds: int(timestamp // window_seconds) * window_seconds
            for window_seconds, _ in active_limits
        }

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "DELETE FROM rate_limit_buckets "
                "WHERE bucket_start + window_seconds <= ?",
                (int(timestamp),),
            )
            for window_seconds, limit in active_limits:
                row = self.connection.execute(
                    "SELECT count FROM rate_limit_buckets "
                    "WHERE user_hash = ? AND window_seconds = ? AND bucket_start = ?",
                    (actor_hash, window_seconds, bucket_starts[window_seconds]),
                ).fetchone()
                if row is not None and row[0] >= limit:
                    self.connection.commit()
                    return False
            for window_seconds, _ in active_limits:
                self.connection.execute(
                    "INSERT INTO rate_limit_buckets("
                    "user_hash,window_seconds,bucket_start,count"
                    ") VALUES(?,?,?,1) "
                    "ON CONFLICT(user_hash,window_seconds,bucket_start) "
                    "DO UPDATE SET count = count + 1",
                    (actor_hash, window_seconds, bucket_starts[window_seconds]),
                )
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            raise

    def record_llm_usage(
        self,
        model: str,
        purpose: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        *,
        day: str | None = None,
    ) -> None:
        token_counts = (prompt_tokens, completion_tokens, total_tokens)
        if any(
            type(value) is not int or not 0 <= value <= _SQLITE_INT_MAX
            for value in token_counts
        ):
            raise ValueError("token counts must fit a non-negative SQLite integer")
        if not model or not purpose:
            raise ValueError("model and purpose must not be empty")
        usage_day = day or datetime.now(timezone.utc).date().isoformat()
        with self.connection:
            self.connection.execute(
                "INSERT INTO llm_usage_daily("
                "day,model,purpose,requests,prompt_tokens,completion_tokens,total_tokens"
                ") VALUES(?,?,?,1,?,?,?) "
                "ON CONFLICT(day,model,purpose) DO UPDATE SET "
                "requests=CASE WHEN requests > 9223372036854775807-excluded.requests "
                "THEN 9223372036854775807 ELSE requests+excluded.requests END,"
                "prompt_tokens=CASE WHEN prompt_tokens > "
                "9223372036854775807-excluded.prompt_tokens "
                "THEN 9223372036854775807 ELSE prompt_tokens+excluded.prompt_tokens END,"
                "completion_tokens=CASE WHEN completion_tokens > "
                "9223372036854775807-excluded.completion_tokens "
                "THEN 9223372036854775807 "
                "ELSE completion_tokens+excluded.completion_tokens END,"
                "total_tokens=CASE WHEN total_tokens > "
                "9223372036854775807-excluded.total_tokens "
                "THEN 9223372036854775807 ELSE total_tokens+excluded.total_tokens END",
                (
                    usage_day,
                    model,
                    purpose,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                ),
            )

    def query_llm_usage(self, *, day: str | None = None) -> list[dict[str, object]]:
        sql = (
            "SELECT day,model,purpose,requests,prompt_tokens,completion_tokens,total_tokens "
            "FROM llm_usage_daily"
        )
        parameters: tuple[str, ...] = ()
        if day is not None:
            sql += " WHERE day = ?"
            parameters = (day,)
        sql += " ORDER BY day,model,purpose"
        return [dict(row) for row in self.connection.execute(sql, parameters).fetchall()]

    def prune_documents(self, prefix: str, retained: Collection[str]) -> int:
        """删除指定 source_id 前缀下未保留的文档及其检索索引。"""
        if re.fullmatch(r"feishu:[^:]+:", prefix) is None:
            raise ValueError("prefix must identify one Feishu space")
        if any(not source_id.startswith(prefix) for source_id in retained):
            raise ValueError("all retained source_ids must start with prefix")

        with self.connection:
            source_rows = self.connection.execute(
                "SELECT source_id FROM documents WHERE substr(source_id, 1, ?) = ?",
                (len(prefix), prefix),
            ).fetchall()
            stale_ids = [row[0] for row in source_rows if row[0] not in retained]
            if not stale_ids:
                return 0

            for source_id in stale_ids:
                if self._fts_available:
                    chunk_rows = self.connection.execute(
                        "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
                    ).fetchall()
                    self.connection.executemany(
                        "DELETE FROM chunks_fts WHERE chunk_id = ?",
                        ((row[0],) for row in chunk_rows),
                    )
                self.connection.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
        return len(stale_ids)

    def document_checksum(self, source_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT checksum FROM documents WHERE source_id = ?", (source_id,)
        ).fetchone()
        return str(row[0]) if row else None

    def set_document_space(self, source_id: str, space_id: str) -> bool:
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE documents SET space_id = ? WHERE source_id = ?",
                (space_id, source_id),
            )
        return cursor.rowcount == 1

    def claim_message_lease(
        self,
        message_id: str,
        retention_seconds: int = 7 * 24 * 60 * 60,
        in_progress_timeout_seconds: int = 10 * 60,
        *,
        now: float | None = None,
    ) -> tuple[str, str | None]:
        if (
            isinstance(retention_seconds, bool)
            or not isinstance(retention_seconds, (int, float))
            or retention_seconds <= 0
        ):
            raise ValueError("retention_seconds must be positive")
        if (
            isinstance(in_progress_timeout_seconds, bool)
            or not isinstance(in_progress_timeout_seconds, (int, float))
            or in_progress_timeout_seconds <= 0
            or in_progress_timeout_seconds >= retention_seconds
        ):
            raise ValueError(
                "in_progress_timeout_seconds must be positive and less than retention_seconds"
            )
        now = time.time() if now is None else now
        _execute_with_lock_retry(self.connection, "BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "DELETE FROM processed_messages WHERE processed_at < ?",
                (now - retention_seconds,),
            )
            claim_token = uuid.uuid4().hex
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO processed_messages("
                "message_id,processed_at,state,claim_token"
                ") VALUES(?,?,'in_progress',?)",
                (message_id, now, claim_token),
            )
            if cursor.rowcount == 1:
                state = "claimed"
            else:
                claim_token = uuid.uuid4().hex
                cursor = self.connection.execute(
                    "UPDATE processed_messages "
                    "SET processed_at = ?, claim_token = ? "
                    "WHERE message_id = ? AND state = 'in_progress' "
                    "AND processed_at <= ?",
                    (
                        now,
                        claim_token,
                        message_id,
                        now - in_progress_timeout_seconds,
                    ),
                )
                if cursor.rowcount == 1:
                    self.connection.commit()
                    return "claimed", claim_token
                row = self.connection.execute(
                    "SELECT state,claim_token FROM processed_messages WHERE message_id = ?",
                    (message_id,),
                ).fetchone()
                if row is None:
                    raise RuntimeError("message claim state disappeared")
                state = str(row[0])
                claim_token = str(row[1]) or None
            self.connection.commit()
            if state == "claimed":
                return state, claim_token
            if state == "in_progress":
                return state, claim_token
            if state == "completed":
                return state, None
            raise RuntimeError("invalid message claim state")
        except Exception:
            self.connection.rollback()
            raise

    def claim_message_state(
        self,
        message_id: str,
        retention_seconds: int = 7 * 24 * 60 * 60,
        in_progress_timeout_seconds: int = 10 * 60,
        *,
        now: float | None = None,
    ) -> str:
        state, _ = self.claim_message_lease(
            message_id,
            retention_seconds,
            in_progress_timeout_seconds,
            now=now,
        )
        return state

    def claim_message(self, message_id: str, retention_seconds: int = 7 * 24 * 60 * 60) -> bool:
        return self.claim_message_state(message_id, retention_seconds) == "claimed"

    def is_message_claim_owner(self, message_id: str, token: str | None) -> bool:
        if token is None:
            return False
        row = self.connection.execute(
            "SELECT 1 FROM processed_messages "
            "WHERE message_id = ? AND state = 'in_progress' AND claim_token = ?",
            (message_id, token),
        ).fetchone()
        return row is not None

    def complete_message(self, message_id: str, token: str | None = None) -> bool:
        with self.connection:
            if token is None:
                # 仅为旧调用方保留；生产消息处理始终传入租约 token。
                cursor = self.connection.execute(
                    "UPDATE processed_messages "
                    "SET state = 'completed', processed_at = ?, claim_token = '' "
                    "WHERE message_id = ?",
                    (time.time(), message_id),
                )
            else:
                cursor = self.connection.execute(
                    "UPDATE processed_messages "
                    "SET state = 'completed', processed_at = ?, claim_token = '' "
                    "WHERE message_id = ? AND state = 'in_progress' "
                    "AND claim_token = ?",
                    (time.time(), message_id, token),
                )
        return cursor.rowcount == 1

    def release_message(self, message_id: str, token: str | None = None) -> bool:
        with self.connection:
            if token is None:
                # 仅为旧调用方保留；生产消息处理始终传入租约 token。
                cursor = self.connection.execute(
                    "DELETE FROM processed_messages WHERE message_id = ?", (message_id,)
                )
            else:
                cursor = self.connection.execute(
                    "DELETE FROM processed_messages "
                    "WHERE message_id = ? AND state = 'in_progress' "
                    "AND claim_token = ?",
                    (message_id, token),
                )
        return cursor.rowcount == 1

    def count_chunks(self, source_id: str | None = None) -> int:
        if source_id is None:
            return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return int(
            self.connection.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,)).fetchone()[0]
        )

    def close(self) -> None:
        self.connection.close()
