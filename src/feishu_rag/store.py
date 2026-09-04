"""SQLite 文档索引和中文友好的轻量检索。"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import time
import unicodedata
import uuid
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Collection, Iterable

from .models import (
    Chunk,
    FaqMatch,
    FaqObservation,
    PreheatCandidate,
    PreheatJob,
    RetrievalScope,
    SearchResult,
)


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
_LOCAL_SOURCE_RE = re.compile(r"^local:([0-9a-f]{64}):")
_SQLITE_INT_MAX = 2**63 - 1
_LOCK_RETRY_ATTEMPTS = 20
_FAQ_PROMOTION_COUNT = 3
_FAQ_WINDOW_DAYS = 15
_FAQ_METRIC_FIELDS = frozenset(
    {
        "eligible_questions",
        "rag_answers",
        "direct_hits",
        "promotions",
        "refreshes",
        "rejected_answers",
        "invalidations",
    }
)


def faq_window_cutoff(today: date, window_days: int = _FAQ_WINDOW_DAYS) -> str:
    """Return inclusive natural-day cutoff (today plus the preceding days)."""
    if type(window_days) is not int or not 1 <= window_days <= 365:
        raise ValueError("window_days must be between 1 and 365")
    return (today - timedelta(days=window_days - 1)).isoformat()


_FAQ_ENTRIES_SCHEMA = """
CREATE TABLE IF NOT EXISTS faq_entries (
    id TEXT PRIMARY KEY,
    intent_key TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    canonical_question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source_signature TEXT NOT NULL,
    source_ids_json TEXT NOT NULL DEFAULT '[]',
    knowledge_revision INTEGER NOT NULL CHECK(knowledge_revision >= 0),
    state TEXT NOT NULL CHECK(state IN ('enabled', 'stale')),
    direct_hits INTEGER NOT NULL DEFAULT 0 CHECK(direct_hits >= 0),
    created_at REAL NOT NULL CHECK(created_at >= 0),
    updated_at REAL NOT NULL CHECK(updated_at >= 0),
    last_hit_at REAL CHECK(last_hit_at IS NULL OR last_hit_at >= 0),
    UNIQUE(scope_key, intent_key)
)
"""
_FAQ_ALIASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS faq_aliases (
    faq_id TEXT NOT NULL REFERENCES faq_entries(id) ON DELETE CASCADE,
    normalized_question TEXT NOT NULL,
    search_text TEXT NOT NULL,
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    total_seen INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(faq_id, normalized_question)
)
"""
_FAQ_OBSERVATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS faq_observation_daily (
    intent_key TEXT NOT NULL,
    scope_key TEXT NOT NULL,
    day TEXT NOT NULL,
    count INTEGER NOT NULL DEFAULT 0,
    normalized_question TEXT NOT NULL,
    source_signature TEXT NOT NULL,
    knowledge_revision INTEGER NOT NULL,
    latest_safe_answer TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(
        scope_key,
        intent_key,
        normalized_question,
        source_signature,
        knowledge_revision,
        day
    )
)
"""


@dataclass(frozen=True)
class PreparedDocument:
    source_id: str
    title: str
    path: str
    checksum: str
    chunks: tuple[Chunk, ...] | None
    space_id: str = ""
    document_code: str = ""
    document_version: str = ""
    effective_date: str = ""
    lifecycle_state: str = "current"
    parser_version: str = ""
    decision_reason: str = "unique-or-unversioned"


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
                """
                CREATE TABLE IF NOT EXISTS knowledge_state (
                    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
                    revision INTEGER NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS local_index_roots (
                    root_key TEXT PRIMARY KEY,
                    root_path TEXT NOT NULL,
                    registered_at REAL NOT NULL,
                    last_snapshot_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS document_versions (
                    source_id TEXT PRIMARY KEY
                        REFERENCES documents(source_id) ON DELETE CASCADE,
                    document_code TEXT NOT NULL,
                    document_version TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL
                        CHECK(lifecycle_state IN ('current','superseded','conflict')),
                    decision_reason TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS faq_preheat_jobs (
                    id TEXT PRIMARY KEY,
                    scope_key TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL CHECK(knowledge_revision >= 0),
                    state TEXT NOT NULL CHECK(state IN ('queued','running','completed','failed')),
                    retry_count INTEGER NOT NULL DEFAULT 0 CHECK(retry_count >= 0),
                    max_retries INTEGER NOT NULL CHECK(max_retries BETWEEN 0 AND 2),
                    created_at REAL NOT NULL CHECK(created_at >= 0),
                    started_at REAL,
                    lease_expires_at REAL,
                    finished_at REAL,
                    candidate_count INTEGER NOT NULL DEFAULT 0 CHECK(candidate_count >= 0),
                    generated_count INTEGER NOT NULL DEFAULT 0 CHECK(generated_count >= 0),
                    failed_count INTEGER NOT NULL DEFAULT 0 CHECK(failed_count >= 0),
                    UNIQUE(scope_key, knowledge_revision)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS faq_preheat_candidates (
                    candidate_signature TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL REFERENCES faq_preheat_jobs(id) ON DELETE CASCADE,
                    scope_key TEXT NOT NULL,
                    knowledge_revision INTEGER NOT NULL CHECK(knowledge_revision >= 0),
                    chunk_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('selected','generated','rejected','failed')),
                    faq_id TEXT,
                    created_at REAL NOT NULL CHECK(created_at >= 0),
                    updated_at REAL NOT NULL CHECK(updated_at >= 0)
                )
                """,
                _FAQ_ENTRIES_SCHEMA,
                _FAQ_ALIASES_SCHEMA,
                _FAQ_OBSERVATION_SCHEMA,
                """
                CREATE TABLE IF NOT EXISTS faq_metrics_daily (
                    scope_key TEXT NOT NULL,
                    day TEXT NOT NULL,
                    eligible_questions INTEGER NOT NULL DEFAULT 0,
                    rag_answers INTEGER NOT NULL DEFAULT 0,
                    direct_hits INTEGER NOT NULL DEFAULT 0,
                    promotions INTEGER NOT NULL DEFAULT 0,
                    refreshes INTEGER NOT NULL DEFAULT 0,
                    rejected_answers INTEGER NOT NULL DEFAULT 0,
                    invalidations INTEGER NOT NULL DEFAULT 0 CHECK(invalidations >= 0),
                    PRIMARY KEY(scope_key, day)
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_faq_entries_scope_state ON faq_entries(scope_key, state)",
                "CREATE INDEX IF NOT EXISTS idx_faq_aliases_normalized_question ON faq_aliases(normalized_question)",
                "CREATE INDEX IF NOT EXISTS idx_faq_observation_daily_day ON faq_observation_daily(day)",
                "CREATE INDEX IF NOT EXISTS idx_faq_metrics_daily_day ON faq_metrics_daily(day)",
            )
            for statement in schema_statements:
                self.connection.execute(statement)
            self.connection.execute(
                "INSERT OR IGNORE INTO knowledge_state(singleton_id,revision,updated_at) "
                "VALUES(1,0,0)"
            )
            metric_columns = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info(faq_metrics_daily)"
                ).fetchall()
            }
            if "invalidations" not in metric_columns:
                self.connection.execute(
                    "ALTER TABLE faq_metrics_daily ADD COLUMN invalidations "
                    "INTEGER NOT NULL DEFAULT 0 CHECK(invalidations >= 0)"
                )
            self._migrate_faq_entries_constraints()
            self._migrate_faq_observation_constraints()
            faq_entry_columns = {
                row[1]
                for row in self.connection.execute(
                    "PRAGMA table_info(faq_entries)"
                ).fetchall()
            }
            if "origin" not in faq_entry_columns:
                self.connection.execute(
                    "ALTER TABLE faq_entries ADD COLUMN origin TEXT NOT NULL "
                    "DEFAULT 'observed' CHECK(origin IN ('observed','preheated'))"
                )
            if "preheat_candidate_signature" not in faq_entry_columns:
                self.connection.execute(
                    "ALTER TABLE faq_entries ADD COLUMN preheat_candidate_signature "
                    "TEXT NOT NULL DEFAULT ''"
                )

            document_columns = {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(documents)").fetchall()
            }
            if "space_id" not in document_columns:
                self.connection.execute(
                    "ALTER TABLE documents ADD COLUMN space_id TEXT NOT NULL DEFAULT ''"
                )
            document_column_migrations = {
                "document_code": "TEXT NOT NULL DEFAULT ''",
                "document_version": "TEXT NOT NULL DEFAULT ''",
                "effective_date": "TEXT NOT NULL DEFAULT ''",
                "lifecycle_state": "TEXT NOT NULL DEFAULT 'current'",
                "parser_version": "TEXT NOT NULL DEFAULT ''",
            }
            for column, declaration in document_column_migrations.items():
                if column not in document_columns:
                    self.connection.execute(
                        f"ALTER TABLE documents ADD COLUMN {column} {declaration}"
                    )
            self.connection.execute(
                "INSERT OR IGNORE INTO document_versions("
                "source_id,document_code,document_version,lifecycle_state,"
                "decision_reason,updated_at) "
                "SELECT source_id,document_code,document_version,lifecycle_state,"
                "'legacy-migration',updated_at FROM documents"
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
        document_code: str = "",
        document_version: str = "",
        effective_date: str = "",
        lifecycle_state: str = "current",
        parser_version: str = "",
        decision_reason: str = "unique-or-unversioned",
    ) -> None:
        chunk_list = list(chunks)
        with self.connection:
            self._upsert_document_in_transaction(
                source_id,
                title,
                path,
                checksum,
                chunk_list,
                space_id=space_id,
                document_code=document_code,
                document_version=document_version,
                effective_date=effective_date,
                lifecycle_state=lifecycle_state,
                parser_version=parser_version,
                decision_reason=decision_reason,
            )

    def _upsert_document_in_transaction(
        self,
        source_id: str,
        title: str,
        path: str,
        checksum: str,
        chunks: Iterable[Chunk],
        *,
        space_id: str = "",
        document_code: str = "",
        document_version: str = "",
        effective_date: str = "",
        lifecycle_state: str = "current",
        parser_version: str = "",
        decision_reason: str = "unique-or-unversioned",
    ) -> None:
        chunk_list = list(chunks)
        old_ids = [
            row[0]
            for row in self.connection.execute(
                "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
            ).fetchall()
        ]
        if self._fts_available and old_ids:
            self.connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?", ((cid,) for cid in old_ids)
            )
        self.connection.execute("DELETE FROM documents WHERE source_id = ?", (source_id,))
        timestamp = time.time()
        self.connection.execute(
            "INSERT INTO documents("
            "source_id,title,path,checksum,updated_at,space_id,document_code,"
            "document_version,effective_date,lifecycle_state,parser_version"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                source_id,
                title,
                path,
                checksum,
                timestamp,
                space_id,
                document_code,
                document_version,
                effective_date,
                lifecycle_state,
                parser_version,
            ),
        )
        self.connection.execute(
            "INSERT INTO document_versions("
            "source_id,document_code,document_version,lifecycle_state,"
            "decision_reason,updated_at) VALUES(?,?,?,?,?,?)",
            (
                source_id,
                document_code,
                document_version,
                lifecycle_state,
                decision_reason,
                timestamp,
            ),
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

    def apply_document_snapshot(
        self,
        prepared_updates: Iterable[PreparedDocument],
        *,
        prune_prefix: str | None = None,
        retained: Collection[str] | None = None,
        local_root: str | Path | None = None,
    ) -> tuple[int, int]:
        updates = tuple(prepared_updates)
        for prepared in updates:
            if prepared.lifecycle_state not in {"current", "superseded", "conflict"}:
                raise ValueError(
                    "lifecycle_state must be current, superseded or conflict"
                )
        local_root_hash: str | None = None
        if local_root is not None:
            if prune_prefix is not None:
                raise ValueError("local_root cannot be combined with prune_prefix")
            if retained is None:
                raise ValueError("retained is required when local_root is provided")
            resolved_root = str(Path(local_root).resolve())
            local_root_hash = sha256(resolved_root.encode("utf-8")).hexdigest()
            prune_prefix = f"local:{local_root_hash}:"
        if prune_prefix is not None:
            is_feishu_prefix = re.fullmatch(r"feishu:[^:]+:", prune_prefix) is not None
            is_local_prefix = _LOCAL_SOURCE_RE.fullmatch(prune_prefix) is not None
            if not (is_feishu_prefix or is_local_prefix):
                raise ValueError("prefix must identify one Feishu space or local root")
            if retained is None:
                raise ValueError("retained is required when prune_prefix is provided")
            if any(not source_id.startswith(prune_prefix) for source_id in retained):
                raise ValueError("all retained source_ids must start with prefix")

        self.connection.execute("BEGIN IMMEDIATE")
        try:
            updated = 0
            for prepared in updates:
                row = self.connection.execute(
                    "SELECT d.checksum,d.space_id,d.document_code,d.document_version,"
                    "d.effective_date,d.lifecycle_state,d.parser_version,"
                    "COALESCE(v.decision_reason,'') AS decision_reason "
                    "FROM documents d LEFT JOIN document_versions v "
                    "ON v.source_id=d.source_id WHERE d.source_id = ?",
                    (prepared.source_id,),
                ).fetchone()
                if row is not None and row[0] == prepared.checksum:
                    if prepared.space_id and row[1] != prepared.space_id:
                        self.connection.execute(
                            "UPDATE documents SET space_id = ? WHERE source_id = ?",
                            (prepared.space_id, prepared.source_id),
                        )
                    stored_metadata = tuple(row[index] for index in range(2, 8))
                    desired_metadata = (
                        prepared.document_code,
                        prepared.document_version,
                        prepared.effective_date,
                        prepared.lifecycle_state,
                        prepared.parser_version,
                        prepared.decision_reason,
                    )
                    if stored_metadata != desired_metadata:
                        timestamp = time.time()
                        self.connection.execute(
                            "UPDATE documents SET document_code=?,document_version=?,"
                            "effective_date=?,lifecycle_state=?,parser_version=?,"
                            "updated_at=? WHERE source_id=?",
                            (*desired_metadata[:5], timestamp, prepared.source_id),
                        )
                        self.connection.execute(
                            "INSERT INTO document_versions("
                            "source_id,document_code,document_version,lifecycle_state,"
                            "decision_reason,updated_at) VALUES(?,?,?,?,?,?) "
                            "ON CONFLICT(source_id) DO UPDATE SET "
                            "document_code=excluded.document_code,"
                            "document_version=excluded.document_version,"
                            "lifecycle_state=excluded.lifecycle_state,"
                            "decision_reason=excluded.decision_reason,"
                            "updated_at=excluded.updated_at",
                            (
                                prepared.source_id,
                                prepared.document_code,
                                prepared.document_version,
                                prepared.lifecycle_state,
                                prepared.decision_reason,
                                timestamp,
                            ),
                        )
                        updated += 1
                    continue
                if prepared.chunks is None:
                    raise ValueError("new or changed documents require chunks")
                self._upsert_document_in_transaction(
                    prepared.source_id,
                    prepared.title,
                    prepared.path,
                    prepared.checksum,
                    prepared.chunks,
                    space_id=prepared.space_id,
                    document_code=prepared.document_code,
                    document_version=prepared.document_version,
                    effective_date=prepared.effective_date,
                    lifecycle_state=prepared.lifecycle_state,
                    parser_version=prepared.parser_version,
                    decision_reason=prepared.decision_reason,
                )
                updated += 1

            deleted = 0
            if prune_prefix is not None:
                deleted = self._prune_documents_in_transaction(prune_prefix, retained or ())
            if local_root_hash is not None:
                any_registry = self.connection.execute(
                    "SELECT 1 FROM local_index_roots LIMIT 1",
                ).fetchone()
                registry = self.connection.execute(
                    "SELECT 1 FROM local_index_roots WHERE root_key = ?",
                    (local_root_hash,),
                ).fetchone()
                resolved_root = str(Path(local_root).resolve())
                timestamp = time.time()
                if any_registry is None:
                    legacy_rows = self.connection.execute(
                        "SELECT source_id FROM documents"
                    ).fetchall()
                    legacy_ids = [
                        str(row[0])
                        for row in legacy_rows
                        if not str(row[0]).startswith(("feishu:", "local:"))
                    ]
                    for source_id in legacy_ids:
                        chunk_rows = self.connection.execute(
                            "SELECT id FROM chunks WHERE source_id = ?", (source_id,)
                        ).fetchall()
                        if self._fts_available:
                            self.connection.executemany(
                                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                                ((row[0],) for row in chunk_rows),
                            )
                        self.connection.execute(
                            "DELETE FROM documents WHERE source_id = ?", (source_id,)
                        )
                    deleted += len(legacy_ids)
                    self.connection.execute(
                        "INSERT INTO local_index_roots(root_key,root_path,registered_at,last_snapshot_at) "
                        "VALUES(?,?,?,?)",
                        (local_root_hash, resolved_root, timestamp, timestamp),
                    )
                elif registry is None:
                    self.connection.execute(
                        "INSERT INTO local_index_roots(root_key,root_path,registered_at,last_snapshot_at) "
                        "VALUES(?,?,?,?)",
                        (local_root_hash, resolved_root, timestamp, timestamp),
                    )
                else:
                    self.connection.execute(
                        "UPDATE local_index_roots SET last_snapshot_at = ? WHERE root_key = ?",
                        (timestamp, local_root_hash),
                    )
            if updated or deleted:
                revision_row = self.connection.execute(
                    "UPDATE knowledge_state SET revision = revision + 1, updated_at = ? "
                    "WHERE singleton_id = 1 RETURNING revision",
                    (time.time(),),
                ).fetchone()
                self._invalidate_faqs_in_transaction(int(revision_row[0]), time.time())
            self.connection.commit()
            return updated, deleted
        except Exception:
            self.connection.rollback()
            raise

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
        conditions = ["documents.lifecycle_state != 'superseded'"]
        scope_parameters: tuple[str, ...] = ()
        if allowed_space_ids is not None:
            scope_parameters = tuple(sorted(allowed_space_ids))
            placeholders = ",".join("?" for _ in scope_parameters)
            conditions.append(f"documents.space_id IN ({placeholders})")
        row_sql += " WHERE " + " AND ".join(conditions)
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
                    "WHERE chunks_fts MATCH ? "
                    "AND documents.lifecycle_state != 'superseded'"
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

    def document_lifecycle_counts(self) -> dict[str, int]:
        counts = {"current": 0, "superseded": 0, "conflict": 0}
        for row in self.connection.execute(
            "SELECT lifecycle_state,COUNT(*) AS count FROM documents "
            "GROUP BY lifecycle_state"
        ).fetchall():
            state = str(row["lifecycle_state"])
            if state in counts:
                counts[state] = int(row["count"])
        return counts

    def enqueue_preheat_job(
        self,
        scope_key: str,
        knowledge_revision: int,
        *,
        max_retries: int = 1,
        now: float | None = None,
    ) -> bool:
        if not isinstance(scope_key, str) or not scope_key.strip():
            raise ValueError("scope_key must not be empty")
        if type(knowledge_revision) is not int or knowledge_revision < 0:
            raise ValueError("knowledge_revision must be a non-negative integer")
        if type(max_retries) is not int or not 0 <= max_retries <= 2:
            raise ValueError("max_retries must be between 0 and 2")
        timestamp = self._validate_faq_now(time.time() if now is None else now)
        with self.connection:
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO faq_preheat_jobs("
                "id,scope_key,knowledge_revision,state,retry_count,max_retries,"
                "created_at) VALUES(?,?,?,'queued',0,?,?)",
                (
                    uuid.uuid4().hex,
                    scope_key.strip(),
                    knowledge_revision,
                    max_retries,
                    timestamp,
                ),
            )
        return cursor.rowcount == 1

    def claim_preheat_job(
        self,
        *,
        now: float | None = None,
        lease_seconds: float = 600.0,
    ) -> PreheatJob | None:
        timestamp = self._validate_faq_now(time.time() if now is None else now)
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        _execute_with_lock_retry(self.connection, "BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "UPDATE faq_preheat_jobs SET state='queued',started_at=NULL,"
                "lease_expires_at=NULL WHERE state='running' "
                "AND lease_expires_at IS NOT NULL AND lease_expires_at <= ? "
                "AND retry_count <= max_retries",
                (timestamp,),
            )
            row = self.connection.execute(
                "SELECT id,scope_key,knowledge_revision,retry_count "
                "FROM faq_preheat_jobs WHERE state='queued' "
                "ORDER BY created_at,id LIMIT 1"
            ).fetchone()
            if row is None:
                self.connection.commit()
                return None
            self.connection.execute(
                "UPDATE faq_preheat_jobs SET state='running',started_at=?,"
                "lease_expires_at=? WHERE id=? AND state='queued'",
                (timestamp, timestamp + lease_seconds, row["id"]),
            )
            self.connection.commit()
            return PreheatJob(
                str(row["id"]),
                str(row["scope_key"]),
                int(row["knowledge_revision"]),
                int(row["retry_count"]),
            )
        except Exception:
            self.connection.rollback()
            raise

    def complete_preheat_job(
        self,
        job_id: str,
        *,
        generated: int,
        failed: int,
        candidate_count: int | None = None,
        now: float | None = None,
    ) -> None:
        if any(type(value) is not int or value < 0 for value in (generated, failed)):
            raise ValueError("generated and failed must be non-negative integers")
        total = generated + failed if candidate_count is None else candidate_count
        if type(total) is not int or total < generated + failed:
            raise ValueError("candidate_count must cover generated and failed")
        timestamp = self._validate_faq_now(time.time() if now is None else now)
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE faq_preheat_jobs SET state='completed',finished_at=?,"
                "lease_expires_at=NULL,candidate_count=?,generated_count=?,"
                "failed_count=? WHERE id=? AND state='running'",
                (timestamp, total, generated, failed, job_id),
            )
        if cursor.rowcount != 1:
            raise ValueError("preheat job is not running")

    def fail_preheat_job(
        self,
        job_id: str,
        *,
        now: float | None = None,
    ) -> bool:
        timestamp = self._validate_faq_now(time.time() if now is None else now)
        _execute_with_lock_retry(self.connection, "BEGIN IMMEDIATE")
        try:
            row = self.connection.execute(
                "SELECT retry_count,max_retries FROM faq_preheat_jobs "
                "WHERE id=? AND state='running'",
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError("preheat job is not running")
            retry_count = int(row["retry_count"]) + 1
            queued = retry_count <= int(row["max_retries"])
            self.connection.execute(
                "UPDATE faq_preheat_jobs SET state=?,retry_count=?,"
                "started_at=NULL,lease_expires_at=NULL,finished_at=? WHERE id=?",
                (
                    "queued" if queued else "failed",
                    retry_count,
                    None if queued else timestamp,
                    job_id,
                ),
            )
            self.connection.commit()
            return queued
        except Exception:
            self.connection.rollback()
            raise

    def preheat_chunks(self, scope_key: str) -> list[Chunk]:
        if not isinstance(scope_key, str) or not scope_key.strip():
            raise ValueError("scope_key must not be empty")
        rows = self.connection.execute(
            "SELECT c.id,c.source_id,c.title,c.content,c.page,c.section,c.search_text "
            "FROM chunks c JOIN documents d ON d.source_id=c.source_id "
            "WHERE d.space_id=? AND d.lifecycle_state!='superseded' "
            "ORDER BY c.source_id,c.id",
            (scope_key,),
        ).fetchall()
        return [
            Chunk(
                str(row["id"]),
                str(row["source_id"]),
                str(row["title"]),
                str(row["content"]),
                row["page"],
                row["section"],
                str(row["search_text"] or ""),
            )
            for row in rows
        ]

    def preheat_candidate_exists(self, signature: str) -> bool:
        return self.connection.execute(
            "SELECT 1 FROM faq_preheat_candidates WHERE candidate_signature=? "
            "AND state='generated'",
            (signature,),
        ).fetchone() is not None

    def record_preheat_candidate(
        self,
        job_id: str,
        candidate: PreheatCandidate,
        state: str,
        *,
        faq_id: str = "",
        now: float | None = None,
    ) -> None:
        if state not in {"selected", "generated", "rejected", "failed"}:
            raise ValueError("invalid preheat candidate state")
        timestamp = self._validate_faq_now(time.time() if now is None else now)
        with self.connection:
            self.connection.execute(
                "INSERT INTO faq_preheat_candidates("
                "candidate_signature,job_id,scope_key,knowledge_revision,chunk_id,"
                "source_id,score,state,faq_id,created_at,updated_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(candidate_signature) DO UPDATE SET "
                "state=excluded.state,faq_id=excluded.faq_id,updated_at=excluded.updated_at",
                (
                    candidate.signature,
                    job_id,
                    candidate.scope_key,
                    candidate.knowledge_revision,
                    candidate.chunk_id,
                    candidate.source_id,
                    candidate.score,
                    state,
                    faq_id,
                    timestamp,
                    timestamp,
                ),
            )

    def upsert_preheated_faq(
        self,
        candidate: PreheatCandidate,
        observation: FaqObservation,
        *,
        canonical_question: str,
        aliases: Iterable[str],
        answer: str,
        now: float | None = None,
    ) -> FaqMatch | None:
        self._validate_faq_observation(observation)
        safe_answer = self._faq_answer(answer)
        question = self._faq_question(observation)
        timestamp = self._validate_faq_now(time.time() if now is None else now)
        alias_values = list(dict.fromkeys([question, *(_normalize(value) for value in aliases)]))
        if any(not value for value in alias_values):
            raise ValueError("aliases must not be empty")
        self._begin_faq_transaction()
        try:
            current_revision = self.connection.execute(
                "SELECT revision FROM knowledge_state WHERE singleton_id=1"
            ).fetchone()[0]
            if int(current_revision) != candidate.knowledge_revision:
                self.connection.commit()
                return None
            source_ids_json = self._faq_source_ids_json(observation)
            existing = self.connection.execute(
                "SELECT id FROM faq_entries WHERE scope_key=? AND intent_key=?",
                (observation.scope_key, observation.intent_key),
            ).fetchone()
            entry_id = str(existing["id"]) if existing is not None else uuid.uuid4().hex
            if existing is None:
                self.connection.execute(
                    "INSERT INTO faq_entries("
                    "id,intent_key,scope_key,canonical_question,answer,source_signature,"
                    "source_ids_json,knowledge_revision,state,direct_hits,created_at,"
                    "updated_at,last_hit_at,origin,preheat_candidate_signature"
                    ") VALUES(?,?,?,?,?,?,?,?,'enabled',0,?,?,NULL,'preheated',?)",
                    (
                        entry_id,
                        observation.intent_key,
                        observation.scope_key,
                        canonical_question.strip(),
                        safe_answer,
                        observation.source_signature,
                        source_ids_json,
                        observation.knowledge_revision,
                        timestamp,
                        timestamp,
                        candidate.signature,
                    ),
                )
            else:
                self.connection.execute(
                    "UPDATE faq_entries SET canonical_question=?,answer=?,"
                    "source_signature=?,source_ids_json=?,knowledge_revision=?,"
                    "state='enabled',updated_at=?,origin='preheated',"
                    "preheat_candidate_signature=? WHERE id=?",
                    (
                        canonical_question.strip(),
                        safe_answer,
                        observation.source_signature,
                        source_ids_json,
                        observation.knowledge_revision,
                        timestamp,
                        candidate.signature,
                        entry_id,
                    ),
                )
                self.connection.execute(
                    "DELETE FROM faq_aliases WHERE faq_id=?",
                    (entry_id,),
                )
            self.connection.executemany(
                "INSERT INTO faq_aliases("
                "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
                ") VALUES(?,?,?,?,?,1)",
                (
                    (
                        entry_id,
                        alias,
                        _pretokenize(alias),
                        timestamp,
                        timestamp,
                    )
                    for alias in alias_values
                ),
            )
            self.connection.commit()
            return FaqMatch(
                entry_id,
                safe_answer,
                observation.intent_key,
                observation.knowledge_revision,
            )
        except Exception:
            self.connection.rollback()
            raise

    def knowledge_revision(self) -> int:
        return int(
            self.connection.execute(
                "SELECT revision FROM knowledge_state WHERE singleton_id = 1"
            ).fetchone()[0]
        )

    def bump_knowledge_revision(self, now: float | None = None) -> int:
        timestamp = time.time() if now is None else float(now)
        try:
            row = self.connection.execute(
                "UPDATE knowledge_state "
                "SET revision = revision + 1, updated_at = ? "
                "WHERE singleton_id = 1 RETURNING revision",
                (timestamp,),
            ).fetchone()
            if row is None:
                raise RuntimeError("knowledge state is not initialized")
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return int(row[0])

    @staticmethod
    def _validate_faq_day(day: str) -> date:
        if not isinstance(day, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) is None:
            raise ValueError("day must be an ISO date")
        try:
            parsed = date.fromisoformat(day)
        except ValueError as exc:
            raise ValueError("day must be an ISO date") from exc
        if parsed.isoformat() != day:
            raise ValueError("day must be an ISO date")
        return parsed

    @staticmethod
    def _validate_faq_now(now: float) -> float:
        try:
            timestamp = float(now)
        except (TypeError, ValueError) as exc:
            raise ValueError("now must be finite and non-negative") from exc
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("now must be finite and non-negative")
        return timestamp

    @staticmethod
    def _validate_faq_observation(observation: FaqObservation) -> None:
        if not isinstance(observation, FaqObservation):
            raise ValueError("observation must be a FaqObservation")
        for field in ("intent_key", "scope_key", "source_signature"):
            value = getattr(observation, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must not be empty")
        if (
            not isinstance(observation.normalized_question, str)
            or not observation.normalized_question.strip()
        ):
            raise ValueError("normalized_question must not be empty")
        if (
            isinstance(observation.knowledge_revision, bool)
            or not isinstance(observation.knowledge_revision, int)
            or observation.knowledge_revision < 0
        ):
            raise ValueError("knowledge_revision must be a non-negative integer")
        if not isinstance(observation.source_ids, tuple) or any(
            not isinstance(source_id, str) or not source_id.strip()
            for source_id in observation.source_ids
        ):
            raise ValueError("source_ids must be a tuple of non-empty strings")

    @staticmethod
    def _faq_source_ids_json(observation: FaqObservation) -> str:
        return json.dumps(
            sorted(set(observation.source_ids)),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _faq_question(observation: FaqObservation) -> str:
        question = _normalize(observation.normalized_question)
        if not question:
            raise ValueError("normalized_question must not be empty")
        return question

    @staticmethod
    def _faq_answer(answer: str) -> str:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must not be empty")
        return answer

    def _begin_faq_transaction(self) -> None:
        if not self.connection.in_transaction:
            _execute_with_lock_retry(self.connection, "BEGIN IMMEDIATE")

    def find_faq_candidates(self, scope_key: str) -> list[sqlite3.Row]:
        """Return all FAQ aliases belonging to one access scope."""
        if not isinstance(scope_key, str) or not scope_key.strip():
            raise ValueError("scope_key must not be empty")
        return list(
            self.connection.execute(
                "SELECT e.id, e.intent_key, e.scope_key, e.canonical_question, "
                "e.answer, e.source_signature, e.source_ids_json, e.knowledge_revision, e.state, "
                "e.direct_hits, e.created_at, e.updated_at, e.last_hit_at, "
                "a.faq_id, a.normalized_question, a.search_text, a.first_seen_at, "
                "a.last_seen_at, a.total_seen "
                "FROM faq_entries AS e "
                "JOIN faq_aliases AS a ON a.faq_id = e.id "
                "WHERE e.scope_key = ? "
                "ORDER BY e.id, a.normalized_question",
                (scope_key,),
            ).fetchall()
        )

    def record_faq_observation(
        self,
        observation: FaqObservation,
        *,
        answer: str,
        day: str,
        now: float,
        promotion_count: int,
        window_days: int = _FAQ_WINDOW_DAYS,
    ) -> FaqMatch | None:
        """Atomically record an observation and promote at the configured threshold."""
        self._validate_faq_observation(observation)
        safe_answer = self._faq_answer(answer)
        current_day = self._validate_faq_day(day)
        timestamp = self._validate_faq_now(now)
        question = self._faq_question(observation)
        source_ids_json = self._faq_source_ids_json(observation)
        if (
            isinstance(promotion_count, bool)
            or not isinstance(promotion_count, int)
            or not 1 <= promotion_count <= 100
        ):
            raise ValueError("promotion_count must be an integer from 1 to 100")
        if (
            isinstance(window_days, bool)
            or not isinstance(window_days, int)
            or window_days <= 0
        ):
            raise ValueError("window_days must be a positive integer")
        first_day = (current_day - timedelta(days=window_days - 1)).isoformat()

        self._begin_faq_transaction()
        try:
            current_revision_row = self.connection.execute(
                "SELECT revision FROM knowledge_state WHERE singleton_id = 1"
            ).fetchone()
            if current_revision_row is None:
                raise RuntimeError("knowledge state is not initialized")
            if observation.knowledge_revision != current_revision_row[0]:
                self.connection.commit()
                return None
            self.connection.execute(
                "INSERT INTO faq_observation_daily("
                "intent_key,scope_key,day,count,normalized_question,source_signature,"
                "knowledge_revision,latest_safe_answer"
                ") VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(scope_key,intent_key,normalized_question,source_signature,knowledge_revision,day) "
                "DO UPDATE SET count = count + 1, "
                "latest_safe_answer = excluded.latest_safe_answer",
                (
                    observation.intent_key,
                    observation.scope_key,
                    day,
                    1,
                    question,
                    observation.source_signature,
                    observation.knowledge_revision,
                    safe_answer,
                ),
            )
            total = self.connection.execute(
                "SELECT COALESCE(SUM(count), 0) FROM faq_observation_daily "
                "WHERE scope_key = ? AND intent_key = ? AND source_signature = ? "
                "AND knowledge_revision = ? AND day >= ? AND day <= ?",
                (
                    observation.scope_key,
                    observation.intent_key,
                    observation.source_signature,
                    observation.knowledge_revision,
                    first_day,
                    day,
                ),
            ).fetchone()[0]

            existing = self.connection.execute(
                "SELECT * FROM faq_entries WHERE scope_key = ? AND intent_key = ?",
                (observation.scope_key, observation.intent_key),
            ).fetchone()
            if existing is not None:
                if (
                    existing["state"] == "stale"
                    or existing["knowledge_revision"] != observation.knowledge_revision
                ):
                    self.connection.execute(
                        "UPDATE faq_entries SET canonical_question = ?, answer = ?, "
                        "source_signature = ?, source_ids_json = ?, knowledge_revision = ?, state = 'enabled', "
                        "updated_at = ? WHERE id = ?",
                        (
                            question,
                            safe_answer,
                            observation.source_signature,
                            source_ids_json,
                            observation.knowledge_revision,
                            timestamp,
                            existing["id"],
                        ),
                    )
                    self.connection.execute(
                        "INSERT INTO faq_aliases("
                        "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
                        ") VALUES(?,?,?,?,?,1) ON CONFLICT(faq_id,normalized_question) DO UPDATE SET "
                        "search_text = excluded.search_text, last_seen_at = excluded.last_seen_at, "
                        "total_seen = total_seen + 1",
                        (existing["id"], question, _pretokenize(question), timestamp, timestamp),
                    )
                    self.connection.execute(
                        "INSERT INTO faq_metrics_daily(scope_key,day,refreshes) VALUES(?,?,1) "
                        "ON CONFLICT(scope_key,day) DO UPDATE SET refreshes = refreshes + 1",
                        (observation.scope_key, day),
                    )
                    self.connection.commit()
                    return FaqMatch(
                        existing["id"], safe_answer, observation.intent_key,
                        observation.knowledge_revision,
                    )
                if (
                    existing["state"] == "enabled"
                    and existing["source_signature"] == observation.source_signature
                    and existing["knowledge_revision"] == observation.knowledge_revision
                ):
                    self.connection.execute(
                        "INSERT INTO faq_aliases("
                        "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
                        ") VALUES(?,?,?,?,?,1) ON CONFLICT(faq_id,normalized_question) DO UPDATE SET "
                        "search_text = excluded.search_text, last_seen_at = excluded.last_seen_at, "
                        "total_seen = total_seen + 1",
                        (existing["id"], question, _pretokenize(question), timestamp, timestamp),
                    )
                self.connection.commit()
                return None

            if total < promotion_count:
                self.connection.commit()
                return None

            entry_id = str(uuid.uuid4())
            self.connection.execute(
                "INSERT INTO faq_entries("
                "id,intent_key,scope_key,canonical_question,answer,source_signature,source_ids_json,"
                "knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at"
                ") VALUES(?,?,?,?,?,?,?,?,'enabled',0,?,?,NULL)",
                (
                    entry_id,
                    observation.intent_key,
                    observation.scope_key,
                    question,
                    safe_answer,
                    observation.source_signature,
                    source_ids_json,
                    observation.knowledge_revision,
                    timestamp,
                    timestamp,
                ),
            )
            rows = self.connection.execute(
                "SELECT normalized_question, SUM(count) AS total_seen "
                "FROM faq_observation_daily WHERE scope_key = ? AND intent_key = ? "
                "AND source_signature = ? AND knowledge_revision = ? "
                "AND day >= ? AND day <= ? GROUP BY normalized_question",
                (
                    observation.scope_key,
                    observation.intent_key,
                    observation.source_signature,
                    observation.knowledge_revision,
                    first_day,
                    day,
                ),
            ).fetchall()
            for row in rows:
                alias_question = row["normalized_question"]
                self.connection.execute(
                    "INSERT INTO faq_aliases("
                    "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
                    ") VALUES(?,?,?,?,?,?)",
                    (
                        entry_id,
                        alias_question,
                        _pretokenize(alias_question),
                        timestamp,
                        timestamp,
                        int(row["total_seen"]),
                    ),
                )
            self.connection.execute(
                "INSERT INTO faq_metrics_daily(scope_key,day,promotions) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET promotions = promotions + 1",
                (observation.scope_key, day),
            )
            self.connection.commit()
            return FaqMatch(
                entry_id, safe_answer, observation.intent_key,
                observation.knowledge_revision,
            )
        except Exception:
            self.connection.rollback()
            raise

    def refresh_stale_faq(
        self,
        entry_id: str,
        observation: FaqObservation,
        *,
        answer: str,
        now: float,
    ) -> FaqMatch:
        """Refresh one stale/versioned FAQ in place while preserving hit history."""
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError("entry_id must not be empty")
        self._validate_faq_observation(observation)
        safe_answer = self._faq_answer(answer)
        timestamp = self._validate_faq_now(now)
        question = self._faq_question(observation)
        source_ids_json = self._faq_source_ids_json(observation)
        self._begin_faq_transaction()
        try:
            current_revision_row = self.connection.execute(
                "SELECT revision FROM knowledge_state WHERE singleton_id = 1"
            ).fetchone()
            if current_revision_row is None:
                raise RuntimeError("knowledge state is not initialized")
            if observation.knowledge_revision != current_revision_row[0]:
                raise ValueError("observation knowledge revision does not match current revision")
            entry = self.connection.execute(
                "SELECT * FROM faq_entries WHERE id = ?", (entry_id,)
            ).fetchone()
            if entry is None:
                raise ValueError("FAQ entry does not exist")
            if (
                entry["scope_key"] != observation.scope_key
                or entry["intent_key"] != observation.intent_key
            ):
                raise ValueError("observation does not match FAQ scope or intent")
            should_refresh = (
                entry["state"] == "stale"
                or entry["knowledge_revision"] != observation.knowledge_revision
            )
            if not should_refresh:
                raise ValueError("FAQ entry is already current")
            if should_refresh:
                self.connection.execute(
                    "UPDATE faq_entries SET canonical_question = ?, answer = ?, "
                    "source_signature = ?, source_ids_json = ?, knowledge_revision = ?, state = 'enabled', "
                    "updated_at = ? WHERE id = ?",
                    (
                        question,
                        safe_answer,
                        observation.source_signature,
                        source_ids_json,
                        observation.knowledge_revision,
                        timestamp,
                        entry_id,
                    ),
                )
                self.connection.execute(
                    "INSERT INTO faq_metrics_daily(scope_key,day,refreshes) VALUES(?,?,1) "
                    "ON CONFLICT(scope_key,day) DO UPDATE SET refreshes = refreshes + 1",
                    (observation.scope_key, datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()),
                )
            self.connection.execute(
                "INSERT INTO faq_aliases("
                "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
                ") VALUES(?,?,?,?,?,1) ON CONFLICT(faq_id,normalized_question) DO UPDATE SET "
                "last_seen_at = excluded.last_seen_at, total_seen = total_seen + 1",
                (entry_id, question, _pretokenize(question), timestamp, timestamp),
            )
            self.connection.commit()
            return FaqMatch(
                entry_id,
                safe_answer if should_refresh else entry["answer"],
                observation.intent_key,
                observation.knowledge_revision,
            )
        except Exception:
            self.connection.rollback()
            raise

    def mark_faq_stale_before_revision(self, revision: int) -> int:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a non-negative integer")
        self._begin_faq_transaction()
        try:
            cursor = self.connection.execute(
                "UPDATE faq_entries SET state = 'stale', updated_at = ? "
                "WHERE state = 'enabled' AND knowledge_revision < ?",
                (time.time(), revision),
            )
            self.connection.commit()
            return cursor.rowcount
        except Exception:
            self.connection.rollback()
            raise

    def record_faq_metric(self, day: str, field: str, *, scope_key: str = "") -> None:
        self._validate_faq_day(day)
        if field not in _FAQ_METRIC_FIELDS:
            raise ValueError("unsupported FAQ metric field")
        if not isinstance(scope_key, str):
            raise ValueError("scope_key must be a string")
        statements = {
            "eligible_questions": (
                "INSERT INTO faq_metrics_daily(scope_key,day,eligible_questions) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET eligible_questions = eligible_questions + 1"
            ),
            "rag_answers": (
                "INSERT INTO faq_metrics_daily(scope_key,day,rag_answers) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET rag_answers = rag_answers + 1"
            ),
            "direct_hits": (
                "INSERT INTO faq_metrics_daily(scope_key,day,direct_hits) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET direct_hits = direct_hits + 1"
            ),
            "promotions": (
                "INSERT INTO faq_metrics_daily(scope_key,day,promotions) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET promotions = promotions + 1"
            ),
            "refreshes": (
                "INSERT INTO faq_metrics_daily(scope_key,day,refreshes) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET refreshes = refreshes + 1"
            ),
            "invalidations": (
                "INSERT INTO faq_metrics_daily(scope_key,day,invalidations) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET invalidations = invalidations + 1"
            ),
            "rejected_answers": (
                "INSERT INTO faq_metrics_daily(scope_key,day,rejected_answers) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET rejected_answers = rejected_answers + 1"
            ),
        }
        self._begin_faq_transaction()
        try:
            self.connection.execute(statements[field], (scope_key, day))
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def record_faq_direct_hit(
        self,
        entry_id: str,
        expected_revision: int | None = None,
        day: str | None = None,
        now: float | None = None,
    ) -> bool:
        """Atomically count a current-version direct FAQ hit and its metrics."""
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError("entry_id must not be empty")
        timestamp = time.time() if now is None else self._validate_faq_now(now)
        metric_day = (
            datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
            if day is None
            else self._validate_faq_day(day).isoformat()
        )
        if expected_revision is not None and (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 0
        ):
            raise ValueError("expected_revision must be a non-negative integer")
        self._begin_faq_transaction()
        try:
            current_revision = self.connection.execute(
                "SELECT revision FROM knowledge_state WHERE singleton_id = 1"
            ).fetchone()
            if current_revision is None:
                raise RuntimeError("knowledge state is not initialized")
            if expected_revision is None:
                expected_revision = int(current_revision[0])
            if expected_revision != current_revision[0]:
                self.connection.commit()
                return False
            row = self.connection.execute(
                "UPDATE faq_entries SET direct_hits = direct_hits + 1, last_hit_at = ? "
                "WHERE id = ? AND state = 'enabled' AND knowledge_revision = ? "
                "RETURNING scope_key",
                (timestamp, entry_id, expected_revision),
            ).fetchone()
            if row is None:
                self.connection.commit()
                return False
            self.connection.execute(
                "INSERT INTO faq_metrics_daily(scope_key,day,direct_hits) VALUES(?,?,1) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET direct_hits = direct_hits + 1",
                (row[0], metric_day),
            )
            self.connection.commit()
            return True
        except Exception:
            self.connection.rollback()
            raise

    def query_faq_metrics(self, *, since_day: str | None = None) -> list[dict[str, object]]:
        if since_day is not None:
            self._validate_faq_day(since_day)
        statement = (
            "SELECT scope_key,day,eligible_questions,rag_answers,direct_hits,promotions,"
            "refreshes,rejected_answers,invalidations FROM faq_metrics_daily"
        )
        parameters: tuple[object, ...] = ()
        if since_day is not None:
            statement += " WHERE day >= ?"
            parameters = (since_day,)
        statement += " ORDER BY day, scope_key"
        return [dict(row) for row in self.connection.execute(statement, parameters).fetchall()]

    def query_faq_summary(self, *, cutoff_day: str, promotion_count: int = _FAQ_PROMOTION_COUNT) -> dict[str, int]:
        """Return anonymous FAQ operations totals for a date window."""
        self._validate_faq_day(cutoff_day)
        if type(promotion_count) is not int or not 1 <= promotion_count <= 100:
            raise ValueError("promotion_count must be between 1 and 100")
        hot_intents = self.connection.execute(
            "SELECT COUNT(DISTINCT scope_key || char(31) || intent_key) FROM ("
            "SELECT scope_key, intent_key, source_signature, knowledge_revision "
            "FROM faq_observation_daily WHERE day >= ? "
            "GROUP BY scope_key, intent_key, source_signature, knowledge_revision "
            "HAVING SUM(count) >= ?)",
            (cutoff_day, promotion_count),
        ).fetchone()[0]
        counts = self.connection.execute(
            "SELECT state, COUNT(*) FROM faq_entries GROUP BY state"
        ).fetchall()
        states = {str(row[0]): int(row[1]) for row in counts}
        totals = self.connection.execute(
            "SELECT COALESCE(SUM(eligible_questions), 0), COALESCE(SUM(direct_hits), 0), "
            "COALESCE(SUM(refreshes), 0) "
            "FROM faq_metrics_daily WHERE day >= ?",
            (cutoff_day,),
        ).fetchone()
        stale = states.get("stale", 0)
        eligible = int(totals[0])
        return {
            "hot_intents": int(hot_intents),
            "enabled_faqs": states.get("enabled", 0),
            "stale_faqs": stale,
            "estimated_deepseek_requests_saved": int(totals[1]),
            "faq_refreshes": int(totals[2]),
            "current_invalid_faqs": stale,
            "direct_hit_rate": (int(totals[1]) / eligible) if eligible else 0.0,
        }

    def cleanup_faq(self, *, cutoff_day: str, stale_cutoff: float) -> dict[str, int]:
        self._validate_faq_day(cutoff_day)
        cutoff = self._validate_faq_now(stale_cutoff)
        self._begin_faq_transaction()
        try:
            observations = self.connection.execute(
                "DELETE FROM faq_observation_daily WHERE day < ?", (cutoff_day,)
            ).rowcount
            stale_entries = self.connection.execute(
                "DELETE FROM faq_entries WHERE state = 'stale' AND updated_at < ?",
                (cutoff,),
            ).rowcount
            self.connection.commit()
            return {"observations": observations, "stale_entries": stale_entries}
        except Exception:
            self.connection.rollback()
            raise

    def _migrate_faq_observation_constraints(self) -> None:
        table_row = self.connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'faq_observation_daily'"
        ).fetchone()
        if table_row is None:
            return

        columns_info = self.connection.execute(
            "PRAGMA table_info(faq_observation_daily)"
        ).fetchall()
        required_columns = {
            "intent_key",
            "scope_key",
            "day",
            "count",
            "normalized_question",
            "source_signature",
            "knowledge_revision",
            "latest_safe_answer",
        }
        columns = {row[1] for row in columns_info}
        if not required_columns.issubset(columns):
            raise RuntimeError(
                "faq_observation_daily migration cannot preserve its existing columns"
            )

        primary_key_columns = [
            row[1] for row in sorted(
                (row for row in columns_info if row[5]), key=lambda row: row[5]
            )
        ]
        has_complete_unique_key = primary_key_columns == [
            "scope_key",
            "intent_key",
            "normalized_question",
            "source_signature",
            "knowledge_revision",
            "day",
        ]
        if has_complete_unique_key:
            return

        def quoted(identifier: str) -> str:
            return '"' + identifier.replace('"', '""') + '"'

        legacy_name = f"faq_observation_daily_legacy_{uuid.uuid4().hex}"
        self.connection.execute(
            "DROP INDEX IF EXISTS idx_faq_observation_daily_day"
        )
        self.connection.execute(
            f"ALTER TABLE faq_observation_daily RENAME TO {quoted(legacy_name)}"
        )
        self.connection.execute(_FAQ_OBSERVATION_SCHEMA)
        self.connection.execute(
            "INSERT INTO faq_observation_daily("
            "intent_key,scope_key,day,count,normalized_question,source_signature,"
            "knowledge_revision,latest_safe_answer"
            f") SELECT intent_key,scope_key,day,count,normalized_question,source_signature,"
            f"knowledge_revision,latest_safe_answer FROM {quoted(legacy_name)}"
        )
        self.connection.execute(f"DROP TABLE {quoted(legacy_name)}")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_faq_observation_daily_day "
            "ON faq_observation_daily(day)"
        )

    def _migrate_faq_entries_constraints(self) -> None:
        table_row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'faq_entries'"
        ).fetchone()
        if table_row is None:
            return

        columns = {
            row[1]
            for row in self.connection.execute("PRAGMA table_info(faq_entries)").fetchall()
        }
        source_column = next(
            (
                row
                for row in self.connection.execute("PRAGMA table_info(faq_entries)").fetchall()
                if row[1] == "source_ids_json"
            ),
            None,
        )
        table_sql = re.sub(r"\s+", "", (table_row[0] or "").casefold())
        required_checks = (
            "check(knowledge_revision>=0)",
            "check(statein('enabled','stale'))",
            "check(direct_hits>=0)",
            "check(created_at>=0)",
            "check(updated_at>=0)",
            "check(last_hit_atisnullorlast_hit_at>=0)",
        )
        valid_schema = (
            all(check in table_sql for check in required_checks)
            and self._faq_scope_intent_unique_constraint_present()
            and (
                source_column is None
                or (bool(source_column[3]) and source_column[4] == "'[]'")
            )
        )
        if valid_schema and "source_ids_json" not in columns:
            self.connection.execute(
                "ALTER TABLE faq_entries ADD COLUMN source_ids_json TEXT NOT NULL DEFAULT '[]'"
            )
            columns.add("source_ids_json")
        if valid_schema:
            self._migrate_faq_aliases_constraints()
            return

        required_columns = {
            "id",
            "intent_key",
            "scope_key",
            "canonical_question",
            "answer",
            "source_signature",
            "knowledge_revision",
            "state",
            "direct_hits",
            "created_at",
            "updated_at",
            "last_hit_at",
        }
        missing_columns = sorted(required_columns - columns)
        if missing_columns:
            raise RuntimeError(
                "faq_entries migration cannot preserve missing columns: "
                + ", ".join(missing_columns)
            )

        rows = self.connection.execute(
            "SELECT id,state,knowledge_revision,direct_hits,created_at,updated_at,last_hit_at "
            "FROM faq_entries"
        ).fetchall()
        for row in rows:
            if row["state"] not in {"enabled", "stale"}:
                raise RuntimeError(
                    f"faq_entries contains invalid state for id {row['id']!r}"
                )
            for field in (
                "knowledge_revision",
                "direct_hits",
                "created_at",
                "updated_at",
                "last_hit_at",
            ):
                value = row[field]
                if value is None and field == "last_hit_at":
                    continue
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    numeric_value = math.nan
                if not math.isfinite(numeric_value) or numeric_value < 0:
                    raise RuntimeError(
                        f"faq_entries contains invalid {field} for id {row['id']!r}"
                    )

        alias_row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'faq_aliases'"
        ).fetchone()
        alias_columns = (
            {
                row[1]
                for row in self.connection.execute("PRAGMA table_info(faq_aliases)").fetchall()
            }
            if alias_row is not None
            else set()
        )
        current_alias_columns = {
            "faq_id",
            "normalized_question",
            "search_text",
            "first_seen_at",
            "last_seen_at",
            "total_seen",
        }
        legacy_alias_columns = {
            "faq_entry_id",
            "normalized_question",
            "alias_question",
            "created_at",
        }
        if alias_row is not None and not (
            current_alias_columns.issubset(alias_columns)
            or legacy_alias_columns.issubset(alias_columns)
        ):
            raise RuntimeError(
                "faq_aliases migration cannot preserve its existing columns"
            )

        def quoted(identifier: str) -> str:
            return '"' + identifier.replace('"', '""') + '"'

        legacy_name = f"faq_entries_legacy_{uuid.uuid4().hex}"
        legacy_alias_name = f"faq_aliases_legacy_{uuid.uuid4().hex}"
        self.connection.execute("DROP INDEX IF EXISTS idx_faq_entries_scope_state")
        self.connection.execute(
            "DROP INDEX IF EXISTS idx_faq_aliases_normalized_question"
        )
        if alias_row is not None:
            self.connection.execute(
                f"ALTER TABLE faq_aliases RENAME TO {quoted(legacy_alias_name)}"
            )
        self.connection.execute(
            f"ALTER TABLE faq_entries RENAME TO {quoted(legacy_name)}"
        )
        self.connection.execute(_FAQ_ENTRIES_SCHEMA)
        source_ids_select = (
            "source_ids_json" if "source_ids_json" in columns else "'[]'"
        )
        self.connection.execute(
            "INSERT INTO faq_entries("
            "id,intent_key,scope_key,canonical_question,answer,source_signature,source_ids_json,"
            "knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at"
            f") SELECT id,intent_key,scope_key,canonical_question,answer,source_signature,{source_ids_select},"
            f"knowledge_revision,state,direct_hits,created_at,updated_at,last_hit_at "
            f"FROM {quoted(legacy_name)}"
        )
        if alias_row is not None:
            self.connection.execute(_FAQ_ALIASES_SCHEMA)
            if current_alias_columns.issubset(alias_columns):
                alias_select = (
                    "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
                )
            else:
                alias_select = (
                    "faq_entry_id,normalized_question,alias_question,created_at,created_at,1"
                )
            self.connection.execute(
                "INSERT INTO faq_aliases("
                "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
                f") SELECT {alias_select} FROM {quoted(legacy_alias_name)}"
            )
            self.connection.execute(f"DROP TABLE {quoted(legacy_alias_name)}")
        self.connection.execute(f"DROP TABLE {quoted(legacy_name)}")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_faq_entries_scope_state "
            "ON faq_entries(scope_key, state)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_faq_aliases_normalized_question "
            "ON faq_aliases(normalized_question)"
        )

    def _faq_scope_intent_unique_constraint_present(self) -> bool:
        """Check the actual unique index columns, including legacy auto-indexes."""
        for index in self.connection.execute("PRAGMA index_list(faq_entries)").fetchall():
            if not bool(index[2]) or (len(index) > 4 and bool(index[4])):
                continue
            index_name = str(index[1]).replace('"', '""')
            columns = [
                row[2]
                for row in self.connection.execute(
                    f'PRAGMA index_info("{index_name}")'
                ).fetchall()
            ]
            if columns == ["scope_key", "intent_key"]:
                return True
        return False

    def _migrate_faq_aliases_constraints(self) -> None:
        """Validate and, when needed, rebuild the alias table in this transaction."""
        table_row = self.connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'faq_aliases'"
        ).fetchone()
        if table_row is None:
            return

        columns_info = self.connection.execute("PRAGMA table_info(faq_aliases)").fetchall()
        columns = {row[1] for row in columns_info}
        required_columns = {
            "faq_id",
            "normalized_question",
            "search_text",
            "first_seen_at",
            "last_seen_at",
            "total_seen",
        }
        has_current_columns = required_columns.issubset(columns)
        primary_key_columns = [
            row[1] for row in sorted(
                (row for row in columns_info if row[5]), key=lambda row: row[5]
            )
        ]
        unique_columns = primary_key_columns == ["faq_id", "normalized_question"]
        if not unique_columns:
            for index in self.connection.execute("PRAGMA index_list(faq_aliases)").fetchall():
                if not bool(index[2]) or (len(index) > 4 and bool(index[4])):
                    continue
                index_name = str(index[1]).replace('"', '""')
                index_columns = [
                    row[2]
                    for row in self.connection.execute(
                        f'PRAGMA index_info("{index_name}")'
                    ).fetchall()
                ]
                if index_columns == ["faq_id", "normalized_question"]:
                    unique_columns = True
                    break

        has_cascade_fk = any(
            row[2] == "faq_entries"
            and row[3] == "faq_id"
            and row[4] == "id"
            and row[6] == "CASCADE"
            for row in self.connection.execute("PRAGMA foreign_key_list(faq_aliases)").fetchall()
        )
        if has_current_columns and unique_columns and has_cascade_fk:
            return

        legacy_columns = {
            "faq_entry_id",
            "normalized_question",
            "alias_question",
            "created_at",
        }
        if not has_current_columns and not legacy_columns.issubset(columns):
            raise RuntimeError(
                "faq_aliases migration cannot preserve its existing columns"
            )

        def quoted(identifier: str) -> str:
            return '"' + identifier.replace('"', '""') + '"'

        legacy_name = f"faq_aliases_legacy_{uuid.uuid4().hex}"
        self.connection.execute(
            "DROP INDEX IF EXISTS idx_faq_aliases_normalized_question"
        )
        self.connection.execute(
            f"ALTER TABLE faq_aliases RENAME TO {quoted(legacy_name)}"
        )
        self.connection.execute(_FAQ_ALIASES_SCHEMA)
        if has_current_columns:
            alias_select = (
                "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
            )
        else:
            alias_select = (
                "faq_entry_id,normalized_question,alias_question,created_at,created_at,1"
            )
        self.connection.execute(
            "INSERT INTO faq_aliases("
            "faq_id,normalized_question,search_text,first_seen_at,last_seen_at,total_seen"
            f") SELECT {alias_select} FROM {quoted(legacy_name)}"
        )
        self.connection.execute(f"DROP TABLE {quoted(legacy_name)}")
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_faq_aliases_normalized_question "
            "ON faq_aliases(normalized_question)"
        )

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
            return self._prune_documents_in_transaction(prefix, retained)

    def _prune_documents_in_transaction(
        self, prefix: str, retained: Collection[str]
    ) -> int:
        source_rows = self.connection.execute(
            "SELECT source_id FROM documents WHERE substr(source_id, 1, ?) = ?",
            (len(prefix), prefix),
        ).fetchall()
        stale_ids = [row[0] for row in source_rows if row[0] not in retained]
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

    def _invalidate_faqs_in_transaction(self, revision: int, timestamp: float) -> int:
        rows = self.connection.execute(
            "SELECT scope_key, COUNT(*) AS count FROM faq_entries "
            "WHERE state = 'enabled' AND knowledge_revision < ? GROUP BY scope_key",
            (revision,),
        ).fetchall()
        invalidated = sum(int(row["count"]) for row in rows)
        if not invalidated:
            return 0
        self.connection.execute(
            "UPDATE faq_entries SET state = 'stale', updated_at = ? "
            "WHERE state = 'enabled' AND knowledge_revision < ?",
            (timestamp, revision),
        )
        metric_day = datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
        for row in rows:
            self.connection.execute(
                "INSERT INTO faq_metrics_daily(scope_key,day,invalidations) VALUES(?,?,?) "
                "ON CONFLICT(scope_key,day) DO UPDATE SET invalidations = invalidations + excluded.invalidations",
                (row["scope_key"], metric_day, int(row["count"])),
            )
        return invalidated

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
                "DELETE FROM processed_messages "
                "WHERE processed_at < ? AND state != 'replying'",
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
            if state in {"in_progress", "replying"}:
                return "in_progress", None
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
        if (
            isinstance(retention_seconds, bool)
            or not isinstance(retention_seconds, (int, float))
            or retention_seconds <= 0
        ):
            raise ValueError("retention_seconds must be positive")
        now = time.time()
        _execute_with_lock_retry(self.connection, "BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "DELETE FROM processed_messages "
                "WHERE processed_at < ? AND claim_token = ''",
                (now - retention_seconds,),
            )
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO processed_messages("
                "message_id,processed_at,state,claim_token"
                ") VALUES(?,?,'completed','')",
                (message_id, now),
            )
            self.connection.commit()
            return cursor.rowcount == 1
        except Exception:
            self.connection.rollback()
            raise

    def is_message_claim_owner(self, message_id: str, token: str | None) -> bool:
        if token is None:
            return False
        row = self.connection.execute(
            "SELECT 1 FROM processed_messages "
            "WHERE message_id = ? AND state IN ('in_progress','replying') "
            "AND claim_token = ?",
            (message_id, token),
        ).fetchone()
        return row is not None

    def begin_message_reply(self, message_id: str, token: str | None) -> bool:
        if token is None:
            return False
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE processed_messages "
                "SET state = 'replying', processed_at = ? "
                "WHERE message_id = ? AND state = 'in_progress' "
                "AND claim_token = ?",
                (time.time(), message_id, token),
            )
        return cursor.rowcount == 1

    def complete_message(self, message_id: str, token: str | None = None) -> bool:
        with self.connection:
            if token is None:
                # 仅为旧调用方保留；生产消息处理始终传入租约 token。
                cursor = self.connection.execute(
                    "UPDATE processed_messages "
                    "SET state = 'completed', processed_at = ?, claim_token = '' "
                    "WHERE message_id = ? AND claim_token = ''",
                    (time.time(), message_id),
                )
            else:
                cursor = self.connection.execute(
                    "UPDATE processed_messages "
                    "SET state = 'completed', processed_at = ?, claim_token = '' "
                    "WHERE message_id = ? AND state = 'replying' "
                    "AND claim_token = ?",
                    (time.time(), message_id, token),
                )
        return cursor.rowcount == 1

    def release_message(self, message_id: str, token: str | None = None) -> bool:
        with self.connection:
            if token is None:
                # 仅为旧调用方保留；生产消息处理始终传入租约 token。
                cursor = self.connection.execute(
                    "DELETE FROM processed_messages "
                    "WHERE message_id = ? AND claim_token = ''",
                    (message_id,),
                )
            else:
                cursor = self.connection.execute(
                    "DELETE FROM processed_messages "
                    "WHERE message_id = ? AND state IN ('in_progress','replying') "
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
