"""SQLite 文档索引和中文友好的轻量检索。"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Iterable

from .models import Chunk, SearchResult


class IndexStore:
    """保存文档元数据和片段，并提供本地检索。"""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._fts_available = True
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                path TEXT NOT NULL,
                checksum TEXT NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                page INTEGER,
                section TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id);
            CREATE TABLE IF NOT EXISTS processed_messages (
                message_id TEXT PRIMARY KEY,
                processed_at REAL NOT NULL
            );
            """
        )
        columns = {row[1] for row in self.connection.execute("PRAGMA table_info(chunks)").fetchall()}
        if "search_text" not in columns:
            self.connection.execute("ALTER TABLE chunks ADD COLUMN search_text TEXT NOT NULL DEFAULT ''")
        try:
            self.connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, title, content)"
            )
        except sqlite3.OperationalError:
            self._fts_available = False
        self.connection.commit()

    def upsert_document(
        self,
        source_id: str,
        title: str,
        path: str,
        checksum: str,
        chunks: Iterable[Chunk],
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
                "INSERT INTO documents(source_id,title,path,checksum,updated_at) VALUES(?,?,?,?,?)",
                (source_id, title, path, checksum, time.time()),
            )
            self.connection.executemany(
                "INSERT INTO chunks(id,source_id,title,content,page,section,search_text) VALUES(?,?,?,?,?,?,?)",
                ((c.id, c.source_id, c.title, c.content, c.page, c.section, c.search_text) for c in chunk_list),
            )
            if self._fts_available:
                self.connection.executemany(
                    "INSERT INTO chunks_fts(chunk_id,title,content) VALUES(?,?,?)",
                    ((c.id, c.title, c.content) for c in chunk_list),
                )

    @staticmethod
    def _terms(query: str) -> list[str]:
        terms: list[str] = []
        for part in re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+", query):
            if re.fullmatch(r"[\u4e00-\u9fff]+", part):
                if len(part) >= 2:
                    terms.append(part)
                    terms.extend(part[i : i + 2] for i in range(len(part) - 1))
                else:
                    terms.append(part)
            else:
                terms.append(part.lower())
        return list(dict.fromkeys(terms))

    def search(self, query: str, top_k: int = 6) -> list[SearchResult]:
        if top_k < 1:
            return []
        terms = self._terms(query)
        if not terms:
            return []
        rows = self.connection.execute(
            "SELECT id,source_id,title,content,page,section,search_text FROM chunks"
        ).fetchall()
        scored: list[SearchResult] = []
        for row in rows:
            title = row["title"].lower()
            content = row["content"].lower()
            search_text = (row["search_text"] or "").lower()
            score = 0.0
            for term in terms:
                needle = term.lower()
                score += title.count(needle) * 3.0
                score += content.count(needle)
                score += search_text.count(needle) * 0.75
            if score:
                scored.append(
                    SearchResult(
                        Chunk(
                            row["id"],
                            row["source_id"],
                            row["title"],
                            row["content"],
                            row["page"],
                            row["section"],
                            row["search_text"] or "",
                        ),
                        score,
                    )
                )
        scored.sort(key=lambda result: (-result.score, result.chunk.id))
        return scored[:top_k]

    def count_documents(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def document_checksum(self, source_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT checksum FROM documents WHERE source_id = ?", (source_id,)
        ).fetchone()
        return str(row[0]) if row else None

    def claim_message(self, message_id: str, retention_seconds: int = 7 * 24 * 60 * 60) -> bool:
        cutoff = time.time() - retention_seconds
        with self.connection:
            self.connection.execute("DELETE FROM processed_messages WHERE processed_at < ?", (cutoff,))
            cursor = self.connection.execute(
                "INSERT OR IGNORE INTO processed_messages(message_id, processed_at) VALUES(?, ?)",
                (message_id, time.time()),
            )
        return cursor.rowcount == 1

    def release_message(self, message_id: str) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM processed_messages WHERE message_id = ?", (message_id,))

    def count_chunks(self, source_id: str | None = None) -> int:
        if source_id is None:
            return int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        return int(
            self.connection.execute("SELECT COUNT(*) FROM chunks WHERE source_id = ?", (source_id,)).fetchone()[0]
        )

    def close(self) -> None:
        self.connection.close()
