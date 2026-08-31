import sqlite3
import threading
from pathlib import Path

import pytest

from feishu_rag.models import Chunk
from feishu_rag.store import IndexStore
import scripts.prepare_v03_rollback as rollback_module
from scripts.prepare_v03_rollback import (
    RollbackPreparationError,
    main,
    prepare_v03_rollback,
)


def _v2_database(path: Path) -> None:
    store = IndexStore(path)
    try:
        store.upsert_document(
            "doc-v4",
            "现行制度",
            "policy.docx",
            "v4",
            [Chunk("chunk-v4", "doc-v4", "现行制度", "现行正文")],
            space_id="space-a",
        )
    finally:
        store.close()


def test_prepare_v03_rollback_backs_up_then_allows_real_v03_writes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    backup_path = tmp_path / "rag.before-v03.sqlite3"
    _v2_database(db_path)

    result = prepare_v03_rollback(db_path, backup_path)

    assert result == backup_path
    with sqlite3.connect(backup_path) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert [
            row[1] for row in backup.execute("PRAGMA table_info(chunks_fts)")
        ] == ["chunk_id", "title_terms", "content_terms", "search_terms"]
        assert backup.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1

    with sqlite3.connect(db_path) as rolled_back:
        assert rolled_back.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone() is None

        # These are the v0.3.0 table definition and write shapes. Extra v0.4
        # columns must retain defaults so the old release can start and write.
        rolled_back.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts "
            "USING fts5(chunk_id UNINDEXED, title, content)"
        )
        rolled_back.execute(
            "INSERT INTO documents(source_id,title,path,checksum,updated_at) "
            "VALUES(?,?,?,?,?)",
            ("doc-v3", "旧版兼容", "legacy.docx", "v3", 1.0),
        )
        rolled_back.execute(
            "INSERT INTO chunks(id,source_id,title,content,page,section) "
            "VALUES(?,?,?,?,?,?)",
            ("chunk-v3", "doc-v3", "旧版兼容", "旧版写入成功", None, None),
        )
        rolled_back.execute(
            "INSERT INTO chunks_fts(chunk_id,title,content) VALUES(?,?,?)",
            ("chunk-v3", "旧版兼容", "旧版写入成功"),
        )
        rolled_back.commit()
        assert [
            row[1] for row in rolled_back.execute("PRAGMA table_info(chunks_fts)")
        ] == ["chunk_id", "title", "content"]


def test_prepare_v03_rollback_never_overwrites_an_existing_backup(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    backup_path = tmp_path / "existing.sqlite3"
    _v2_database(db_path)
    backup_path.write_bytes(b"keep-me")

    with pytest.raises(RollbackPreparationError, match="backup"):
        prepare_v03_rollback(db_path, backup_path)

    assert backup_path.read_bytes() == b"keep-me"
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone() is not None


def test_prepare_v03_rollback_backup_captures_committed_wal_pages(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    backup_path = tmp_path / "rag.before-v03.sqlite3"
    _v2_database(db_path)
    writer = sqlite3.connect(db_path)
    try:
        writer.execute("PRAGMA journal_mode = WAL")
        writer.execute("PRAGMA wal_autocheckpoint = 0")
        writer.execute(
            "INSERT INTO documents(source_id,title,path,checksum,updated_at) "
            "VALUES(?,?,?,?,?)",
            ("wal-doc", "WAL 文档", "wal.docx", "wal", 2.0),
        )
        writer.commit()

        prepare_v03_rollback(db_path, backup_path)

        with sqlite3.connect(backup_path) as backup:
            assert backup.execute(
                "SELECT title FROM documents WHERE source_id = 'wal-doc'"
            ).fetchone()[0] == "WAL 文档"
    finally:
        writer.close()


def test_prepare_v03_rollback_holds_write_lock_across_backup_and_drop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    backup_path = tmp_path / "rag.before-v03.sqlite3"
    _v2_database(db_path)
    original_backup = rollback_module._create_backup
    writer_outcome: list[str] = []

    def backup_then_attempt_write(source, path):
        original_backup(source, path)

        def write_during_window() -> None:
            connection = sqlite3.connect(db_path, timeout=0.05)
            try:
                connection.execute("PRAGMA busy_timeout = 50")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO documents("
                    "source_id,title,path,checksum,updated_at"
                    ") VALUES(?,?,?,?,?)",
                    ("late-doc", "窗口写入", "late.docx", "late", 3.0),
                )
                connection.commit()
                writer_outcome.append("committed")
            except sqlite3.OperationalError:
                writer_outcome.append("locked")
            finally:
                connection.close()

        writer = threading.Thread(target=write_during_window)
        writer.start()
        writer.join(timeout=2)
        assert not writer.is_alive()

    monkeypatch.setattr(rollback_module, "_create_backup", backup_then_attempt_write)

    prepare_v03_rollback(db_path, backup_path)

    assert writer_outcome == ["locked"]
    with sqlite3.connect(db_path) as source:
        assert source.execute(
            "SELECT COUNT(*) FROM documents WHERE source_id = 'late-doc'"
        ).fetchone()[0] == 0
    with sqlite3.connect(backup_path) as backup:
        assert backup.execute(
            "SELECT COUNT(*) FROM documents WHERE source_id = 'late-doc'"
        ).fetchone()[0] == 0


def test_cli_requires_execute_flag_before_creating_backup_or_dropping_fts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    backup_path = tmp_path / "rag.before-v03.sqlite3"
    _v2_database(db_path)

    assert main([str(db_path), "--backup", str(backup_path)]) == 0

    assert "status=ready" in capsys.readouterr().out
    assert not backup_path.exists()
    with sqlite3.connect(db_path) as connection:
        assert [
            row[1] for row in connection.execute("PRAGMA table_info(chunks_fts)")
        ] == ["chunk_id", "title_terms", "content_terms", "search_terms"]

    assert main(
        [str(db_path), "--backup", str(backup_path), "--execute"]
    ) == 0
    assert backup_path.exists()


def test_prepare_v03_rollback_rejects_unexpected_fts_schema_without_backup(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    backup_path = tmp_path / "should-not-exist.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "CREATE VIRTUAL TABLE chunks_fts "
            "USING fts5(chunk_id UNINDEXED, title, content)"
        )

    with pytest.raises(RollbackPreparationError, match="schema"):
        prepare_v03_rollback(db_path, backup_path)

    assert not backup_path.exists()
    with sqlite3.connect(db_path) as connection:
        assert [
            row[1] for row in connection.execute("PRAGMA table_info(chunks_fts)")
        ] == ["chunk_id", "title", "content"]


@pytest.mark.parametrize("same_spelling", [True, False])
def test_prepare_v03_rollback_requires_distinct_existing_database_and_backup(
    tmp_path: Path, same_spelling: bool
) -> None:
    db_path = tmp_path / "rag.sqlite3"
    _v2_database(db_path)
    backup_path = db_path if same_spelling else tmp_path / "missing" / "backup.sqlite3"

    with pytest.raises(RollbackPreparationError):
        prepare_v03_rollback(db_path, backup_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
        ).fetchone() is not None
