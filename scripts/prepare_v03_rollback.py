"""安全准备 SQLite 数据库，以便从 v0.4 回滚到 v0.3。"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Sequence


class RollbackPreparationError(RuntimeError):
    """数据库不满足安全回滚前置条件。"""


_V04_FTS_COLUMNS = ("chunk_id", "title_terms", "content_terms", "search_terms")


def _fts_columns(connection: sqlite3.Connection) -> tuple[str, ...]:
    return tuple(
        row[1] for row in connection.execute("PRAGMA table_info(chunks_fts)")
    )


def _require_valid_v04_database(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
        raise RollbackPreparationError("database integrity check failed")
    if _fts_columns(connection) != _V04_FTS_COLUMNS:
        raise RollbackPreparationError("unexpected chunks_fts schema")


def _create_backup(
    source: sqlite3.Connection,
    backup_path: Path,
) -> None:
    try:
        descriptor = os.open(
            backup_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as exc:
        raise RollbackPreparationError("backup path already exists") from exc
    except OSError as exc:
        raise RollbackPreparationError("cannot create backup") from exc
    os.close(descriptor)
    try:
        with sqlite3.connect(backup_path) as backup:
            source.backup(backup)
            _require_valid_v04_database(backup)
        try:
            os.chmod(backup_path, 0o600)
        except OSError:
            pass
    except Exception:
        backup_path.unlink(missing_ok=True)
        raise


def _preflight_paths(
    db_path: str | Path,
    backup_path: str | Path,
) -> tuple[Path, Path]:
    database = Path(db_path)
    backup = Path(backup_path)
    if not database.is_file():
        raise RollbackPreparationError("database does not exist")
    if not backup.parent.is_dir():
        raise RollbackPreparationError("backup directory does not exist")
    database = database.resolve()
    backup = backup.resolve()
    if database == backup:
        raise RollbackPreparationError("backup path must differ from database")
    if backup.exists():
        raise RollbackPreparationError("backup path already exists")
    with sqlite3.connect(database, timeout=30.0) as source:
        source.execute("PRAGMA busy_timeout = 30000")
        _require_valid_v04_database(source)
    return database, backup


def prepare_v03_rollback(
    db_path: str | Path,
    backup_path: str | Path,
) -> Path:
    database, backup = _preflight_paths(db_path, backup_path)

    with sqlite3.connect(database, timeout=30.0) as guard:
        guard.execute("PRAGMA busy_timeout = 30000")
        try:
            # Prevent any writer from committing between the backup snapshot and
            # the schema transition. Reads remain available while this reserved
            # lock is held, so a separate connection can use SQLite Backup API.
            guard.execute("BEGIN IMMEDIATE")
            _require_valid_v04_database(guard)
            document_count = guard.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            chunk_count = guard.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            with sqlite3.connect(database, timeout=30.0) as snapshot:
                snapshot.execute("PRAGMA busy_timeout = 30000")
                _require_valid_v04_database(snapshot)
                _create_backup(snapshot, backup)

            _require_valid_v04_database(guard)
            guard.execute("DROP TABLE chunks_fts")
            if _fts_columns(guard):
                raise RollbackPreparationError("chunks_fts removal failed")
            if (
                guard.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
                != document_count
            ):
                raise RollbackPreparationError(
                    "document count changed during rollback preparation"
                )
            if guard.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] != chunk_count:
                raise RollbackPreparationError("chunk count changed during rollback preparation")
            guard.commit()
        except Exception as exc:
            guard.rollback()
            if isinstance(exc, RollbackPreparationError):
                raise
            raise RollbackPreparationError("database rollback preparation failed") from exc
    return backup


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="备份 v0.4 SQLite 数据库并移除 v2 FTS，以便安全切换到 v0.3"
    )
    parser.add_argument("db", type=Path)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际创建备份并删除 v2 chunks_fts；缺省仅做只读预检",
    )
    args = parser.parse_args(argv)
    try:
        if not args.execute:
            _, backup = _preflight_paths(args.db, args.backup)
            print(f"status=ready execute_required=true backup={backup}")
            return 0
        backup = prepare_v03_rollback(args.db, args.backup)
    except RollbackPreparationError as exc:
        parser.error(str(exc))
    print(f"status=prepared backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
