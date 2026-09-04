"""只读输出聚合后的知识资料缺口。"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Sequence


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="输出知识资料缺口汇总")
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
    parser.add_argument("--min-count", type=_positive, default=2)
    parser.add_argument("--limit", type=_positive, default=50)
    args = parser.parse_args(argv)
    database = args.db.resolve()
    if not database.is_file():
        return 2
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro",
            uri=True,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        rows = connection.execute(
            "SELECT scope_key,gap_type,count,display_question,knowledge_revision "
            "FROM question_gaps WHERE count>=? "
            "ORDER BY count DESC,last_seen_at DESC,scope_key,gap_type LIMIT ?",
            (args.min_count, args.limit),
        ).fetchall()
    except sqlite3.Error:
        return 2
    finally:
        if connection is not None:
            connection.close()
    for row in rows:
        print(
            f"{row['scope_key']}\t{row['gap_type']}\t{row['count']}\t"
            f"{row['display_question']}\trevision={row['knowledge_revision']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
