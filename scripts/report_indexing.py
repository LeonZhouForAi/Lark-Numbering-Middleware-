"""只读汇总文档格式与版本生命周期。"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Sequence


_STATES = ("current", "superseded", "conflict")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="输出本地索引版本状态")
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
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
            "SELECT source_id,path,document_code,document_version,lifecycle_state "
            "FROM documents ORDER BY source_id"
        ).fetchall()
    except sqlite3.Error:
        return 2
    finally:
        if connection is not None:
            connection.close()

    state_counts = Counter(str(row["lifecycle_state"]) for row in rows)
    print(
        f"documents={len(rows)} "
        + " ".join(f"{state}={state_counts[state]}" for state in _STATES)
    )
    format_counts = Counter(Path(str(row["path"])).suffix.lower() for row in rows)
    formats = ",".join(
        f"{suffix or '(none)'}:{count}"
        for suffix, count in sorted(format_counts.items())
    )
    print(f"formats={formats}")
    for row in rows:
        if row["lifecycle_state"] == "conflict":
            print(
                f"{row['source_id']}\t{row['document_code']}\t"
                f"{row['document_version']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
