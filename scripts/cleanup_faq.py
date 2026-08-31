"""Remove expired FAQ observations and stale entries."""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

from feishu_rag.store import IndexStore

_WINDOW_DAYS = 15


def _date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid date; expected YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise argparse.ArgumentTypeError("invalid date; expected YYYY-MM-DD")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清理过期 FAQ 数据")
    parser.add_argument("db", type=Path)
    parser.add_argument("--today", type=_date, default=date.today().isoformat())
    args = parser.parse_args(argv)
    today = date.fromisoformat(args.today)
    cutoff_day = (today - timedelta(days=_WINDOW_DAYS)).isoformat()
    store = IndexStore(args.db)
    try:
        deleted = store.cleanup_faq(
            cutoff_day=cutoff_day,
            stale_cutoff=time.time() - _WINDOW_DAYS * 24 * 60 * 60,
        )
    finally:
        store.close()
    print(f"observations_deleted={deleted['observations']}")
    print(f"stale_entries_deleted={deleted['stale_entries']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
