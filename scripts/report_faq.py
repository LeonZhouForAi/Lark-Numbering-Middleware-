"""Print anonymous, daily FAQ aggregate metrics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

from feishu_rag.store import IndexStore


def _date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid date; expected YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise argparse.ArgumentTypeError("invalid date; expected YYYY-MM-DD")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="报告匿名 FAQ 聚合指标")
    parser.add_argument("db", type=Path)
    parser.add_argument("--since", type=_date)
    args = parser.parse_args(argv)
    store = IndexStore(args.db)
    try:
        rows = store.query_faq_metrics(since_day=args.since)
    finally:
        store.close()
    print("day\tscope_key\teligible_questions\trag_answers\tdirect_hits\tdirect_hit_rate\tpromotions\trefreshes\trejected_answers")
    for row in rows:
        rag_answers = int(row["rag_answers"])
        output = {
            "day": row["day"],
            "scope_key": row["scope_key"],
            "eligible_questions": int(row["eligible_questions"]),
            "rag_answers": rag_answers,
            "direct_hits": int(row["direct_hits"]),
            "direct_hit_rate": (int(row["direct_hits"]) / rag_answers) if rag_answers else 0.0,
            "promotions": int(row["promotions"]),
            "refreshes": int(row["refreshes"]),
            "rejected_answers": int(row["rejected_answers"]),
        }
        print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
