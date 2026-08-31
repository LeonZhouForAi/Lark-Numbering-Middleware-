"""Print anonymous, daily FAQ aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from feishu_rag.store import IndexStore, faq_window_cutoff


def _date(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid date; expected YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise argparse.ArgumentTypeError("invalid date; expected YYYY-MM-DD")
    return value


def _promotion_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("promotion count must be an integer from 1 to 100") from exc
    if not 1 <= count <= 100:
        raise argparse.ArgumentTypeError("promotion count must be an integer from 1 to 100")
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="报告匿名 FAQ 聚合指标")
    parser.add_argument("db", type=Path)
    parser.add_argument("--since", type=_date)
    default_promotion = os.environ.get("RAG_FAQ_PROMOTION_COUNT", "3")
    parser.add_argument("--promotion-count", type=_promotion_count, default=_promotion_count(default_promotion))
    args = parser.parse_args(argv)
    store = IndexStore(args.db)
    try:
        rows = store.query_faq_metrics(since_day=args.since)
        cutoff_day = faq_window_cutoff(date.today())
        summary = store.query_faq_summary(cutoff_day=cutoff_day, promotion_count=args.promotion_count)
    finally:
        store.close()
    print("day\tscope_key\teligible_questions\trag_answers\tdirect_hits\tdirect_hit_rate\tpromotions\trefreshes\trejected_answers")
    for row in rows:
        eligible_questions = int(row["eligible_questions"])
        rag_answers = int(row["rag_answers"])
        output = {
            "day": row["day"],
            "scope_key": row["scope_key"],
            "eligible_questions": eligible_questions,
            "rag_answers": rag_answers,
            "direct_hits": int(row["direct_hits"]),
            "direct_hit_rate": (int(row["direct_hits"]) / eligible_questions)
            if eligible_questions
            else 0.0,
            "promotions": int(row["promotions"]),
            "refreshes": int(row["refreshes"]),
            "rejected_answers": int(row["rejected_answers"]),
        }
        print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    print("summary\t" + json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
