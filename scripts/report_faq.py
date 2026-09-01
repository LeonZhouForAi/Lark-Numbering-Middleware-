"""Print anonymous, daily FAQ aggregate metrics."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from feishu_rag.store import faq_window_cutoff


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


def _read_report(db: Path, since: str | None, cutoff: str, promotion_count: int):
    uri = f"file:{db.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        required = {"faq_metrics_daily", "faq_entries", "faq_observation_daily"}
        actual = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        missing = required - actual
        if missing:
            raise RuntimeError("database is missing FAQ tables: " + ", ".join(sorted(missing)))
        query = "SELECT scope_key,day,eligible_questions,rag_answers,direct_hits,promotions,refreshes,rejected_answers,invalidations FROM faq_metrics_daily"
        params = ()
        if since is not None:
            query += " WHERE day >= ?"
            params = (since,)
        rows = [dict(row) for row in connection.execute(query + " ORDER BY day, scope_key", params)]
        hot = connection.execute(
            "SELECT COUNT(DISTINCT scope_key || char(31) || intent_key) FROM ("
            "SELECT scope_key,intent_key,source_signature,knowledge_revision FROM faq_observation_daily "
            "WHERE day >= ? GROUP BY scope_key,intent_key,source_signature,knowledge_revision HAVING SUM(count) >= ?)",
            (cutoff, promotion_count),
        ).fetchone()[0]
        states = {row[0]: row[1] for row in connection.execute("SELECT state,COUNT(*) FROM faq_entries GROUP BY state")}
        eligible, direct, refreshes = connection.execute(
            "SELECT COALESCE(SUM(eligible_questions),0),COALESCE(SUM(direct_hits),0),COALESCE(SUM(refreshes),0) "
            "FROM faq_metrics_daily WHERE day >= ?", (cutoff,)
        ).fetchone()
        stale = int(states.get("stale", 0))
        summary = {"hot_intents": int(hot), "enabled_faqs": int(states.get("enabled", 0)), "stale_faqs": stale,
                   "estimated_deepseek_requests_saved": int(direct), "faq_refreshes": int(refreshes),
                   "current_invalid_faqs": stale, "direct_hit_rate": int(direct) / int(eligible) if eligible else 0.0}
        return rows, summary
    finally:
        connection.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="报告匿名 FAQ 聚合指标")
    parser.add_argument("db", type=Path)
    parser.add_argument("--since", type=_date)
    default_promotion = os.environ.get("RAG_FAQ_PROMOTION_COUNT", "3")
    parser.add_argument("--promotion-count", type=_promotion_count, default=_promotion_count(default_promotion))
    args = parser.parse_args(argv)
    cutoff_day = faq_window_cutoff(date.today())
    rows, summary = _read_report(args.db, args.since, cutoff_day, args.promotion_count)
    print("day\tscope_key\teligible_questions\trag_answers\tdirect_hits\tdirect_hit_rate\tpromotions\trefreshes\trejected_answers\tinvalidations")
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
            "invalidations": int(row["invalidations"]),
        }
        print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))
    print("summary\t" + json.dumps(summary, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
