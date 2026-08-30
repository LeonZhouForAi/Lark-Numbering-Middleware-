"""按日报告 LLM Token 用量和估算成本。"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

from feishu_rag.store import IndexStore


def _price(value: str) -> Decimal:
    try:
        price = Decimal(value)
    except InvalidOperation as exc:
        raise argparse.ArgumentTypeError("单价必须是数字") from exc
    if not price.is_finite() or price < 0:
        raise argparse.ArgumentTypeError("单价必须是有限非负数")
    return price


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="报告每日 LLM Token 用量")
    parser.add_argument("db", type=Path)
    parser.add_argument("--input-price", type=_price, required=True, help="每百万输入 Token 单价")
    parser.add_argument("--output-price", type=_price, required=True, help="每百万输出 Token 单价")
    args = parser.parse_args(argv)

    store = IndexStore(args.db)
    try:
        rows = store.query_llm_usage()
    finally:
        store.close()
    print(
        "day\tmodel\tpurpose\trequests\tprompt_tokens\tcompletion_tokens\t"
        "total_tokens\testimated_cost"
    )
    million = Decimal(1_000_000)
    for row in rows:
        cost = (
            Decimal(row["prompt_tokens"]) * args.input_price
            + Decimal(row["completion_tokens"]) * args.output_price
        ) / million
        print(
            f'{row["day"]}\t{row["model"]}\t{row["purpose"]}\t{row["requests"]}\t'
            f'{row["prompt_tokens"]}\t{row["completion_tokens"]}\t{row["total_tokens"]}\t'
            f"{cost:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
