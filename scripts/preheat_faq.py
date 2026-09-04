"""处理待执行的 FAQ 预热作业。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from feishu_rag.config import Settings
from feishu_rag.llm import DeepSeekClient
from feishu_rag.preheat import PreheatWorker
from feishu_rag.retry import RetryPolicy
from feishu_rag.store import IndexStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="处理一个 FAQ 预热作业")
    parser.add_argument("db", type=Path, nargs="?", default=Path("./data/rag.sqlite3"))
    parser.add_argument("--once", action="store_true", help="处理一个作业后退出")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    store = IndexStore(args.db)
    try:
        llm = DeepSeekClient(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
            retry_policy=RetryPolicy(
                max_attempts=settings.api_retry_max_attempts,
                base_delay=settings.api_retry_base_delay,
            ),
            usage_sink=store,
        )
        worker = PreheatWorker(
            store,
            llm,
            max_per_scope=settings.rag_faq_preheat_max_per_space,
            workers=settings.rag_faq_preheat_workers,
        )
        result = worker.run_once()
        print(
            f"job_id={result.job_id or 'none'} candidates={result.candidates} "
            f"generated={result.generated} failed={result.failed}"
        )
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
