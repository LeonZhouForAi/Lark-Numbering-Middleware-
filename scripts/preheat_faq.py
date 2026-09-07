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
    if not getattr(settings, "rag_faq_preheat_enabled", True):
        print("preheat_disabled=true")
        return 0
    def record_usage(*values, **options):
        usage_store = IndexStore(args.db)
        try:
            usage_store.record_llm_usage(*values, **options)
        finally:
            usage_store.close()

    store = IndexStore(args.db)
    try:
        if callable(getattr(store, "enqueue_preheat_job", None)):
            store.enqueue_preheat_job(
                "global", store.knowledge_revision(),
                max_retries=getattr(settings, "rag_faq_preheat_max_retries", 1),
            )
        llm = DeepSeekClient(
            settings.deepseek_api_key,
            settings.deepseek_base_url,
            settings.deepseek_model,
            retry_policy=RetryPolicy(
                max_attempts=settings.api_retry_max_attempts,
                base_delay=settings.api_retry_base_delay,
            ),
            usage_sink=record_usage,
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
