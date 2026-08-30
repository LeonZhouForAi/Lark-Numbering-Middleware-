"""用不含制度正文的金标问题验收检索和回答质量。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

from feishu_rag.config import Settings
from feishu_rag.evaluation import _is_insufficient, evaluate_cases, load_cases
from feishu_rag.llm import DeepSeekClient
from feishu_rag.rag import RagService
from feishu_rag.retry import RetryPolicy
from feishu_rag.store import IndexStore


QUESTIONS = ["财务报销流程是什么", "供应商管理程序是什么", "员工入职流程是什么"]
DEFAULT_CASES = Path("eval/golden_questions.json")


def evaluate_questions(store: IndexStore, rag: RagService, questions: list[str]) -> list[dict[str, object]]:
    """保留旧版脚本调用方式，输出不含回答正文的简要结果。"""
    report: list[dict[str, object]] = []
    for index, question in enumerate(questions, start=1):
        matches = store.search(
            question,
            top_k=rag.top_k,
            min_relevance=rag.min_relevance,
        )
        answer = rag.answer(question)
        report.append(
            {
                "case_id": f"legacy-{index:03d}",
                "matches": len(matches),
                "answer_chars": len(answer.text),
                "insufficient": _is_insufficient(answer.text),
                "citations": len(answer.citations),
            }
        )
    return report


def threshold_failed(
    report: dict[str, object],
    *,
    min_hit_rate: float | None = None,
    min_mrr: float | None = None,
    max_leak_rate: float | None = None,
) -> bool:
    """判断质量门禁；未设置的阈值不参与判定。"""
    return (
        (min_hit_rate is not None and float(report["hit_rate"]) < min_hit_rate)
        or (min_mrr is not None and float(report["mrr"]) < min_mrr)
        or (max_leak_rate is not None and float(report["source_leak_rate"]) > max_leak_rate)
    )


def _rate(value: str) -> float:
    try:
        rate = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threshold must be a number between 0 and 1") from exc
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be a finite number between 0 and 1")
    return rate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="验收知识库金标问题，不输出制度正文")
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
    parser.add_argument("--question", action="append", dest="questions")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--min-hit-rate", type=_rate)
    parser.add_argument("--min-mrr", type=_rate)
    parser.add_argument("--max-leak-rate", type=_rate)
    parser.add_argument("--min-relevance", type=_rate)
    args = parser.parse_args(argv)

    if args.questions and any(value is not None for value in (args.min_hit_rate, args.min_mrr, args.max_leak_rate)):
        parser.error("--question cannot be combined with quality thresholds")

    store = IndexStore(args.db)
    try:
        if args.questions:
            if args.retrieval_only:
                min_relevance = (
                    args.min_relevance if args.min_relevance is not None else 0.42
                )
                report = [
                    {
                        "case_id": f"legacy-{index:03d}",
                        "matches": len(
                            store.search(
                                question,
                                top_k=6,
                                min_relevance=min_relevance,
                            )
                        ),
                    }
                    for index, question in enumerate(args.questions, start=1)
                ]
            else:
                settings = Settings.from_env()
                min_relevance = (
                    args.min_relevance
                    if args.min_relevance is not None
                    else settings.rag_min_relevance
                )
                rag = RagService(
                    store,
                    DeepSeekClient(
                        settings.deepseek_api_key,
                        settings.deepseek_base_url,
                        settings.deepseek_model,
                        retry_policy=RetryPolicy(
                            max_attempts=settings.api_retry_max_attempts,
                            base_delay=settings.api_retry_base_delay,
                        ),
                        usage_sink=store,
                    ),
                    top_k=settings.rag_top_k,
                    min_relevance=min_relevance,
                    question_max_chars=settings.rag_question_max_chars,
                )
                report = evaluate_questions(store, rag, args.questions)
            print(json.dumps(report, ensure_ascii=False))
            return 0

        rag = None
        top_k = 6
        min_relevance = args.min_relevance if args.min_relevance is not None else 0.42
        if not args.retrieval_only:
            settings = Settings.from_env()
            top_k = settings.rag_top_k
            min_relevance = (
                args.min_relevance
                if args.min_relevance is not None
                else settings.rag_min_relevance
            )
            rag = RagService(
                store,
                DeepSeekClient(
                    settings.deepseek_api_key,
                    settings.deepseek_base_url,
                    settings.deepseek_model,
                    retry_policy=RetryPolicy(
                        max_attempts=settings.api_retry_max_attempts,
                        base_delay=settings.api_retry_base_delay,
                    ),
                    usage_sink=store,
                ),
                top_k=top_k,
                min_relevance=min_relevance,
                question_max_chars=settings.rag_question_max_chars,
            )
        report = evaluate_cases(
            store,
            load_cases(args.cases),
            rag,
            top_k=top_k,
            min_relevance=min_relevance,
        ).to_report()
        print(json.dumps(report, ensure_ascii=False))
        return int(threshold_failed(report, min_hit_rate=args.min_hit_rate, min_mrr=args.min_mrr, max_leak_rate=args.max_leak_rate))
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
