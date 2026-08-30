"""用固定问题验收检索和回答质量，不输出制度正文。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feishu_rag.config import Settings
from feishu_rag.llm import DeepSeekClient
from feishu_rag.rag import RagService
from feishu_rag.store import IndexStore


QUESTIONS = [
    "财务报销流程是什么",
    "供应商管理程序是什么",
    "员工入职流程是什么",
]


def evaluate_questions(store: IndexStore, rag: RagService, questions: list[str]) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for question in questions:
        matches = store.search(question, top_k=rag.top_k)
        answer = rag.answer(question)
        report.append(
            {
                "question": question,
                "matches": len(matches),
                "answer_chars": len(answer.text),
                "insufficient": "现有资料不足" in answer.text,
                "citations": len(answer.citations),
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="验收三个知识库问题")
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
    parser.add_argument("--question", action="append", dest="questions")
    args = parser.parse_args()
    settings = Settings.from_env()
    store = IndexStore(args.db)
    try:
        rag = RagService(
            store,
            DeepSeekClient(settings.deepseek_api_key, settings.deepseek_base_url, settings.deepseek_model),
            settings.rag_top_k,
        )
        print(json.dumps(evaluate_questions(store, rag, args.questions or QUESTIONS), ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
