from __future__ import annotations

from feishu_rag.rag import RagAnswer
from feishu_rag.store import IndexStore
from scripts.evaluate_chunking import evaluate_questions


class FakeRag:
    def __init__(self, store):
        self.store = store
        self.top_k = 6

    def answer(self, question):
        return RagAnswer("根据制度回答", [])


def test_evaluation_reports_matches_and_insufficient_flag(tmp_path) -> None:
    store = IndexStore(tmp_path / "rag.sqlite3")
    try:
        from feishu_rag.models import Chunk

        store.upsert_document(
            "finance.txt",
            "财务制度",
            "finance.txt",
            "v1",
            [Chunk("c1", "finance.txt", "财务制度", "报销流程需要发票。")],
        )
        report = evaluate_questions(store, FakeRag(store), ["财务报销流程是什么"])
    finally:
        store.close()

    assert report[0]["matches"] >= 1
    assert report[0]["insufficient"] is False
    assert "answer" not in report[0]
