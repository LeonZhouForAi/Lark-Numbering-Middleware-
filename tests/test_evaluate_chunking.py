from __future__ import annotations

from feishu_rag.rag import RagAnswer
from feishu_rag.store import IndexStore
import pytest

from scripts.evaluate_chunking import evaluate_questions, main, threshold_failed


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
    assert report[0]["case_id"] == "legacy-001"
    assert "question" not in report[0]


def test_thresholds_return_failure_when_summary_does_not_meet_gate() -> None:
    report = {"hit_rate": 0.7, "mrr": 0.5, "source_leak_rate": 0.1}

    assert threshold_failed(report, min_hit_rate=0.8) is True
    assert threshold_failed(report, min_mrr=0.6) is True
    assert threshold_failed(report, max_leak_rate=0.05) is True
    assert threshold_failed(report, min_hit_rate=0.7, min_mrr=0.5, max_leak_rate=0.1) is False


@pytest.mark.parametrize("option", ["--min-hit-rate", "--min-mrr", "--max-leak-rate"])
@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_cli_rejects_non_finite_and_out_of_range_thresholds(option: str, value: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        main([option, value])


@pytest.mark.parametrize("option", ["--min-hit-rate", "--min-mrr", "--max-leak-rate"])
def test_cli_rejects_question_mode_with_thresholds(option: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--question", "匿名问题", option, "0.5"])
