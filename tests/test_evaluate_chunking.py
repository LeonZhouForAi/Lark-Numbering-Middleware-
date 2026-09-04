from __future__ import annotations

from types import SimpleNamespace

from feishu_rag.rag import RagAnswer
from feishu_rag.store import IndexStore
import pytest

import scripts.evaluate_chunking as evaluation_script
from scripts.evaluate_chunking import evaluate_questions, main, threshold_failed


class FakeRag:
    def __init__(self, store, min_relevance=0.42):
        self.store = store
        self.top_k = 6
        self.min_relevance = min_relevance

    def answer(self, question):
        return RagAnswer("根据制度回答", [])


class RecordingStore:
    calls = []

    def __init__(self, _path=None):
        self.calls = []
        type(self).calls = self.calls

    def search(self, query, top_k=6, min_relevance=0.42):
        self.calls.append(
            {"query": query, "top_k": top_k, "min_relevance": min_relevance}
        )
        return []

    def close(self):
        pass


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


def test_evaluate_questions_uses_rag_min_relevance_for_direct_search() -> None:
    store = RecordingStore()

    evaluate_questions(store, FakeRag(store, min_relevance=0.73), ["报销要求"])

    assert store.calls == [
        {"query": "报销要求", "top_k": 6, "min_relevance": 0.73}
    ]


def test_retrieval_only_question_cli_passes_custom_min_relevance(monkeypatch) -> None:
    monkeypatch.setattr(evaluation_script, "IndexStore", RecordingStore)

    assert main(["--retrieval-only", "--question", "报销要求", "--min-relevance", "0.73"]) == 0
    assert RecordingStore.calls == [
        {"query": "报销要求", "top_k": 6, "min_relevance": 0.73}
    ]


def test_retrieval_only_question_cli_defaults_min_relevance(monkeypatch) -> None:
    monkeypatch.setattr(evaluation_script, "IndexStore", RecordingStore)

    assert main(["--retrieval-only", "--question", "报销要求"]) == 0
    assert RecordingStore.calls == [
        {"query": "报销要求", "top_k": 6, "min_relevance": 0.42}
    ]


def test_non_retrieval_cli_passes_settings_min_relevance_everywhere(monkeypatch) -> None:
    seen = {}
    settings = SimpleNamespace(
        deepseek_api_key="secret",
        deepseek_base_url="https://example.invalid",
        deepseek_model="model",
        rag_top_k=5,
        rag_min_relevance=0.73,
        rag_question_max_chars=321,
        api_retry_max_attempts=3,
        api_retry_base_delay=0.5,
    )

    class FakeSettings:
        @staticmethod
        def from_env():
            return settings

    class FakeRagService:
        def __init__(self, store, llm, *, top_k, min_relevance, question_max_chars):
            seen["rag"] = (top_k, min_relevance, question_max_chars)

    class FakeSummary:
        def to_report(self):
            return {"hit_rate": 1.0, "mrr": 1.0, "source_leak_rate": 0.0}

    def fake_evaluate_cases(store, cases, rag, *, top_k, min_relevance):
        seen["evaluation"] = (top_k, min_relevance)
        return FakeSummary()

    monkeypatch.setattr(evaluation_script, "IndexStore", RecordingStore)
    monkeypatch.setattr(evaluation_script, "Settings", FakeSettings)
    monkeypatch.setattr(evaluation_script, "DeepSeekClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation_script, "RagService", FakeRagService)
    monkeypatch.setattr(evaluation_script, "load_cases", lambda path: [])
    monkeypatch.setattr(evaluation_script, "evaluate_cases", fake_evaluate_cases)

    assert main([]) == 0
    assert seen == {"rag": (5, 0.73, 321), "evaluation": (5, 0.73)}


def test_explicit_cli_min_relevance_overrides_settings_everywhere(monkeypatch) -> None:
    seen = {}
    settings = SimpleNamespace(
        deepseek_api_key="secret",
        deepseek_base_url="https://example.invalid",
        deepseek_model="model",
        rag_top_k=5,
        rag_min_relevance=0.73,
        rag_question_max_chars=321,
        api_retry_max_attempts=3,
        api_retry_base_delay=0.5,
    )

    class FakeSettings:
        @staticmethod
        def from_env():
            return settings

    class FakeRagService:
        def __init__(self, store, llm, *, top_k, min_relevance, question_max_chars):
            self.top_k = top_k
            self.min_relevance = min_relevance
            seen["rag"] = (min_relevance, question_max_chars)

        def answer(self, question):
            return RagAnswer("根据制度回答", [])

    class FakeSummary:
        def to_report(self):
            return {"hit_rate": 1.0, "mrr": 1.0, "source_leak_rate": 0.0}

    def fake_evaluate_cases(store, cases, rag, *, top_k, min_relevance):
        seen["evaluation"] = min_relevance
        return FakeSummary()

    def fake_evaluate_questions(store, rag, questions):
        seen["legacy"] = rag.min_relevance
        return []

    monkeypatch.setattr(evaluation_script, "IndexStore", RecordingStore)
    monkeypatch.setattr(evaluation_script, "Settings", FakeSettings)
    monkeypatch.setattr(evaluation_script, "DeepSeekClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation_script, "RagService", FakeRagService)
    monkeypatch.setattr(evaluation_script, "load_cases", lambda path: [])
    monkeypatch.setattr(evaluation_script, "evaluate_cases", fake_evaluate_cases)
    monkeypatch.setattr(evaluation_script, "evaluate_questions", fake_evaluate_questions)

    assert main(["--min-relevance", "0.61"]) == 0
    assert seen == {"rag": (0.61, 321), "evaluation": 0.61}

    seen.clear()
    assert main(["--question", "报销要求", "--min-relevance", "0.61"]) == 0
    assert seen == {"rag": (0.61, 321), "legacy": 0.61}


def test_thresholds_return_failure_when_summary_does_not_meet_gate() -> None:
    report = {
        "hit_rate": 0.7,
        "mrr": 0.5,
        "source_leak_rate": 0.1,
        "status_accuracy": 0.8,
        "answerable_insufficient_rate": 0.2,
        "unanswerable_answer_rate": 0.1,
    }

    assert threshold_failed(report, min_hit_rate=0.8) is True
    assert threshold_failed(report, min_mrr=0.6) is True
    assert threshold_failed(report, max_leak_rate=0.05) is True
    assert threshold_failed(report, min_status_accuracy=0.9) is True
    assert threshold_failed(report, max_answerable_insufficient_rate=0.1) is True
    assert threshold_failed(report, max_unanswerable_answer_rate=0.05) is True
    assert threshold_failed(
        report,
        min_hit_rate=0.7,
        min_mrr=0.5,
        max_leak_rate=0.1,
        min_status_accuracy=0.8,
        max_answerable_insufficient_rate=0.2,
        max_unanswerable_answer_rate=0.1,
    ) is False


@pytest.mark.parametrize(
    "option",
    [
        "--min-hit-rate",
        "--min-mrr",
        "--max-leak-rate",
        "--min-relevance",
        "--min-status-accuracy",
        "--max-answerable-insufficient-rate",
        "--max-unanswerable-answer-rate",
    ],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_cli_rejects_non_finite_and_out_of_range_thresholds(option: str, value: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        main([option, value])


@pytest.mark.parametrize(
    "option",
    [
        "--min-hit-rate",
        "--min-mrr",
        "--max-leak-rate",
        "--min-status-accuracy",
        "--max-answerable-insufficient-rate",
        "--max-unanswerable-answer-rate",
    ],
)
def test_cli_rejects_question_mode_with_thresholds(option: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--question", "匿名问题", option, "0.5"])


@pytest.mark.parametrize(
    "option",
    [
        "--min-status-accuracy",
        "--max-answerable-insufficient-rate",
        "--max-unanswerable-answer-rate",
    ],
)
def test_retrieval_only_rejects_answer_quality_thresholds(option: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["--retrieval-only", option, "0.5"])
