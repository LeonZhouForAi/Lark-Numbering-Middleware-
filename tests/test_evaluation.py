from __future__ import annotations

import json

import pytest

from feishu_rag.evaluation import EvaluationCase, _is_insufficient, evaluate_cases, load_cases
from feishu_rag.models import Chunk, SearchResult
from feishu_rag.rag import RagAnswer


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def search(
        self,
        question: str,
        top_k: int,
        min_relevance: float = 0.42,
    ) -> list[SearchResult]:
        self.calls.append(
            {"question": question, "top_k": top_k, "min_relevance": min_relevance}
        )
        return self.results[:top_k]


class FakeRag:
    def __init__(self, answers: dict[str, str], top_k: int = 3) -> None:
        self.answers = answers
        self.top_k = top_k

    def answer(self, question: str) -> RagAnswer:
        return RagAnswer(self.answers[question], [])


def _result(title: str, source_id: str) -> SearchResult:
    return SearchResult(Chunk(f"{source_id}-chunk", source_id, title, "不应出现在报告中"), 1.0)


def _case(**overrides: object) -> EvaluationCase:
    fields: dict[str, object] = {
        "id": "finance-001",
        "category": "财务",
        "question": "报销怎么弄",
        "answerable": True,
        "expected_titles": ["财务报销"],
        "expected_source_ids": [],
        "forbidden_titles": [],
        "expected_terms": ["发票"],
        "forbidden_terms": [],
    }
    fields.update(overrides)
    return EvaluationCase(**fields)


def test_load_cases_rejects_missing_fields_wrong_types_and_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "cases.json"
    valid = _case().__dict__.copy()
    del valid["category"]
    path.write_text(json.dumps([valid]), encoding="utf-8")
    with pytest.raises(ValueError, match="category"):
        load_cases(path)

    valid = _case().__dict__.copy()
    valid["answerable"] = "true"
    path.write_text(json.dumps([valid]), encoding="utf-8")
    with pytest.raises(ValueError, match="answerable"):
        load_cases(path)

    valid = _case().__dict__.copy()
    path.write_text(json.dumps([valid, valid]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(path)


def test_load_cases_rejects_unknown_fields_without_echoing_question(tmp_path) -> None:
    path = tmp_path / "cases.json"
    valid = _case().__dict__.copy()
    valid["unexpected"] = "不得出现在错误信息中的问题正文"
    path.write_text(json.dumps([valid]), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown fields: unexpected") as exc_info:
        load_cases(path)

    assert "不得出现在错误信息中的问题正文" not in str(exc_info.value)


def test_evaluation_calculates_metrics_and_strips_sensitive_report_fields() -> None:
    cases = [
        _case(),
        _case(
            id="procurement-001",
            category="采购",
            question="供应商怎么选",
            expected_titles=["供应商管理"],
            forbidden_titles=["财务报销"],
            forbidden_terms=["越权"],
        ),
        _case(
            id="admin-001",
            category="行政",
            question="有没有宠物补贴",
            answerable=False,
            expected_titles=[],
            expected_terms=[],
        ),
    ]
    store = FakeStore([_result("财务报销", "finance-1"), _result("供应商管理", "procurement-1")])
    rag = FakeRag(
        {
            "报销怎么弄": "现有资料不足，无法确认。",
            "供应商怎么选": "请遵循流程。来源：[1]",
            "有没有宠物补贴": "可以申请。",
        }
    )

    summary = evaluate_cases(store, cases, rag)
    report = summary.to_report()

    assert summary.hit_rate == pytest.approx(1.0)
    assert summary.mrr == pytest.approx(0.75)
    assert summary.answerable_insufficient_rate == pytest.approx(0.5)
    assert summary.unanswerable_answer_rate == pytest.approx(1.0)
    assert summary.forbidden_title_rate == pytest.approx(1.0)
    assert summary.source_leak_rate == pytest.approx(1.0 / 3)
    assert report["results"][0]["case_id"] == "finance-001"
    assert report["results"][1]["expected_terms_hit"] is False
    assert report["results"][1]["forbidden_terms_hit"] is False
    assert report["results"][0]["forbidden_title_evaluated"] is False
    assert report["results"][1]["forbidden_title_evaluated"] is True
    report_text = json.dumps(report, ensure_ascii=False)
    assert '"answer"' not in report_text
    assert '"content"' not in report_text
    assert '"prompt"' not in report_text
    assert "不应出现在报告中" not in report_text


def test_retrieval_only_skips_answer_quality_metrics() -> None:
    summary = evaluate_cases(FakeStore([_result("财务报销", "finance-1")]), [_case()])

    result = summary.to_report()["results"][0]
    assert summary.hit_rate == 1.0
    assert result["answer_evaluated"] is False
    assert result["source_leak"] is False


def test_evaluate_cases_passes_custom_min_relevance_to_store() -> None:
    store = FakeStore([_result("财务报销", "finance-1")])

    evaluate_cases(store, [_case()], top_k=2, min_relevance=0.73)

    assert store.calls == [
        {"question": "报销怎么弄", "top_k": 2, "min_relevance": 0.73}
    ]


@pytest.mark.parametrize(
    "text",
    ["", "   ", "现有资料不足", "暂无依据", "无法回答", "无法确定", "没有相关资料", "未找到对应内容", "不能回答"],
)
def test_insufficient_detector_recognizes_empty_and_fixed_refusals(text: str) -> None:
    assert _is_insufficient(text) is True


def test_empty_answers_are_insufficient_for_answerable_cases_but_not_unanswerable_answers() -> None:
    cases = [_case(), _case(id="unknown-001", answerable=False, expected_titles=[], expected_terms=[])]
    rag = FakeRag({"报销怎么弄": " ", "unknown-001": ""})
    cases[1] = _case(id="unknown-001", question="unknown-001", answerable=False, expected_titles=[], expected_terms=[])

    summary = evaluate_cases(FakeStore([]), cases, rag)

    assert summary.answerable_insufficient_rate == 1.0
    assert summary.unanswerable_answer_rate == 0.0
