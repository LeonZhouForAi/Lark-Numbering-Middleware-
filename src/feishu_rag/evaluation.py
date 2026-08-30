"""不含制度正文的检索与回答质量评估。"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


class SearchStore(Protocol):
    def search(self, query: str, top_k: int = 6) -> Sequence[Any]: ...


class AnswerService(Protocol):
    top_k: int

    def answer(self, question: str) -> Any: ...


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    category: str
    question: str
    answerable: bool
    expected_titles: list[str]
    expected_source_ids: list[str]
    forbidden_titles: list[str]
    expected_terms: list[str]
    forbidden_terms: list[str]


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    answerable: bool
    retrieval_evaluated: bool
    hit: bool
    reciprocal_rank: float
    forbidden_title_hit: bool
    answer_evaluated: bool
    insufficient: bool
    answerable_insufficient: bool
    unanswerable_answer: bool
    expected_terms_hit: bool
    forbidden_terms_hit: bool
    source_leak: bool
    latency_ms: float

    def to_report(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSummary:
    results: list[EvaluationResult]

    @property
    def hit_rate(self) -> float:
        scored = [result for result in self.results if result.retrieval_evaluated]
        return _average([float(result.hit) for result in scored])

    @property
    def mrr(self) -> float:
        return _average([result.reciprocal_rank for result in self.results if result.retrieval_evaluated])

    @property
    def answerable_insufficient_rate(self) -> float:
        return _average(
            [float(result.answerable_insufficient) for result in self.results if result.answer_evaluated and result.answerable]
        )

    @property
    def unanswerable_answer_rate(self) -> float:
        return _average(
            [float(result.unanswerable_answer) for result in self.results if result.answer_evaluated and not result.answerable]
        )

    @property
    def forbidden_title_rate(self) -> float:
        return _average([float(result.forbidden_title_hit) for result in self.results])

    @property
    def source_leak_rate(self) -> float:
        return _average([float(result.source_leak) for result in self.results if result.answer_evaluated])

    def to_report(self) -> dict[str, object]:
        return {
            "case_count": len(self.results),
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "answerable_insufficient_rate": self.answerable_insufficient_rate,
            "unanswerable_answer_rate": self.unanswerable_answer_rate,
            "forbidden_title_rate": self.forbidden_title_rate,
            "source_leak_rate": self.source_leak_rate,
            "results": [result.to_report() for result in self.results],
        }


_REQUIRED_FIELDS: dict[str, type[object]] = {
    "id": str,
    "category": str,
    "question": str,
    "answerable": bool,
    "expected_titles": list,
    "expected_source_ids": list,
    "forbidden_titles": list,
    "expected_terms": list,
    "forbidden_terms": list,
}
_INSUFFICIENT_RE = re.compile(
    r"现有资料不足|暂无依据|知识库中暂无|无法根据(?:现有)?资料|无法回答|无法确定|没有相关资料|未找到(?:相关|对应)?(?:资料|内容)?|不能回答",
    re.IGNORECASE,
)
_SOURCE_LEAK_RE = re.compile(r"来源|参考资料|依据文档|引用|出处|资料来源|\[\s*\d+\s*\]", re.IGNORECASE)


def load_cases(path: str | Path) -> list[EvaluationCase]:
    """加载并严格校验金标 JSON，拒绝不完整或含重复 ID 的数据。"""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evaluation cases: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("evaluation cases must be a JSON list")

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(data):
        if not isinstance(value, dict):
            raise ValueError(f"case {index} must be an object")
        unknown_fields = sorted(set(value) - set(_REQUIRED_FIELDS))
        if unknown_fields:
            raise ValueError(f"case {index} unknown fields: {', '.join(unknown_fields)}")
        for field, expected_type in _REQUIRED_FIELDS.items():
            if field not in value:
                raise ValueError(f"case {index} missing required field: {field}")
            field_value = value[field]
            if expected_type is bool:
                valid = type(field_value) is bool
            else:
                valid = type(field_value) is expected_type
            if not valid:
                raise ValueError(f"case {index} field {field} has invalid type")
            if expected_type is str and not field_value.strip():
                raise ValueError(f"case {index} field {field} must not be empty")
            if expected_type is list and any(type(item) is not str or not item.strip() for item in field_value):
                raise ValueError(f"case {index} field {field} must contain non-empty strings")
        case_id = value["id"]
        if case_id in seen_ids:
            raise ValueError(f"duplicate evaluation case id: {case_id}")
        seen_ids.add(case_id)
        cases.append(EvaluationCase(**{field: value[field] for field in _REQUIRED_FIELDS}))
    return cases


def evaluate_cases(
    store: SearchStore,
    cases: Sequence[EvaluationCase],
    rag: AnswerService | None = None,
    *,
    top_k: int | None = None,
) -> EvaluationSummary:
    """评估检索结果，且只在传入 RAG 服务时评估回答质量。"""
    effective_top_k = top_k if top_k is not None else getattr(rag, "top_k", 6)
    results: list[EvaluationResult] = []
    for case in cases:
        started = time.perf_counter()
        matches = store.search(case.question, top_k=effective_top_k)
        retrieval_evaluated = bool(case.expected_titles or case.expected_source_ids)
        hit, reciprocal_rank = _retrieval_score(case, matches)
        forbidden_title_hit = any(
            _matches_title(_chunk_value(match, "title"), case.forbidden_titles) for match in matches
        )
        answer_text = ""
        answer_evaluated = rag is not None
        if rag is not None:
            answer_text = str(rag.answer(case.question).text)
        insufficient = _is_insufficient(answer_text)
        expected_terms_hit = bool(case.expected_terms) and all(term in answer_text for term in case.expected_terms)
        forbidden_terms_hit = any(term in answer_text for term in case.forbidden_terms)
        source_leak = bool(_SOURCE_LEAK_RE.search(answer_text))
        results.append(
            EvaluationResult(
                case_id=case.id,
                answerable=case.answerable,
                retrieval_evaluated=retrieval_evaluated,
                hit=hit,
                reciprocal_rank=reciprocal_rank,
                forbidden_title_hit=forbidden_title_hit,
                answer_evaluated=answer_evaluated,
                insufficient=insufficient,
                answerable_insufficient=answer_evaluated and case.answerable and insufficient,
                unanswerable_answer=answer_evaluated and not case.answerable and bool(answer_text.strip()) and not insufficient,
                expected_terms_hit=expected_terms_hit,
                forbidden_terms_hit=forbidden_terms_hit,
                source_leak=source_leak,
                latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )
        )
    return EvaluationSummary(results)


def _retrieval_score(case: EvaluationCase, matches: Sequence[Any]) -> tuple[bool, float]:
    if not case.expected_titles and not case.expected_source_ids:
        return False, 0.0
    for index, match in enumerate(matches, start=1):
        if _matches_title(_chunk_value(match, "title"), case.expected_titles) or _matches_source(
            _chunk_value(match, "source_id"), case.expected_source_ids
        ):
            return True, 1.0 / index
    return False, 0.0


def _is_insufficient(text: str) -> bool:
    return not text.strip() or bool(_INSUFFICIENT_RE.search(text))


def _chunk_value(match: Any, name: str) -> str:
    return str(getattr(getattr(match, "chunk", match), name, ""))


def _matches_title(title: str, expected_titles: Sequence[str]) -> bool:
    normalized = title.strip().casefold()
    return normalized in {candidate.strip().casefold() for candidate in expected_titles}


def _matches_source(source_id: str, expected_source_ids: Sequence[str]) -> bool:
    normalized = source_id.strip().casefold()
    return normalized in {candidate.strip().casefold() for candidate in expected_source_ids}


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
