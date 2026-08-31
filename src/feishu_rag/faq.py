"""Local, deterministic FAQ observation and lookup service."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Iterable

from .models import FaqMatch, FaqObservation, RetrievalScope, SearchResult
from .store import _core_terms, _meaningful_query, _normalize, _tokens


# These replacements are deliberately small and reviewable.  They are used
# only to group FAQ questions; the original question is never rewritten for a
# generated answer.
_SYNONYMS = (
    ("新的供应商", "供应商"),
    ("新供应商", "供应商"),
    ("费用报销", "报销"),
    ("开发", "准入"),
    ("导入", "准入"),
    ("录入", "准入"),
    ("怎么走", "流程"),
    ("怎么办", "流程"),
    ("如何办理", "流程"),
    ("流程是什么", "流程"),
)


def _row_value(row: Any, name: str, default: Any = None) -> Any:
    try:
        return row[name]
    except (KeyError, IndexError, TypeError):
        return getattr(row, name, default)


class FaqService:
    """Coordinates local FAQ features with the storage layer."""

    def __init__(
        self,
        store: Any,
        enabled: bool,
        promotion_count: int,
        window_days: int,
        min_text_similarity: float,
        min_source_overlap: float,
    ) -> None:
        self.store = store
        self.enabled = bool(enabled)
        self.promotion_count = promotion_count
        self.window_days = window_days
        self.min_text_similarity = min_text_similarity
        self.min_source_overlap = min_source_overlap

    @staticmethod
    def _scope_key(scope: RetrievalScope | None) -> str:
        if scope is None or scope.allowed_space_ids is None:
            return "global"
        ids = sorted({str(space_id) for space_id in scope.allowed_space_ids})
        if not ids:
            return ""
        return sha256(",".join(ids).encode("utf-8")).hexdigest()

    @staticmethod
    def _question_features(question: str) -> tuple[str, str]:
        if not isinstance(question, str):
            return "", ""
        text = _normalize(question)
        for source, target in _SYNONYMS:
            text = text.replace(source, f" {target} ")
        meaningful = _meaningful_query(text)
        terms = _core_terms(meaningful)
        normalized = " ".join(terms)
        if not terms:
            return "", ""
        intent_key = sha256(",".join(sorted(terms)).encode("utf-8")).hexdigest()
        return normalized, intent_key

    @staticmethod
    def _source_ids(results: Iterable[SearchResult]) -> tuple[str, ...]:
        unique: list[str] = []
        seen: set[str] = set()
        for result in results:
            source_id = getattr(getattr(result, "chunk", None), "source_id", "")
            if isinstance(source_id, str) and source_id and source_id not in seen:
                seen.add(source_id)
                unique.append(source_id)
            if len(unique) == 3:
                break
        return tuple(sorted(unique))

    @staticmethod
    def _signature(source_ids: tuple[str, ...]) -> str:
        if not source_ids:
            return ""
        return sha256(",".join(source_ids).encode("utf-8")).hexdigest()

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(_tokens(text)) if isinstance(text, str) and text else set()

    @staticmethod
    def _source_overlap(candidate: Any, current_source_ids: tuple[str, ...]) -> float:
        source_ids_json = _row_value(candidate, "source_ids_json")
        if not isinstance(source_ids_json, str):
            return 0.0
        try:
            candidate_ids = json.loads(source_ids_json)
        except (TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        if (
            not isinstance(candidate_ids, list)
            or not candidate_ids
            or any(not isinstance(source_id, str) or not source_id for source_id in candidate_ids)
        ):
            return 0.0
        candidate_set = frozenset(candidate_ids)
        current_set = frozenset(current_source_ids)
        union = candidate_set | current_set
        return len(candidate_set & current_set) / len(union) if union else 1.0

    def describe(
        self,
        question: str,
        results: list[SearchResult],
        scope: RetrievalScope | None,
    ) -> FaqObservation:
        normalized_question, intent_key = self._question_features(question)
        source_ids = self._source_ids(results)
        source_signature = self._signature(source_ids)
        scope_key = self._scope_key(scope)
        return FaqObservation(
            intent_key=intent_key,
            scope_key=scope_key,
            normalized_question=normalized_question,
            source_signature=source_signature,
            knowledge_revision=int(self.store.knowledge_revision()),
            source_ids=source_ids,
        )

    def lookup_observation(self, observation: FaqObservation) -> FaqMatch | None:
        if not self.enabled:
            return None
        if (
            not observation.intent_key
            or not observation.normalized_question
            or not observation.source_signature
            or not observation.scope_key
        ):
            return None

        candidates = self.store.find_faq_candidates(observation.scope_key)
        current_revision = observation.knowledge_revision
        text_terms = self._token_set(observation.normalized_question)
        stale_marked = False
        best: tuple[float, float, Any] | None = None
        for candidate in candidates:
            if _row_value(candidate, "state", "enabled") != "enabled":
                continue
            if _row_value(candidate, "intent_key") != observation.intent_key:
                continue
            candidate_question = _row_value(
                candidate, "normalized_question", observation.normalized_question
            )
            candidate_terms = self._token_set(candidate_question)
            union = text_terms | candidate_terms
            text_similarity = (
                len(text_terms & candidate_terms) / len(union) if union else 0.0
            )
            if text_similarity < self.min_text_similarity:
                continue
            source_overlap = self._source_overlap(candidate, observation.source_ids)
            if source_overlap < self.min_source_overlap:
                continue
            if _row_value(candidate, "knowledge_revision") != current_revision:
                if not stale_marked:
                    self.store.mark_faq_stale_before_revision(current_revision)
                    stale_marked = True
                continue
            score = (text_similarity, source_overlap)
            if best is None or score > best[:2]:
                best = (text_similarity, source_overlap, candidate)
        if stale_marked:
            return None
        if best is not None:
            candidate = best[2]
            return FaqMatch(
                str(_row_value(candidate, "id")),
                str(_row_value(candidate, "answer", "")),
                observation.intent_key,
                observation.knowledge_revision,
            )
        return None

    def lookup(
        self,
        question: str,
        results: list[SearchResult],
        scope: RetrievalScope | None,
    ) -> FaqMatch | None:
        """Look up a FAQ while preserving the original public API."""
        return self.lookup_observation(self.describe(question, results, scope))

    def record_safe_answer(
        self, observation: FaqObservation, answer: str
    ) -> FaqMatch | None:
        if (
            not self.enabled
            or not isinstance(observation, FaqObservation)
            or not observation.intent_key
            or not observation.normalized_question
            or not observation.source_signature
            or not observation.scope_key
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            return None
        now = datetime.now(timezone.utc)
        return self.store.record_faq_observation(
            observation,
            answer=answer,
            day=now.date().isoformat(),
            now=now.timestamp(),
            promotion_count=self.promotion_count,
            window_days=self.window_days,
        )
