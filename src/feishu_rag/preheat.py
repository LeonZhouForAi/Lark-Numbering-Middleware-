"""FAQ 预热候选筛选与后台生成。"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from .faq import FaqService
from .models import (
    Chunk,
    FaqObservation,
    PreheatCandidate,
    RetrievalScope,
)


_TITLE_SIGNALS = (
    "流程",
    "审批",
    "报销",
    "供应商",
    "准入",
    "认证",
    "表单",
    "时限",
    "职责",
    "程序",
    "规范",
)
_EXCLUDED_TITLES = ("目录", "附件索引", "模板说明", "导航")
_STEP_RE = re.compile(r"(?m)^\s*(?:\d+[.、)]|[一二三四五六七八九十]+[、.])\s*")
_BODY_SIGNALS_RE = re.compile(r"部门|工作日|小时|条件|责任|提交|审核|审批|表单")
_DOCUMENT_CODE_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]+){2,}-\d{3}\b", re.IGNORECASE)


def candidate_signature(
    scope_key: str,
    chunk: Chunk,
    knowledge_revision: int,
) -> str:
    content_hash = sha256(chunk.content.encode("utf-8")).hexdigest()
    raw = (
        f"{scope_key}:{chunk.id}:{chunk.source_id}:"
        f"{content_hash}:{knowledge_revision}"
    )
    return sha256(raw.encode("utf-8")).hexdigest()


def _candidate_score(chunk: Chunk) -> int | None:
    title_context = f"{chunk.title} {chunk.section or ''}"
    content = chunk.content.strip()
    if len(content) < 40 or any(signal in title_context for signal in _EXCLUDED_TITLES):
        return None
    score = sum(3 for signal in _TITLE_SIGNALS if signal in title_context)
    if _STEP_RE.search(content):
        score += 2
    if _BODY_SIGNALS_RE.search(content):
        score += 2
    if 300 <= len(content) <= 1500:
        score += 1
    if _DOCUMENT_CODE_RE.search(title_context):
        score += 1
    return score if score > 0 else None


def select_preheat_candidates(
    chunks: Iterable[Chunk],
    *,
    scope_key: str,
    knowledge_revision: int,
    max_per_scope: int,
) -> list[PreheatCandidate]:
    if not scope_key.strip():
        raise ValueError("scope_key must not be empty")
    if type(knowledge_revision) is not int or knowledge_revision < 0:
        raise ValueError("knowledge_revision must be a non-negative integer")
    if type(max_per_scope) is not int or not 1 <= max_per_scope <= 50:
        raise ValueError("max_per_scope must be between 1 and 50")

    ranked: list[tuple[int, Chunk]] = []
    for chunk in chunks:
        score = _candidate_score(chunk)
        if score is not None:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: (-item[0], item[1].source_id, item[1].id))

    candidates: list[PreheatCandidate] = []
    seen_sources: set[str] = set()
    for score, chunk in ranked:
        if chunk.source_id in seen_sources:
            continue
        seen_sources.add(chunk.source_id)
        candidates.append(
            PreheatCandidate(
                signature=candidate_signature(
                    scope_key,
                    chunk,
                    knowledge_revision,
                ),
                scope_key=scope_key,
                knowledge_revision=knowledge_revision,
                chunk_id=chunk.id,
                source_id=chunk.source_id,
                title=chunk.title,
                content=chunk.content,
                score=score,
            )
        )
        if len(candidates) >= max_per_scope:
            break
    return candidates


@dataclass(frozen=True)
class PreheatRunResult:
    job_id: str
    generated: int
    failed: int
    candidates: int


@dataclass(frozen=True)
class _GeneratedFaq:
    canonical_question: str
    aliases: tuple[str, ...]
    answer: str


class PreheatWorker:
    def __init__(
        self,
        store,
        llm,
        *,
        max_per_scope: int = 10,
        workers: int = 2,
    ) -> None:
        if not 1 <= max_per_scope <= 50:
            raise ValueError("max_per_scope must be between 1 and 50")
        if not 1 <= workers <= 8:
            raise ValueError("workers must be between 1 and 8")
        self.store = store
        self.llm = llm
        self.max_per_scope = max_per_scope
        self.workers = workers

    @staticmethod
    def _validated_generation(response: object) -> _GeneratedFaq:
        required = {
            "canonical_question",
            "aliases",
            "answer",
            "evidence_sufficient",
        }
        if not isinstance(response, dict) or set(response) != required:
            raise ValueError("invalid preheat response fields")
        canonical = response["canonical_question"]
        aliases = response["aliases"]
        answer = response["answer"]
        evidence = response["evidence_sufficient"]
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError("invalid canonical question")
        if not isinstance(aliases, list) or len(aliases) > 5 or any(
            not isinstance(alias, str) or not alias.strip() for alias in aliases
        ):
            raise ValueError("invalid aliases")
        if not isinstance(answer, str) or not answer.strip() or evidence is not True:
            raise ValueError("invalid preheat answer")
        texts = [canonical, answer, *aliases]
        if any(FaqService.contains_personal_identifier(text) for text in texts):
            raise ValueError("personal identifier in preheat output")
        from .rag import INSUFFICIENT_ANSWER, UNSAFE_ANSWER, RagService

        cleaned = RagService._clean_answer(answer)
        if cleaned in {INSUFFICIENT_ANSWER, UNSAFE_ANSWER}:
            raise ValueError("unsafe preheat answer")
        return _GeneratedFaq(
            canonical.strip(),
            tuple(alias.strip() for alias in aliases),
            cleaned,
        )

    def _generate(self, candidate: PreheatCandidate) -> _GeneratedFaq:
        response = self.llm.complete_json(
            "你是公司知识库 FAQ 预热器。仅依据资料生成一个标准问题、最多五个同义问法和简洁答案。"
            "不得补造事实,不得输出来源。只返回 canonical_question、aliases、answer、"
            "evidence_sufficient 四个字段的 JSON。",
            f"资料：{candidate.content}",
            purpose="preheat",
        )
        return self._validated_generation(response)

    def run_once(self) -> PreheatRunResult:
        job = self.store.claim_preheat_job()
        if job is None:
            return PreheatRunResult("", 0, 0, 0)
        if self.store.knowledge_revision() != job.knowledge_revision:
            self.store.complete_preheat_job(job.id, generated=0, failed=0)
            return PreheatRunResult(job.id, 0, 0, 0)

        candidates = select_preheat_candidates(
            self.store.preheat_chunks(job.scope_key),
            scope_key=job.scope_key,
            knowledge_revision=job.knowledge_revision,
            max_per_scope=self.max_per_scope,
        )
        pending = [
            candidate
            for candidate in candidates
            if not self.store.preheat_candidate_exists(candidate.signature)
        ]
        generated_count = 0
        failed_count = 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self._generate, candidate) for candidate in pending]
            for candidate, future in zip(pending, futures, strict=True):
                try:
                    generated = future.result()
                    normalized, intent_key = FaqService._question_features(
                        generated.canonical_question
                    )
                    normalized_aliases = []
                    for alias in generated.aliases:
                        alias_normalized, alias_intent = FaqService._question_features(alias)
                        if alias_normalized and alias_intent == intent_key:
                            normalized_aliases.append(alias_normalized)
                    faq_scope = FaqService._scope_key(
                        RetrievalScope(frozenset({job.scope_key}))
                    )
                    source_ids = (candidate.source_id,)
                    observation = FaqObservation(
                        intent_key=intent_key,
                        scope_key=faq_scope,
                        normalized_question=normalized,
                        source_signature=FaqService._signature(source_ids),
                        knowledge_revision=job.knowledge_revision,
                        source_ids=source_ids,
                    )
                    match = self.store.upsert_preheated_faq(
                        candidate,
                        observation,
                        canonical_question=generated.canonical_question,
                        aliases=normalized_aliases,
                        answer=generated.answer,
                    )
                    if match is None:
                        raise ValueError("knowledge revision changed")
                    self.store.record_preheat_candidate(
                        job.id,
                        candidate,
                        "generated",
                        faq_id=match.entry_id,
                    )
                    generated_count += 1
                except Exception:
                    self.store.record_preheat_candidate(
                        job.id,
                        candidate,
                        "failed",
                    )
                    failed_count += 1
        self.store.complete_preheat_job(
            job.id,
            generated=generated_count,
            failed=failed_count,
            candidate_count=len(pending),
        )
        return PreheatRunResult(
            job.id,
            generated_count,
            failed_count,
            len(pending),
        )
