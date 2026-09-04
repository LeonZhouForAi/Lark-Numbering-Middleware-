"""FAQ 预热候选筛选与后台生成。"""

from __future__ import annotations

import re
from hashlib import sha256
from typing import Iterable

from .models import Chunk, PreheatCandidate


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
