"""DeepSeek 语义分组与原文切片。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .chunker import chunk_text
from .models import Chunk


class SemanticChunkError(ValueError):
    """模型返回的切片分组无法安全应用。"""


@dataclass(frozen=True)
class AtomicUnit:
    unit_id: str
    text: str
    page: int | None = None
    section: str | None = None


class SemanticPlanner(Protocol):
    def plan(self, units: Sequence[AtomicUnit]) -> dict[str, Any]: ...


def _validate_groups(response: dict[str, Any], units: Sequence[AtomicUnit]) -> list[dict[str, Any]]:
    groups = response.get("groups")
    if not isinstance(groups, list) or not groups:
        raise SemanticChunkError("DeepSeek 未返回有效分组")
    expected_ids = [unit.unit_id for unit in units]
    positions = {unit_id: index for index, unit_id in enumerate(expected_ids)}
    seen: list[str] = []
    validated: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict):
            raise SemanticChunkError("语义分组格式无效")
        unit_ids = group.get("unit_ids")
        title = group.get("title")
        keywords = group.get("keywords")
        summary = group.get("summary")
        if (
            not isinstance(unit_ids, list)
            or not unit_ids
            or not all(isinstance(unit_id, str) for unit_id in unit_ids)
            or not isinstance(title, str)
            or not isinstance(keywords, list)
            or not all(isinstance(keyword, str) for keyword in keywords)
            or not isinstance(summary, str)
        ):
            raise SemanticChunkError("语义分组字段类型无效")
        if any(unit_id not in positions for unit_id in unit_ids):
            raise SemanticChunkError("语义分组包含未知段落")
        if any(unit_id in seen for unit_id in unit_ids):
            raise SemanticChunkError("语义分组重复引用段落")
        indexes = [positions[unit_id] for unit_id in unit_ids]
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            raise SemanticChunkError("语义分组必须引用连续段落")
        seen.extend(unit_ids)
        validated.append({"unit_ids": unit_ids, "title": title, "keywords": keywords, "summary": summary})
    if seen != expected_ids:
        raise SemanticChunkError("语义分组遗漏段落或顺序不一致")
    return validated


def semantic_chunks(
    units: Sequence[AtomicUnit],
    source_id: str,
    document_title: str,
    planner: SemanticPlanner,
    max_chars: int = 900,
) -> list[Chunk]:
    if not units:
        return []
    groups = _validate_groups(planner.plan(units), units)
    by_id = {unit.unit_id: unit for unit in units}
    chunks: list[Chunk] = []
    for group_index, group in enumerate(groups):
        group_units = [by_id[unit_id] for unit_id in group["unit_ids"]]
        content = "\n\n".join(unit.text.strip() for unit in group_units).strip()
        if not content:
            raise SemanticChunkError("语义分组正文为空")
        search_text = " ".join([group["title"], *group["keywords"], group["summary"]]).strip()
        group_source = f"{source_id}:group-{group_index}"
        base_chunks = chunk_text(
            content,
            source_id=group_source,
            title=document_title,
            max_chars=max_chars,
            overlap=0,
            page=group_units[0].page,
            section=group_units[0].section,
        )
        chunks.extend(
            Chunk(chunk.id, chunk.source_id, chunk.title, chunk.content, chunk.page, chunk.section, search_text)
            for chunk in base_chunks
        )
    return chunks


class DeepSeekPlanner:
    """将原子段落按批次交给 DeepSeek 进行语义分组。"""

    def __init__(self, llm, batch_chars: int = 12000):
        if batch_chars < 2000:
            raise ValueError("batch_chars 必须至少为 2000")
        self.llm = llm
        self.batch_chars = batch_chars

    def plan(self, units: Sequence[AtomicUnit]) -> dict[str, Any]:
        all_groups: list[dict[str, Any]] = []
        batch: list[AtomicUnit] = []
        batch_size = 0
        for unit in units:
            unit_size = len(unit.text) + len(unit.unit_id) + 32
            if batch and batch_size + unit_size > self.batch_chars:
                all_groups.extend(self._plan_batch(batch))
                batch = []
                batch_size = 0
            batch.append(unit)
            batch_size += unit_size
        if batch:
            all_groups.extend(self._plan_batch(batch))
        return {"groups": all_groups}

    def _plan_batch(self, units: Sequence[AtomicUnit]) -> list[dict[str, Any]]:
        payload = "\n".join(json.dumps({"id": unit.unit_id, "text": unit.text}, ensure_ascii=False) for unit in units)
        system = (
            "你是制度文档切片规划器。只能把输入段落按连续编号分组，不能改写正文。"
            "每个编号必须恰好出现一次。只返回 JSON 对象，字段为 groups；每组包含 unit_ids、title、keywords、summary。"
        )
        result = self.llm.complete_json(system, f"输入段落：\n{payload}")
        return _validate_groups(result, units)
