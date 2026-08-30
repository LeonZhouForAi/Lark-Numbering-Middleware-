from __future__ import annotations

import pytest

from feishu_rag.semantic_chunker import (
    AtomicUnit,
    DeepSeekPlanner,
    SemanticChunkError,
    semantic_chunks,
)


class FakePlanner:
    def __init__(self, response):
        self.response = response

    def plan(self, units):
        return self.response


class RecordingLLM:
    def __init__(self):
        self.prompts = []

    def complete_json(self, system_prompt, user_prompt):
        self.prompts.append(user_prompt)
        units = [
            AtomicUnit(item["id"], item["text"])
            for item in (__import__("json").loads(line) for line in user_prompt.splitlines()[1:])
        ]
        return {
            "groups": [
                {
                    "unit_ids": [unit.unit_id for unit in units],
                    "title": "批次",
                    "keywords": [],
                    "summary": "",
                }
            ]
        }


def test_semantic_groups_reassemble_original_units_without_rewriting() -> None:
    units = [
        AtomicUnit("u1", "第一条：提交申请。", 1, "申请"),
        AtomicUnit("u2", "第二条：主管审批。", 1, "审批"),
    ]
    planner = FakePlanner(
        {
            "groups": [
                {
                    "unit_ids": ["u1", "u2"],
                    "title": "申请与审批",
                    "keywords": ["申请", "主管审批"],
                    "summary": "申请后由主管审批",
                }
            ]
        }
    )

    chunks = semantic_chunks(units, "finance", "报销制度", planner, 900)

    assert chunks[0].content == "第一条：提交申请。\n\n第二条：主管审批。"
    assert "申请与审批" in chunks[0].search_text
    assert "主管审批" in chunks[0].search_text


@pytest.mark.parametrize(
    "groups",
    [
        [{"unit_ids": ["u1"], "title": "标题", "keywords": [], "summary": "摘要"}],
        [{"unit_ids": ["u1", "u1"], "title": "标题", "keywords": [], "summary": "摘要"}],
        [{"unit_ids": ["u3"], "title": "标题", "keywords": [], "summary": "摘要"}],
        [{"unit_ids": ["u2"], "title": "标题", "keywords": [], "summary": "摘要"}],
        [{"unit_ids": [], "title": "标题", "keywords": [], "summary": "摘要"}],
    ],
)
def test_semantic_groups_reject_missing_duplicate_unknown_or_non_contiguous_units(groups) -> None:
    units = [AtomicUnit("u1", "第一段", 1, None), AtomicUnit("u2", "第二段", 1, None)]

    with pytest.raises(SemanticChunkError):
        semantic_chunks(units, "source", "文档", FakePlanner({"groups": groups}), 900)


def test_semantic_group_split_keeps_original_text_when_over_limit() -> None:
    units = [AtomicUnit("u1", "甲" * 60, 1, None), AtomicUnit("u2", "乙" * 60, 1, None)]
    planner = FakePlanner(
        {"groups": [{"unit_ids": ["u1", "u2"], "title": "合并", "keywords": [], "summary": ""}]}
    )

    chunks = semantic_chunks(units, "source", "文档", planner, 80)

    assert len(chunks) > 1
    assert "".join(chunk.content for chunk in chunks).replace("\n", "") == "甲" * 60 + "乙" * 60


def test_deepseek_planner_splits_long_input_into_bounded_batches() -> None:
    llm = RecordingLLM()
    planner = DeepSeekPlanner(llm, batch_chars=2000)
    units = [AtomicUnit(f"u{index}", "制度段落" * 100) for index in range(8)]

    result = planner.plan(units)

    assert len(llm.prompts) > 1
    assert len(result["groups"]) == len(llm.prompts)
