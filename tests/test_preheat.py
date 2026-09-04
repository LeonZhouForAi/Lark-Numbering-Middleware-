from feishu_rag.models import Chunk
from feishu_rag.preheat import candidate_signature, select_preheat_candidates


def _chunk(index: int, *, source: str | None = None, title: str = "供应商开发流程", content: str | None = None) -> Chunk:
    body = content or (
        "供应商准入条件如下：\n1. 采购部审核资质。\n2. 品质部完成现场认证。\n"
        "3. 申请人在三个工作日内提交审批表单。" + "补充要求。" * 30
    )
    return Chunk(
        f"chunk-{index:02d}",
        source or f"source-{index:02d}",
        title,
        body,
    )


def test_select_candidates_keeps_one_chunk_per_source_and_ten_per_scope() -> None:
    chunks = [_chunk(index) for index in range(12)]
    chunks.append(_chunk(99, source="source-00"))

    candidates = select_preheat_candidates(
        chunks,
        scope_key="space-a",
        knowledge_revision=3,
        max_per_scope=10,
    )

    assert len(candidates) == 10
    assert len({candidate.source_id for candidate in candidates}) == 10
    assert {candidate.scope_key for candidate in candidates} == {"space-a"}
    assert {candidate.knowledge_revision for candidate in candidates} == {3}


def test_process_steps_rank_above_directory_and_short_chunks() -> None:
    candidates = select_preheat_candidates(
        [
            _chunk(1),
            _chunk(2, title="附件目录", content="附件一、附件二"),
            _chunk(3, title="模板说明", content="请填写模板"),
        ],
        scope_key="space-a",
        knowledge_revision=1,
        max_per_scope=10,
    )

    assert [candidate.chunk_id for candidate in candidates] == ["chunk-01"]


def test_candidate_order_and_signature_are_deterministic() -> None:
    chunks = [_chunk(2), _chunk(1)]

    first = select_preheat_candidates(
        chunks,
        scope_key="space-a",
        knowledge_revision=4,
        max_per_scope=10,
    )
    second = select_preheat_candidates(
        list(reversed(chunks)),
        scope_key="space-a",
        knowledge_revision=4,
        max_per_scope=10,
    )

    assert first == second
    assert [candidate.chunk_id for candidate in first] == ["chunk-01", "chunk-02"]
    assert first[0].signature == candidate_signature(
        "space-a", chunks[1], knowledge_revision=4
    )


def test_candidate_signature_changes_with_content_or_revision() -> None:
    chunk = _chunk(1)
    changed = Chunk(chunk.id, chunk.source_id, chunk.title, chunk.content + "新要求")

    assert candidate_signature("space-a", chunk, 1) != candidate_signature(
        "space-a", changed, 1
    )
    assert candidate_signature("space-a", chunk, 1) != candidate_signature(
        "space-a", chunk, 2
    )
