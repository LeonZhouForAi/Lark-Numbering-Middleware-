from feishu_rag.faq import FaqService
from feishu_rag.models import Chunk, RetrievalScope, SearchResult
from feishu_rag.preheat import (
    PreheatWorker,
    candidate_signature,
    select_preheat_candidates,
)
from feishu_rag.store import IndexStore


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


class FakePreheatLLM:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls = []

    def complete_json(self, system_prompt, user_prompt, *, purpose="answer"):
        self.calls.append((system_prompt, user_prompt, purpose))
        return self.response


def _seed_preheat_job(store: IndexStore) -> Chunk:
    chunk = _chunk(1)
    store.upsert_document(
        chunk.source_id,
        chunk.title,
        "supplier.txt",
        "v1",
        [chunk],
        space_id="space-a",
    )
    store.bump_knowledge_revision(now=1.0)
    store.enqueue_preheat_job("space-a", 1, max_retries=1, now=2.0)
    return chunk


def test_preheated_candidate_creates_enabled_faq_without_observations(tmp_path) -> None:
    store = IndexStore(tmp_path / "rag.sqlite3")
    chunk = _seed_preheat_job(store)
    llm = FakePreheatLLM(
        {
            "canonical_question": "供应商开发流程是什么？",
            "aliases": ["新供应商怎么导入？"],
            "answer": "先审核资质，再完成现场认证并提交审批。",
            "evidence_sufficient": True,
        }
    )
    try:
        result = PreheatWorker(store, llm, max_per_scope=10, workers=2).run_once()

        assert result.generated == 1
        assert store.connection.execute(
            "SELECT COUNT(*) FROM faq_entries WHERE origin='preheated' AND state='enabled'"
        ).fetchone()[0] == 1
        faq = FaqService(store, True, 3, 15, 0.6, 0.8)
        match = faq.lookup(
            "新供应商怎么导入？",
            [SearchResult(chunk, 1.0)],
            RetrievalScope(frozenset({"space-a"})),
        )
        assert match is not None
        assert match.answer == "先审核资质,再完成现场认证并提交审批。"
    finally:
        store.close()


def test_preheat_rejects_personal_identifier_in_answer(tmp_path) -> None:
    store = IndexStore(tmp_path / "rag.sqlite3")
    _seed_preheat_job(store)
    llm = FakePreheatLLM(
        {
            "canonical_question": "供应商开发流程是什么？",
            "aliases": [],
            "answer": "请联系张三办理供应商准入。",
            "evidence_sufficient": True,
        }
    )
    try:
        result = PreheatWorker(store, llm, max_per_scope=10, workers=2).run_once()

        assert result.generated == 0
        assert result.failed == 1
        assert store.connection.execute("SELECT COUNT(*) FROM faq_entries").fetchone()[0] == 0
    finally:
        store.close()


def test_stale_preheat_job_does_not_call_llm(tmp_path) -> None:
    store = IndexStore(tmp_path / "rag.sqlite3")
    _seed_preheat_job(store)
    store.bump_knowledge_revision(now=3.0)
    llm = FakePreheatLLM({})
    try:
        result = PreheatWorker(store, llm, max_per_scope=10, workers=2).run_once()

        assert result.generated == 0
        assert result.failed == 0
        assert llm.calls == []
    finally:
        store.close()


def test_preheat_cli_outputs_counts_without_generated_content(
    monkeypatch, capsys, tmp_path
) -> None:
    from types import SimpleNamespace

    from scripts import preheat_faq

    settings = SimpleNamespace(
        deepseek_api_key="secret",
        deepseek_base_url="https://example.invalid",
        deepseek_model="model",
        api_retry_max_attempts=1,
        api_retry_base_delay=0.0,
        rag_faq_preheat_max_per_space=10,
        rag_faq_preheat_workers=2,
    )
    fake_store = SimpleNamespace(close=lambda: None)

    class FakeWorker:
        def __init__(self, store, llm, *, max_per_scope, workers):
            assert store is fake_store
            assert (max_per_scope, workers) == (10, 2)

        def run_once(self):
            return SimpleNamespace(
                job_id="job-1", candidates=3, generated=2, failed=1
            )

    monkeypatch.setattr(preheat_faq.Settings, "from_env", lambda: settings)
    monkeypatch.setattr(preheat_faq, "IndexStore", lambda path: fake_store)
    monkeypatch.setattr(preheat_faq, "DeepSeekClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(preheat_faq, "PreheatWorker", FakeWorker)

    assert preheat_faq.main([str(tmp_path / "rag.sqlite3"), "--once"]) == 0

    output = capsys.readouterr().out
    assert output == "job_id=job-1 candidates=3 generated=2 failed=1\n"
    assert "标准问题" not in output
