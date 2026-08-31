import unittest
from dataclasses import FrozenInstanceError
import json
import tempfile
from pathlib import Path

from feishu_rag.faq import FaqService
from feishu_rag.models import Chunk, FaqMatch, FaqObservation, RetrievalScope, SearchResult
from feishu_rag.store import IndexStore


def _results(*source_ids: str) -> list[SearchResult]:
    return [
        SearchResult(Chunk(f"{source_id}-chunk", source_id, "供应商准入", "流程内容"), 0.95)
        for source_id in source_ids
    ]


class _FakeStore:
    def __init__(self, candidates=None, revision=4):
        self.candidates = list(candidates or [])
        self.revision = revision
        self.recorded = []
        self.marked = []

    def knowledge_revision(self):
        return self.revision

    def find_faq_candidates(self, scope_key):
        return self.candidates

    def mark_faq_stale_before_revision(self, revision):
        self.marked.append(revision)
        return 1

    def record_faq_observation(self, observation, **kwargs):
        self.recorded.append((observation, kwargs))
        return FaqMatch("faq-1", kwargs["answer"], observation.intent_key)


class FaqModelsTests(unittest.TestCase):
    def test_faq_match_is_immutable(self):
        match = FaqMatch(entry_id="faq-1", answer="answer", intent_key="reset-password")
        self.assertEqual(match.entry_id, "faq-1")
        with self.assertRaises(FrozenInstanceError):
            match.answer = "changed"

    def test_faq_observation_has_expected_fields_and_is_immutable(self):
        observation = FaqObservation(
            intent_key="reset-password",
            scope_key="space-a",
            source_signature="sig",
            knowledge_revision=4,
            normalized_question="密码怎么重置",
        )
        self.assertEqual(observation.knowledge_revision, 4)
        self.assertEqual(observation.normalized_question, "密码怎么重置")
        self.assertEqual(observation.source_ids, ())
        with self.assertRaises(FrozenInstanceError):
            observation.scope_key = "space-b"


class FaqServiceTests(unittest.TestCase):
    def _service(self, store=None, **kwargs):
        return FaqService(
            store or _FakeStore(),
            enabled=True,
            promotion_count=3,
            window_days=15,
            min_text_similarity=0.82,
            min_source_overlap=0.80,
            **kwargs,
        )

    def test_supplier_synonyms_share_intent(self):
        service = self._service()
        first = service.describe("供应商开发流程是什么", _results("supplier"), None)
        second = service.describe("新供应商怎么导入", _results("supplier"), None)
        self.assertEqual(first.intent_key, second.intent_key)

    def test_scope_keys_are_isolated_and_order_independent(self):
        service = self._service()
        first = service.describe(
            "采购审批流程", _results("finance"), RetrievalScope(frozenset({"b", "a"}))
        )
        second = service.describe(
            "采购审批流程", _results("finance"), RetrievalScope(frozenset({"a", "b"}))
        )
        other = service.describe(
            "采购审批流程", _results("finance"), RetrievalScope(frozenset({"a"}))
        )
        self.assertEqual(first.scope_key, second.scope_key)
        self.assertNotEqual(first.scope_key, other.scope_key)

    def test_source_signature_is_stable_and_uses_first_three_unique_sources(self):
        service = self._service()
        first = service.describe("供应商开发", _results("c", "a", "b", "d", "a"), None)
        second = service.describe("供应商开发", _results("a", "b", "c", "d"), None)
        self.assertEqual(first.source_signature, second.source_signature)

    def test_nfkc_and_question_shells_are_normalized(self):
        service = self._service()
        first = service.describe("ＡＢＣ是什么", _results("supplier"), None)
        second = service.describe("abc", _results("supplier"), None)
        self.assertEqual(first.intent_key, second.intent_key)

    def test_no_results_are_not_recorded(self):
        store = _FakeStore()
        service = self._service(store)
        observation = service.describe("供应商开发", [], None)
        self.assertIsNone(service.record_safe_answer(observation, "答案"))
        self.assertEqual(store.recorded, [])

    def test_disabled_service_does_not_record(self):
        store = _FakeStore()
        service = FaqService(store, False, 3, 15, 0.82, 0.8)
        observation = service.describe("供应商开发", _results("supplier"), None)
        self.assertIsNone(service.lookup("供应商开发", _results("supplier"), None))
        self.assertIsNone(service.record_safe_answer(observation, "答案"))
        self.assertEqual(store.recorded, [])

    def test_lookup_requires_current_revision_and_marks_old_entry_stale(self):
        store = _FakeStore(revision=4)
        service = self._service(store)
        observation = service.describe("供应商开发", _results("supplier"), None)
        store.candidates = [{
            "id": "faq-1", "intent_key": observation.intent_key, "answer": "旧答案",
            "normalized_question": observation.normalized_question,
            "source_signature": observation.source_signature,
            "source_ids_json": json.dumps(list(observation.source_ids)),
            "knowledge_revision": 3, "state": "enabled",
        }]
        self.assertIsNone(service.lookup("供应商开发", _results("supplier"), None))
        self.assertEqual(store.marked, [4])

    def test_unrelated_old_candidate_does_not_block_current_matching_candidate(self):
        store = _FakeStore(revision=4)
        service = self._service(store)
        observation = service.describe("供应商开发", _results("supplier"), None)
        store.candidates = [
            {
                "id": "old-faq", "answer": "旧答案", "normalized_question": "财务报销",
                "source_signature": "finance", "source_ids_json": json.dumps(["finance"]),
                "knowledge_revision": 3, "state": "enabled",
            },
            {
                "id": "current-faq", "answer": "新答案",
                "normalized_question": observation.normalized_question,
                "source_signature": observation.source_signature,
                "source_ids_json": json.dumps(list(observation.source_ids)),
                "knowledge_revision": 4, "state": "enabled",
            },
        ]
        self.assertEqual(
            service.lookup("供应商开发", _results("supplier"), None),
            FaqMatch("current-faq", "新答案", observation.intent_key),
        )
        self.assertEqual(store.marked, [])

    def test_lookup_rejects_different_source_signature(self):
        store = _FakeStore()
        service = self._service(store)
        observation = service.describe("供应商开发", _results("supplier"), None)
        store.candidates = [{
            "id": "faq-1", "intent_key": observation.intent_key, "answer": "答案",
            "normalized_question": observation.normalized_question,
            "search_text": observation.normalized_question,
            "source_signature": "other", "knowledge_revision": 4, "state": "enabled",
        }]
        self.assertIsNone(service.lookup("供应商开发", _results("supplier"), None))

    def test_lookup_rejects_low_source_overlap_even_when_text_is_similar(self):
        service = self._service()
        observation = service.describe("供应商开发", _results("source-a", "source-c"), None)
        service.store.candidates = [{
            "id": "faq-1", "intent_key": observation.intent_key, "answer": "答案",
            "normalized_question": observation.normalized_question,
            "search_text": observation.normalized_question,
            "source_signature": "other", "source_ids": ["source-a", "source-b"],
            "source_ids_json": json.dumps(["source-a", "source-b"]),
            "knowledge_revision": 4, "state": "enabled",
        }]
        self.assertIsNone(
            service.lookup("供应商开发", _results("source-a", "source-c"), None)
        )

    def test_lookup_accepts_current_matching_entry(self):
        service = self._service()
        observation = service.describe("供应商开发", _results("supplier"), None)
        store = service.store
        store.candidates = [{
            "id": "faq-1", "intent_key": observation.intent_key, "answer": "答案",
            "normalized_question": observation.normalized_question,
            "search_text": " ".join(observation.normalized_question.split()),
            "source_signature": observation.source_signature,
            "source_ids_json": json.dumps(list(observation.source_ids)),
            "knowledge_revision": 4, "state": "enabled",
        }]
        match = service.lookup("供应商开发", _results("supplier"), None)
        self.assertEqual(match, FaqMatch("faq-1", "答案", observation.intent_key))

    def test_record_safe_answer_forwards_promotion_configuration(self):
        store = _FakeStore()
        service = FaqService(store, True, 7, 30, 0.82, 0.8)
        observation = service.describe("供应商开发", _results("supplier"), None)
        match = service.record_safe_answer(observation, "安全答案")
        self.assertIsNotNone(match)
        _, kwargs = store.recorded[0]
        self.assertEqual(kwargs["promotion_count"], 7)
        self.assertEqual(kwargs["window_days"], 30)
        self.assertRegex(kwargs["day"], r"^\d{4}-\d{2}-\d{2}$")

    def test_describe_persists_sorted_first_three_unique_source_ids(self):
        service = self._service()
        observation = service.describe("供应商开发", _results("c", "a", "b", "d", "a"), None)
        self.assertEqual(observation.source_ids, ("a", "b", "c"))

    def test_non_string_question_is_rejected_safely(self):
        service = self._service()
        self.assertIsNone(service.lookup(None, _results("supplier"), None))

    def test_real_store_lookup_survives_service_restart_and_uses_all_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "faq.sqlite3"
            store = IndexStore(db_path)
            try:
                service = FaqService(store, True, 2, 15, 0.70, 0.80)
                results = _results("supplier-a", "supplier-b")
                observation = service.describe("供应商开发流程是什么", results, None)
                service.record_safe_answer(observation, "安全答案")
                service.record_safe_answer(observation, "安全答案")
                alias = service.describe("供应商开发怎么做", results, None)
                service.record_safe_answer(alias, "安全答案")
                store.close()

                restarted = IndexStore(db_path)
                try:
                    fresh_service = FaqService(restarted, True, 2, 15, 0.70, 0.80)
                    match = fresh_service.lookup("供应商开发怎么执行", results, None)
                    self.assertIsNotNone(match)
                finally:
                    restarted.close()
            finally:
                if store.connection is not None:
                    try:
                        store.close()
                    except Exception:
                        pass

    def test_real_store_lookup_rejects_invalid_persisted_source_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "faq.sqlite3")
            try:
                service = FaqService(store, True, 1, 15, 0.5, 0.8)
                observation = service.describe("供应商开发", _results("supplier"), None)
                service.record_safe_answer(observation, "安全答案")
                store.connection.execute(
                    "UPDATE faq_entries SET source_ids_json = ?", ("not-json",)
                )
                store.connection.commit()
                self.assertIsNone(service.lookup("供应商开发", _results("supplier"), None))
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
