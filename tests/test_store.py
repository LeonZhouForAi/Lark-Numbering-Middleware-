import tempfile
import unittest
from pathlib import Path

from feishu_rag.models import Chunk
from feishu_rag.store import IndexStore


class StoreTests(unittest.TestCase):
    def test_prune_documents_removes_only_stale_documents_in_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "feishu:space-a:keep", "保留", "keep.txt", "v1", [Chunk("keep-chunk", "feishu:space-a:keep", "保留", "保留内容")]
                )
                store.upsert_document(
                    "feishu:space-a:stale", "过期", "stale.txt", "v1", [Chunk("stale-chunk", "feishu:space-a:stale", "过期", "过期内容")]
                )
                store.upsert_document(
                    "feishu:space-b:other", "其他", "other.txt", "v1", [Chunk("other-chunk", "feishu:space-b:other", "其他", "其他内容")]
                )
                store.upsert_document(
                    "local.txt", "本地", "local.txt", "v1", [Chunk("local-chunk", "local.txt", "本地", "本地内容")]
                )

                self.assertEqual(
                    store.prune_documents("feishu:space-a:", {"feishu:space-a:keep"}),
                    1,
                )
                self.assertEqual(store.search("过期"), [])
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'stale-chunk'").fetchone()[0], 0)
                self.assertEqual(store.count_chunks("feishu:space-a:stale"), 0)
                self.assertEqual(store.document_checksum("feishu:space-a:keep"), "v1")
                self.assertEqual(store.count_chunks("feishu:space-a:keep"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'keep-chunk'").fetchone()[0], 1)
                self.assertEqual(store.document_checksum("feishu:space-b:other"), "v1")
                self.assertEqual(store.count_chunks("feishu:space-b:other"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'other-chunk'").fetchone()[0], 1)
                self.assertEqual(store.document_checksum("local.txt"), "v1")
                self.assertEqual(store.count_chunks("local.txt"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'local-chunk'").fetchone()[0], 1)
                self.assertEqual(store.prune_documents("feishu:space-a:", {"feishu:space-a:keep"}), 0)
            finally:
                store.close()

    def test_prune_documents_validates_prefix_and_retained_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                with self.assertRaises(ValueError):
                    store.prune_documents("", set())
                with self.assertRaises(ValueError):
                    store.prune_documents("feishu:space-a:", {"feishu:space-b:other"})
            finally:
                store.close()

    def test_prune_documents_with_empty_retained_clears_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "feishu:space-a:one", "一", "one.txt", "v1", [Chunk("one-chunk", "feishu:space-a:one", "一", "内容一")]
                )
                store.upsert_document(
                    "feishu:space-b:two", "二", "two.txt", "v1", [Chunk("two-chunk", "feishu:space-b:two", "二", "内容二")]
                )

                self.assertEqual(store.prune_documents("feishu:space-a:", set()), 1)
                self.assertIsNone(store.document_checksum("feishu:space-a:one"))
                self.assertEqual(store.document_checksum("feishu:space-b:two"), "v1")
            finally:
                store.close()

    def test_prune_documents_requires_single_feishu_space_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                for prefix in ("", "local:", "feishu:", "feishu:space:a:", "FEISHU:space-a:"):
                    with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                        store.prune_documents(prefix, set())
            finally:
                store.close()

    def test_prune_documents_uses_case_sensitive_literal_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "FEISHU:space-a:upper", "大写", "upper.txt", "v1", [Chunk("upper-chunk", "FEISHU:space-a:upper", "大写", "大写内容")]
                )
                self.assertEqual(store.prune_documents("feishu:space-a:", set()), 0)
                self.assertEqual(store.document_checksum("FEISHU:space-a:upper"), "v1")
                self.assertEqual(store.count_chunks("FEISHU:space-a:upper"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'upper-chunk'").fetchone()[0], 1)
            finally:
                store.close()

    def test_prune_documents_handles_more_than_sqlite_variable_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                for index in range(1200):
                    source_id = f"feishu:space-a:stale-{index}"
                    store.upsert_document(
                        source_id, "过期", f"{index}.txt", "v1", [Chunk(f"chunk-{index}", source_id, "过期", "过期内容")]
                    )
                self.assertEqual(store.prune_documents("feishu:space-a:", set()), 1200)
                self.assertEqual(store.count_documents(), 0)
                self.assertEqual(store.count_chunks(), 0)
            finally:
                store.close()

    def test_prune_documents_rolls_back_fts_when_document_delete_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "feishu:space-a:stale", "过期", "stale.txt", "v1", [Chunk("rollback-chunk", "feishu:space-a:stale", "过期", "过期内容")]
                )
                store.connection.execute(
                    """
                    CREATE TRIGGER fail_document_delete
                    BEFORE DELETE ON documents
                    BEGIN
                        SELECT RAISE(ABORT, 'blocked');
                    END
                    """
                )
                with self.assertRaisesRegex(Exception, "blocked"):
                    store.prune_documents("feishu:space-a:", set())
                self.assertEqual(store.document_checksum("feishu:space-a:stale"), "v1")
                self.assertEqual(store.count_chunks("feishu:space-a:stale"), 1)
                self.assertEqual(store.connection.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunk_id = 'rollback-chunk'").fetchone()[0], 1)
            finally:
                store.close()

    def test_sqlite_connection_uses_concurrency_safe_pragmas(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertEqual(
                    store.connection.execute("PRAGMA journal_mode").fetchone()[0],
                    "wal",
                )
                self.assertGreaterEqual(
                    store.connection.execute("PRAGMA busy_timeout").fetchone()[0],
                    30000,
                )
                self.assertEqual(
                    store.connection.execute("PRAGMA synchronous").fetchone()[0],
                    1,
                )
            finally:
                store.close()

    def test_search_text_is_searchable_but_original_content_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "财务制度",
                    "finance.pdf",
                    "v1",
                    [
                        Chunk(
                            "c-meta",
                            "finance.pdf",
                            "财务制度",
                            "提交费用申请单。",
                            search_text="报销 付款 审批流程",
                        )
                    ],
                )
                result = store.search("报销审批", 3)[0].chunk
                self.assertEqual(result.content, "提交费用申请单。")
                self.assertEqual(result.search_text, "报销 付款 审批流程")
            finally:
                store.close()

    def test_message_claim_prevents_duplicate_processing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                self.assertTrue(store.claim_message("om_duplicate"))
                self.assertFalse(store.claim_message("om_duplicate"))
                store.release_message("om_duplicate")
                self.assertTrue(store.claim_message("om_duplicate"))
            finally:
                store.close()

    def test_search_returns_chinese_keyword_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    source_id="finance/reimbursement.pdf",
                    title="财务报销制度",
                    path="finance/reimbursement.pdf",
                    checksum="v1",
                    chunks=[
                        Chunk("c1", "finance/reimbursement.pdf", "财务报销制度", "差旅费报销需要提供发票。", 2, "差旅")
                    ],
                )

                results = store.search("发票报销", top_k=3)

                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].chunk.title, "财务报销制度")
                self.assertEqual(results[0].chunk.page, 2)
            finally:
                store.close()

    def test_reindex_replaces_previous_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    source_id="policy.txt",
                    title="旧政策",
                    path="policy.txt",
                    checksum="v1",
                    chunks=[Chunk("old", "policy.txt", "旧政策", "旧内容")],
                )
                store.upsert_document(
                    source_id="policy.txt",
                    title="新制度",
                    path="policy.txt",
                    checksum="v2",
                    chunks=[Chunk("new", "policy.txt", "新制度", "新内容")],
                )

                self.assertEqual(store.count_documents(), 1)
                self.assertEqual(store.count_chunks("policy.txt"), 1)
                self.assertEqual(store.search("旧政策"), [])
                self.assertEqual(store.search("新内容")[0].chunk.title, "新制度")
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
