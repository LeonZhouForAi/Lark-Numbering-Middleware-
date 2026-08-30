import tempfile
import unittest
from pathlib import Path

from feishu_rag.models import Chunk
from feishu_rag.store import IndexStore


class StoreTests(unittest.TestCase):
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
