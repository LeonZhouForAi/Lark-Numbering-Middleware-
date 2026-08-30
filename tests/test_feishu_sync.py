import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feishu_rag.store import IndexStore
from feishu_rag.sync import sync_wiki_space


class FakeFeishuClient:
    def __init__(self):
        self.content = "财务报销制度要求提交发票。"

    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        return {
            "data": {
                "items": [
                    {"node_token": "node-1", "obj_token": "doc-1", "obj_type": "docx", "title": "财务报销制度"},
                    {"node_token": "node-file", "obj_token": "file-1", "obj_type": "file", "title": "流程图.xlsx"},
                ],
                "has_more": False,
            }
        }

    def get_document_raw_content(self, document_id):
        return {"data": {"content": self.content}}


class NestedFeishuClient:
    def __init__(self):
        self.requested_parents = []

    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        self.requested_parents.append(parent_node_token)
        if parent_node_token is None:
            items = [
                {
                    "node_token": "folder-1",
                    "obj_token": "folder-doc",
                    "obj_type": "docx",
                    "title": "03_流程与表单",
                    "has_child": True,
                }
            ]
        else:
            items = [
                {
                    "node_token": "child-1",
                    "obj_token": "child-doc",
                    "obj_type": "docx",
                    "title": "费用报销流程",
                    "has_child": False,
                },
                {
                    "node_token": "file-1",
                    "obj_token": "file-token-1",
                    "obj_type": "file",
                    "title": "付款流程.txt",
                    "has_child": False,
                },
            ]
        return {"data": {"items": items, "has_more": False}}

    def get_document_raw_content(self, document_id):
        content = "目录" if document_id == "folder-doc" else "费用报销应先填写报销单并附发票。"
        return {"data": {"content": content}}

    def download_file(self, file_token):
        assert file_token == "file-token-1"
        return "付款申请应先完成部门审批。".encode("utf-8")


class FeishuSyncTests(unittest.TestCase):
    def test_sync_recursively_indexes_documents_inside_folders(self):
        client = NestedFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                result = sync_wiki_space("space-1", client, store)
                self.assertEqual(result.nodes_seen, 3)
                self.assertEqual(result.indexed, 2)
                self.assertIn("folder-1", client.requested_parents)
                self.assertEqual(store.search("填写报销单")[0].chunk.title, "费用报销流程")
                self.assertEqual(store.search("部门审批")[0].chunk.title, "付款流程.txt")
                self.assertEqual(store.count_documents(), 2)
            finally:
                store.close()

    def test_sync_indexes_docx_nodes_and_skips_unsupported_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                result = sync_wiki_space("space-1", FakeFeishuClient(), store)
                self.assertEqual(result.nodes_seen, 2)
                self.assertEqual(result.indexed, 1)
                self.assertEqual(result.skipped, 1)
                self.assertEqual(store.search("发票")[0].chunk.source_id, "feishu:space-1:node-1")
            finally:
                store.close()

    def test_sync_updates_existing_node_without_duplicate_documents(self):
        client = FakeFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store)
                client.content = "新版报销制度要求电子发票。"
                sync_wiki_space("space-1", client, store)
                self.assertEqual(store.count_documents(), 1)
                self.assertEqual(store.search("电子发票")[0].chunk.title, "财务报销制度")
            finally:
                store.close()

    def test_sync_does_not_reparse_unchanged_file_nodes(self):
        client = NestedFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store)
                with patch("feishu_rag.sync.extract_sections", side_effect=AssertionError("reparsed")):
                    sync_wiki_space("space-1", client, store)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
