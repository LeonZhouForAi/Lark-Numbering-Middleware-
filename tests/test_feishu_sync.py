import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from feishu_rag.logging_utils import configure_logging
from feishu_rag.ingest import Section, index_file
from feishu_rag.models import RetrievalScope
from feishu_rag.store import IndexStore
from feishu_rag import sync as sync_module
from feishu_rag.sync import SyncResult, sync_wiki_space


class FakeFeishuClient:
    def __init__(self):
        self.content = "财务报销制度要求提交发票。"

    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        return {
            "data": {
                "items": [
                    {
                        "node_token": "node-1",
                        "obj_token": "doc-1",
                        "obj_type": "docx",
                        "title": "财务报销制度",
                        "has_child": False,
                    },
                    {
                        "node_token": "node-file",
                        "obj_token": "file-1",
                        "obj_type": "file",
                        "title": "流程图.xlsx",
                        "has_child": False,
                    },
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


class SemanticPlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, units):
        self.calls += 1
        return {
            "groups": [
                {
                    "unit_ids": [unit.unit_id for unit in units],
                    "title": "语义标题",
                    "keywords": ["语义关键词"],
                    "summary": "语义摘要",
                }
            ]
        }


class FailingPlanner:
    def plan(self, units):
        raise RuntimeError("DeepSeek unavailable")


class EmptyFeishuClient:
    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        return {"data": {"items": [], "has_more": False}}


class StaticResponseFeishuClient:
    def __init__(self, response):
        self.response = response

    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        return self.response


class MutableFileFeishuClient:
    def __init__(self):
        self.content = "有效附件内容。".encode("utf-8")
        self.title = "附件.txt"

    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        return {
            "data": {
                "items": [
                    {
                        "node_token": "file-node",
                        "obj_token": "file-token",
                        "obj_type": "file",
                        "title": self.title,
                        "has_child": False,
                    }
                ],
                "has_more": False,
            }
        }

    def download_file(self, file_token):
        return self.content


class UnsupportedParentFeishuClient:
    def __init__(self):
        self.requested_parents = []

    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        self.requested_parents.append(parent_node_token)
        if parent_node_token is None:
            items = [
                {
                    "node_token": "sheet-parent",
                    "obj_token": "sheet-1",
                    "obj_type": "sheet",
                    "title": "不支持的表格父节点",
                    "has_child": True,
                }
            ]
        else:
            items = [
                {
                    "node_token": "supported-child",
                    "obj_token": "child-doc",
                    "obj_type": "docx",
                    "title": "可同步子文档",
                    "has_child": False,
                }
            ]
        return {"data": {"items": items, "has_more": False}}

    def get_document_raw_content(self, document_id):
        return {"data": {"content": "父节点类型不受支持时，子文档仍应同步。"}}


class PaginationFailureClient:
    def __init__(self, failure):
        self.failure = failure

    def list_wiki_nodes(self, space_id, page_token=None, page_size=50, parent_node_token=None):
        if page_token is None:
            response = {
                "data": {
                    "items": [
                        {
                            "node_token": "new-node",
                            "obj_token": "new-doc",
                            "obj_type": "docx",
                            "title": "新文档",
                            "has_child": False,
                        }
                    ],
                    "has_more": True,
                }
            }
            if self.failure != "missing_token":
                response["data"]["page_token"] = "next-page"
            return response
        if self.failure == "second_page":
            raise RuntimeError("second page unavailable")
        return {"data": {"items": [], "has_more": True, "page_token": "next-page"}}

    def get_document_raw_content(self, document_id):
        return {"data": {"content": "第一页的新内容允许先写入。"}}


class FeishuSyncTests(unittest.TestCase):
    def test_feishu_sync_error_is_runtime_error(self):
        error_type = getattr(sync_module, "FeishuSyncError", None)
        self.assertIsNotNone(error_type)
        self.assertTrue(issubclass(error_type, RuntimeError))

    def test_sync_uses_semantic_planner_metadata(self):
        client = FakeFeishuClient()
        planner = SemanticPlanner()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                result = sync_wiki_space("space-1", client, store, semantic_planner=planner)
                self.assertEqual(result.indexed, 1)
                self.assertEqual(planner.calls, 1)
                self.assertEqual(store.search("语义关键词")[0].chunk.content, client.content)
            finally:
                store.close()

    def test_sync_falls_back_to_local_chunks_when_planner_fails(self):
        client = FakeFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                with self.assertLogs("feishu_rag.sync", level="WARNING") as logs:
                    result = sync_wiki_space("space-1", client, store, semantic_planner=FailingPlanner())
                self.assertEqual(result.indexed, 1)
                self.assertTrue(store.search("财务报销制度要求提交发票。"))
                output = "\n".join(logs.output)
                self.assertIn("semantic_chunk_fallback", output)
                self.assertIn("feishu:space-1:node-1", output)
                self.assertIn("RuntimeError", output)
                self.assertNotIn(client.content, output)
            finally:
                store.close()

    def test_sync_stores_actual_feishu_space_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", FakeFeishuClient(), store)

                self.assertEqual(
                    store.connection.execute(
                        "SELECT DISTINCT space_id FROM documents"
                    ).fetchall()[0][0],
                    "space-1",
                )
            finally:
                store.close()

    def test_unchanged_docx_backfills_legacy_space_without_rechunking(self):
        client = FakeFeishuClient()
        planner = SemanticPlanner()
        scope = RetrievalScope(frozenset({"space-1"}))
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store, semantic_planner=planner)
                store.connection.execute(
                    "UPDATE documents SET space_id = '' "
                    "WHERE source_id = 'feishu:space-1:node-1'"
                )
                store.connection.commit()

                result = sync_wiki_space(
                    "space-1", client, store, semantic_planner=planner
                )

                self.assertEqual(result.indexed, 0)
                self.assertEqual(planner.calls, 1)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT space_id FROM documents "
                        "WHERE source_id = 'feishu:space-1:node-1'"
                    ).fetchone()[0],
                    "space-1",
                )
                self.assertTrue(store.search("财务报销", scope=scope))
            finally:
                store.close()

    def test_unchanged_file_backfills_legacy_space_without_reparsing(self):
        client = MutableFileFeishuClient()
        planner = SemanticPlanner()
        scope = RetrievalScope(frozenset({"space-1"}))
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store, semantic_planner=planner)
                store.connection.execute(
                    "UPDATE documents SET space_id = '' "
                    "WHERE source_id = 'feishu:space-1:file-node'"
                )
                store.connection.commit()

                with patch(
                    "feishu_rag.sync.extract_sections",
                    side_effect=AssertionError("unchanged file must not be reparsed"),
                ):
                    result = sync_wiki_space(
                        "space-1", client, store, semantic_planner=planner
                    )

                self.assertEqual(result.indexed, 0)
                self.assertEqual(planner.calls, 1)
                self.assertEqual(
                    store.connection.execute(
                        "SELECT space_id FROM documents "
                        "WHERE source_id = 'feishu:space-1:file-node'"
                    ).fetchone()[0],
                    "space-1",
                )
                self.assertTrue(store.search("有效附件", scope=scope))
            finally:
                store.close()

    def test_configure_logging_uses_safe_default_for_unknown_level(self):
        with patch("feishu_rag.logging_utils.logging.basicConfig") as basic_config:
            configure_logging("not-a-level")

        kwargs = basic_config.call_args.kwargs
        self.assertEqual(kwargs["level"], "INFO")
        self.assertIn("%(asctime)s", kwargs["format"])
        self.assertIn("%(levelname)s", kwargs["format"])
        self.assertIn("%(name)s", kwargs["format"])
        self.assertIn("%(message)s", kwargs["format"])
        self.assertTrue(kwargs["force"])

    def test_configure_logging_applies_the_latest_level(self):
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ | {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(project_root / "src"),
        }
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging; from feishu_rag.logging_utils import configure_logging; "
                "configure_logging('INFO'); configure_logging('DEBUG'); "
                "print(logging.getLogger().getEffectiveLevel())",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )

        self.assertEqual(result.stdout.strip(), "10")

    def test_local_indexing_without_planner_does_not_log_semantic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "policy.txt"
            path.write_text("本地模式的正文", encoding="utf-8")
            store = IndexStore(root / "rag.sqlite3")
            try:
                with self.assertNoLogs("feishu_rag.ingest", level="WARNING"):
                    indexed = index_file(path, root, store)
                self.assertTrue(indexed)
            finally:
                store.close()

    def test_local_indexing_logs_redacted_fallback_for_failing_planner(self):
        content = "不应写入日志的本地正文"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "policy.txt"
            path.write_text(content, encoding="utf-8")
            store = IndexStore(root / "rag.sqlite3")
            try:
                with self.assertLogs("feishu_rag.ingest", level="WARNING") as logs:
                    indexed = index_file(path, root, store, semantic_planner=FailingPlanner())
                self.assertTrue(indexed)
                output = "\n".join(logs.output)
                self.assertIn("semantic_chunk_fallback", output)
                self.assertIn("policy.txt", output)
                self.assertIn("RuntimeError", output)
                self.assertNotIn(content, output)
            finally:
                store.close()

    def test_sync_skips_unchanged_docx_without_calling_planner(self):
        client = FakeFeishuClient()
        first_planner = SemanticPlanner()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space(
                    "space-1",
                    client,
                    store,
                    semantic_planner=first_planner,
                    chunk_strategy_version="hybrid-v1",
                    chunk_model="deepseek-v4-flash",
                )
                second_planner = SemanticPlanner()
                result = sync_wiki_space(
                    "space-1",
                    client,
                    store,
                    semantic_planner=second_planner,
                    chunk_strategy_version="hybrid-v1",
                    chunk_model="deepseek-v4-flash",
                )
                self.assertEqual(result.indexed, 0)
                self.assertEqual(result.deleted, 0)
                self.assertEqual(second_planner.calls, 0)
                self.assertEqual(store.count_documents(), 1)
                self.assertTrue(store.search("财务报销制度要求提交发票。"))
            finally:
                store.close()

    def test_sync_recursively_indexes_documents_inside_folders(self):
        client = NestedFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                result = sync_wiki_space("space-1", client, store)
                self.assertEqual(result.nodes_seen, 3)
                self.assertEqual(result.indexed, 3)
                self.assertEqual(result.deleted, 0)
                self.assertIn("folder-1", client.requested_parents)
                self.assertEqual(store.search("目录")[0].chunk.title, "03_流程与表单")
                self.assertEqual(store.search("填写报销单")[0].chunk.title, "费用报销流程")
                self.assertEqual(store.search("部门审批")[0].chunk.title, "付款流程.txt")
                self.assertEqual(store.count_documents(), 3)
            finally:
                store.close()

    def test_sync_recurses_through_unsupported_parent_nodes(self):
        client = UnsupportedParentFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                result = sync_wiki_space("space-1", client, store)
                self.assertEqual(result.nodes_seen, 2)
                self.assertEqual(result.indexed, 1)
                self.assertEqual(result.skipped, 1)
                self.assertIn("sheet-parent", client.requested_parents)
                self.assertTrue(store.search("子文档仍应同步"))
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

    def test_sync_prunes_documents_removed_from_the_space(self):
        client = FakeFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store)
                client.list_wiki_nodes = EmptyFeishuClient().list_wiki_nodes

                result = sync_wiki_space("space-1", client, store)

                self.assertEqual(result.deleted, 1)
                self.assertFalse(store.search("财务报销制度要求提交发票。"))
                self.assertEqual(store.count_documents(), 0)
            finally:
                store.close()

    def test_successful_empty_space_prunes_only_that_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-a", FakeFeishuClient(), store)
                sync_wiki_space("space-b", FakeFeishuClient(), store)

                result = sync_wiki_space("space-a", EmptyFeishuClient(), store)

                self.assertEqual(result.deleted, 1)
                self.assertIsNone(store.document_checksum("feishu:space-a:node-1"))
                self.assertIsNotNone(store.document_checksum("feishu:space-b:node-1"))
                self.assertEqual(store.count_documents(), 1)
            finally:
                store.close()

    def test_second_page_failure_does_not_prune_existing_documents(self):
        self._assert_pagination_failure_does_not_prune("second_page", "RuntimeError")

    def test_missing_page_token_does_not_prune_existing_documents(self):
        self._assert_pagination_failure_does_not_prune("missing_token")

    def test_repeated_page_token_does_not_prune_existing_documents(self):
        self._assert_pagination_failure_does_not_prune("repeated_token")

    def _assert_pagination_failure_does_not_prune(self, failure, error_name="FeishuSyncError"):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", FakeFeishuClient(), store)

                with self.assertRaises(Exception) as raised:
                    sync_wiki_space("space-1", PaginationFailureClient(failure), store)

                self.assertEqual(type(raised.exception).__name__, error_name)
                self.assertTrue(store.search("财务报销制度要求提交发票。"))
            finally:
                store.close()

    def test_missing_data_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune({"code": 0})

    def test_non_list_items_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {"data": {"items": {}, "has_more": False}}
        )

    def test_missing_has_more_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune({"data": {"items": []}})

    def test_invalid_node_token_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "",
                            "obj_token": "doc-1",
                            "obj_type": "docx",
                            "title": "无效节点",
                        }
                    ],
                    "has_more": False,
                }
            }
        )

    def test_missing_obj_type_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "invalid-type-node",
                            "obj_token": "doc-1",
                            "title": "缺少类型",
                        }
                    ],
                    "has_more": False,
                }
            }
        )

    def test_non_string_obj_type_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "invalid-type-node",
                            "obj_token": "doc-1",
                            "obj_type": 123,
                            "title": "错误类型",
                        }
                    ],
                    "has_more": False,
                }
            }
        )

    def test_non_boolean_has_child_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "invalid-child-node",
                            "obj_token": "doc-1",
                            "obj_type": "docx",
                            "has_child": "false",
                            "title": "错误子节点标记",
                        }
                    ],
                    "has_more": False,
                }
            }
        )

    def test_missing_has_child_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "missing-child-node",
                            "obj_token": "doc-1",
                            "obj_type": "docx",
                            "title": "缺少子节点标记",
                        }
                    ],
                    "has_more": False,
                }
            }
        )

    def test_malformed_duplicate_node_is_validated_before_deduplication(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "duplicate-node",
                            "obj_token": "sheet-1",
                            "obj_type": "sheet",
                            "has_child": False,
                            "title": "合法节点",
                        },
                        {
                            "node_token": "duplicate-node",
                            "obj_token": "sheet-1",
                            "has_child": False,
                            "title": "畸形重复节点",
                        },
                    ],
                    "has_more": False,
                }
            }
        )

    def test_conflicting_duplicate_object_type_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "node-1",
                            "obj_token": "doc-1",
                            "obj_type": "sheet",
                            "has_child": False,
                            "title": "财务报销制度",
                        },
                        {
                            "node_token": "node-1",
                            "obj_token": "doc-1",
                            "obj_type": "docx",
                            "has_child": False,
                            "title": "财务报销制度",
                        },
                    ],
                    "has_more": False,
                }
            }
        )

    def test_conflicting_duplicate_has_child_does_not_prune_existing_documents(self):
        self._assert_invalid_snapshot_does_not_prune(
            {
                "data": {
                    "items": [
                        {
                            "node_token": "node-1",
                            "obj_token": "doc-1",
                            "obj_type": "sheet",
                            "has_child": False,
                            "title": "财务报销制度",
                        },
                        {
                            "node_token": "node-1",
                            "obj_token": "doc-1",
                            "obj_type": "sheet",
                            "has_child": True,
                            "title": "财务报销制度",
                        },
                    ],
                    "has_more": False,
                }
            }
        )

    def _assert_invalid_snapshot_does_not_prune(self, response):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", FakeFeishuClient(), store)

                with self.assertRaises(sync_module.FeishuSyncError):
                    sync_wiki_space("space-1", StaticResponseFeishuClient(response), store)

                self.assertTrue(store.search("财务报销制度要求提交发票。"))
                self.assertEqual(store.count_documents(), 1)
            finally:
                store.close()

    def test_empty_docx_content_prunes_the_previous_version(self):
        client = FakeFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store)
                client.content = ""

                result = sync_wiki_space("space-1", client, store)

                self.assertEqual(result.deleted, 1)
                self.assertIsNone(store.document_checksum("feishu:space-1:node-1"))
                self.assertFalse(store.search("财务报销制度要求提交发票。"))
            finally:
                store.close()

    def test_missing_raw_content_fields_do_not_prune_existing_document(self):
        responses = [
            {"code": 0},
            {"data": {}},
            {"data": {"document": {}}},
        ]
        for response in responses:
            with self.subTest(response=response):
                self._assert_invalid_raw_content_does_not_prune(response)

    def test_invalid_raw_content_types_do_not_prune_existing_document(self):
        responses = [
            {"data": {"content": 123}},
            {"data": {"content": [{"missing": "text"}]}},
            {"data": {"content": [{"text": 123}]}},
            {"data": {"content": [object()]}},
        ]
        for response in responses:
            with self.subTest(response=response):
                self._assert_invalid_raw_content_does_not_prune(response)

    def _assert_invalid_raw_content_does_not_prune(self, response):
        client = FakeFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store)
                client.get_document_raw_content = lambda document_id: response

                with self.assertRaises(sync_module.FeishuSyncError):
                    sync_wiki_space("space-1", client, store)

                self.assertTrue(store.search("财务报销制度要求提交发票。"))
                self.assertEqual(store.count_documents(), 1)
            finally:
                store.close()

    def test_empty_file_parse_prunes_the_previous_version(self):
        client = MutableFileFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                sync_wiki_space("space-1", client, store)
                client.content = "附件已变更但无法解析。".encode("utf-8")

                with patch("feishu_rag.sync.extract_sections", return_value=[]):
                    result = sync_wiki_space("space-1", client, store)

                self.assertEqual(result.deleted, 1)
                self.assertIsNone(store.document_checksum("feishu:space-1:file-node"))
                self.assertFalse(store.search("有效附件内容"))
            finally:
                store.close()

    def test_sync_passes_disabled_ocr_to_attachment_extraction(self):
        client = MutableFileFeishuClient()
        client.title = "扫描附件.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                with patch(
                    "feishu_rag.sync.extract_sections",
                    return_value=[Section(text="无需 OCR 的附件内容。")],
                ) as extract:
                    result = sync_wiki_space("space-1", client, store, enable_ocr=False)

                self.assertEqual(result.indexed, 1)
                extract.assert_called_once()
                self.assertFalse(extract.call_args.kwargs["enable_ocr"])
            finally:
                store.close()

    def test_pdf_ocr_mode_change_reindexes_attachment_and_same_mode_skips(self):
        client = MutableFileFeishuClient()
        client.title = "扫描附件.pdf"
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                with patch(
                    "feishu_rag.sync.extract_sections",
                    return_value=[Section(text="PDF 附件内容。")],
                ) as extract:
                    first = sync_wiki_space("space-1", client, store, enable_ocr=False)
                    repeated_false = sync_wiki_space("space-1", client, store, enable_ocr=False)
                    changed = sync_wiki_space("space-1", client, store, enable_ocr=True)
                    repeated_true = sync_wiki_space("space-1", client, store, enable_ocr=True)

                self.assertEqual(first.indexed, 1)
                self.assertEqual(repeated_false.indexed, 0)
                self.assertEqual(changed.indexed, 1)
                self.assertEqual(repeated_true.indexed, 0)
                self.assertEqual(extract.call_count, 2)
                self.assertEqual(
                    [call.kwargs["enable_ocr"] for call in extract.call_args_list],
                    [False, True],
                )
            finally:
                store.close()

    def test_non_pdf_ocr_mode_change_keeps_attachment_cache(self):
        client = MutableFileFeishuClient()
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                with patch(
                    "feishu_rag.sync.extract_sections",
                    return_value=[Section(text="文本附件内容。")],
                ) as extract:
                    first = sync_wiki_space("space-1", client, store, enable_ocr=False)
                    changed = sync_wiki_space("space-1", client, store, enable_ocr=True)

                self.assertEqual(first.indexed, 1)
                self.assertEqual(changed.indexed, 0)
                self.assertEqual(extract.call_count, 1)
            finally:
                store.close()

    def test_cli_output_and_completion_log_include_deleted_count(self):
        store = MagicMock()
        settings = MagicMock(
            feishu_space_id="space-1",
            feishu_app_id="app-id",
            feishu_app_secret="app-secret",
            rag_semantic_chunking=False,
            rag_enable_ocr=False,
            rag_chunk_strategy_version="local-v1",
            deepseek_chunk_model="",
            api_retry_max_attempts=3,
            api_retry_base_delay=0.5,
            log_level="INFO",
        )
        stdout = StringIO()
        with (
            patch("feishu_rag.sync.Settings.from_env", return_value=settings),
            patch("feishu_rag.sync.configure_logging"),
            patch("feishu_rag.sync.FeishuClient"),
            patch("feishu_rag.sync.IndexStore", return_value=store),
            patch("feishu_rag.sync.sync_wiki_space", return_value=SyncResult(3, 2, 1, 4)) as sync,
            patch.object(sys, "argv", ["feishu-rag-sync"]),
            self.assertLogs("feishu_rag.sync", level="INFO") as logs,
            redirect_stdout(stdout),
        ):
            sync_module.main()

        self.assertIn("deleted=4", stdout.getvalue())
        self.assertIn("deleted=4", "\n".join(logs.output))
        self.assertFalse(sync.call_args.kwargs["enable_ocr"])
        store.close.assert_called_once_with()

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
