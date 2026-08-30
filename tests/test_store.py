import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from feishu_rag.models import Chunk, RetrievalScope
from feishu_rag.store import IndexStore, _pretokenize


class StoreTests(unittest.TestCase):
    def test_retrieval_scope_preserves_an_explicit_immutable_space_set(self):
        allowed = frozenset({"space-a", "space-b"})
        self.assertEqual(RetrievalScope(allowed).allowed_space_ids, allowed)

    def test_pretokenize_normalizes_nfkc_case_and_chinese_terms(self):
        terms = _pretokenize("ＡＢＣ１２ 报销")
        self.assertIn("abc12", terms.split())
        self.assertIn("报销", terms.split())
        self.assertIn("报", terms.split())
        self.assertIn("销", terms.split())

    def test_initialization_migrates_old_fts_schema_and_backfills_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT,
                    search_text TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO documents VALUES('legacy.pdf', '报销指南', 'legacy.pdf', 'v1', 0);
                INSERT INTO chunks VALUES(
                    'legacy-chunk', 'legacy.pdf', '报销指南', '提交申请单。', NULL, NULL,
                    '电子发票 核验'
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title, content);
                INSERT INTO chunks_fts VALUES('legacy-chunk', '报销指南', '提交申请单。');
                """
            )
            connection.commit()
            connection.close()

            store = IndexStore(db_path)
            try:
                columns = [
                    row[1]
                    for row in store.connection.execute("PRAGMA table_info(chunks_fts)").fetchall()
                ]
                self.assertEqual(
                    columns,
                    ["chunk_id", "title_terms", "content_terms", "search_terms"],
                )
                fts_row = store.connection.execute(
                    "SELECT title_terms,content_terms,search_terms FROM chunks_fts "
                    "WHERE chunk_id = 'legacy-chunk'"
                ).fetchone()
                self.assertIn("报销", fts_row[0].split())
                self.assertIn("申请", fts_row[1].split())
                self.assertIn("发票", fts_row[2].split())
                self.assertEqual(store.search("电子发票")[0].chunk.id, "legacy-chunk")
            finally:
                store.close()

    def test_failed_fts_backfill_rolls_back_and_next_start_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            connection = sqlite3.connect(db_path)
            connection.executescript(
                """
                CREATE TABLE documents (
                    source_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    path TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES documents(source_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    page INTEGER,
                    section TEXT,
                    search_text TEXT NOT NULL DEFAULT ''
                );
                INSERT INTO documents VALUES('one.pdf', '报销指南', 'one.pdf', 'v1', 0);
                INSERT INTO documents VALUES('two.pdf', '发票指南', 'two.pdf', 'v1', 0);
                INSERT INTO chunks VALUES(
                    'one', 'one.pdf', '报销指南', '提交申请。', NULL, NULL, ''
                );
                INSERT INTO chunks VALUES(
                    'two', 'two.pdf', '发票指南', '核验发票。', NULL, NULL, ''
                );
                CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title, content);
                INSERT INTO chunks_fts VALUES('one', '报销指南', '提交申请。');
                INSERT INTO chunks_fts VALUES('two', '发票指南', '核验发票。');
                """
            )
            connection.commit()
            connection.close()

            call_count = 0

            def fail_during_second_row(text):
                nonlocal call_count
                call_count += 1
                if call_count == 5:
                    raise sqlite3.OperationalError("injected backfill failure")
                return _pretokenize(text)

            with patch("feishu_rag.store._pretokenize", side_effect=fail_during_second_row):
                failed_store = IndexStore(db_path)
            try:
                self.assertFalse(failed_store._fts_available)
                columns = [
                    row[1]
                    for row in failed_store.connection.execute(
                        "PRAGMA table_info(chunks_fts)"
                    ).fetchall()
                ]
                self.assertEqual(columns, ["chunk_id", "title", "content"])
                self.assertEqual(
                    failed_store.connection.execute(
                        "SELECT COUNT(*) FROM chunks_fts"
                    ).fetchone()[0],
                    2,
                )
            finally:
                failed_store.close()

            recovered_store = IndexStore(db_path)
            try:
                columns = [
                    row[1]
                    for row in recovered_store.connection.execute(
                        "PRAGMA table_info(chunks_fts)"
                    ).fetchall()
                ]
                self.assertEqual(
                    columns,
                    ["chunk_id", "title_terms", "content_terms", "search_terms"],
                )
                self.assertEqual(
                    {
                        row[0]
                        for row in recovered_store.connection.execute(
                            "SELECT chunk_id FROM chunks_fts"
                        ).fetchall()
                    },
                    {"one", "two"},
                )
            finally:
                recovered_store.close()

    def test_initialization_rebuilds_v2_fts_when_ids_do_not_match_chunks(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "rag.sqlite3"
            store = IndexStore(db_path)
            try:
                store.upsert_document(
                    "one.pdf",
                    "报销指南",
                    "one.pdf",
                    "v1",
                    [Chunk("one", "one.pdf", "报销指南", "提交申请。")],
                )
                store.upsert_document(
                    "two.pdf",
                    "发票指南",
                    "two.pdf",
                    "v1",
                    [Chunk("two", "two.pdf", "发票指南", "核验发票。")],
                )
                store.connection.execute("DELETE FROM chunks_fts WHERE chunk_id = 'two'")
                store.connection.execute(
                    "INSERT INTO chunks_fts VALUES('orphan', '孤儿', '孤儿', '')"
                )
                store.connection.commit()
            finally:
                store.close()

            recovered_store = IndexStore(db_path)
            try:
                self.assertEqual(
                    [
                        row[0]
                        for row in recovered_store.connection.execute(
                            "SELECT chunk_id FROM chunks_fts ORDER BY chunk_id"
                        ).fetchall()
                    ],
                    ["one", "two"],
                )
            finally:
                recovered_store.close()

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

    def test_search_ranks_the_specific_chinese_document_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "travel.pdf",
                    "差旅报销办法",
                    "travel.pdf",
                    "v1",
                    [Chunk("travel", "travel.pdf", "差旅报销办法", "差旅发票须经部门审批。")],
                )
                store.upsert_document(
                    "invoice.pdf",
                    "发票开具说明",
                    "invoice.pdf",
                    "v1",
                    [Chunk("invoice", "invoice.pdf", "发票开具说明", "电子发票可以下载。")],
                )

                results = store.search("请问差旅发票怎么报销", top_k=2)

                self.assertEqual(results[0].chunk.id, "travel")
                self.assertGreaterEqual(results[0].score, 0.42)
                self.assertLessEqual(results[0].score, 1.0)
            finally:
                store.close()

    def test_search_returns_no_result_for_generic_or_unrelated_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "财务制度",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "财务制度", "报销流程按规定办理。")],
                )
                for query in ("请问是什么制度", "流程规定办法", "量子芯片温度"):
                    with self.subTest(query=query):
                        self.assertEqual(store.search(query), [])
            finally:
                store.close()

    def test_search_strips_common_which_question_shells(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "invoice.pdf",
                    "电子发票管理规定",
                    "invoice.pdf",
                    "v1",
                    [Chunk("invoice", "invoice.pdf", "电子发票管理规定", "电子发票需要核验。")],
                )
                for query in ("电子发票有什么规定", "请问电子发票有哪些规定"):
                    with self.subTest(query=query):
                        self.assertEqual(store.search(query)[0].chunk.id, "invoice")
            finally:
                store.close()

    def test_search_treats_match_syntax_as_plain_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "报销指引",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "报销指引", "提交电子发票。")],
                )
                results = store.search('电子发票" OR * NOT (')
                self.assertEqual(results[0].chunk.id, "finance")
            finally:
                store.close()

    def test_search_falls_back_to_literal_scan_when_fts_is_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "报销指引",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "报销指引", "提交电子发票。")],
                )
                store._fts_available = False
                self.assertEqual(store.search("电子发票")[0].chunk.id, "finance")
            finally:
                store.close()

    def test_search_applies_configurable_confidence_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "报销指引",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "报销指引", "提交电子发票。")],
                )
                result = store.search("电子发票", min_relevance=0.0)[0]
                self.assertGreaterEqual(result.score, 0.0)
                self.assertLessEqual(result.score, 1.0)
                self.assertEqual(store.search("电子发票", min_relevance=1.0), [])
            finally:
                store.close()

    def test_search_accepts_scope_without_claiming_to_filter_before_acl_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                store.upsert_document(
                    "finance.pdf",
                    "报销指引",
                    "finance.pdf",
                    "v1",
                    [Chunk("finance", "finance.pdf", "报销指引", "提交电子发票。")],
                )
                scope = RetrievalScope(frozenset({"space-a"}))
                self.assertEqual(store.search("电子发票", scope=scope)[0].chunk.id, "finance")
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
