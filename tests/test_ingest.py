import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from docx import Document

from feishu_rag.ingest import (
    DocumentExtractionError,
    Section,
    UnsupportedFileError,
    _ocr_pdf_page,
    _read_pdf,
    extract_sections,
    index_file,
    index_directory,
)
from feishu_rag.store import IndexStore


class IngestTests(unittest.TestCase):
    def test_pdf_ocr_mode_change_reindexes_file_and_same_mode_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "scan.pdf"
            path.write_bytes(b"same-pdf-bytes")
            store = IndexStore(root / "rag.sqlite3")
            try:
                with patch(
                    "feishu_rag.ingest.extract_sections",
                    return_value=[Section(text="PDF 正文")],
                ) as extract:
                    self.assertTrue(index_file(path, root, store, enable_ocr=False))
                    self.assertFalse(index_file(path, root, store, enable_ocr=False))
                    self.assertTrue(index_file(path, root, store, enable_ocr=True))
                    self.assertFalse(index_file(path, root, store, enable_ocr=True))

                self.assertEqual(extract.call_count, 2)
                self.assertEqual(
                    [call.kwargs["enable_ocr"] for call in extract.call_args_list],
                    [False, True],
                )
            finally:
                store.close()

    def test_non_pdf_ocr_mode_change_keeps_file_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "policy.txt"
            path.write_text("同一份文本内容", encoding="utf-8")
            store = IndexStore(root / "rag.sqlite3")
            try:
                with patch(
                    "feishu_rag.ingest.extract_sections",
                    return_value=[Section(text="同一份文本内容")],
                ) as extract:
                    self.assertTrue(index_file(path, root, store, enable_ocr=False))
                    self.assertFalse(index_file(path, root, store, enable_ocr=True))

                self.assertEqual(extract.call_count, 1)
            finally:
                store.close()

    def test_index_file_can_use_semantic_planner_metadata(self):
        class Planner:
            def plan(self, units):
                return {
                    "groups": [
                        {
                            "unit_ids": [unit.unit_id for unit in units],
                            "title": "付款语义标题",
                            "keywords": ["付款关键词"],
                            "summary": "付款语义摘要",
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            docs = root / "docs"
            docs.mkdir()
            path = docs / "policy.txt"
            path.write_text("付款申请应由主管审批。", encoding="utf-8")
            store = IndexStore(root / "rag.sqlite3")
            try:
                count = index_directory(docs, store, semantic_planner=Planner())
                self.assertEqual(count, 1)
                result = store.search("付款关键词")[0].chunk
                self.assertEqual(result.content, "付款申请应由主管审批。")
            finally:
                store.close()

    def test_extracts_text_and_docx_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            txt = root / "制度.txt"
            txt.write_text("报销制度\n\n需要发票。", encoding="utf-8")
            docx_path = root / "流程.docx"
            doc = Document()
            doc.add_paragraph("付款申请流程")
            doc.add_paragraph("申请人提交审批。")
            doc.save(docx_path)

            txt_sections = extract_sections(txt)
            docx_sections = extract_sections(docx_path)

            self.assertIn("报销制度", "\n".join(section.text for section in txt_sections))
            self.assertIn("付款申请流程", "\n".join(section.text for section in docx_sections))

    def test_extracts_docx_paragraphs_and_table_rows_in_document_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "流程.docx"
            document = Document()
            document.add_paragraph("段落 A")
            table = document.add_table(rows=2, cols=2)
            table.cell(0, 0).text = "申请人"
            table.cell(0, 1).text = "先提交"
            table.cell(0, 1).add_paragraph("再审批")
            table.cell(1, 0).text = ""
            table.cell(1, 1).text = ""
            document.add_paragraph("段落 B")
            document.save(path)

            sections = extract_sections(path)

            self.assertEqual(
                [section.text for section in sections],
                ["段落 A\n\n申请人 | 先提交 再审批\n\n段落 B"],
            )

    def test_extracts_text_from_docx_that_contains_only_a_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "表格.docx"
            document = Document()
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "费用类型"
            table.cell(0, 1).text = "交通费"
            document.save(path)

            sections = extract_sections(path)

            self.assertEqual([section.text for section in sections], ["费用类型 | 交通费"])

    def test_extracts_each_merged_docx_table_cell_only_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "合并单元格.docx"
            document = Document()
            table = document.add_table(rows=3, cols=3)
            table.cell(0, 0).merge(table.cell(0, 1)).text = "横向合并"
            table.cell(0, 2).text = "第一行末"
            table.cell(1, 0).merge(table.cell(2, 0)).text = "纵向合并"
            table.cell(1, 1).text = "第二行中"
            table.cell(1, 2).text = "第二行末"
            table.cell(2, 1).text = "第三行中"
            table.cell(2, 2).text = "第三行末"
            document.save(path)

            sections = extract_sections(path)

            self.assertEqual(
                [section.text for section in sections],
                [
                    "横向合并 | 第一行末\n\n"
                    "纵向合并 | 第二行中 | 第二行末\n\n"
                    "第三行中 | 第三行末"
                ],
            )

    def test_extracts_visible_text_from_a_nested_docx_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "嵌套表格.docx"
            document = Document()
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "外层标题"
            nested_cell = table.cell(0, 1)
            nested_cell.text = "外层说明"
            nested_table = nested_cell.add_table(rows=1, cols=2)
            nested_table.cell(0, 0).text = "内层左"
            nested_table.cell(0, 1).text = "内层右"
            document.save(path)

            sections = extract_sections(path)

            self.assertEqual(
                [section.text for section in sections],
                ["外层标题 | 外层说明 / 内层左 | 内层右"],
            )

    def test_extracts_all_cells_from_a_large_unmerged_docx_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "大表格.docx"
            document = Document()
            table = document.add_table(rows=50, cols=10)
            expected_tokens = {f"cell-{row}-{column}" for row in range(50) for column in range(10)}
            for row in range(50):
                for column in range(10):
                    table.cell(row, column).text = f"cell-{row}-{column}"
            document.save(path)

            sections = extract_sections(path)

            extracted_tokens = [
                cell for line in sections[0].text.splitlines() if line for cell in line.split(" | ")
            ]
            self.assertEqual(len(extracted_tokens), 500)
            self.assertEqual(set(extracted_tokens), expected_tokens)

    def test_read_pdf_uses_ocr_only_for_blank_pages_and_preserves_page_order(self):
        class Page:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class Reader:
            pages = [Page("原生第一页"), Page(""), Page("原生第三页")]

        path = Path("扫描件.pdf")
        with (
            patch("pypdf.PdfReader", return_value=Reader()),
            patch("feishu_rag.ingest._ocr_pdf_page", return_value="第二页 OCR") as ocr,
        ):
            sections = _read_pdf(path)

        self.assertEqual(
            [(section.text, section.page) for section in sections],
            [("原生第一页", 1), ("第二页 OCR", 2), ("原生第三页", 3)],
        )
        ocr.assert_called_once_with(path, 2)

    def test_read_pdf_does_not_ocr_blank_pages_when_ocr_is_disabled(self):
        class Page:
            def extract_text(self):
                return ""

        class Reader:
            pages = [Page()]

        with (
            patch("pypdf.PdfReader", return_value=Reader()),
            patch("feishu_rag.ingest._ocr_pdf_page") as ocr,
        ):
            sections = _read_pdf(Path("扫描件.pdf"), enable_ocr=False)

        self.assertEqual(sections, [])
        ocr.assert_not_called()

    def test_read_pdf_propagates_ocr_failure_instead_of_returning_partial_text(self):
        class Page:
            def __init__(self, text):
                self.text = text

            def extract_text(self):
                return self.text

        class Reader:
            pages = [Page("可提取文本"), Page("")]

        with (
            patch("pypdf.PdfReader", return_value=Reader()),
            patch(
                "feishu_rag.ingest._ocr_pdf_page",
                side_effect=DocumentExtractionError("OCR 不可用"),
            ),
        ):
            with self.assertRaisesRegex(DocumentExtractionError, "OCR 不可用"):
                _read_pdf(Path("混合.pdf"))

    def test_ocr_pdf_page_renders_only_the_requested_page_and_falls_back_to_english(self):
        image = object()
        path = Path("扫描件.pdf")
        with (
            patch("pdf2image.convert_from_path", return_value=[image]) as convert,
            patch(
                "pytesseract.image_to_string",
                side_effect=[RuntimeError("missing chi_sim"), "recognized text"],
            ) as recognize,
        ):
            text = _ocr_pdf_page(path, 3)

        self.assertEqual(text, "recognized text")
        convert.assert_called_once_with(str(path), first_page=3, last_page=3, dpi=220)
        self.assertEqual(
            [call.kwargs["lang"] for call in recognize.call_args_list],
            ["chi_sim", "eng"],
        )

    def test_ocr_pdf_page_ignores_successful_empty_text(self):
        with (
            patch("pdf2image.convert_from_path", return_value=[object()]),
            patch("pytesseract.image_to_string", return_value="  "),
        ):
            text = _ocr_pdf_page(Path("空白扫描件.pdf"), 1)

        self.assertIsNone(text)

    def test_ocr_pdf_page_raises_when_rendering_or_all_languages_fail(self):
        with patch("pdf2image.convert_from_path", side_effect=RuntimeError("poppler missing")):
            with self.assertRaises(DocumentExtractionError):
                _ocr_pdf_page(Path("扫描件.pdf"), 1)

        with (
            patch("pdf2image.convert_from_path", return_value=[object()]),
            patch("pytesseract.image_to_string", side_effect=RuntimeError("tesseract missing")),
        ):
            with self.assertRaises(DocumentExtractionError):
                _ocr_pdf_page(Path("扫描件.pdf"), 1)

    def test_ocr_pdf_page_raises_when_ocr_dependencies_are_unavailable(self):
        import builtins

        original_import = builtins.__import__

        def import_without_pytesseract(name, *args, **kwargs):
            if name == "pytesseract":
                raise ImportError("pytesseract missing")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_pytesseract):
            with self.assertRaises(DocumentExtractionError):
                _ocr_pdf_page(Path("扫描件.pdf"), 1)

    def test_rejects_legacy_word_and_wps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for suffix in (".doc", ".wps"):
                path = root / f"legacy{suffix}"
                path.write_bytes(b"legacy")
                with self.assertRaisesRegex(UnsupportedFileError, "先转换为 DOCX"):
                    extract_sections(path)

    def test_indexes_supported_files_recursively(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "财务").mkdir()
            (root / "财务" / "报销.md").write_text("报销申请需要发票。", encoding="utf-8")
            store = IndexStore(root / "index.sqlite3")
            try:
                indexed = index_directory(root, store, max_chars=100)
                self.assertEqual(indexed, 1)
                self.assertEqual(store.count_documents(), 1)
                self.assertEqual(store.search("发票报销")[0].chunk.title, "报销")
            finally:
                store.close()

    def test_index_directory_bumps_revision_once_only_when_files_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.txt").write_text("第一份制度。", encoding="utf-8")
            (root / "two.md").write_text("第二份制度。", encoding="utf-8")
            store = IndexStore(root / "index.sqlite3")
            try:
                self.assertEqual(index_directory(root, store), 2)
                self.assertEqual(store.knowledge_revision(), 1)

                self.assertEqual(index_directory(root, store), 0)
                self.assertEqual(store.knowledge_revision(), 1)
            finally:
                store.close()

    def test_index_directory_does_not_bump_when_indexing_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "one.txt").write_text("第一份制度。", encoding="utf-8")
            (root / "two.txt").write_text("第二份制度。", encoding="utf-8")
            store = IndexStore(root / "index.sqlite3")
            try:
                with patch(
                    "feishu_rag.ingest.index_file",
                    side_effect=[True, RuntimeError("index failed")],
                ):
                    with self.assertRaisesRegex(RuntimeError, "index failed"):
                        index_directory(root, store)
                self.assertEqual(store.knowledge_revision(), 0)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
