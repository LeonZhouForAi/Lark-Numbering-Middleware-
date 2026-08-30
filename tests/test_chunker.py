import unittest

from feishu_rag.chunker import chunk_text


class ChunkerTests(unittest.TestCase):
    def test_chunks_keep_source_and_respect_limit(self):
        text = "第一段讲费用报销制度。\n\n第二段讲付款申请流程。\n\n第三段讲审批权限。"

        chunks = chunk_text(
            text,
            source_id="finance/reimbursement.pdf",
            title="财务报销制度",
            max_chars=20,
            overlap=4,
            page=3,
            section="流程",
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.content), 20)
            self.assertEqual(chunk.source_id, "finance/reimbursement.pdf")
            self.assertEqual(chunk.title, "财务报销制度")
            self.assertEqual(chunk.page, 3)
            self.assertEqual(chunk.section, "流程")

    def test_long_paragraph_is_split_without_empty_chunks(self):
        chunks = chunk_text("报销" * 80, "a.txt", "长文档", max_chars=30, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.content for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
