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

    def test_long_paragraph_keeps_every_source_character(self):
        text = "".join(chr(0x4E00 + index) for index in range(100))

        chunks = chunk_text(text, "a.txt", "长文档", max_chars=30, overlap=5)

        self.assertTrue(all(len(chunk.content) <= 30 for chunk in chunks))
        self.assertTrue(set(text).issubset(set().union(*(set(chunk.content) for chunk in chunks))))

    def test_repeated_long_paragraph_keeps_numbered_tokens(self):
        text = "".join(f"条{index:03d}款" for index in range(20))

        chunks = chunk_text(text, "a.txt", "长文档", max_chars=30, overlap=5)

        combined = "".join(chunk.content for chunk in chunks)
        self.assertTrue(all(len(chunk.content) <= 30 for chunk in chunks))
        for index in range(20):
            self.assertIn(f"条{index:03d}款", combined)

    def test_long_paragraph_with_zero_overlap_matches_non_overlapping_windows(self):
        text = "报销" * 50

        chunks = chunk_text(text, "a.txt", "长文档", max_chars=30, overlap=0)

        expected = [text[start : start + 30] for start in (0, 30, 60, 90)]
        self.assertEqual([chunk.content for chunk in chunks], expected)

    def test_paragraph_at_max_chars_is_kept_intact(self):
        text = "报销" * 15

        chunks = chunk_text(text, "a.txt", "长文档", max_chars=30, overlap=5)

        self.assertEqual([chunk.content for chunk in chunks], [text])


if __name__ == "__main__":
    unittest.main()
