import tempfile
import unittest
from pathlib import Path

from feishu_rag.llm import DeepSeekClient
from feishu_rag.models import Chunk
from feishu_rag.rag import RagService
from feishu_rag.store import IndexStore


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "根据制度，员工需要先提交申请。"


class RagTests(unittest.TestCase):
    def test_no_evidence_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            llm = FakeLLM()
            try:
                answer = RagService(store, llm).answer("完全不存在的流程")
            finally:
                store.close()

        self.assertIn("暂无依据", answer.text)
        self.assertEqual(llm.calls, [])
        self.assertEqual(answer.citations, [])

    def test_hit_answer_hides_sources_but_keeps_internal_citations(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            store.upsert_document(
                "finance.txt",
                "财务报销制度",
                "finance.txt",
                "v1",
                [Chunk("c1", "finance.txt", "财务报销制度", "报销需要提交发票。", 2, "报销")],
            )
            llm = FakeLLM()
            try:
                answer = RagService(store, llm).answer("报销需要什么")
            finally:
                store.close()

        self.assertNotIn("来源：", answer.text)
        self.assertNotIn("[1]", answer.text)
        self.assertIn("仅依据资料", llm.calls[0][0])
        self.assertIn("不得输出资料编号", llm.calls[0][0])
        self.assertEqual(len(answer.citations), 1)


class DeepSeekClientTests(unittest.TestCase):
    def test_posts_chat_completion_payload(self):
        seen = {}

        def transport(url, headers, payload, timeout):
            seen.update(url=url, headers=headers, payload=payload, timeout=timeout)
            return 200, b'{"choices":[{"message":{"content":"ok"}}]}'

        client = DeepSeekClient(
            api_key="secret-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            transport=transport,
        )

        result = client.complete("system", "question")

        self.assertEqual(result, "ok")
        self.assertEqual(seen["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(seen["payload"]["model"], "deepseek-v4-flash")
        self.assertEqual(seen["payload"]["messages"][-1]["content"], "question")
        self.assertEqual(seen["headers"]["Authorization"], "Bearer secret-key")
        self.assertNotIn("secret-key", repr(client))


if __name__ == "__main__":
    unittest.main()
