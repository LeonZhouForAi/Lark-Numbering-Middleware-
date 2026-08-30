import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from feishu_rag.rag import RagAnswer
from feishu_rag.store import IndexStore
from feishu_rag.web import handle_event, verify_signature


class FakeRag:
    def __init__(self, store=None):
        self.questions = []
        self.store = store

    def answer(self, question):
        self.questions.append(question)
        return RagAnswer("请先提交申请。", [])


class FakeFeishu:
    def __init__(self):
        self.replies = []

    def reply_text(self, message_id, text):
        self.replies.append((message_id, text))


class WebhookTests(unittest.TestCase):
    def test_url_verification_returns_challenge(self):
        payload = {"type": "url_verification", "token": "verify", "challenge": "abc"}

        result = handle_event(payload, FakeRag(), FakeFeishu(), verification_token="verify")

        self.assertEqual(result, {"challenge": "abc"})

    def test_signature_matches_feishu_formula(self):
        timestamp = "1700000000"
        nonce = "nonce"
        body = '{"event":"test"}'
        encrypt_key = "encrypt-key"
        signature = hashlib.sha256((timestamp + nonce + encrypt_key + body).encode("utf-8")).hexdigest()

        self.assertTrue(verify_signature(timestamp, nonce, body, encrypt_key, signature))
        self.assertFalse(verify_signature(timestamp, nonce, body, encrypt_key, "bad"))

    def test_text_message_is_answered_once(self):
        rag = FakeRag()
        feishu = FakeFeishu()
        payload = {
            "header": {"event_type": "im.message.receive_v1"},
            "event": {
                "message": {
                    "message_id": "om_123",
                    "message_type": "text",
                    "content": json.dumps({"text": "报销怎么走"}, ensure_ascii=False),
                }
            },
        }

        result = handle_event(payload, rag, feishu, verification_token="verify")

        self.assertEqual(result, {"status": "ok"})
        self.assertEqual(rag.questions, ["报销怎么走"])
        self.assertEqual(feishu.replies, [("om_123", "请先提交申请。")])

    def test_duplicate_message_id_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = IndexStore(Path(tmp) / "rag.sqlite3")
            try:
                rag = FakeRag(store)
                feishu = FakeFeishu()
                payload = {
                    "header": {"event_type": "im.message.receive_v1"},
                    "event": {
                        "message": {
                            "message_id": "om_duplicate",
                            "message_type": "text",
                            "content": json.dumps({"text": "报销怎么走"}, ensure_ascii=False),
                        }
                    },
                }

                first = handle_event(payload, rag, feishu, verification_token="verify")
                second = handle_event(payload, rag, feishu, verification_token="verify")

                self.assertEqual(first, {"status": "ok"})
                self.assertEqual(second, {"status": "duplicate"})
                self.assertEqual(rag.questions, ["报销怎么走"])
                self.assertEqual(feishu.replies, [("om_duplicate", "请先提交申请。")])
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
