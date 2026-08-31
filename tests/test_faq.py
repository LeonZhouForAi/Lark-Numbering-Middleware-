import unittest
from dataclasses import FrozenInstanceError

from feishu_rag.models import FaqMatch, FaqObservation


class FaqModelsTests(unittest.TestCase):
    def test_faq_match_is_immutable(self):
        match = FaqMatch(entry_id="faq-1", answer="answer", intent_key="reset-password")
        self.assertEqual(match.entry_id, "faq-1")
        with self.assertRaises(FrozenInstanceError):
            match.answer = "changed"

    def test_faq_observation_has_expected_fields_and_is_immutable(self):
        observation = FaqObservation(
            intent_key="reset-password",
            scope_key="space-a",
            source_signature="sig",
            knowledge_revision=4,
            normalized_question="密码怎么重置",
        )
        self.assertEqual(observation.knowledge_revision, 4)
        self.assertEqual(observation.normalized_question, "密码怎么重置")
        with self.assertRaises(FrozenInstanceError):
            observation.scope_key = "space-b"


if __name__ == "__main__":
    unittest.main()
