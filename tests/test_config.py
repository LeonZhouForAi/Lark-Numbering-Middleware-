import os
import unittest
from unittest.mock import patch

from feishu_rag.config import ConfigError, Settings


class SettingsTests(unittest.TestCase):
    def _base_env(self):
        return {
            "DEEPSEEK_API_KEY": "deepseek-secret-value",
            "FEISHU_APP_ID": "cli_test",
            "FEISHU_APP_SECRET": "secret_test",
            "FEISHU_VERIFICATION_TOKEN": "verify_test",
        }

    def test_requires_deepseek_api_key(self):
        env = self._base_env()
        env.pop("DEEPSEEK_API_KEY")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ConfigError, "DEEPSEEK_API_KEY"):
                Settings.from_env()

    def test_repr_does_not_expose_api_key(self):
        with patch.dict(os.environ, self._base_env(), clear=True):
            settings = Settings.from_env()
        self.assertNotIn("deepseek-secret-value", repr(settings))

    def test_ocr_can_be_disabled_by_environment(self):
        env = self._base_env()
        env["RAG_ENABLE_OCR"] = "false"
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertFalse(settings.rag_enable_ocr)

    def test_long_connection_does_not_require_webhook_verification_token(self):
        env = self._base_env()
        env.pop("FEISHU_VERIFICATION_TOKEN")
        with patch.dict(os.environ, env, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.feishu_verification_token, "")


if __name__ == "__main__":
    unittest.main()
