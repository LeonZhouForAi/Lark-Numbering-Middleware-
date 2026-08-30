"""服务配置与凭据读取。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping


class ConfigError(ValueError):
    """配置缺失或格式不正确。"""


def _required(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "").strip()
    if not value:
        raise ConfigError(f"缺少必填环境变量: {name}")
    return value


def _as_bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} 必须是 true/false")


@dataclass(frozen=True, repr=False)
class Settings:
    """运行时配置；密钥字段不会出现在 repr 中。"""

    deepseek_api_key: str = field(repr=False)
    feishu_app_id: str
    feishu_app_secret: str = field(repr=False)
    feishu_verification_token: str = field(repr=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"
    feishu_encrypt_key: str = field(default="", repr=False)
    feishu_space_id: str = ""
    rag_db_path: str = "./data/rag.sqlite3"
    rag_top_k: int = 6
    rag_max_chars: int = 900
    rag_enable_ocr: bool = True
    rag_semantic_chunking: bool = True
    deepseek_chunk_model: str = "deepseek-v4-flash"
    deepseek_chunk_batch_chars: int = 12000
    rag_chunk_strategy_version: str = "hybrid-v1"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        if environ is None:
            try:
                from dotenv import load_dotenv

                load_dotenv()
            except ImportError:
                pass
        env = os.environ if environ is None else environ
        try:
            top_k = int(env.get("RAG_TOP_K", "6"))
            max_chars = int(env.get("RAG_MAX_CHARS", "900"))
            chunk_batch_chars = int(env.get("DEEPSEEK_CHUNK_BATCH_CHARS", "12000"))
        except ValueError as exc:
            raise ConfigError("RAG_TOP_K、RAG_MAX_CHARS 和 DEEPSEEK_CHUNK_BATCH_CHARS 必须是整数") from exc
        if top_k < 1 or max_chars < 100:
            raise ConfigError("RAG_TOP_K 必须大于 0，RAG_MAX_CHARS 必须不小于 100")
        if chunk_batch_chars < 2000:
            raise ConfigError("DEEPSEEK_CHUNK_BATCH_CHARS 必须至少为 2000")
        strategy_version = env.get("RAG_CHUNK_STRATEGY_VERSION", "hybrid-v1").strip()
        if not strategy_version:
            raise ConfigError("RAG_CHUNK_STRATEGY_VERSION 不能为空")
        return cls(
            deepseek_api_key=_required(env, "DEEPSEEK_API_KEY"),
            feishu_app_id=_required(env, "FEISHU_APP_ID"),
            feishu_app_secret=_required(env, "FEISHU_APP_SECRET"),
            feishu_verification_token=env.get("FEISHU_VERIFICATION_TOKEN", "").strip(),
            deepseek_base_url=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
            deepseek_model=env.get("DEEPSEEK_MODEL", "deepseek-v4-flash").strip(),
            feishu_encrypt_key=env.get("FEISHU_ENCRYPT_KEY", "").strip(),
            feishu_space_id=env.get("FEISHU_SPACE_ID", "").strip(),
            rag_db_path=env.get("RAG_DB_PATH", "./data/rag.sqlite3").strip(),
            rag_top_k=top_k,
            rag_max_chars=max_chars,
            rag_enable_ocr=_as_bool(env.get("RAG_ENABLE_OCR", "true"), "RAG_ENABLE_OCR"),
            rag_semantic_chunking=_as_bool(
                env.get("RAG_SEMANTIC_CHUNKING", "true"), "RAG_SEMANTIC_CHUNKING"
            ),
            deepseek_chunk_model=env.get("DEEPSEEK_CHUNK_MODEL", "deepseek-v4-flash").strip(),
            deepseek_chunk_batch_chars=chunk_batch_chars,
            rag_chunk_strategy_version=strategy_version,
            log_level=env.get("LOG_LEVEL", "INFO").upper(),
        )

    def __repr__(self) -> str:
        return (
            "Settings("
            f"deepseek_base_url={self.deepseek_base_url!r}, "
            f"deepseek_model={self.deepseek_model!r}, "
            f"feishu_app_id={self.feishu_app_id!r}, "
            f"feishu_space_id={self.feishu_space_id!r}, "
            f"rag_db_path={self.rag_db_path!r}, "
            f"rag_top_k={self.rag_top_k!r}, rag_max_chars={self.rag_max_chars!r}, "
            f"rag_semantic_chunking={self.rag_semantic_chunking!r}, "
            f"deepseek_chunk_model={self.deepseek_chunk_model!r}, "
            f"rag_chunk_strategy_version={self.rag_chunk_strategy_version!r}, "
            f"log_level={self.log_level!r})"
        )
