"""从飞书知识库同步可读取的 DOCX 节点到本地索引。"""

from __future__ import annotations

import argparse
import hashlib
import html
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .feishu_client import FeishuClient
from .chunker import chunk_text
from .ingest import SUPPORTED_SUFFIXES, Section, extract_sections
from .llm import DeepSeekClient
from .logging_utils import configure_logging
from .models import Chunk
from .retry import RetryPolicy
from .semantic_chunker import AtomicUnit, DeepSeekPlanner, SemanticPlanner, semantic_chunks
from .store import IndexStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncResult:
    nodes_seen: int
    indexed: int
    skipped: int
    deleted: int = 0


class FeishuSyncError(RuntimeError):
    """飞书知识库分页响应不完整，无法安全完成同步。"""


def _wiki_page_data(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise FeishuSyncError("飞书知识库列表响应不是对象")
    data = response.get("data")
    if not isinstance(data, dict):
        raise FeishuSyncError("飞书知识库列表响应缺少有效 data")
    if not isinstance(data.get("items"), list):
        raise FeishuSyncError("飞书知识库列表响应缺少有效 items")
    if not isinstance(data.get("has_more"), bool):
        raise FeishuSyncError("飞书知识库列表响应缺少有效 has_more")
    return data


def _response_data(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data", response)
    return data if isinstance(data, dict) else {}


def _extract_text(response: dict[str, Any]) -> str:
    data = _response_data(response)
    candidate: Any = data.get("content")
    if candidate is None and isinstance(data.get("document"), dict):
        candidate = data["document"].get("content")
    if isinstance(candidate, list):
        candidate = "\n".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in candidate)
    if not isinstance(candidate, str):
        return ""
    return html.unescape(re.sub(r"<[^>]+>", " ", candidate)).strip()


def _extract_sync_text(response: Any) -> str:
    if not isinstance(response, dict):
        raise FeishuSyncError("飞书文档正文响应不是对象")
    data = response.get("data")
    if not isinstance(data, dict):
        raise FeishuSyncError("飞书文档正文响应缺少有效 data")
    if "content" in data:
        candidate = data["content"]
    else:
        document = data.get("document")
        if not isinstance(document, dict) or "content" not in document:
            raise FeishuSyncError("飞书文档正文响应缺少 content")
        candidate = document["content"]
    if isinstance(candidate, list):
        for item in candidate:
            if isinstance(item, str):
                continue
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                raise FeishuSyncError("飞书文档正文 content 列表项无效")
    elif not isinstance(candidate, str):
        raise FeishuSyncError("飞书文档正文 content 类型无效")
    return _extract_text({"data": {"content": candidate}})


def _sections_to_units(sections: list[Section], source_id: str, max_unit_chars: int = 600) -> list[AtomicUnit]:
    units: list[AtomicUnit] = []
    for section in sections:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", section.text) if part.strip()]
        for paragraph in paragraphs:
            if len(paragraph) <= max_unit_chars:
                pieces = [paragraph]
            else:
                pieces = [paragraph[index : index + max_unit_chars] for index in range(0, len(paragraph), max_unit_chars)]
            for piece in pieces:
                units.append(AtomicUnit(f"{source_id}:unit-{len(units)}", piece, section.page, section.section))
    return units


def _local_chunks(sections: list[Section], source_id: str, title: str, max_chars: int) -> list[Chunk]:
    overlap = max(0, min(120, max_chars // 5))
    chunks = []
    for section in sections:
        chunks.extend(
            chunk_text(
                section.text,
                source_id=source_id,
                title=title,
                max_chars=max_chars,
                overlap=overlap,
                page=section.page,
                section=section.section,
            )
        )
    return chunks


def _hybrid_chunks(
    sections: list[Section],
    source_id: str,
    title: str,
    max_chars: int,
    semantic_planner: SemanticPlanner | None,
) -> list[Chunk]:
    if semantic_planner is None:
        return _local_chunks(sections, source_id, title, max_chars)
    units = _sections_to_units(sections, source_id)
    if not units:
        return []
    return semantic_chunks(units, source_id, title, semantic_planner, max_chars)


def sync_wiki_space(
    space_id: str,
    client: FeishuClient,
    store: IndexStore,
    max_chars: int = 900,
    semantic_planner: SemanticPlanner | None = None,
    chunk_strategy_version: str = "local-v1",
    chunk_model: str = "",
    enable_ocr: bool = True,
) -> SyncResult:
    nodes_seen = indexed = skipped = 0
    pending_parents: list[str | None] = [None]
    seen_nodes: dict[str, tuple[str, str, bool, str]] = {}
    retained_source_ids: set[str] = set()
    while pending_parents:
        parent_node_token = pending_parents.pop()
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        while True:
            response = client.list_wiki_nodes(
                space_id,
                page_token=page_token,
                parent_node_token=parent_node_token,
            )
            data = _wiki_page_data(response)
            items = data["items"]
            for node in items:
                if not isinstance(node, dict):
                    raise FeishuSyncError("飞书知识库节点不是对象")
                node_token = node.get("node_token")
                if not isinstance(node_token, str) or not node_token.strip():
                    raise FeishuSyncError("飞书知识库节点缺少有效 node_token")
                object_type = node.get("obj_type")
                if not isinstance(object_type, str) or not object_type.strip():
                    raise FeishuSyncError("飞书知识库节点缺少有效 obj_type")
                has_child = node.get("has_child")
                if not isinstance(has_child, bool):
                    raise FeishuSyncError("飞书知识库节点缺少有效 has_child")
                object_token = node.get("obj_token") or node.get("document_id")
                if not isinstance(object_token, str) or not object_token.strip():
                    raise FeishuSyncError("飞书知识库节点缺少有效 object_token")
                object_type = object_type.strip().lower()
                title = str(node.get("title") or "未命名飞书文档")
                fingerprint = (object_type, object_token, has_child, title)
                previous_fingerprint = seen_nodes.get(node_token)
                if previous_fingerprint is not None:
                    if previous_fingerprint != fingerprint:
                        raise FeishuSyncError("飞书知识库重复 node_token 的节点信息冲突")
                    continue
                seen_nodes[node_token] = fingerprint
                nodes_seen += 1
                if has_child:
                    pending_parents.append(node_token)
                if object_type == "file":
                    suffix = Path(title).suffix.lower()
                    if suffix not in SUPPORTED_SUFFIXES:
                        skipped += 1
                        continue
                    source_id = f"feishu:{space_id}:{node_token}"
                    raw_file = client.download_file(object_token)
                    content_checksum = hashlib.sha256(raw_file).hexdigest()
                    ocr_cache_mode = str(enable_ocr).lower() if suffix == ".pdf" else "na"
                    checksum = hashlib.sha256(
                        f"{content_checksum}:{chunk_strategy_version}:{chunk_model}:ocr={ocr_cache_mode}".encode("utf-8")
                    ).hexdigest()
                    if store.document_checksum(source_id) == checksum:
                        retained_source_ids.add(source_id)
                        skipped += 1
                        continue
                    with tempfile.TemporaryDirectory() as tmp:
                        downloaded = Path(tmp) / f"downloaded{suffix}"
                        downloaded.write_bytes(raw_file)
                        sections = extract_sections(downloaded, enable_ocr=enable_ocr)
                    try:
                        chunks = _hybrid_chunks(sections, source_id, title, max_chars, semantic_planner)
                    except Exception as exc:
                        logger.warning(
                            "semantic_chunk_fallback source_id=%s error_type=%s",
                            source_id,
                            type(exc).__name__,
                        )
                        chunks = _local_chunks(sections, source_id, title, max_chars)
                    if not chunks:
                        skipped += 1
                        continue
                    store.upsert_document(source_id, title, f"wiki/{space_id}/{node_token}", checksum, chunks)
                    retained_source_ids.add(source_id)
                    indexed += 1
                    continue
                if object_type not in {"docx", "doc"}:
                    skipped += 1
                    continue
                source_id = f"feishu:{space_id}:{node_token}"
                text = _extract_sync_text(client.get_document_raw_content(object_token))
                if not text:
                    skipped += 1
                    continue
                content_checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
                checksum = hashlib.sha256(
                    f"{content_checksum}:{chunk_strategy_version}:{chunk_model}".encode("utf-8")
                ).hexdigest()
                if store.document_checksum(source_id) == checksum:
                    retained_source_ids.add(source_id)
                    skipped += 1
                    continue
                sections = [Section(text=text)]
                try:
                    chunks = _hybrid_chunks(sections, source_id, title, max_chars, semantic_planner)
                except Exception as exc:
                    logger.warning(
                        "semantic_chunk_fallback source_id=%s error_type=%s",
                        source_id,
                        type(exc).__name__,
                    )
                    chunks = _local_chunks(sections, source_id, title, max_chars)
                if not chunks:
                    skipped += 1
                    continue
                store.upsert_document(source_id, title, f"wiki/{space_id}/{node_token}", checksum, chunks)
                retained_source_ids.add(source_id)
                indexed += 1
            has_more = bool(data.get("has_more"))
            if not has_more:
                break
            next_token = data.get("page_token")
            if not isinstance(next_token, str) or not next_token.strip():
                raise FeishuSyncError("has_more=true 时缺少有效 page_token")
            if next_token in seen_page_tokens:
                raise FeishuSyncError("has_more=true 时 page_token 重复")
            seen_page_tokens.add(next_token)
            page_token = next_token
    deleted = store.prune_documents(f"feishu:{space_id}:", retained_source_ids)
    return SyncResult(nodes_seen, indexed, skipped, deleted)


def main() -> None:
    parser = argparse.ArgumentParser(description="同步飞书知识库 DOCX 节点")
    parser.add_argument("--space-id", default="")
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    space_id = args.space_id or settings.feishu_space_id
    if not space_id:
        raise SystemExit("请提供 --space-id 或设置 FEISHU_SPACE_ID")
    store = IndexStore(args.db)
    retry_policy = RetryPolicy(
        max_attempts=settings.api_retry_max_attempts,
        base_delay=settings.api_retry_base_delay,
    )
    client = FeishuClient(
        settings.feishu_app_id,
        settings.feishu_app_secret,
        retry_policy=retry_policy,
    )
    semantic_planner = None
    if settings.rag_semantic_chunking:
        semantic_planner = DeepSeekPlanner(
            DeepSeekClient(
                settings.deepseek_api_key,
                settings.deepseek_base_url,
                settings.deepseek_chunk_model,
                retry_policy=retry_policy,
                usage_sink=store,
            ),
            batch_chars=settings.deepseek_chunk_batch_chars,
        )
    try:
        result = sync_wiki_space(
            space_id,
            client,
            store,
            max_chars=args.max_chars,
            enable_ocr=settings.rag_enable_ocr,
            semantic_planner=semantic_planner,
            chunk_strategy_version=settings.rag_chunk_strategy_version,
            chunk_model=settings.deepseek_chunk_model if semantic_planner else "",
        )
        logger.info(
            "sync_completed nodes_seen=%d indexed=%d skipped=%d deleted=%d",
            result.nodes_seen,
            result.indexed,
            result.skipped,
            result.deleted,
        )
        print(
            f"nodes_seen={result.nodes_seen} indexed={result.indexed} "
            f"skipped={result.skipped} deleted={result.deleted}"
        )
    finally:
        store.close()


if __name__ == "__main__":
    main()
