"""从飞书知识库同步可读取的 DOCX 节点到本地索引。"""

from __future__ import annotations

import argparse
import hashlib
import html
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .feishu_client import FeishuClient
from .chunker import chunk_text
from .ingest import SUPPORTED_SUFFIXES, extract_sections
from .store import IndexStore


@dataclass(frozen=True)
class SyncResult:
    nodes_seen: int
    indexed: int
    skipped: int


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


def sync_wiki_space(space_id: str, client: FeishuClient, store: IndexStore, max_chars: int = 900) -> SyncResult:
    nodes_seen = indexed = skipped = 0
    pending_parents: list[str | None] = [None]
    seen_nodes: set[str] = set()
    while pending_parents:
        parent_node_token = pending_parents.pop()
        page_token: str | None = None
        while True:
            response = client.list_wiki_nodes(
                space_id,
                page_token=page_token,
                parent_node_token=parent_node_token,
            )
            data = _response_data(response)
            items = data.get("items", [])
            if not isinstance(items, list):
                items = []
            for node in items:
                if not isinstance(node, dict):
                    skipped += 1
                    continue
                node_token = node.get("node_token")
                if not isinstance(node_token, str) or node_token in seen_nodes:
                    continue
                seen_nodes.add(node_token)
                nodes_seen += 1
                if node.get("has_child"):
                    pending_parents.append(node_token)
                    skipped += 1
                    continue
                object_type = str(node.get("obj_type", "")).lower()
                object_token = node.get("obj_token") or node.get("document_id")
                title = str(node.get("title") or "未命名飞书文档")
                if not isinstance(object_token, str):
                    skipped += 1
                    continue
                if object_type == "file":
                    suffix = Path(title).suffix.lower()
                    if suffix not in SUPPORTED_SUFFIXES:
                        skipped += 1
                        continue
                    raw_file = client.download_file(object_token)
                    source_id = f"feishu:{space_id}:{node_token}"
                    checksum = hashlib.sha256(raw_file).hexdigest()
                    if store.document_checksum(source_id) == checksum:
                        skipped += 1
                        continue
                    with tempfile.TemporaryDirectory() as tmp:
                        downloaded = Path(tmp) / f"downloaded{suffix}"
                        downloaded.write_bytes(raw_file)
                        sections = extract_sections(downloaded)
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
                    if not chunks:
                        skipped += 1
                        continue
                    store.upsert_document(source_id, title, f"wiki/{space_id}/{node_token}", checksum, chunks)
                    indexed += 1
                    continue
                if object_type not in {"docx", "doc"}:
                    skipped += 1
                    continue
                text = _extract_text(client.get_document_raw_content(object_token))
                if not text:
                    skipped += 1
                    continue
                source_id = f"feishu:{space_id}:{node_token}"
                chunks = chunk_text(text, source_id=source_id, title=title, max_chars=max_chars)
                checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
                store.upsert_document(source_id, title, f"wiki/{space_id}/{node_token}", checksum, chunks)
                indexed += 1
            has_more = bool(data.get("has_more"))
            next_token = data.get("page_token")
            if not has_more or not isinstance(next_token, str) or not next_token or next_token == page_token:
                break
            page_token = next_token
    return SyncResult(nodes_seen, indexed, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description="同步飞书知识库 DOCX 节点")
    parser.add_argument("--space-id", default="")
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()
    settings = Settings.from_env()
    space_id = args.space_id or settings.feishu_space_id
    if not space_id:
        raise SystemExit("请提供 --space-id 或设置 FEISHU_SPACE_ID")
    client = FeishuClient(settings.feishu_app_id, settings.feishu_app_secret)
    store = IndexStore(args.db)
    try:
        result = sync_wiki_space(space_id, client, store, max_chars=args.max_chars)
        print(f"nodes_seen={result.nodes_seen} indexed={result.indexed} skipped={result.skipped}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
