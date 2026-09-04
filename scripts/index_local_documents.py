"""索引服务器上的本地文档目录。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from feishu_rag.ingest import index_directory
from feishu_rag.store import IndexStore


def _chunk_strategy_version(environ=os.environ) -> str:
    raw_value = environ.get("RAG_CHUNK_STRATEGY_VERSION")
    if raw_value is None:
        return "hybrid-v4"
    value = raw_value.strip()
    if not value:
        raise SystemExit("RAG_CHUNK_STRATEGY_VERSION 不能为空")
    return value


def _parser_version(environ=os.environ) -> str:
    raw_value = environ.get("RAG_PARSER_VERSION")
    if raw_value is None:
        return "parser-v2"
    value = raw_value.strip()
    if not value:
        raise SystemExit("RAG_PARSER_VERSION 不能为空")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()
    store = IndexStore(args.db)
    try:
        count = index_directory(
            args.root,
            store,
            max_chars=args.max_chars,
            enable_ocr=not args.no_ocr,
            chunk_strategy_version=_chunk_strategy_version(),
            parser_version=_parser_version(),
        )
        print(f"indexed_files={count} indexed_chunks={store.count_chunks()}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
