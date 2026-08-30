"""索引服务器上的本地文档目录。"""

from __future__ import annotations

import argparse
from pathlib import Path

from feishu_rag.ingest import index_directory
from feishu_rag.store import IndexStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"))
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()
    store = IndexStore(args.db)
    try:
        count = index_directory(args.root, store, max_chars=args.max_chars, enable_ocr=not args.no_ocr)
        print(f"indexed_files={count} indexed_chunks={store.count_chunks()}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
