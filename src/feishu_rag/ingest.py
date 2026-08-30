"""本地文档解析和索引入口。"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from .chunker import chunk_text
from .store import IndexStore


class UnsupportedFileError(ValueError):
    """文件格式需要先转换。"""


@dataclass(frozen=True)
class Section:
    text: str
    page: int | None = None
    section: str | None = None


SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def _read_pdf_ocr(path: Path) -> list[Section]:
    """在服务器安装 OCR 依赖时识别扫描型 PDF；依赖缺失则安全返回空列表。"""

    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return []
    try:
        images = convert_from_path(str(path), dpi=220)
    except Exception:
        return []
    sections: list[Section] = []
    for page_number, image in enumerate(images, start=1):
        try:
            text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
        except Exception:
            try:
                text = pytesseract.image_to_string(image, lang="eng").strip()
            except Exception:
                text = ""
        if text:
            sections.append(Section(text=text, page=page_number))
    return sections


def _read_pdf(path: Path, enable_ocr: bool = True) -> list[Section]:
    from pypdf import PdfReader

    sections: list[Section] = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append(Section(text=text, page=page_number))
    if sections or not enable_ocr:
        return sections
    return _read_pdf_ocr(path)


def _read_docx(path: Path) -> list[Section]:
    from docx import Document

    document = Document(str(path))
    text = "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
    return [Section(text=text)] if text else []


def extract_sections(path: str | Path, enable_ocr: bool = True) -> list[Section]:
    """按格式抽取文本；`.doc` 和 `.wps` 明确要求先转 DOCX。"""

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".doc", ".wps"}:
        raise UnsupportedFileError(f"{file_path.name} 先转换为 DOCX 后再索引")
    if suffix in {".txt", ".md", ".markdown"}:
        text = file_path.read_text(encoding="utf-8")
        return [Section(text=text)] if text.strip() else []
    if suffix == ".pdf":
        return _read_pdf(file_path, enable_ocr=enable_ocr)
    if suffix == ".docx":
        return _read_docx(file_path)
    raise UnsupportedFileError(f"不支持的文件格式: {file_path.suffix or '(无扩展名)'}")


def index_file(
    path: Path,
    root: Path,
    store: IndexStore,
    max_chars: int = 900,
    enable_ocr: bool = True,
) -> bool:
    sections = extract_sections(path, enable_ocr=enable_ocr)
    source_id = path.relative_to(root).as_posix()
    title = path.stem
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
        return False
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    store.upsert_document(source_id, title, source_id, checksum, chunks)
    return True


def index_directory(
    root: str | Path,
    store: IndexStore,
    max_chars: int = 900,
    enable_ocr: bool = True,
) -> int:
    """递归索引目录，返回成功索引的文件数量。"""

    root_path = Path(root).resolve()
    indexed = 0
    for path in sorted(root_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if index_file(path, root_path, store, max_chars=max_chars, enable_ocr=enable_ocr):
            indexed += 1
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="索引本地 PDF/DOCX/Markdown/TXT 文件")
    parser.add_argument("root", type=Path, help="文档目录")
    parser.add_argument("--db", type=Path, default=Path("./data/rag.sqlite3"), help="SQLite 数据库路径")
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--no-ocr", action="store_true", help="禁用扫描 PDF OCR")
    args = parser.parse_args()
    store = IndexStore(args.db)
    try:
        count = index_directory(args.root, store, max_chars=args.max_chars, enable_ocr=not args.no_ocr)
        print(f"indexed_files={count} indexed_chunks={store.count_chunks()}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
