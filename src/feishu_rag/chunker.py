"""将文档文本切分成适合检索的片段。"""

from __future__ import annotations

import hashlib
import re

from .models import Chunk


def _split_long_unit(unit: str, max_chars: int, overlap: int) -> list[str]:
    if len(unit) <= max_chars:
        return [unit]
    pieces: list[str] = []
    step = max_chars - overlap
    start = 0
    while start < len(unit):
        piece = unit[start : start + max_chars].strip()
        if piece:
            pieces.append(piece)
        if start + max_chars >= len(unit):
            break
        start += step
    return pieces


def chunk_text(
    text: str,
    source_id: str,
    title: str,
    max_chars: int = 900,
    overlap: int = 120,
    page: int | None = None,
    section: str | None = None,
) -> list[Chunk]:
    """按段落切分文本，并为每片生成稳定 ID。"""

    if max_chars < 1:
        raise ValueError("max_chars 必须大于 0")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap 必须大于等于 0 且小于 max_chars")

    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_long_unit(paragraph, max_chars, overlap))

    contents: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else f"{current}\n\n{unit}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            contents.append(current)
        carry = current[-overlap:] if overlap and current else ""
        current = f"{carry}\n\n{unit}".strip() if carry else unit
        if len(current) > max_chars:
            current = current[:max_chars].rstrip()
    if current:
        contents.append(current)

    chunks: list[Chunk] = []
    for index, content in enumerate(contents):
        chunk_id = hashlib.sha256(f"{source_id}:{index}:{content}".encode("utf-8")).hexdigest()[:24]
        chunks.append(Chunk(chunk_id, source_id, title, content, page, section))
    return chunks
