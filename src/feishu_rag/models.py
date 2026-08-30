"""RAG 使用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """可被检索的一段文档内容。"""

    id: str
    source_id: str
    title: str
    content: str
    page: int | None = None
    section: str | None = None

    @property
    def location(self) -> str:
        if self.page is not None and self.section:
            return f"第 {self.page} 页 · {self.section}"
        if self.page is not None:
            return f"第 {self.page} 页"
        return self.section or "正文"


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
