"""RAG 使用的数据结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalScope:
    """检索可访问空间；None 表示维持当前全库行为。"""

    allowed_space_ids: frozenset[str] | None = None


@dataclass(frozen=True)
class Chunk:
    """可被检索的一段文档内容。"""

    id: str
    source_id: str
    title: str
    content: str
    page: int | None = None
    section: str | None = None
    search_text: str = ""

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


@dataclass(frozen=True)
class FaqMatch:
    """命中的高频问答条目。"""

    entry_id: str
    answer: str
    intent_key: str
    knowledge_revision: int = 0


@dataclass(frozen=True)
class FaqObservation:
    """用于统计高频问答的观测记录。"""

    intent_key: str
    scope_key: str
    normalized_question: str
    source_signature: str
    knowledge_revision: int
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreheatJob:
    id: str
    scope_key: str
    knowledge_revision: int
    retry_count: int = 0


@dataclass(frozen=True)
class PreheatCandidate:
    signature: str
    scope_key: str
    knowledge_revision: int
    chunk_id: str
    source_id: str
    title: str
    content: str
    score: int


@dataclass(frozen=True)
class StructuredFact:
    source_id: str
    sheet_name: str
    row_number: int
    fact_type: str
    part_number: str = ""
    series_name: str = ""
    process_stage: str = ""
    operation_name: str = ""
    metric_name: str = ""
    numeric_value: float | None = None
    text_value: str = ""
    unit: str = ""
