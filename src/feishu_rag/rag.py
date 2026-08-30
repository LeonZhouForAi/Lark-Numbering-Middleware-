"""检索增强生成编排。"""

from __future__ import annotations

from dataclasses import dataclass

from .models import SearchResult
from .store import IndexStore


@dataclass(frozen=True)
class Citation:
    index: int
    title: str
    location: str
    source_id: str


@dataclass(frozen=True)
class RagAnswer:
    text: str
    citations: list[Citation]


class RagService:
    def __init__(self, store: IndexStore, llm, top_k: int = 6):
        self.store = store
        self.llm = llm
        self.top_k = top_k

    @staticmethod
    def _context(results: list[SearchResult]) -> tuple[str, list[Citation]]:
        citations: list[Citation] = []
        blocks: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            citations.append(Citation(index, chunk.title, chunk.location, chunk.source_id))
            blocks.append(f"[{index}] {chunk.title}（{chunk.location}）\n{chunk.content}")
        return "\n\n".join(blocks), citations

    def answer(self, question: str) -> RagAnswer:
        question = question.strip()
        if not question:
            return RagAnswer("请输入要查询的问题。", [])
        results = self.store.search(question, top_k=self.top_k)
        if not results:
            return RagAnswer("知识库中暂无依据，请换一种问法或联系文控管理员。", [])

        context, citations = self._context(results)
        system_prompt = (
            "你是公司内部知识库助手。仅依据资料回答，不得补造制度、金额、日期或审批人。"
            "资料不足时明确说明‘现有资料不足’，不要用常识替代。回答简洁，保留必要条件。"
            "回答中可使用资料编号，例如 [1]。"
        )
        user_prompt = f"问题：{question}\n\n资料：\n{context}"
        generated = self.llm.complete(system_prompt, user_prompt)
        source_lines = "\n".join(f"[{c.index}] {c.title}（{c.location}）" for c in citations)
        return RagAnswer(f"{generated}\n\n来源：\n{source_lines}", citations)
