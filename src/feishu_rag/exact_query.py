"""IE 受控表型的精确数值查询。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


_PART_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}(?![A-Za-z0-9])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExactAnswer:
    text: str
    status: str = "answerable"


def _value(row: Any, name: str) -> str:
    try:
        value = row[name]
    except (KeyError, IndexError, TypeError):
        value = getattr(row, name, "")
    return "" if value is None else str(value)


def _number(row: Any) -> str:
    try:
        value = row["numeric_value"]
    except (KeyError, IndexError, TypeError):
        value = getattr(row, "numeric_value", None)
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else format(numeric, "g")


class ExactQueryService:
    def __init__(self, store: Any) -> None:
        self.store = store

    def _rows(self, fact_type: str) -> list[Any]:
        return list(self.store.structured_fact_rows(fact_type))

    def answer(self, question: str) -> ExactAnswer | None:
        normalized = unicodedata.normalize("NFKC", question).strip()
        lowered = normalized.casefold()
        if "upph" in lowered or "uph" in lowered:
            rows = self._rows("upph")
            matches = [
                row
                for row in rows
                if _value(row, "operation_name")
                and _value(row, "operation_name").casefold() in lowered
            ]
            if len(matches) == 1:
                row = matches[0]
                return ExactAnswer(
                    f"{_value(row, 'operation_name')} UPPH：{_number(row)}"
                )
            if not matches:
                return ExactAnswer("请说明要查询的具体工序？", "ambiguous")
            return ExactAnswer("请说明要查询的具体工序？", "ambiguous")

        if "工时" not in normalized:
            return None
        part_match = _PART_NUMBER_RE.search(normalized)
        if part_match is not None:
            part_number = part_match.group(0).upper()
            rows = [
                row
                for row in self._rows("part_hours")
                if _value(row, "part_number").upper() == part_number
            ]
            if not rows:
                return None
            stage = "合计" if "总工时" in normalized else self._matched_stage(normalized, rows)
            if not stage:
                return ExactAnswer("请说明要查询的工段？", "ambiguous")
            matches = [row for row in rows if _value(row, "process_stage") == stage]
            if len(matches) == 1:
                return ExactAnswer(f"{part_number} {stage}工时：{_number(matches[0])}")
            return None

        rows = self._rows("series_hours")
        series_names = sorted(
            {_value(row, "series_name") for row in rows if _value(row, "series_name")},
            key=len,
            reverse=True,
        )
        series = next((name for name in series_names if name in normalized), "")
        if not series:
            if "系列" in normalized:
                return ExactAnswer("请说明要查询的产品系列？", "ambiguous")
            return None
        series_rows = [row for row in rows if _value(row, "series_name") == series]
        stage = "合计" if "总工时" in normalized else self._matched_stage(
            normalized, series_rows
        )
        if not stage:
            return ExactAnswer("请说明要查询的工段？", "ambiguous")
        matches = [row for row in series_rows if _value(row, "process_stage") == stage]
        if len(matches) == 1:
            return ExactAnswer(f"{series} {stage}工时：{_number(matches[0])}")
        return None

    @staticmethod
    def _matched_stage(question: str, rows: list[Any]) -> str:
        stages = sorted(
            {_value(row, "process_stage") for row in rows if _value(row, "process_stage")},
            key=len,
            reverse=True,
        )
        return next((stage for stage in stages if stage in question), "")
