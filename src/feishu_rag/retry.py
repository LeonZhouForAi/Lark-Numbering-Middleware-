"""外部 API 的有限指数退避策略。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 4.0
    sleep: Callable[[float], None] = field(
        default=time.sleep, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not math.isfinite(self.base_delay) or self.base_delay < 0:
            raise ValueError("base_delay must be finite and non-negative")
        if not math.isfinite(self.max_delay) or self.max_delay < 0:
            raise ValueError("max_delay must be finite and non-negative")

    def wait(self, retry_number: int) -> None:
        """等待第 retry_number 次重试；首次重试编号为 1。"""
        if retry_number < 1:
            raise ValueError("retry_number must be at least 1")
        self.sleep(min(self.base_delay * (2 ** (retry_number - 1)), self.max_delay))


def run_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy,
    *,
    retry_result: Callable[[T], bool],
    retry_exception: Callable[[Exception], bool],
) -> T:
    for attempt in range(1, policy.max_attempts + 1):
        try:
            result = operation()
        except Exception as exc:
            if attempt >= policy.max_attempts or not retry_exception(exc):
                raise
        else:
            if attempt >= policy.max_attempts or not retry_result(result):
                return result
        policy.wait(attempt)
    raise RuntimeError("retry loop exhausted")  # pragma: no cover
