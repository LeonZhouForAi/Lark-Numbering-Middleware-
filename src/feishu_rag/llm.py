"""DeepSeek OpenAI 兼容 Chat Completions 客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from .retry import RetryPolicy


_SQLITE_INT_MAX = 2**63 - 1


class DeepSeekError(RuntimeError):
    """模型调用失败，错误信息不包含密钥。"""


Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, bytes]]


class _DeepSeekNetworkError(RuntimeError):
    pass


def _urllib_transport(url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise _DeepSeekNetworkError from exc


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        transport: Transport | None = None,
        timeout: float = 30.0,
        retry_policy: RetryPolicy | None = None,
        usage_sink: Any | None = None,
    ):
        if not api_key.strip():
            raise DeepSeekError("缺少 DeepSeek API Key")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport or _urllib_transport
        self.retry_policy = retry_policy or RetryPolicy()
        self._usage_sink = usage_sink

    def __repr__(self) -> str:
        return f"DeepSeekClient(base_url={self.base_url!r}, model={self.model!r})"

    @staticmethod
    def _token_count(value: Any) -> int:
        return (
            value
            if type(value) is int and 0 <= value <= _SQLITE_INT_MAX
            else 0
        )

    def _record_usage(self, data: dict[str, Any], purpose: str) -> None:
        if self._usage_sink is None:
            return
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = self._token_count(usage.get("prompt_tokens"))
        completion_tokens = self._token_count(usage.get("completion_tokens"))
        total_value = usage.get("total_tokens")
        total_tokens = (
            self._token_count(total_value)
            if total_value is not None
            else min(prompt_tokens + completion_tokens, _SQLITE_INT_MAX)
        )
        recorder = getattr(self._usage_sink, "record_llm_usage", self._usage_sink)
        recorder(
            self.model,
            purpose,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

    def _chat_completion(self, payload: dict[str, Any], purpose: str) -> str:
        def request() -> tuple[int, bytes]:
            return self._transport(
                f"{self.base_url}/chat/completions",
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                payload,
                self.timeout,
            )

        try:
            status, raw = request()
        except Exception as exc:
            raise DeepSeekError(
                "DeepSeek 请求结果不明，已停止自动重试；费用请以服务商账单为准"
            ) from exc
        if status == 401:
            raise DeepSeekError("DeepSeek API Key 无效或已过期")
        if status == 429:
            raise DeepSeekError("DeepSeek API 额度或频率受限，请稍后重试")
        if status >= 500:
            raise DeepSeekError("DeepSeek 服务暂时不可用，请稍后重试")
        if status >= 400:
            raise DeepSeekError(f"DeepSeek API 请求失败（HTTP {status}）")
        try:
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise TypeError
            self._record_usage(data, purpose)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekError("DeepSeek 返回了无法解析的结果") from exc
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekError("DeepSeek 返回了空答案")
        return content.strip()

    def complete(
        self, system_prompt: str, user_prompt: str, *, purpose: str = "answer"
    ) -> str:
        return self._chat_completion(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 1200,
            },
            purpose,
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        purpose: str = "chunking",
    ) -> dict[str, Any]:
        content = self._chat_completion(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "temperature": 0,
                "max_tokens": 2000,
                "thinking": {"type": "disabled"},
                "response_format": {"type": "json_object"},
            },
            purpose,
        )
        try:
            result = json.loads(content)
        except (TypeError, ValueError) as exc:
            raise DeepSeekError("DeepSeek 返回了无法解析的 JSON") from exc
        if not isinstance(result, dict):
            raise DeepSeekError("DeepSeek JSON 输出必须是对象")
        return result
