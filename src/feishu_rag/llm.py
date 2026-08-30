"""DeepSeek OpenAI 兼容 Chat Completions 客户端。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


class DeepSeekError(RuntimeError):
    """模型调用失败，错误信息不包含密钥。"""


Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, bytes]]


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
        raise DeepSeekError("无法连接 DeepSeek API，请检查服务器网络") from exc


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-flash",
        transport: Transport | None = None,
        timeout: float = 30.0,
    ):
        if not api_key.strip():
            raise DeepSeekError("缺少 DeepSeek API Key")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport or _urllib_transport

    def __repr__(self) -> str:
        return f"DeepSeekClient(base_url={self.base_url!r}, model={self.model!r})"

    def _chat_completion(self, payload: dict[str, Any]) -> str:
        status, raw = self._transport(
            f"{self.base_url}/chat/completions",
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            payload,
            self.timeout,
        )
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
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise DeepSeekError("DeepSeek 返回了无法解析的结果") from exc
        if not isinstance(content, str) or not content.strip():
            raise DeepSeekError("DeepSeek 返回了空答案")
        return content.strip()

    def complete(self, system_prompt: str, user_prompt: str) -> str:
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
            }
        )

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
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
            }
        )
        try:
            result = json.loads(content)
        except (TypeError, ValueError) as exc:
            raise DeepSeekError("DeepSeek 返回了无法解析的 JSON") from exc
        if not isinstance(result, dict):
            raise DeepSeekError("DeepSeek JSON 输出必须是对象")
        return result
