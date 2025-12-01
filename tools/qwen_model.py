"""Async Qwen (DashScope) client wrapper for future tool integrations."""

import inspect
import os
from typing import Callable, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class QwenModelConfig:
    """
    Minimal wrapper around DashScope's OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "qwen3-235b-a22b",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        system_prompt: str = "You are a helpful assistant.",
    ):
        candidate_key = api_key or os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not candidate_key:
            raise ValueError(
                "未找到 DashScope API Key，请传入 api_key 或设置 QWEN_API_KEY/DASHSCOPE_API_KEY 环境变量。"
            )

        self.async_client = AsyncOpenAI(api_key=candidate_key, base_url=base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def process_input(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = True,
        on_stream: Optional[Callable[[str], None]] = None,
    ) -> "QwenResponse":
        model = model_name or self.model_name
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if not stream:
            resp = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return QwenResponse(resp.choices[0].message.content)

        collected_text = ""
        try:
            stream_resp = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:  # pragma: no cover - thin wrapper
            return QwenResponse(f"对话创建错误: {exc}")

        try:
            async for chunk in stream_resp:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    collected_text += delta
                    if on_stream:
                        maybe_awaitable = on_stream(delta)
                        if inspect.isawaitable(maybe_awaitable):
                            await maybe_awaitable
        except Exception as exc:  # pragma: no cover - thin wrapper
            return QwenResponse(f"数据请求错误: {exc}")

        return QwenResponse(collected_text)


class QwenResponse:
    """Very small response adapter so downstream code can use `.text`."""

    def __init__(self, response_data: str):
        self.text = response_data

