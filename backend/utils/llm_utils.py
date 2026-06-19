"""OpenRouter LLM client."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Union

import aiohttp

UserInput = Union[str, list[dict[str, str]]]
logger = logging.getLogger(__name__)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = "deepseek/deepseek-v4-flash"
_NOTHINK_TOKEN = "/nothink"


def _build_messages(system: str, user: UserInput, enable_thinking: bool = True) -> list[dict]:
    if not enable_thinking:
        system = f"{system}\n\n{_NOTHINK_TOKEN}" if system else _NOTHINK_TOKEN
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    if isinstance(user, str):
        messages.append({"role": "user", "content": user})
    else:
        messages.extend(user)
    return messages


def _strip_think_tags(text: str, show_thought: bool = True) -> str:
    if not text:
        return ""
    tag = "think"
    pattern = rf"<{tag}>(.*?)</{tag}>"
    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.IGNORECASE))
    if not matches:
        return text.strip()
    if show_thought:
        thoughts = [m.group(1).strip() for m in matches if m.group(1).strip()]
        if thoughts:
            logger.info("Model thought:\n%s", "\n\n".join(thoughts))
    return text[matches[-1].end():].strip()


class OpenRouterLLM:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        enable_thinking: bool = False,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required.")
        self.model = model or os.getenv("OPENROUTER_LLM_MODEL", DEFAULT_LLM_MODEL)
        self._enable_thinking = enable_thinking

    async def _chat(self, messages: list[dict], response_format: dict | None = None) -> str:
        payload: dict = {"model": self.model, "messages": messages}
        if response_format:
            payload["response_format"] = response_format
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(OPENROUTER_CHAT_URL, headers=headers, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"OpenRouter chat failed ({resp.status}): {body}")
                data = await resp.json()
        return data["choices"][0]["message"]["content"]

    async def generate(self, system: str, user: UserInput, show_thought: bool = True) -> str:
        messages = _build_messages(system, user, enable_thinking=self._enable_thinking)
        content = await self._chat(messages)
        return _strip_think_tags(content, show_thought=show_thought)

    async def generate_structured(self, system: str, user: UserInput, schema: dict) -> dict:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "response", "strict": True, "schema": schema},
        }
        messages = _build_messages(system, user, enable_thinking=self._enable_thinking)
        content = await self._chat(messages, response_format=response_format)
        try:
            return json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return {}
