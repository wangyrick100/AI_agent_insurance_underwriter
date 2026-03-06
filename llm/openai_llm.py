"""OpenAI-backed LLM provider."""

from typing import Optional

from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """Thin wrapper around the openai Chat Completions API."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        try:
            import openai  # noqa: F401 — validated at import time
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAILLM. "
                "Install it with: pip install openai"
            ) from exc

        import openai as _openai

        self._client = _openai.OpenAI(api_key=api_key)
        self.model = model

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
