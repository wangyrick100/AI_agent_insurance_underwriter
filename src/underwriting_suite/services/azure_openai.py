"""Azure OpenAI service wrapper.

Provides chat completions and embedding generation through
Azure OpenAI / Azure AI Foundry deployments.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncAzureOpenAI

from underwriting_suite.config import settings

logger = logging.getLogger(__name__)

_client: AsyncAzureOpenAI | None = None


def _get_client() -> AsyncAzureOpenAI:
    """Lazy singleton for the Azure OpenAI async client."""
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
        )
    return _client


async def get_chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    response_format: str | None = None,
) -> str:
    """Call Azure OpenAI chat completion.

    Args:
        messages: OpenAI-style message list.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens.
        response_format: "json_object" to enforce JSON output.

    Returns:
        The assistant's reply text.
    """
    client = _get_client()

    kwargs: dict[str, Any] = {
        "model": settings.azure_openai_chat_deployment,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = await client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        logger.debug("OpenAI completion: %d tokens used", response.usage.total_tokens if response.usage else 0)
        return content
    except Exception as e:
        logger.error("Azure OpenAI call failed: %s", str(e))
        # Return a fallback JSON for json_object mode
        if response_format == "json_object":
            return '{"error": "LLM call failed"}'
        return f"Error: {str(e)}"


async def get_embeddings(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Args:
        text: The text to embed.

    Returns:
        List of floats (embedding vector).
    """
    client = _get_client()

    try:
        response = await client.embeddings.create(
            model=settings.azure_openai_embedding_deployment,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error("Azure OpenAI embedding failed: %s", str(e))
        # Return zero vector as fallback (dimensions for text-embedding-3-large)
        return [0.0] * 3072
