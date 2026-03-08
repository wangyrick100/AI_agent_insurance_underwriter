"""Azure OpenAI service wrapper.

Production-grade wrapper providing:
  • Lazy-singleton async client
  • Automatic retry with jittered exponential back-off
  • Circuit-breaker pattern (trip → half-open probe → recovery)
  • Per-call token counting and cost estimation
  • Structured-output helper with Pydantic validation
  • Prompt-token budget guard to avoid context-window overflow
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai import AsyncAzureOpenAI

from underwriting_suite.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ═══════════════════════════════════════════════
#  Circuit-breaker state (module-level singleton)
# ═══════════════════════════════════════════════

@dataclass
class _CircuitBreaker:
    """Simple per-process circuit breaker for the LLM backend."""
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed | open | half-open

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= settings.circuit_breaker_failure_threshold:
            self.state = "open"
            logger.critical(
                "Circuit breaker OPEN after %d consecutive LLM failures", self.failure_count
            )

    def allow_request(self) -> bool:
        if self.state == "closed":
            return True
        elapsed = time.time() - self.last_failure_time
        if elapsed >= settings.circuit_breaker_recovery_timeout:
            self.state = "half-open"
            logger.info("Circuit breaker half-open – allowing probe request")
            return True
        return False


_breaker = _CircuitBreaker()


# ═══════════════════════════════════════════════
#  Usage tracking (lightweight, per-process)
# ═══════════════════════════════════════════════

@dataclass
class LLMUsageRecord:
    """Accumulated token / cost counters for a single session."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def estimated_cost_usd(self) -> float:
        return (
            (self.prompt_tokens / 1_000) * settings.cost_per_1k_prompt_tokens
            + (self.completion_tokens / 1_000) * settings.cost_per_1k_completion_tokens
        )


# Session-scoped usage buckets: session_id → LLMUsageRecord
_session_usage: dict[str, LLMUsageRecord] = {}


def get_session_usage(session_id: str) -> LLMUsageRecord:
    """Return the accumulated usage record for *session_id*."""
    return _session_usage.setdefault(session_id, LLMUsageRecord())


def reset_session_usage(session_id: str) -> None:
    _session_usage.pop(session_id, None)


# ═══════════════════════════════════════════════
#  Client singleton
# ═══════════════════════════════════════════════

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


# ═══════════════════════════════════════════════
#  Retry helper
# ═══════════════════════════════════════════════

async def _retry_with_backoff(coro_factory, *, retries: int | None = None):
    """Execute *coro_factory()* with jittered exponential back-off.

    ``coro_factory`` is a zero-arg callable returning an awaitable so that
    each retry creates a fresh coroutine.
    """
    max_attempts = retries if retries is not None else settings.llm_retry_attempts
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        if not _breaker.allow_request():
            raise RuntimeError(
                "Circuit breaker is OPEN – LLM calls are temporarily disabled. "
                f"Recovery in ≤{settings.circuit_breaker_recovery_timeout}s."
            )
        try:
            result = await coro_factory()
            _breaker.record_success()
            return result
        except Exception as exc:
            last_exc = exc
            _breaker.record_failure()
            if attempt < max_attempts:
                backoff = min(
                    settings.llm_retry_backoff_base * (2 ** (attempt - 1)),
                    settings.llm_retry_backoff_max,
                )
                jitter = random.uniform(0, backoff * 0.3)
                wait = backoff + jitter
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s – retrying in %.1fs",
                    attempt, max_attempts, str(exc)[:200], wait,
                )
                await asyncio.sleep(wait)
            else:
                logger.error(
                    "LLM call failed after %d attempts: %s", max_attempts, str(exc)[:300]
                )

    raise last_exc  # type: ignore[misc]


# ═══════════════════════════════════════════════
#  Chat completion (core)
# ═══════════════════════════════════════════════

async def get_chat_completion(
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int | None = None,
    response_format: str | None = None,
    *,
    session_id: str | None = None,
    retries: int | None = None,
) -> str:
    """Call Azure OpenAI chat completion with retry + circuit-breaker.

    Args:
        messages: OpenAI-style message list.
        temperature: Sampling temperature.
        max_tokens: Maximum response tokens (defaults to config).
        response_format: ``"json_object"`` to enforce JSON output.
        session_id: Optional – tracks token usage per session.
        retries: Override for retry count.

    Returns:
        The assistant's reply text.
    """
    effective_max_tokens = max_tokens or settings.max_completion_tokens
    client = _get_client()

    kwargs: dict[str, Any] = {
        "model": settings.azure_openai_chat_deployment,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": effective_max_tokens,
    }
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    start = time.time()

    async def _call():
        return await client.chat.completions.create(**kwargs)

    try:
        response = await _retry_with_backoff(_call, retries=retries)
        content = response.choices[0].message.content or ""

        # ── Token tracking ──────────────────────
        usage = response.usage
        prompt_tok = usage.prompt_tokens if usage else 0
        completion_tok = usage.completion_tokens if usage else 0
        total_tok = usage.total_tokens if usage else 0
        latency_ms = (time.time() - start) * 1000

        logger.debug(
            "OpenAI completion: prompt=%d completion=%d total=%d latency=%.0fms",
            prompt_tok, completion_tok, total_tok, latency_ms,
        )

        if session_id and settings.enable_cost_tracking:
            record = get_session_usage(session_id)
            record.prompt_tokens += prompt_tok
            record.completion_tokens += completion_tok
            record.total_tokens += total_tok
            record.call_count += 1
            record.total_latency_ms += latency_ms

            # Budget guard
            if record.total_tokens > settings.token_budget_per_session:
                logger.warning(
                    "Session %s exceeded token budget (%d/%d)",
                    session_id, record.total_tokens, settings.token_budget_per_session,
                )

        return content

    except Exception as e:
        logger.error("Azure OpenAI call failed: %s", str(e))
        if response_format == "json_object":
            return '{"error": "LLM call failed"}'
        return f"Error: {str(e)}"


# ═══════════════════════════════════════════════
#  Structured output helper
# ═══════════════════════════════════════════════

async def get_structured_completion(
    messages: list[dict[str, str]],
    response_model: type[T],
    temperature: float = 0.0,
    *,
    session_id: str | None = None,
) -> T | None:
    """Chat completion that parses & validates into a Pydantic model.

    Returns ``None`` if parsing/validation fails so callers can handle
    gracefully rather than crashing.
    """
    import json as _json
    from pydantic import ValidationError

    raw = await get_chat_completion(
        messages,
        temperature=temperature,
        response_format="json_object",
        session_id=session_id,
    )
    try:
        data = _json.loads(raw)
        return response_model(**data)  # type: ignore[call-arg]
    except (_json.JSONDecodeError, ValidationError, TypeError) as exc:
        logger.warning("Structured completion parse failed (%s): %s", response_model.__name__, exc)
        return None


# ═══════════════════════════════════════════════
#  Embeddings
# ═══════════════════════════════════════════════

async def get_embeddings(text: str, *, session_id: str | None = None) -> list[float]:
    """Generate an embedding vector for the given text.

    Uses retry logic and falls back to a zero-vector on failure.
    """
    client = _get_client()

    async def _call():
        return await client.embeddings.create(
            model=settings.azure_openai_embedding_deployment,
            input=text,
        )

    try:
        response = await _retry_with_backoff(_call)
        return response.data[0].embedding
    except Exception as e:
        logger.error("Azure OpenAI embedding failed: %s", str(e))
        return [0.0] * settings.azure_openai_embedding_dimensions


async def get_embeddings_batch(
    texts: list[str], *, session_id: str | None = None
) -> list[list[float]]:
    """Generate embeddings for a batch of texts in a single API call."""
    if not texts:
        return []

    client = _get_client()

    async def _call():
        return await client.embeddings.create(
            model=settings.azure_openai_embedding_deployment,
            input=texts,
        )

    try:
        response = await _retry_with_backoff(_call)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
    except Exception as e:
        logger.error("Azure OpenAI batch embedding failed: %s", str(e))
        dim = settings.azure_openai_embedding_dimensions
        return [[0.0] * dim for _ in texts]
