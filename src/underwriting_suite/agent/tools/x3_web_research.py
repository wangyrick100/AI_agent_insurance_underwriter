"""X3 – Restricted Web Research Agent.

Performs web research limited to allowlisted underwriting-relevant domains.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import httpx

from underwriting_suite.agent.schemas import UnderwriterBrief, WebSource, X3Input
from underwriting_suite.config import settings
from underwriting_suite.services.azure_openai import get_chat_completion

logger = logging.getLogger(__name__)

WEB_RESEARCH_SYSTEM_PROMPT = """\
You are AgentX3WebResearch, an underwriting web research assistant.

TASK:
Given a medical/underwriting query, synthesise a brief from allowlisted
medical and regulatory sources.

OUTPUT FORMAT (JSON):
{
  "summary": "concise research findings",
  "sources": [
    {"url": "https://...", "title": "...", "snippet": "relevant excerpt"}
  ]
}

RULES:
1. Only reference domains from the allowlist.
2. Provide factual, evidence-based summaries.
3. Include source citations for every major claim.
4. This is research support only – not clinical advice.
"""


def _is_domain_allowed(url: str) -> bool:
    """Check if a URL's domain is in the allowlist."""
    try:
        domain = urlparse(url).netloc.lower()
        return any(domain.endswith(d) for d in settings.allowlisted_domains)
    except Exception:
        return False


async def x3_web(input_data: dict[str, Any]) -> dict[str, Any]:
    """Perform restricted web research on underwriting-relevant topics.

    Args:
        input_data: Must conform to X3Input schema.

    Returns:
        Serialised UnderwriterBrief.
    """
    parsed = X3Input(**input_data)
    logger.info("X3 web research started | query=%s", parsed.query[:80])

    allowed_domains_str = ", ".join(settings.allowlisted_domains)

    messages = [
        {"role": "system", "content": WEB_RESEARCH_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Research query: {parsed.query}\n"
                f"Topic scope: {parsed.topic_scope or 'general underwriting'}\n\n"
                f"Allowed domains: {allowed_domains_str}\n\n"
                "Provide a research brief with cited sources from the allowed domains only."
            ),
        },
    ]

    raw = await get_chat_completion(messages, temperature=0.2, response_format="json_object")

    import json

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("X3: LLM returned invalid JSON")
        return UnderwriterBrief(
            query=parsed.query,
            summary="Unable to complete research – LLM output error.",
        ).model_dump()

    # Filter sources to only allowlisted domains
    sources: list[WebSource] = []
    for src in result.get("sources", []):
        url = src.get("url", "")
        if _is_domain_allowed(url):
            sources.append(WebSource(url=url, title=src.get("title", ""), snippet=src.get("snippet", "")))
        else:
            logger.warning("X3: Filtered non-allowlisted source: %s", url)

    brief = UnderwriterBrief(
        query=parsed.query,
        summary=result.get("summary", ""),
        sources=sources,
    )
    logger.info("X3 web research complete | %d sources", len(sources))
    return brief.model_dump()
