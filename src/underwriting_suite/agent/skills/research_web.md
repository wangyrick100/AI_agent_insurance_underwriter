---
name: research_web
version: "2.0.0"
tool: research_web
category: research
cost_tier: medium
tags:
  - web
  - research
  - bing
  - medical
  - regulatory
  - allowlist
---

# Skill: research_web

## Purpose

Perform restricted web research on underwriting-relevant topics using an
allowlisted set of medical, actuarial, and regulatory domains. Supports
multi-query expansion, source credibility tiering, deduplication, and result
caching to avoid redundant API calls.

## When to Invoke

- The underwriter or Supervisor needs current clinical guidelines,
  drug interaction data, mortality studies, or regulatory information
  that is not present in the vector index.
- A diagnosis or medication code needs to be contextualised with
  published medical literature.
- Actuarial tables or premium rate references are required.

## Do NOT Invoke When

- The question can be answered from ingested documents – use `query_rag` instead.
- The query is about applicant-specific data – use `read_sql` or `query_rag`.
- The topic is outside the underwriting/medical/regulatory domain.

## Inputs

| Field           | Type   | Required | Description                                                 |
|-----------------|--------|----------|-------------------------------------------------------------|
| `query`         | `str`  | Yes      | Research question in natural language                       |
| `topic_scope`   | `str`  | No       | Optional narrower to guide query expansion                  |
| `max_sources`   | `int`  | No       | Maximum sources to return (default: 6)                      |
| `require_tier1` | `bool` | No       | When `true`, only includes peer-reviewed / authoritative sources |

## Outputs

```json
{
  "query": "hypertension mortality impact life insurance",
  "results": [
    {
      "title": "Hypertension and Cardiovascular Mortality – NEJM 2023",
      "url": "https://www.nejm.org/doi/...",
      "snippet": "Uncontrolled hypertension increases all-cause mortality by ...",
      "credibility_tier": 1,
      "domain": "nejm.org"
    }
  ],
  "source_count": 4,
  "from_cache": false,
  "confidence": 0.81
}
```

## Credibility Tiers

| Tier | Examples                                                      |
|------|---------------------------------------------------------------|
| 1    | PubMed, NEJM, Lancet, WHO, CDC, FDA, SOA, CMS                |
| 2    | Major health systems (Mayo, Cleveland Clinic), NICE, CMAJ    |
| 3    | Reputable health news, underwriting trade publications       |

## Usage Example (Supervisor ReAct JSON)

```json
{
  "thought_summary": "Need mortality data for uncontrolled T2 diabetes.",
  "next_tool": "research_web",
  "tool_input": {
    "query": "type 2 diabetes uncontrolled mortality life insurance underwriting",
    "require_tier1": true,
    "max_sources": 5
  }
}
```

## Notes

- Only domains on the internal allowlist are queried; the allowlist is managed
  in `config.py` (`ALLOWED_RESEARCH_DOMAINS`).
- Results are cached per query for 24 h to reduce API costs.
- Prompt-injection content in web results is stripped before returning.
