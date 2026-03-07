"""NotebookLM source-quality policy helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from typing import Any, Iterator
from urllib.parse import urlparse


ENV_SOURCE_POLICY_JSON = "NOTEBOOK_SOURCE_POLICY_JSON"

_DEFAULT_BLOCKED_DOMAINS = (
    "reddit.com",
    "tradingview.com",
    "scribd.com",
    "quora.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "pinterest.com",
    "x.com",
    "twitter.com",
)
_DEFAULT_BLOCKED_URL_SUBSTRINGS = (
    "/script/",
    "/ideas/",
)


def resolve_source_policy(raw: Any = None) -> dict[str, Any]:
    """Return a normalized source filter policy."""
    cfg = dict(raw) if isinstance(raw, dict) else {}
    max_imports_per_query = max(1, int(cfg.get("max_imports_per_query", 8)))
    max_per_domain = max(1, int(cfg.get("max_per_domain", 2)))
    require_https = bool(cfg.get("require_https", True))
    allowed_domains = _clean_domain_list(cfg.get("allowed_domains") or [])
    blocked_domains = _clean_domain_list(cfg.get("blocked_domains") or _DEFAULT_BLOCKED_DOMAINS)
    blocked_url_substrings = _clean_text_list(
        cfg.get("blocked_url_substrings") or _DEFAULT_BLOCKED_URL_SUBSTRINGS,
    )
    return {
        "name": str(cfg.get("name", "quant_research_default")).strip() or "quant_research_default",
        "max_imports_per_query": max_imports_per_query,
        "max_per_domain": min(max_per_domain, max_imports_per_query),
        "require_https": require_https,
        "allowed_domains": allowed_domains,
        "blocked_domains": blocked_domains,
        "blocked_url_substrings": blocked_url_substrings,
    }


def load_source_policy_from_env() -> dict[str, Any]:
    raw = os.getenv(ENV_SOURCE_POLICY_JSON, "").strip()
    if not raw:
        return resolve_source_policy()
    try:
        payload = json.loads(raw)
    except Exception:
        return resolve_source_policy()
    return resolve_source_policy(payload)


@contextmanager
def notebook_source_policy_context(policy: dict[str, Any] | None) -> Iterator[None]:
    """Expose the current source policy to child notebook query processes."""
    previous = os.environ.get(ENV_SOURCE_POLICY_JSON)
    try:
        if policy:
            os.environ[ENV_SOURCE_POLICY_JSON] = json.dumps(resolve_source_policy(policy), sort_keys=True)
        else:
            os.environ.pop(ENV_SOURCE_POLICY_JSON, None)
        yield
    finally:
        if previous is None:
            os.environ.pop(ENV_SOURCE_POLICY_JSON, None)
        else:
            os.environ[ENV_SOURCE_POLICY_JSON] = previous


def filter_research_sources(
    sources: list[dict[str, Any]] | None,
    policy: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter raw NotebookLM research sources using a configurable quality policy."""
    resolved = resolve_source_policy(policy)
    rows = list(sources) if isinstance(sources, list) else []

    approved: list[dict[str, Any]] = []
    approved_domains: list[str] = []
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}
    rejected_reason_counts: dict[str, int] = {}
    rejected_domains: set[str] = set()
    discovered_with_url = 0

    for source in rows:
        url = _normalize_url(source.get("url"))
        if not url:
            _increment(rejected_reason_counts, "missing_url")
            continue
        discovered_with_url += 1

        parsed = urlparse(url)
        domain = _normalize_domain(parsed.netloc)
        if not domain:
            _increment(rejected_reason_counts, "invalid_domain")
            continue
        if resolved["require_https"] and parsed.scheme.lower() != "https":
            _increment(rejected_reason_counts, "non_https")
            rejected_domains.add(domain)
            continue
        if resolved["allowed_domains"] and not _domain_matches_any(domain, resolved["allowed_domains"]):
            _increment(rejected_reason_counts, "domain_not_allowed")
            rejected_domains.add(domain)
            continue
        if _domain_matches_any(domain, resolved["blocked_domains"]):
            _increment(rejected_reason_counts, "blocked_domain")
            rejected_domains.add(domain)
            continue
        if any(token in url for token in resolved["blocked_url_substrings"]):
            _increment(rejected_reason_counts, "blocked_url_pattern")
            rejected_domains.add(domain)
            continue
        if url in seen_urls:
            _increment(rejected_reason_counts, "duplicate_url")
            rejected_domains.add(domain)
            continue
        if domain_counts.get(domain, 0) >= int(resolved["max_per_domain"]):
            _increment(rejected_reason_counts, "domain_cap")
            rejected_domains.add(domain)
            continue
        if len(approved) >= int(resolved["max_imports_per_query"]):
            _increment(rejected_reason_counts, "query_cap")
            rejected_domains.add(domain)
            continue

        row = dict(source)
        row["url"] = url
        row["domain"] = domain
        approved.append(row)
        seen_urls.add(url)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        if domain not in approved_domains:
            approved_domains.append(domain)

    return approved, {
        "policy_name": str(resolved["name"]),
        "discovered_sources": discovered_with_url,
        "approved_sources": len(approved),
        "rejected_sources": max(0, discovered_with_url - len(approved)),
        "approved_domains": approved_domains,
        "rejected_domains": sorted(rejected_domains),
        "rejected_reason_counts": dict(sorted(rejected_reason_counts.items())),
    }


def _normalize_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/")
    normalized = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
        fragment="",
    )
    return normalized.geturl()


def _normalize_domain(value: Any) -> str:
    domain = str(value or "").strip().lower()
    if not domain:
        return ""
    if "@" in domain:
        return ""
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _domain_matches_any(domain: str, candidates: list[str]) -> bool:
    return any(domain == candidate or domain.endswith(f".{candidate}") for candidate in candidates)


def _clean_domain_list(values: Any) -> list[str]:
    out: list[str] = []
    for value in _clean_text_list(values):
        normalized = _normalize_domain(value)
        if normalized and normalized not in out:
            out.append(normalized)
    return out


def _clean_text_list(values: Any) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[str] = []
    for value in values:
        text = str(value or "").strip().lower()
        if text and text not in out:
            out.append(text)
    return out


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = int(counter.get(key, 0)) + 1


__all__ = [
    "ENV_SOURCE_POLICY_JSON",
    "filter_research_sources",
    "load_source_policy_from_env",
    "notebook_source_policy_context",
    "resolve_source_policy",
]
