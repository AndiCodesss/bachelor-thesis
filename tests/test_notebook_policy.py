from __future__ import annotations

from research.lib.notebook_policy import filter_research_sources, resolve_source_policy


def test_filter_research_sources_blocks_low_signal_domains_and_caps_per_domain():
    policy = resolve_source_policy(
        {
            "max_imports_per_query": 3,
            "max_per_domain": 1,
            "blocked_domains": ["reddit.com"],
            "blocked_url_substrings": ["/ideas/"],
        },
    )
    approved, stats = filter_research_sources(
        [
            {"url": "https://reddit.com/r/futures/comments/abc"},
            {"url": "https://example.com/research/one"},
            {"url": "https://example.com/research/two"},
            {"url": "https://good.org/research/not-blocked"},
            {"url": "https://market.net/notes/3"},
            {"url": "https://tradingview.com/ideas/nq-scalp"},
        ],
        policy,
    )
    assert [row["domain"] for row in approved] == ["example.com", "good.org", "market.net"]
    assert stats["approved_sources"] == 3
    assert stats["approved_domains"] == ["example.com", "good.org", "market.net"]
    assert stats["rejected_reason_counts"]["blocked_domain"] == 1
    assert stats["rejected_reason_counts"]["domain_cap"] == 1
    assert stats["rejected_reason_counts"]["blocked_url_pattern"] == 1


def test_filter_research_sources_respects_allowlist_and_https():
    policy = resolve_source_policy(
        {
            "allowed_domains": ["cmegroup.com", "ssrn.com"],
            "blocked_domains": [],
            "blocked_url_substrings": [],
        },
    )
    approved, stats = filter_research_sources(
        [
            {"url": "https://www.cmegroup.com/education/articles.html"},
            {"url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1"},
            {"url": "http://cmegroup.com/insecure"},
            {"url": "https://randomblog.example.com/post"},
        ],
        policy,
    )
    assert [row["domain"] for row in approved] == ["cmegroup.com", "papers.ssrn.com"]
    assert stats["rejected_reason_counts"]["non_https"] == 1
    assert stats["rejected_reason_counts"]["domain_not_allowed"] == 1
