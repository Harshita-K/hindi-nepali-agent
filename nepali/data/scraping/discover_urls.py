"""Auto-discover Nepali article URLs from RSS feeds, verify each by actually
fetching + extracting it, and append the survivors to urls_manual.txt.
Replaces hand-copy-pasting links one at a time.

Run from repo root, then run scrape_news.py to actually collect the text:
    python nepali/data/scraping/discover_urls.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.url_discovery import discover_and_verify, write_verified_urls  # noqa: E402

URLS_FILE = Path(__file__).resolve().parent / "urls_manual.txt"

# BBC Nepali removed: bbc.com's robots.txt explicitly disallows
# ClaudeBot/anthropic-ai/Claude-Web (applies to the whole domain, not just
# bbc.com/nepali). OnlineKhabar and Setopati carry no such restriction.
RSS_FEEDS = [
    "https://www.onlinekhabar.com/feed",
    "https://www.setopati.com/feed",
]

if __name__ == "__main__":
    verified = discover_and_verify(rss_feeds=RSS_FEEDS, max_candidates=400, max_verified=200)
    n_added = write_verified_urls(verified, URLS_FILE)
    print(f"\nAdded {n_added} new verified URLs to {URLS_FILE}")
    print(f"({len(verified)} total verified this run; some may already have been present)")
