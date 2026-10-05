"""Auto-discover Hindi article URLs from RSS feeds, verify each by actually
fetching + extracting it, and append the survivors to urls_manual.txt.

NOTE: as of this writing, no compliant Hindi outlet we've found publishes a
working small RSS feed -- BBC Hindi, Amar Ujala, and Dainik Bhaskar all
explicitly disallow ClaudeBot/anthropic-ai in robots.txt (see bulk_scrape.py
docstring for the full findings) and are excluded here for that reason, not
technical failure. For actual Hindi volume, use bulk_scrape.py (Patrika +
Jagran sitemaps) and scrape_archive_org.py (books) instead -- this script is
kept only in case a compliant RSS feed turns up later.

Run from repo root, then run scrape_news.py to actually collect the text:
    python hindi/data/scraping/discover_urls.py
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.url_discovery import discover_and_verify, write_verified_urls  # noqa: E402

URLS_FILE = Path(__file__).resolve().parent / "urls_manual.txt"

# Empty on purpose -- see module docstring. Add a feed here only after
# checking its robots.txt doesn't disallow ClaudeBot/anthropic-ai/Claude-Web.
RSS_FEEDS: list[str] = []

if __name__ == "__main__":
    if not RSS_FEEDS:
        raise SystemExit(
            "No compliant RSS feeds configured -- see this script's docstring. "
            "Use bulk_scrape.py and scrape_archive_org.py for Hindi volume instead."
        )
    verified = discover_and_verify(rss_feeds=RSS_FEEDS, max_candidates=400, max_verified=200)
    n_added = write_verified_urls(verified, URLS_FILE)
    print(f"\nAdded {n_added} new verified URLs to {URLS_FILE}")
    print(f"({len(verified)} total verified this run; some may already have been present)")
