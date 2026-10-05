"""Large-scale MANUAL Hindi collection -- this is what actually moves the
token count, unlike RSS (~30-50 items/feed). Two compliant sources:

  Patrika (patrika.com)  -- robots.txt explicitly ALLOWS ClaudeBot/anthropic-ai.
    Daily Google-News sitemaps (google-sitemap-YYYY-MM-DD.xml) hold ~300
    article URLs each and are available going back over a year.
  Jagran (jagran.com)    -- robots.txt: `User-agent: * / Allow: /` (no bot-specific block).
    Flat sitemap.xml (~800 URLs, mixed articles + section fronts; the crawl
    step naturally drops anything that isn't a real article).

Excluded on purpose: amarujala.com, bhaskar.com, bbc.com (Hindi *and*
Nepali), hindi.news18.com, nepalpress.com -- all of these name ClaudeBot /
anthropic-ai / Claude-Web explicitly as disallowed in robots.txt.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes straight to data/raw/{patrika,jagran}_bulk.jsonl (source_type=manual).

Run from repo root:
    python hindi/data/scraping/bulk_scrape.py --patrika-days 120
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk  # noqa: E402
from common.url_discovery import discover_from_date_sitemaps, discover_from_sitemap  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # hindi/

PATRIKA_SITEMAP_TEMPLATE = "https://www.patrika.com/date/google-sitemap-{date}.xml"
JAGRAN_SITEMAP = "https://www.jagran.com/sitemap.xml"


def recent_dates(n_days: int) -> list[str]:
    today = date.today()
    return [(today - timedelta(days=d)).isoformat() for d in range(1, n_days + 1)]


def main(patrika_days: int, max_workers: int, min_interval: float) -> None:
    print(f"=== Patrika: discovering URLs from the last {patrika_days} days of sitemaps ===")
    patrika_urls = discover_from_date_sitemaps(PATRIKA_SITEMAP_TEMPLATE, recent_dates(patrika_days))
    n = crawl_urls_bulk(
        patrika_urls, LANG_DIR / "data" / "raw" / "patrika_bulk.jsonl", source_name="patrika_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Patrika: {n} new articles written this run.\n")

    print("=== Jagran: discovering URLs from sitemap.xml ===")
    jagran_urls = discover_from_sitemap(JAGRAN_SITEMAP, limit=2000)
    n = crawl_urls_bulk(
        jagran_urls, LANG_DIR / "data" / "raw" / "jagran_bulk.jsonl", source_name="jagran_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Jagran: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patrika-days", type=int, default=120,
                         help="how many days of Patrika daily sitemaps to pull (~300 URLs/day)")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=1.2,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.patrika_days, args.max_workers, args.min_interval)
