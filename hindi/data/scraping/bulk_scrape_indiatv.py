"""Large-scale MANUAL Hindi collection from India TV (indiatv.in).

robots.txt: `User-agent: * / Allow: /` with only specific path exclusions
(testfiles, widgets, print, etc.) -- no bot-specific block for ClaudeBot/
anthropic-ai/Claude-Web. Confirmed separately from patrika.com's domain, so
running this alongside hindi/data/scraping/bulk_scrape.py adds genuinely
parallel throughput rather than competing for the same per-domain
rate-limit slot.

Daily sitemaps (xmlsitemap/sitemap/generic-articles-YYYY-MM-DD.xml) hold
~130 article URLs each and are confirmed available back to at least
2024-01-01 (900+ days of history).

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/indiatv_bulk.jsonl (source_type=manual).

Run from repo root:
    python hindi/data/scraping/bulk_scrape_indiatv.py --days 400
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk  # noqa: E402
from common.url_discovery import discover_from_date_sitemaps  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # hindi/

SITEMAP_TEMPLATE = "https://www.indiatv.in/xmlsitemap/sitemap/generic-articles-{date}.xml"


def recent_dates(n_days: int) -> list[str]:
    today = date.today()
    return [(today - timedelta(days=d)).isoformat() for d in range(1, n_days + 1)]


def main(days: int, max_workers: int, min_interval: float) -> None:
    print(f"=== India TV: discovering URLs from the last {days} days of sitemaps ===")
    urls = discover_from_date_sitemaps(SITEMAP_TEMPLATE, recent_dates(days))
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "indiatv_bulk.jsonl", source_name="indiatv_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"India TV: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=400,
                         help="how many days of India TV daily sitemaps to pull (~130 URLs/day)")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.days, args.max_workers, args.min_interval)
