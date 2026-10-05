"""Large-scale MANUAL Hindi collection from ABP Live (abplive.com),
standalone so it runs in PARALLEL with Patrika/Jagran/India TV (different
domain, independent per-domain throttle).

robots.txt: `User-agent: * / Allow: /`, no bot-specific block (unlike
amarujala.com/bhaskar.com/zeenews, which explicitly disallow ClaudeBot or
similar AI crawlers -- see hindi/data/scraping/bulk_scrape.py's exclusion
list, same reasoning applies here).

Daily sitemap at /news-{DD-MM-YYYY}.xml holds ~300-500 article URLs each
(mixed Hindi content: news, astrology, entertainment etc). Confirmed by
direct sampling: dense back through at least mid-2024, thinning out by
early 2023 -- default day range stays well within the dense window.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/abplive_bulk.jsonl (source_type=manual).

Run from repo root:
    python hindi/data/scraping/bulk_scrape_abplive.py --days 500
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

SITEMAP_TEMPLATE = "https://www.abplive.com/news-{date}.xml"


def recent_dates(n_days: int) -> list[str]:
    today = date.today()
    return [(today - timedelta(days=d)).strftime("%d-%m-%Y") for d in range(1, n_days + 1)]


def main(days: int, max_workers: int, min_interval: float) -> None:
    print(f"=== ABP Live: discovering URLs from the last {days} days of sitemaps ===")
    urls = discover_from_date_sitemaps(SITEMAP_TEMPLATE, recent_dates(days))
    print(f"{len(urls)} candidate URLs discovered")
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "abplive_bulk.jsonl", source_name="abplive_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"ABP Live: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=500,
                         help="how many days back to pull sitemaps for (dense through ~mid-2024)")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.days, args.max_workers, args.min_interval)
