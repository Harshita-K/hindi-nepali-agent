"""Large-scale MANUAL Nepali collection from OnlineKhabar (onlinekhabar.com),
split out as its own standalone script so it runs in PARALLEL with
nepali/data/scraping/bulk_scrape.py's Setopati crawl (different domain,
independent per-domain throttle) rather than waiting for Setopati's huge
candidate pool to finish first.

robots.txt is `User-agent: *` with no rules at all. Monthly archive pages
(/YYYY/MM, then /YYYY/MM/page/N for N>1) list ~200 articles/page, confirmed
available back to at least 2024/01. Articles here average ~12,900 chars
(~6x richer than Setopati's ~2,700 chars/doc).

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/onlinekhabar_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape_onlinekhabar.py --months-back 28
"""
import argparse
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk  # noqa: E402
from common.url_discovery import discover_from_paginated_archive  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

LINK_PATTERN = r'https://www\.onlinekhabar\.com/\d{4}/\d{2}/\d+/[a-z0-9-]+'


def month_url_fn(year: int, month: int):
    base = f"https://www.onlinekhabar.com/{year}/{month:02d}"
    def _fn(page: int) -> str:
        return base if page == 1 else f"{base}/page/{page}"
    return _fn


def recent_year_months(n_months: int) -> list[tuple[int, int]]:
    y, m = date.today().year, date.today().month
    out = []
    for _ in range(n_months):
        out.append((y, m))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return out


def main(months_back: int, max_workers: int, min_interval: float) -> None:
    print(f"=== OnlineKhabar: paginated archive for the last {months_back} months ===")
    all_urls: list[str] = []
    for year, month in recent_year_months(months_back):
        urls = discover_from_paginated_archive(month_url_fn(year, month), LINK_PATTERN, max_pages=60)
        print(f"  {year}/{month:02d}: {len(urls)} links")
        all_urls.extend(urls)
    n = crawl_urls_bulk(
        all_urls, LANG_DIR / "data" / "raw" / "onlinekhabar_bulk.jsonl", source_name="onlinekhabar_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"OnlineKhabar: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--months-back", type=int, default=28)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.months_back, args.max_workers, args.min_interval)
