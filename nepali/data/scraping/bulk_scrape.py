"""Large-scale MANUAL Nepali collection -- this is what actually moves the
token count, unlike RSS (~30-50 items/feed). Two compliant sources:

  Setopati (setopati.com)     -- robots.txt only blocks /login /cms /preview
    /mail, no bot-specific rules. Article URLs are /<any-category-slug>/<id>
    and the site routes purely by numeric ID (the slug doesn't have to
    match the article's real category), so a wide ID range can be sampled
    directly -- no sitemap needed.
  OnlineKhabar (onlinekhabar.com) -- robots.txt is `User-agent: *` with no
    rules at all. Monthly archive pages (/YYYY/MM, then /YYYY/MM/page/N for
    N>1) list ~200 articles/page, confirmed available back to at least 2024/01.

Excluded on purpose: bbc.com (covers bbc.com/nepali too) and
nepalpress.com -- both name ClaudeBot / anthropic-ai / Claude-Web explicitly
as disallowed in robots.txt.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes straight to data/raw/{setopati,onlinekhabar}_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape.py --setopati-stride 4 --months-back 24
"""
import argparse
import re
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk, fetch_html  # noqa: E402
from common.url_discovery import discover_from_paginated_archive, id_range_urls  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

SETOPATI_TEMPLATE = "https://www.setopati.com/politics/{id}"
ONLINEKHABAR_LINK_PATTERN = r'https://www\.onlinekhabar\.com/\d{4}/\d{2}/\d+/[a-z0-9-]+'


def setopati_current_max_id() -> int:
    """Discover today's newest article ID from the RSS feed so the range
    stays current without hardcoding a number that goes stale.
    """
    html = fetch_html("https://www.setopati.com/feed")
    ids = [int(i) for i in re.findall(r"setopati\.com/[a-zA-Z/-]+/(\d+)", html or "")]
    if not ids:
        raise RuntimeError("Could not determine current Setopati article ID from RSS feed")
    return max(ids)


def onlinekhabar_month_url_fn(year: int, month: int):
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


def main(setopati_id_start: int, setopati_stride: int, months_back: int, max_workers: int, min_interval: float) -> None:
    print("=== Setopati: building ID-range candidates ===")
    max_id = setopati_current_max_id()
    print(f"Current max article ID: {max_id}")
    ids = range(setopati_id_start, max_id, setopati_stride)
    setopati_urls = id_range_urls(SETOPATI_TEMPLATE, ids)
    print(f"{len(setopati_urls)} candidate URLs (stride={setopati_stride})")
    n = crawl_urls_bulk(
        setopati_urls, LANG_DIR / "data" / "raw" / "setopati_bulk.jsonl", source_name="setopati_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Setopati: {n} new articles written this run.\n")

    print(f"=== OnlineKhabar: paginated archive for the last {months_back} months ===")
    all_urls: list[str] = []
    for year, month in recent_year_months(months_back):
        urls = discover_from_paginated_archive(
            onlinekhabar_month_url_fn(year, month), ONLINEKHABAR_LINK_PATTERN, max_pages=60,
        )
        print(f"  {year}/{month:02d}: {len(urls)} links")
        all_urls.extend(urls)
    n = crawl_urls_bulk(
        all_urls, LANG_DIR / "data" / "raw" / "onlinekhabar_bulk.jsonl", source_name="onlinekhabar_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"OnlineKhabar: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setopati-id-start", type=int, default=6000,
                         help="IDs below ~5000 are almost entirely 404 (confirmed by direct sampling: "
                              "0/800+ hits under 5000, 90-100% hit rate at 5000+) -- skip that dead zone")
    parser.add_argument("--setopati-stride", type=int, default=4,
                         help="sample every Nth ID to control request volume")
    parser.add_argument("--months-back", type=int, default=24)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=1.2,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.setopati_id_start, args.setopati_stride, args.months_back, args.max_workers, args.min_interval)
