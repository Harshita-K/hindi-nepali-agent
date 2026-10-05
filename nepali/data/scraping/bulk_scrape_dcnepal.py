"""Large-scale MANUAL Nepali collection from DC Nepal (dcnepal.com).

robots.txt: only `Disallow: /wp-admin/` under `User-agent: *` -- no
bot-specific block. Confirmed separately from setopati.com/onlinekhabar.com's
domains, so running this alongside nepali/data/scraping/bulk_scrape.py adds
genuinely parallel throughput rather than competing for the same
per-domain rate-limit slot.

WordPress's native wp-sitemap.xml index: 123 sub-sitemaps of up to 2000
URLs each (potentially 200K+ candidate articles).

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/dcnepal_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape_dcnepal.py --max-urls 30000
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk  # noqa: E402
from common.url_discovery import discover_from_sitemap  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

SITEMAP_INDEX = "https://www.dcnepal.com/wp-sitemap.xml"


def main(max_urls: int, max_workers: int, min_interval: float) -> None:
    print(f"=== DC Nepal: discovering URLs from wp-sitemap.xml (up to {max_urls}) ===")
    urls = discover_from_sitemap(SITEMAP_INDEX, limit=max_urls, max_subsitemaps=200)
    print(f"{len(urls)} candidate URLs discovered")
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "dcnepal_bulk.jsonl", source_name="dcnepal_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"DC Nepal: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-urls", type=int, default=30000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.max_urls, args.max_workers, args.min_interval)
