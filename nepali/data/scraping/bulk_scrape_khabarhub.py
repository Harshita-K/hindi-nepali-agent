"""Large-scale MANUAL Nepali collection from Khabarhub (khabarhub.com),
standalone so it runs in PARALLEL with Setopati/OnlineKhabar/DC Nepal
(different domain, independent per-domain throttle).

robots.txt: only `Disallow: /wp-admin/` under `User-agent: *` -- no
bot-specific block. WordPress's native wp-sitemap.xml index: 137
sub-sitemaps of up to 2000 URLs each (potentially 270K+ candidate articles) --
same proven pattern as dcnepal.com, which had a 99.7% success rate.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/khabarhub_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape_khabarhub.py --max-urls 50000
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk  # noqa: E402
from common.url_discovery import discover_from_sitemap  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

SITEMAP_INDEX = "https://khabarhub.com/wp-sitemap.xml"


def main(max_urls: int, max_workers: int, min_interval: float) -> None:
    print(f"=== Khabarhub: discovering URLs from wp-sitemap.xml (up to {max_urls}) ===")
    urls = discover_from_sitemap(SITEMAP_INDEX, limit=max_urls, max_subsitemaps=200)
    print(f"{len(urls)} candidate URLs discovered")
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "khabarhub_bulk.jsonl", source_name="khabarhub_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Khabarhub: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-urls", type=int, default=200000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.max_urls, args.max_workers, args.min_interval)
