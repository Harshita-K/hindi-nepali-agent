"""Large-scale MANUAL Nepali collection from Rajdhani Daily (rajdhanidaily.com),
standalone so it runs in PARALLEL with the other Nepali sources (different
domain, independent per-domain throttle).

robots.txt: `Disallow:` (empty, nothing blocked), no bot-specific rules.
Standard Yoast SEO WordPress sitemap index (post-sitemap.xml,
post-sitemap2.xml, post-sitemap3.xml, ...), articles at /id/{id}/ going
back to at least 2019.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/rajdhani_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape_rajdhani.py --max-urls 100000
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk  # noqa: E402
from common.url_discovery import discover_from_sitemap  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

SITEMAP_INDEX = "https://rajdhanidaily.com/sitemap_index.xml"


def main(max_urls: int, max_workers: int, min_interval: float) -> None:
    print(f"=== Rajdhani Daily: discovering URLs from sitemap_index.xml (up to {max_urls}) ===")
    urls = discover_from_sitemap(SITEMAP_INDEX, limit=max_urls, max_subsitemaps=200)
    print(f"{len(urls)} candidate URLs discovered")
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "rajdhani_bulk.jsonl", source_name="rajdhani_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Rajdhani Daily: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-urls", type=int, default=100000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.max_urls, args.max_workers, args.min_interval)
