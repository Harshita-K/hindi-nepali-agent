"""Large-scale MANUAL Nepali collection from Ratopati (ratopati.com),
standalone so it runs in PARALLEL with the other Nepali sources (different
domain, independent per-domain throttle).

robots.txt: `Disallow:` (empty, i.e. nothing blocked) but `Crawl-delay: 20`
-- much slower than the other sources' self-imposed 0.8s, but this one is
site-mandated so it's respected as the default rather than overridden.

Article URLs are /story/{id}/{any-slug} and the site routes purely by
numeric ID (confirmed: a dummy slug still resolves) -- same proven pattern
as Setopati, no sitemap needed.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/ratopati_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape_ratopati.py --stride 3
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import crawl_urls_bulk, fetch_html  # noqa: E402
from common.url_discovery import id_range_urls  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

TEMPLATE = "https://www.ratopati.com/story/{id}/x"


def current_max_id(retries: int = 5, delay: float = 15.0) -> int:
    import time

    for attempt in range(retries):
        html = fetch_html("https://www.ratopati.com/feed")
        ids = [int(i) for i in re.findall(r"ratopati\.com/story/(\d+)", html or "")]
        if ids:
            return max(ids)
        print(f"[RSS feed fetch failed, retrying in {delay}s ({attempt + 1}/{retries})]")
        time.sleep(delay)
    raise RuntimeError("Could not determine current Ratopati article ID from RSS feed after retries")


def main(id_start: int, stride: int, max_workers: int, min_interval: float) -> None:
    print("=== Ratopati: building ID-range candidates ===")
    max_id = current_max_id()
    print(f"Current max article ID: {max_id}")
    ids = range(id_start, max_id, stride)
    urls = id_range_urls(TEMPLATE, ids)
    print(f"{len(urls)} candidate URLs (stride={stride})")
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "ratopati_bulk.jsonl", source_name="ratopati_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Ratopati: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-start", type=int, default=1000,
                         help="low IDs may be sparse/dead -- crawl_urls_bulk just skips failed fetches")
    parser.add_argument("--stride", type=int, default=3, help="sample every Nth ID to control request volume")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--min-interval", type=float, default=20.0,
                         help="seconds between requests to the same domain (site mandates Crawl-delay: 20)")
    args = parser.parse_args()
    main(args.id_start, args.stride, args.max_workers, args.min_interval)
