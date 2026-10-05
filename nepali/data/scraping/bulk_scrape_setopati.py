"""Large-scale MANUAL Nepali collection from Setopati (setopati.com), split
out as its own standalone script so it runs in PARALLEL with DC Nepal and
OnlineKhabar (different domains, independent per-domain throttles) instead
of any of them queuing behind another in a single sequential script.

robots.txt only blocks /login /cms /preview /mail, no bot-specific rules.
Article URLs are /<any-category-slug>/<id> and the site routes purely by
numeric ID (the slug doesn't have to match the article's real category), so
a wide ID range can be sampled directly -- no sitemap needed.

IDs below ~5000 are almost entirely 404 (confirmed by direct sampling:
0/800+ hits under 5000, 90-100% hit rate at 5000+) -- default start skips
that dead zone.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/setopati_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape_setopati.py --stride 5
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

TEMPLATE = "https://www.setopati.com/politics/{id}"


def current_max_id(retries: int = 5, delay: float = 15.0) -> int:
    """Discover today's newest article ID from the RSS feed so the range
    stays current without hardcoding a number that goes stale.

    The feed has occasionally returned a transient 500 -- retry with a
    delay rather than crashing the whole run over a momentary blip.
    """
    import time

    for attempt in range(retries):
        html = fetch_html("https://www.setopati.com/feed")
        ids = [int(i) for i in re.findall(r"setopati\.com/[a-zA-Z/-]+/(\d+)", html or "")]
        if ids:
            return max(ids)
        print(f"[RSS feed fetch failed, retrying in {delay}s ({attempt + 1}/{retries})]")
        time.sleep(delay)
    raise RuntimeError("Could not determine current Setopati article ID from RSS feed after retries")


def main(id_start: int, stride: int, max_workers: int, min_interval: float) -> None:
    print("=== Setopati: building ID-range candidates ===")
    max_id = current_max_id()
    print(f"Current max article ID: {max_id}")
    ids = range(id_start, max_id, stride)
    urls = id_range_urls(TEMPLATE, ids)
    print(f"{len(urls)} candidate URLs (stride={stride})")
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "setopati_bulk.jsonl", source_name="setopati_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Setopati: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id-start", type=int, default=6000,
                         help="IDs below ~5000 are almost entirely 404 -- skip that dead zone")
    parser.add_argument("--stride", type=int, default=5, help="sample every Nth ID to control request volume")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.id_start, args.stride, args.max_workers, args.min_interval)
