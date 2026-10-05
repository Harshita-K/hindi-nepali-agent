"""Large-scale MANUAL Nepali collection from Ekantipur (ekantipur.com),
standalone so it runs in PARALLEL with Setopati/OnlineKhabar/DC Nepal/
Khabarhub (different domain, independent per-domain throttle).

robots.txt has no bot-specific disallow and no ai-train content-signal --
neither grants nor restricts under the site's own stated rules (unlike
karobardaily.com, which explicitly blocks ClaudeBot).

Sitemap structure is custom (not WordPress wp-sitemap): the index at
/sitemap_index lists 501 sub-sitemaps at /sitemap_index/newsdetail/{n}, each
holding ~1000 <url> entries. Each story is listed TWICE -- a Nepali URL and
an English translation at the same path + "/en/" -- so English entries are
filtered out here. Confirmed empirically: sub-sitemaps 1-133 hold real
content (today back to ~Jan 2025, ~500 Nepali articles per page); 134+ are
consistently empty, so discovery stops after a run of empty pages rather
than working through all 501 blindly.

Resumable: safe to Ctrl-C and re-run, already-scraped URLs are skipped.
Writes to data/raw/ekantipur_bulk.jsonl (source_type=manual).

Run from repo root:
    python nepali/data/scraping/bulk_scrape_ekantipur.py
"""
import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from common.scraping_utils import USER_AGENT, crawl_urls_bulk  # noqa: E402

LANG_DIR = Path(__file__).resolve().parents[2]  # nepali/

SITEMAP_INDEX = "https://ekantipur.com/sitemap_index"
SUBSITEMAP_TEMPLATE = "https://ekantipur.com/sitemap_index/newsdetail/{n}"


def discover_ekantipur_urls(max_subsitemaps: int, stop_after_empty: int) -> list[str]:
    import requests

    r = requests.get(SITEMAP_INDEX, headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    n_sub = len(re.findall(r"<loc>", r.text))
    print(f"Sitemap index lists {n_sub} sub-sitemaps")

    urls: list[str] = []
    empty_streak = 0
    for n in range(1, min(n_sub, max_subsitemaps) + 1):
        try:
            resp = requests.get(SUBSITEMAP_TEMPLATE.format(n=n), headers={"User-Agent": USER_AGENT}, timeout=15)
            resp.raise_for_status()
            locs = re.findall(r"<loc>(.*?)</loc>", resp.text)
        except Exception as e:
            print(f"[sub-sitemap {n} failed] {e}")
            locs = []
        ne_locs = [u for u in locs if "/en/" not in u]
        if not ne_locs:
            empty_streak += 1
            if empty_streak >= stop_after_empty:
                print(f"[stopping] {stop_after_empty} consecutive empty sub-sitemaps at page {n}")
                break
        else:
            empty_streak = 0
            urls.extend(ne_locs)
    return urls


def main(max_subsitemaps: int, stop_after_empty: int, max_workers: int, min_interval: float) -> None:
    print("=== Ekantipur: discovering URLs from custom sitemap index ===")
    urls = discover_ekantipur_urls(max_subsitemaps, stop_after_empty)
    print(f"{len(urls)} candidate Nepali-language URLs discovered")
    n = crawl_urls_bulk(
        urls, LANG_DIR / "data" / "raw" / "ekantipur_bulk.jsonl", source_name="ekantipur_manual",
        min_interval_per_domain=min_interval, max_workers=max_workers,
    )
    print(f"Ekantipur: {n} new articles written this run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-subsitemaps", type=int, default=200,
                         help="upper bound on sub-sitemap pages to try (index lists 501, but content ends ~133)")
    parser.add_argument("--stop-after-empty", type=int, default=5,
                         help="stop discovery after this many consecutive empty sub-sitemaps")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--min-interval", type=float, default=0.8,
                         help="seconds between requests to the same domain")
    args = parser.parse_args()
    main(args.max_subsitemaps, args.stop_after_empty, args.max_workers, args.min_interval)
