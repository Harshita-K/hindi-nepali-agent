"""Discover candidate article URLs automatically (RSS feeds / sitemaps),
then VERIFY each one by actually fetching it and confirming trafilatura can
pull out a real article body. Only verified URLs get written to
urls_manual.txt -- this replaces hand-copy-pasting links one at a time.

"Verified" here means: robots.txt allows it, the page fetches successfully,
and extracted article text clears a minimum length. It does not mean
fact-checked; you're still responsible for skimming the report for garbage.
"""
import json
from pathlib import Path
from urllib.parse import urljoin
from xml.etree import ElementTree

import feedparser
import requests

from common.scraping_utils import USER_AGENT, allowed_by_robots, extract_article_text, fetch_html

SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def discover_from_rss(feed_url: str) -> list[str]:
    """Return article links found in an RSS/Atom feed."""
    parsed = feedparser.parse(feed_url, request_headers={"User-Agent": USER_AGENT})
    return [entry.link for entry in parsed.entries if getattr(entry, "link", None)]


def discover_from_sitemap(sitemap_url: str, limit: int = 500, max_subsitemaps: int = 500) -> list[str]:
    """Return <loc> URLs from a sitemap.xml (or sitemap index, one level deep).

    `max_subsitemaps` bounds how many sub-sitemaps get recursed into for a
    sitemap INDEX (some sites, e.g. WordPress's wp-sitemap.xml, split into
    100+ sub-sitemaps of ~2000 URLs each) -- the `limit` total-URL cap also
    applies and usually triggers first, this is just a safety ceiling.
    """
    try:
        resp = requests.get(sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=15)
        resp.raise_for_status()
        # Some sites (e.g. dcnepal.com's WordPress sitemap plugin) emit a
        # stray leading newline before the XML declaration, which ElementTree
        # rejects outright ("XML or text declaration not at start of entity").
        root = ElementTree.fromstring(resp.content.lstrip())
    except Exception as e:
        print(f"[sitemap failed] {sitemap_url}: {e}")
        return []

    locs = [el.text for el in root.findall(".//sm:loc", SITEMAP_NS) if el.text]
    if locs and locs[0].endswith(".xml"):  # sitemap index -> recurse one level
        urls = []
        for sub in locs[:max_subsitemaps]:
            urls.extend(discover_from_sitemap(sub, limit=limit, max_subsitemaps=max_subsitemaps))
            if len(urls) >= limit:
                break
        return urls[:limit]
    return locs[:limit]


def verify_url(url: str, min_chars: int = 200) -> str | None:
    """Fetch + extract; return the extracted text if it looks like a real
    article, else None. This is what separates "discovered" from "verified".
    """
    if not allowed_by_robots(url):
        return None
    html = fetch_html(url)
    if html is None:
        return None
    text = extract_article_text(html, url)
    if text and len(text) >= min_chars:
        return text
    return None


def discover_and_verify(
    rss_feeds: list[str] | None = None,
    sitemaps: list[str] | None = None,
    max_candidates: int = 300,
    max_verified: int = 150,
    min_chars: int = 200,
) -> list[dict]:
    """Full pipeline: gather candidate URLs from feeds/sitemaps, dedup, then
    verify each until max_verified is reached. Returns a list of
    {"url":..., "n_chars": ...} for URLs that passed verification.
    """
    candidates: list[str] = []
    for feed in rss_feeds or []:
        found = discover_from_rss(feed)
        print(f"[rss] {feed}: {len(found)} candidate links")
        candidates.extend(found)
    for sm in sitemaps or []:
        found = discover_from_sitemap(sm)
        print(f"[sitemap] {sm}: {len(found)} candidate links")
        candidates.extend(found)

    seen, deduped = set(), []
    for u in candidates:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    deduped = deduped[:max_candidates]

    verified = []
    for url in deduped:
        text = verify_url(url, min_chars=min_chars)
        if text is not None:
            verified.append({"url": url, "n_chars": len(text)})
            print(f"[verified {len(verified)}/{max_verified}] {url} ({len(text)} chars)")
        else:
            print(f"[rejected] {url}")
        if len(verified) >= max_verified:
            break
    return verified


def discover_from_date_sitemaps(url_template: str, dates: list[str]) -> list[str]:
    """Aggregate URLs across many per-day sitemap files, e.g. Patrika's
    https://www.patrika.com/date/google-sitemap-{date}.xml (one working day
    of history at a time -- confirmed available back well over a year).
    `url_template` must contain a single "{date}" placeholder.
    """
    import time as _time

    urls: list[str] = []
    for date_str in dates:
        found = discover_from_sitemap(url_template.format(date=date_str))
        urls.extend(found)
        _time.sleep(0.3)  # courtesy delay between sitemap-index fetches
    print(f"[date-sitemaps] {len(dates)} days -> {len(urls)} candidate URLs")
    return urls


def id_range_urls(url_template: str, ids: range) -> list[str]:
    """Build candidate URLs from a sequential-ID pattern, e.g. Setopati's
    https://www.setopati.com/politics/{id} (the category slug in the path
    doesn't need to match the article's real category -- the site routes by
    ID). No network calls here; existence is settled during the crawl.
    """
    return [url_template.format(id=i) for i in ids]


def discover_from_paginated_archive(
    page_url_fn,
    link_pattern: str,
    max_pages: int = 200,
) -> list[str]:
    """Crawl a site's own paginated date-archive (e.g. OnlineKhabar's
    /YYYY/MM (+ /page/N for N>1) listing pages), extracting article links
    with a regex, until a page yields zero new links or a fetch fails.

    `page_url_fn`: callable page_number(int, starting at 1) -> url string
    (a plain callable, not a template string, since page 1 is often a bare
    URL while later pages need a distinct pattern -- see per-site scripts).
    `link_pattern` is a regex whose full match is the article URL.
    """
    import re
    import time as _time

    urls: list[str] = []
    seen: set[str] = set()
    for page in range(1, max_pages + 1):
        html = fetch_html(page_url_fn(page))
        _time.sleep(1.0)  # politeness delay between archive-listing-page fetches
        if html is None:
            break
        found = set(re.findall(link_pattern, html))
        new = found - seen
        if not new:
            break
        seen |= new
        urls.extend(sorted(new))
    return urls


def write_verified_urls(verified: list[dict], urls_manual_path: Path) -> int:
    """Append verified URLs to urls_manual.txt, deduping against what's
    already there. Returns how many new URLs were added.
    """
    existing = set()
    if urls_manual_path.exists():
        existing = {
            ln.strip() for ln in urls_manual_path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        }
    new_urls = [v["url"] for v in verified if v["url"] not in existing]
    if new_urls:
        with urls_manual_path.open("a", encoding="utf-8") as f:
            f.write(f"\n# auto-discovered + verified ({len(new_urls)} URLs)\n")
            for u in new_urls:
                f.write(u + "\n")
    return len(new_urls)
