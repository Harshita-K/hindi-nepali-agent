"""Polite web-scraping utilities for MANUAL data collection.

Used to satisfy the >=20% manual-collection requirement by scraping and
cleaning pages ourselves (as opposed to downloading pre-built corpora like
Wikipedia dumps or OSCAR/CC100, which count as "downloaded").

Respects robots.txt and rate-limits requests. Extraction uses trafilatura,
which strips nav/ads/boilerplate and returns just the article body.
"""
import json
import threading
import time
import urllib.robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
import trafilatura

USER_AGENT = "LMA-coursework-bot/1.0 (educational corpus collection)"
_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}


def _robots_for(url: str) -> urllib.robotparser.RobotFileParser:
    """Fetch robots.txt using OUR real User-Agent, not urllib's default.

    `RobotFileParser.read()` calls bare `urlopen()` with no headers, which
    identifies as generic "Python-urllib/x.y". Several sites (e.g.
    onlinekhabar.com) 403 that generic UA as basic bot-detection -- nothing
    to do with any AI-crawler policy -- and robotparser's convention is to
    treat a 401/403 on robots.txt as "disallow everything", which would
    misrepresent a genuinely permissive robots.txt as a block. Fetching with
    the same identity our real requests use avoids that false negative.
    """
    origin = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    if origin not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(origin + "/robots.txt")
        try:
            resp = requests.get(origin + "/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=10)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            elif resp.status_code in (401, 403):
                rp.disallow_all = True
            # any other status (404 etc.) -> leave parser in its default
            # (permissive) state, matching robots.txt convention
        except requests.RequestException:
            pass  # unreachable robots.txt -> fall back to permissive
        _robots_cache[origin] = rp
    return _robots_cache[origin]


def allowed_by_robots(url: str) -> bool:
    return _robots_for(url).can_fetch(USER_AGENT, url)


def fetch_html(url: str, timeout: int = 15) -> str | None:
    if not allowed_by_robots(url):
        print(f"[skip: robots.txt disallows] {url}")
        return None
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        resp.raise_for_status()
        # requests defaults to ISO-8859-1 when a text/* response has no
        # charset in its Content-Type header (some sites, e.g. some
        # onlinekhabar.com article pages, omit it despite the page itself
        # declaring <meta charset="UTF-8">). That silently mojibake's every
        # non-ASCII character. apparent_encoding (content-sniffed) catches
        # this; only trust the header's encoding when a charset was actually
        # declared in it.
        if "charset" not in resp.headers.get("content-type", "").lower():
            resp.encoding = resp.apparent_encoding
        return resp.text
    except requests.RequestException as e:
        print(f"[fetch failed] {url}: {e}")
        return None


def extract_article_text(html: str, url: str) -> str | None:
    return trafilatura.extract(html, url=url, favor_precision=True)


def crawl_urls(urls: list[str], out_path: Path, source_name: str, delay_seconds: float = 2.0) -> int:
    """Fetch+extract each URL, appending JSONL records to out_path.

    Each record is tagged source_type="manual" since this is scraping +
    cleaning that we do ourselves (per the assignment's manual-collection
    definition). Returns count of documents written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for url in urls:
            html = fetch_html(url)
            if html is None:
                continue
            text = extract_article_text(html, url)
            if not text or len(text) < 200:
                continue
            record = {
                "text": text,
                "source": source_name,
                "source_type": "manual",
                "url": url,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
            time.sleep(delay_seconds)  # be polite; avoid hammering the site
    return n_written


def load_url_list(path: Path) -> list[str]:
    """One URL per line, '#' comments allowed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


# --- bulk crawling: concurrent, per-domain rate-limited, resumable ---
# For thousands-of-URLs sources (sitemap archives, ID-range enumeration,
# paginated archives). A per-domain lock+timestamp keeps requests to any
# single site spaced `min_interval_per_domain` seconds apart regardless of
# thread count, so max_workers only buys parallelism *across* domains.

_domain_locks: dict[str, threading.Lock] = {}
_domain_last_request: dict[str, float] = {}
_domain_lock_guard = threading.Lock()


def _throttle(url: str, min_interval: float) -> None:
    domain = urlparse(url).netloc
    with _domain_lock_guard:
        lock = _domain_locks.setdefault(domain, threading.Lock())
    with lock:
        wait = min_interval - (time.time() - _domain_last_request.get(domain, 0.0))
        if wait > 0:
            time.sleep(wait)
        _domain_last_request[domain] = time.time()


def load_scraped_urls(out_path: Path) -> set[str]:
    """URLs already present in a JSONL output file (for resuming a bulk crawl)."""
    if not out_path.exists():
        return set()
    urls = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                urls.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return urls


def crawl_urls_bulk(
    urls: list[str],
    out_path: Path,
    source_name: str,
    min_interval_per_domain: float = 1.2,
    max_workers: int = 8,
    min_chars: int = 200,
) -> int:
    """Concurrent, resumable crawl for large URL sets (thousands+).

    Safe to Ctrl-C and re-run: already-written URLs (tracked by reading
    out_path) are skipped. Each successful fetch is written immediately so
    progress isn't lost on interruption.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already = load_scraped_urls(out_path)
    todo = [u for u in dict.fromkeys(urls) if u not in already]  # dedup, preserve order
    print(f"{len(already)} already scraped, {len(todo)} remaining out of {len(urls)} candidates")

    write_lock = threading.Lock()
    n_written = 0

    def worker(url: str) -> tuple[str, str | None]:
        _throttle(url, min_interval_per_domain)
        if not allowed_by_robots(url):
            return url, None
        html = fetch_html(url)
        if html is None:
            return url, None
        text = extract_article_text(html, url)
        return url, (text if text and len(text) >= min_chars else None)

    with out_path.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, u) for u in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            url, text = fut.result()
            if text is not None:
                record = {"text": text, "source": source_name, "source_type": "manual", "url": url}
                with write_lock:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    n_written += 1
            if i % 100 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] processed, {n_written} kept so far")
    return n_written
