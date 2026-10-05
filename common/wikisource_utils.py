"""Bulk text collection from Wikisource (MANUAL source -- community-
transcribed/proofread public-domain books and documents; NOT scanned images,
so no OCR is involved, unlike the archive.org path).

Structural note: a "work" on Wikisource is a top-level title (often just an
index/table-of-contents page) whose actual content lives in "Title/subpage"
chapter pages (e.g. "कटोरा भर खून/१"). But even those subpages' WIKITEXT is
just a `<pages index="..." from=N to=M />` transclusion tag pointing at the
underlying scanned Page: namespace -- it only expands into real prose when
MediaWiki *renders* the page. So `prop=extracts` (which reads wikitext-ish
content) returns ~0 chars; this uses `action=parse&prop=text` (rendered
HTML) and strips tags, which is the only way to get the actual text. That
means one request per page rather than a batched generator query.

Requires a descriptive User-Agent (Wikimedia's API etiquette policy; the
default requests/urllib UA gets 403/429'd) and a courtesy rate limit.
"""
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = "LMA-coursework-bot/1.0 (educational corpus collection; contact: coursework project)"
API_URL_TEMPLATE = "https://{wiki_code}.wikisource.org/w/api.php"
REQUEST_DELAY = 2.0  # seconds between requests -- Wikimedia API etiquette


def _get(wiki_code: str, params: dict, max_retries: int = 6) -> dict:
    """429s from Wikimedia can persist for a while after a burst of requests
    (even a well-behaved one afterwards can still get throttled for a bit) --
    retry with growing backoff rather than treating it as fatal.
    """
    for attempt in range(max_retries):
        resp = requests.get(
            API_URL_TEMPLATE.format(wiki_code=wiki_code),
            params={**params, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        if resp.status_code == 429:
            wait = min(10 * (attempt + 1), 60)
            print(f"[429, backing off {wait}s (attempt {attempt + 1}/{max_retries})]")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return resp.json()
    raise requests.RequestException(f"Exceeded {max_retries} retries on 429")


def list_content_subpages(wiki_code: str, limit: int | None = None) -> list[str]:
    """Titles containing '/' -- top-level titles are usually just index/TOC
    pages carrying no real body text.
    """
    titles: list[str] = []
    apcontinue = None
    while limit is None or len(titles) < limit:
        params = {"action": "query", "list": "allpages", "apnamespace": 0, "aplimit": 500}
        if apcontinue:
            params["apcontinue"] = apcontinue
        body = _get(wiki_code, params)
        for p in body.get("query", {}).get("allpages", []):
            if "/" in p["title"]:
                titles.append(p["title"])
        apcontinue = body.get("continue", {}).get("apcontinue")
        if not apcontinue:
            break
    return titles


def fetch_rendered_text(wiki_code: str, title: str) -> str | None:
    """Render the page (expanding <pages> transclusions) and strip HTML."""
    try:
        body = _get(wiki_code, {"action": "parse", "page": title, "prop": "text"})
    except requests.RequestException:
        return None
    if "parse" not in body:
        return None
    html = body["parse"]["text"]["*"]
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ")
    return re.sub(r"\s+", " ", text).strip()


def collect_wikisource(wiki_code: str, out_path: Path, max_pages: int = 20000, min_chars: int = 200) -> int:
    """Resumable: already-collected titles (read from out_path) are skipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already.add(json.loads(line).get("title"))

    print("Listing content subpages...")
    titles = list_content_subpages(wiki_code)
    print(f"{len(titles)} candidate subpages found")

    n_written = 0
    with out_path.open("a", encoding="utf-8") as out_f:
        for title in titles:
            if title in already or n_written >= max_pages:
                continue
            already.add(title)
            text = fetch_rendered_text(wiki_code, title)
            if not text or len(text) < min_chars:
                continue
            record = {
                "text": text,
                "source": f"{wiki_code}_wikisource",
                "source_type": "manual",
                "title": title,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            n_written += 1
            if n_written % 100 == 0:
                print(f"[{n_written}/{max_pages}] latest: {title} ({len(text)} chars)")
    return n_written
