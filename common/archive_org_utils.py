"""Bulk book collection from archive.org (MANUAL source -- this is the
token-dense counterpart to news scraping: one book can be worth hundreds of
articles). robots.txt has no AI-bot-specific restrictions (only
`Disallow: /control/` under `User-agent: *`), so this is safe to automate.

archive.org has a long tail of spam/junk items carrying generic language
metadata (gibberish identifiers, no real content). Sorting search results by
downloads-descending is a cheap, effective filter: real books accumulate
real downloads, spam essentially never does.

Most legitimate book items already have a pre-OCR'd `<identifier>_djvu.txt`
file, which we prefer over running our own Tesseract pass (faster, and
archive.org's OCR is generally solid for printed books). Falls back to
downloading the PDF and running common.pdf_ocr_utils on it if no djvu text
exists.
"""
import json
import time
from pathlib import Path

import requests

from common.pdf_ocr_utils import extract_pdf_auto

USER_AGENT = "LMA-coursework-bot/1.0 (educational corpus collection)"
SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{filename}"


class DeepPagingLimitReached(Exception):
    """archive.org's advancedsearch API hard-caps sorted pagination at 10,000
    results ([DEEP_PAGING] error) -- this is permanent for a given query, not
    transient, so callers should stop paging rather than retry.
    """


def search_books(language_code: str, rows: int = 50, start: int = 0) -> list[dict]:
    """language_code: ISO 639 code, e.g. 'hin' or 'nep' (NOT the English
    name -- 'language:Nepali' pulls in far more spam than 'language:nep').
    """
    params = {
        "q": f"language:({language_code}) AND mediatype:(texts)",
        "fl[]": ["identifier", "title", "downloads"],
        "sort[]": "downloads desc",
        "rows": rows,
        "start": start,
        "output": "json",
    }
    resp = requests.get(SEARCH_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    body = resp.json()
    if "error" in body:
        if "DEEP_PAGING" in body["error"]:
            raise DeepPagingLimitReached(body["error"])
        raise ValueError(body["error"])
    return body["response"]["docs"]


def get_item_files(identifier: str) -> list[dict]:
    resp = requests.get(METADATA_URL.format(identifier=identifier),
                         headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return resp.json().get("files", [])


def fetch_book_text(identifier: str, tesseract_lang: str, tmp_pdf_dir: Path) -> tuple[str | None, str]:
    """Returns (text, method): 'djvu_ocr' (archive.org's own pre-extracted
    text), 'digital_pdf' / 'own_ocr' (we downloaded the PDF ourselves), or
    (None, 'none') if nothing usable was found.
    """
    files = get_item_files(identifier)

    djvu_file = next((f["name"] for f in files if f["name"].endswith("_djvu.txt")), None)
    if djvu_file:
        resp = requests.get(DOWNLOAD_URL.format(identifier=identifier, filename=djvu_file),
                             headers={"User-Agent": USER_AGENT}, timeout=60)
        if resp.ok:
            # archive.org's djvu.txt responses don't always declare charset
            # in Content-Type, which makes requests default to ISO-8859-1
            # and mojibake the text (same root cause as the OnlineKhabar fix
            # in common/scraping_utils.fetch_html).
            if "charset" not in resp.headers.get("content-type", "").lower():
                resp.encoding = resp.apparent_encoding
            return resp.text, "djvu_ocr"

    pdf_file = next((f["name"] for f in files if f["name"].endswith(".pdf")), None)
    if pdf_file:
        tmp_pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = tmp_pdf_dir / f"{identifier}.pdf"
        resp = requests.get(DOWNLOAD_URL.format(identifier=identifier, filename=pdf_file),
                             headers={"User-Agent": USER_AGENT}, timeout=120, stream=True)
        if resp.ok:
            with pdf_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            text, method = extract_pdf_auto(pdf_path, tesseract_lang)
            pdf_path.unlink(missing_ok=True)
            return text, ("own_ocr" if method == "ocr" else "digital_pdf")

    return None, "none"


def collect_books(
    language_code: str,
    tesseract_lang: str,
    out_path: Path,
    tmp_pdf_dir: Path,
    max_books: int = 500,
    min_chars: int = 2000,
) -> int:
    """Resumable: already-collected identifiers (read from out_path) are skipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    already: set[str] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    already.add(json.loads(line).get("identifier"))

    n_written = 0
    start = 0
    page_size = 50
    consecutive_failures = 0
    with out_path.open("a", encoding="utf-8") as out_f:
        while n_written < max_books:
            try:
                docs = search_books(language_code, rows=page_size, start=start)
                consecutive_failures = 0
            except DeepPagingLimitReached:
                print(f"[archive.org deep-paging limit hit at start={start} -- "
                      f"this is permanent for this query, stopping pagination here]")
                break
            except (requests.RequestException, KeyError, ValueError) as e:
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print(f"[giving up after 5 consecutive failures at start={start}]: {e}")
                    break
                print(f"[search page failed ({consecutive_failures}/5), retrying] start={start}: {e}")
                time.sleep(3)
                continue
            if not docs:
                break
            start += page_size
            for doc in docs:
                identifier = doc["identifier"]
                if identifier in already:
                    continue
                already.add(identifier)
                try:
                    text, method = fetch_book_text(identifier, tesseract_lang, tmp_pdf_dir)
                except (requests.RequestException, KeyError, ValueError) as e:
                    print(f"[skip] {identifier}: fetch/parse error ({e})")
                    continue
                if text is None or len(text) < min_chars:
                    print(f"[skip] {identifier}: no usable text")
                    continue
                record = {
                    "text": text,
                    "source": f"archive_org_{language_code}",
                    "source_type": "manual",
                    "identifier": identifier,
                    "title": doc.get("title", ""),
                    "extraction_method": method,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                n_written += 1
                print(f"[{n_written}/{max_books}] {identifier} ({method}, {len(text)} chars)")
                time.sleep(0.5)
                if n_written >= max_books:
                    break
    return n_written
