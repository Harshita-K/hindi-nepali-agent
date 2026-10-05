"""Wikipedia dump download + extraction (counts as DOWNLOADED, not manual).

Steps:
  1. Download the latest pages-articles dump for the given wiki code from
     https://dumps.wikimedia.org/{wiki_code}wiki/latest/
  2. Run wikiextractor to strip markup and produce clean JSON per article.
  3. Convert to our standard JSONL schema (source_type="downloaded").
"""
import json
import subprocess
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DUMP_URL_TEMPLATE = "https://dumps.wikimedia.org/{wiki_code}wiki/latest/{wiki_code}wiki-latest-pages-articles.xml.bz2"


def download_dump(wiki_code: str, out_dir: Path) -> Path:
    """wiki_code: 'hi' for Hindi, 'ne' for Nepali."""
    out_dir.mkdir(parents=True, exist_ok=True)
    url = DUMP_URL_TEMPLATE.format(wiki_code=wiki_code)
    dest = out_dir / f"{wiki_code}wiki-latest-pages-articles.xml.bz2"
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return dest

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
        )
    }

    print(f"Downloading {url} ...")
    with requests.get(url, stream=True, headers=headers, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with dest.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def run_wikiextractor(dump_path: Path, extracted_dir: Path) -> None:
    extracted_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "wikiextractor.WikiExtractor",
        str(dump_path), "-o", str(extracted_dir), "--json", "--no-templates",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def convert_extracted_to_jsonl(extracted_dir: Path, out_path: Path, source_name: str) -> int:
    """wikiextractor writes many small files of one-JSON-object-per-line
    (fields: id, title, text). Re-tag into our schema and merge into one file.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for sub_path in sorted(extracted_dir.rglob("wiki_*")):
            with sub_path.open(encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text = obj.get("text", "").strip()
                    if len(text) < 200:
                        continue
                    record = {
                        "text": text,
                        "source": source_name,
                        "source_type": "downloaded",
                        "title": obj.get("title", ""),
                        "wiki_id": obj.get("id", ""),
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
    return n_written


def build_wikipedia_corpus(wiki_code: str, work_dir: Path, out_path: Path, source_name: str) -> int:
    dump_path = download_dump(wiki_code, work_dir / "dump")
    extracted_dir = work_dir / "extracted"
    run_wikiextractor(dump_path, extracted_dir)
    return convert_extracted_to_jsonl(extracted_dir, out_path, source_name)
