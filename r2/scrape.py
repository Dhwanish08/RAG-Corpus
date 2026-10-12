"""Scrape a Wikipedia category into a JSONL corpus via the MediaWiki API.

Usage:
    python scrape.py --category "Cricket World Cup" --limit 400 --out ../data/corpus.jsonl
"""
import argparse
import json
import sys
import time

import requests

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "R2-RAG-poisoning-project/1.0 (student research; contact via github)"}
MIN_CHARS = 500  # skip stubs, they add noise without adding facts


def category_members(category: str, limit: int):
    """Yield page titles in a category (non-recursive, skips subcategories)."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": 500,
        "cmtype": "page",
        "format": "json",
    }
    seen = 0
    while seen < limit:
        for attempt in range(6):
            r = requests.get(API, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                time.sleep(2**attempt)
                continue
            r.raise_for_status()
            break
        data = r.json()
        for m in data["query"]["categorymembers"]:
            yield m["title"]
            seen += 1
            if seen >= limit:
                return
        cont = data.get("continue")
        if not cont:
            return
        params.update(cont)


def fetch_extract(title: str) -> str:
    """Fetch the full plain-text extract for one page.

    The MediaWiki extracts API silently returns only ONE page's full extract
    per request when explaintext is used without exintro, even if you pass
    multiple titles -- so this fetches one title at a time, not batched.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    for attempt in range(8):
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 429:
            time.sleep(min(2**attempt, 60))
            continue
        r.raise_for_status()
        pages = r.json()["query"]["pages"]
        for p in pages.values():
            if "missing" not in p:
                return p.get("extract", "")
        return ""
    r.raise_for_status()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True, help='Wikipedia category name, no "Category:" prefix')
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out", default="../data/corpus.jsonl")
    args = ap.parse_args()

    titles = list(category_members(args.category, args.limit))
    print(f"found {len(titles)} page titles", file=sys.stderr)

    # resumable: skip titles already fetched by a previous (possibly interrupted) run
    done = set()
    try:
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["title"])
    except FileNotFoundError:
        pass
    todo = [t for t in titles if t not in done]
    print(f"{len(done)} already fetched, {len(todo)} remaining", file=sys.stderr)

    n_written = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, title in enumerate(todo, 1):
            text = fetch_extract(title)
            if len(text) >= MIN_CHARS:
                f.write(json.dumps({"title": title, "text": text}) + "\n")
                f.flush()
                n_written += 1
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} pages fetched", file=sys.stderr)
            time.sleep(0.3)  # be polite to the API

    print(f"wrote {n_written} new articles to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
