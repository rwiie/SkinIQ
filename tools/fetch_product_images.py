"""Find a product photo on INKEEDecoder (formerly INCIDecoder) for every catalogue product.

Writes data/knowledge/product_images.csv (brand, product, page_url, image_url, match), which
the app uses to show product photos. Only links are stored: images are loaded from
INKEEDecoder's image storage when shown, with a link back to the product page as credit.

Polite crawling: robots.txt allows /products/ and /search; one request per second
(--delay), a descriptive User-Agent, and resumable (products already in the CSV are
skipped). Run from the repository root:

    python tools/fetch_product_images.py            # all products (~30-60 min)
    python tools/fetch_product_images.py --limit 20 # try a few
"""

from __future__ import annotations

import argparse
import csv
import html
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from skiniq.product_images import IMAGES_CSV  # noqa: E402
from skiniq.products import load_products  # noqa: E402

SITE = "https://inkeedecoder.com"
HEADERS = {"User-Agent": "SkinIQ-student-prototype/1.0 (product photo lookup; 1 request per second)"}
FIELDS = ["brand", "product", "page_url", "image_url", "match", "page_title"]
MIN_SIMILARITY = 0.75  # applied after the word check in similarity()

_IMAGE = re.compile(r'https://incidecoder-content\.storage\.googleapis\.com/[^"\s]+?_front_photo_300x300@2x\.jpeg')
_IMAGE_ANY = re.compile(r'https://incidecoder-content\.storage\.googleapis\.com/[^"\s]+?_front_photo[^"\s/]*?\.(?:jpe?g|png|webp)')
_TITLE = re.compile(r"<title>(.*?) ingredients \(Explained\)", re.S)
_RESULT = re.compile(r'<a href="(/products/[^"]+)" class="klavika simpletextlistitem">(.*?)</a>', re.S)


def normalise(text: str) -> str:
    text = re.sub(r"[™®©]", " ", html.unescape(text))  # 'Ceramidin™' -> 'Ceramidin'
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"['`]", "", text.lower())  # A'Pieu -> apieu
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def core(text: str) -> str:
    """Name without sun-protection ratings, which are written many ways ('SPF50+/PA++++')."""
    t = normalise(re.sub(r"(?i)\bspf\s*\d*\+*|\bpa\s*\++", " ", text))
    t = re.sub(r"(\w)is(ing|er|e)\b", r"\1iz\2", t)  # moisturising -> moisturizing
    return re.sub(r"\s+", " ", t).strip()


def slugify(text: str) -> str:
    return normalise(text).replace(" ", "-")


_FILLER = {"with", "and", "the", "for", "of", "tm"}  # "tm": the ™ sign written out


def _words(text: str) -> set[str]:
    return {w for w in core(text).split() if w not in _FILLER}


def similarity(found: str, wanted: str) -> float:
    """0 unless the names are the same product, else how close they are (for ranking).

    Same product: either every word of the wanted name is in the found name, which adds at
    most two words (e.g. 'jojoba', '2X'), or the found name only drops one word of a long
    (5+ word) name and adds none (e.g. a sub-brand). A swapped word means another product or
    variant, so 'Moisturizing Foam Cleanser' never matches 'Moisturizing Cream Cleanser' and
    'Sheet Mask (Olive)' never matches 'Sheet Mask (Lime)'.
    """
    have, want = _words(found), _words(wanted)
    missing, extra = want - have, have - want
    ok = (not missing and len(extra) <= 2) or (len(missing) == 1 and not extra and len(want) >= 5)
    if not ok:
        return 0.0
    return SequenceMatcher(None, " ".join(sorted(have)), " ".join(sorted(want))).ratio()


class Fetcher:
    def __init__(self, delay: float):
        self.delay = delay
        self.last = 0.0
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get(self, path: str, **params) -> requests.Response | None:
        wait = self.delay - (time.monotonic() - self.last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(3):
            self.last = time.monotonic()
            try:
                r = self.session.get(SITE + path, params=params or None, timeout=20)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(15 * (attempt + 1))
                continue
            return r
        return None


def page_image(text: str) -> str:
    m = _IMAGE.search(text) or _IMAGE_ANY.search(text)
    return m.group(0) if m else ""


def page_title(text: str) -> str:
    m = _TITLE.search(text)
    return html.unescape(m.group(1)).strip() if m else ""


NOT_FOUND = {"page_url": "", "image_url": "", "match": "none", "page_title": ""}


def find(fetcher: Fetcher, brand: str, name: str) -> dict:
    """Page and photo for one product. A page is used only if its own title names the same
    product: search result names and page addresses can be out of date on the site."""
    title = f"{brand} {name}"
    # 1) the page address made from the name
    r = fetcher.get(f"/products/{slugify(title)}")
    if r is not None and r.status_code == 200:
        found = page_title(r.text)
        if found and similarity(found, title) >= MIN_SIMILARITY:
            return {"page_url": r.url, "image_url": page_image(r.text), "match": "address", "page_title": found}
    # 2) site search (punctuation and SPF/PA ratings confuse it), best-matching result names
    results = []
    plain = re.sub(r"\s+", " ", re.sub(r"(?i)\bspf\s*\d*\+*|\bpa\s*\++|[/()]", " ", title)).strip()
    for query in dict.fromkeys([plain, core(title), " ".join(plain.split()[:5])]):
        r = fetcher.get("/search", query=query)
        if r is None or r.status_code != 200:
            continue
        results = [(similarity(text, title), href, text) for href, text in _RESULT.findall(r.text)
                   if href != "/products/create"]
        if results and max(results)[0] >= MIN_SIMILARITY:
            break
    for score, href, _ in sorted(results, reverse=True)[:3]:
        if score < MIN_SIMILARITY:
            break
        r = fetcher.get(href)
        if r is None or r.status_code != 200:
            continue
        found = page_title(r.text)
        if found and similarity(found, title) >= MIN_SIMILARITY:  # the page itself must agree
            return {"page_url": r.url, "image_url": page_image(r.text), "match": f"search {score:.2f}",
                    "page_title": found}
    return dict(NOT_FOUND)


def recheck(path: Path, fetcher: Fetcher) -> None:
    """Validate every stored match against its page's own title (rewrites the CSV)."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    changed = kept = 0
    for i, row in enumerate(rows, 1):
        if row.get("page_url") and not row.get("page_title"):
            title = f"{row['brand']} {row['product']}"
            r = fetcher.get(row["page_url"].replace(SITE, ""))
            found = page_title(r.text) if r is not None and r.status_code == 200 else ""
            if found and similarity(found, title) >= MIN_SIMILARITY:
                row["page_title"] = found
                kept += 1
            else:  # wrong product (or page gone): search again with the stricter rule
                row.update(find(fetcher, row["brand"], row["product"]))
                row["match"] += f" (was {found or 'missing page'})"
                changed += 1
        if i % 100 == 0 or i == len(rows):
            print(f"{i}/{len(rows)} checked, {kept} confirmed, {changed} replaced or removed", flush=True)
            _write(path, rows)


def _write(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in FIELDS} for row in rows)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="stop after this many new products")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests (default 1)")
    ap.add_argument("--out", type=Path, default=IMAGES_CSV)
    ap.add_argument("--recheck", action="store_true",
                    help="confirm every stored match against its page title; replace wrong ones")
    args = ap.parse_args()
    if args.recheck:
        recheck(args.out, Fetcher(args.delay))
        return

    done = set()
    if args.out.exists():
        with args.out.open(newline="", encoding="utf-8") as f:
            done = {(r["brand"], r["product"]) for r in csv.DictReader(f)}
    todo = list(dict.fromkeys((p.brand, p.name) for p in load_products() if (p.brand, p.name) not in done))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} already done, {len(todo)} to look up", flush=True)

    new_file = not args.out.exists()
    fetcher = Fetcher(args.delay)
    with args.out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        found = 0
        for i, (brand, name) in enumerate(todo, 1):
            row = {"brand": brand, "product": name, **find(fetcher, brand, name)}
            writer.writerow(row)
            f.flush()
            found += bool(row["image_url"])
            if i % 25 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} looked up, {found} with a photo", flush=True)


if __name__ == "__main__":
    main()
