"""Product photos from INKEEDecoder (formerly INCIDecoder).

`data/knowledge/product_images.csv` is built once by `tools/fetch_product_images.py`
and holds, per catalogue product, the INKEEDecoder product page and the URL of its
front photo. Only links are stored; the image is loaded from INKEEDecoder's storage
when shown, and the app links back to the product page as the source.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

IMAGES_CSV = Path(__file__).resolve().parent.parent / "data" / "knowledge" / "product_images.csv"
SOURCE_NAME = "INKEEDecoder"


@lru_cache(maxsize=1)
def _table(path: str = str(IMAGES_CSV)) -> dict[tuple[str, str], dict[str, str]]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        return {(r["brand"], r["product"]): r for r in csv.DictReader(f) if r.get("image_url")}


def product_image(brand: str, name: str) -> dict[str, str] | None:
    """{'image_url', 'page_url'} for a catalogue product, or None if no photo was found."""
    row = _table().get((brand, name))
    if not row:
        return None
    return {"image_url": row["image_url"], "page_url": row["page_url"]}
