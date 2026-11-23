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


# ---- tidy product photo: remove the plain background, trim, centre on a square ----------------

PHOTO_SIZE = 320          # output square, px
PHOTO_FILL = 0.86         # product's longer side as a share of the square (even margins)
_LIGHT = 232              # background pixels: every channel at least this bright...
_CHROMA = 18              # ...and nearly grey (max - min channel)


def tidy_photo(data: bytes, size: int = PHOTO_SIZE, fill: float = PHOTO_FILL) -> bytes:
    """PNG of the product on a transparent square, trimmed and centred with even margins.

    The background is removed only when the photo has a plain light (white/grey) background:
    near-white pixels connected to the photo's border become transparent, but the fill stops at
    the product's outline (edges), so white bottles and labels are kept. Photos with any other
    background are trimmed and centred with their background. Raises ValueError for data that
    is not an image.
    """
    import io

    import cv2
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError("not an image") from e
    rgb = np.asarray(im)
    h, w = rgb.shape[:2]
    mx, mn = rgb.max(2).astype(int), rgb.min(2).astype(int)
    plain = (mn >= _LIGHT) & (mx - mn <= _CHROMA)
    sides = [plain[0], plain[-1], plain[:, 0], plain[:, -1]]
    alpha = np.full((h, w), 255, np.uint8)
    if np.concatenate(sides).mean() >= 0.6 or sum(s.mean() >= 0.9 for s in sides) >= 2:
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.dilate(cv2.Canny(grey, 20, 60), np.ones((3, 3), np.uint8))
        passable = (plain & (edges == 0)).astype(np.uint8)
        _, lab = cv2.connectedComponents(passable, connectivity=4)
        touching = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        bg = np.isin(lab, list(touching))
        # background-coloured outline pixels next to the background join it
        bg |= (cv2.dilate(bg.astype(np.uint8), np.ones((3, 3), np.uint8)) == 1) & plain
        n, lab2, stats, _ = cv2.connectedComponentsWithStats((~bg).astype(np.uint8), connectivity=8)
        if n > 1:  # drop specks (noise, faint shadows) far smaller than the product
            biggest = stats[1:, cv2.CC_STAT_AREA].max()
            for k in range(1, n):
                if stats[k, cv2.CC_STAT_AREA] < 0.02 * biggest:
                    bg[lab2 == k] = True
        alpha[bg] = 0
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)  # soft edge
    ys, xs = np.nonzero(alpha > 16)
    if len(xs) == 0:  # nothing left: keep the whole photo
        alpha[:] = 255
        ys, xs = np.nonzero(alpha)
    crop = np.dstack([rgb, alpha])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    product = Image.fromarray(crop, "RGBA")
    scale = fill * size / max(product.size)
    product = product.resize((max(1, round(product.width * scale)), max(1, round(product.height * scale))),
                             Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(product, ((size - product.width) // 2, (size - product.height) // 2))
    out = io.BytesIO()
    canvas.save(out, format="PNG", optimize=True)
    return out.getvalue()
