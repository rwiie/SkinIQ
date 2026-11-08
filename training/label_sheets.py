"""Contact sheets for AI-assisted annotation: numbered photos with their prefilled
(known) concerns, so an annotator can label several photos per view.

    python training/label_sheets.py --keys keys.txt --out sheets/ --per-sheet 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import DATA_DIR  # noqa: E402

TILE = 400
CAPTION = 34


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_sheets(queue: pd.DataFrame, keys: list[str], images: Path, out: Path,
                per_sheet: int = 6, cols: int = 3, start: int = 1) -> list[dict]:
    """Write sheet_<n>.png files; return one entry per tile (sheet, number, key)."""
    out.mkdir(parents=True, exist_ok=True)
    q = queue.set_index("key")
    font, small = _font(26), _font(15)
    index = []
    for s, i in enumerate(range(0, len(keys), per_sheet), start=start):
        chunk = keys[i:i + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * TILE, rows * (TILE + CAPTION)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, key in enumerate(chunk):
            item = q.loc[key]
            x, y = (j % cols) * TILE, (j // cols) * (TILE + CAPTION)
            img = Image.open(images / item["image"]).convert("RGB")
            img.thumbnail((TILE, TILE))
            sheet.paste(img, (x + (TILE - img.width) // 2, y + CAPTION + (TILE - img.height) // 2))
            prefill = item["prefill"] if isinstance(item["prefill"], str) and item["prefill"] else "-"
            draw.text((x + 6, y + 3), f"{j + 1}", fill="red", font=font)
            draw.text((x + 40, y + 9), f"{item['dataset']} | known: {prefill}", fill="black", font=small)
            index.append({"sheet": s, "number": j + 1, "key": key})
        sheet.save(out / f"sheet_{s:03d}.png")
    (out / "index.json").write_text(json.dumps(index, indent=1))
    return index


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--keys", type=Path, required=True, help="text file, one queue key per line")
    p.add_argument("--queue", type=Path, default=DATA_DIR / "labels" / "annotation_queue.csv")
    p.add_argument("--images", type=Path, default=DATA_DIR / "processed")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-sheet", type=int, default=6)
    args = p.parse_args()
    keys = [k.strip() for k in args.keys.read_text().splitlines() if k.strip()]
    index = make_sheets(pd.read_csv(args.queue), keys, args.images, args.out, args.per_sheet)
    print(f"{len(index)} photos on {index[-1]['sheet']} sheets -> {args.out}")


if __name__ == "__main__":
    main()
