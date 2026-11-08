"""Multi-label annotation: label storage, annotation session and the Colab widget UI.

Used by training/annotate_colab.ipynb. Every answer is written to the labels CSV
immediately (atomic replace), so closing Colab never loses work; reopening resumes
at the first unlabelled photo in the queue.
"""

from __future__ import annotations

import io
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skiniq.config import SKIN_CONDITIONS  # noqa: E402

LABEL_COLUMNS = [
    "key", "dataset", "split", "image", "md5", *SKIN_CONDITIONS,
    "unsure", "unusable", "annotator", "labelled_at",
]
CONDITION_TITLES = {
    "acne": "Acne", "blackheads": "Blackheads", "hyperpigmentation": "Hyperpigmentation",
    "redness": "Redness", "enlarged_pores": "Enlarged pores", "wrinkles": "Wrinkles",
}


class LabelStore:
    """Labels CSV keyed by photo-group key; saved after every change."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.rows: dict[str, dict] = {}
        if self.path.exists() and self.path.stat().st_size > 0:
            for row in pd.read_csv(self.path, dtype={"annotator": str}).fillna("").to_dict("records"):
                self.rows[row["key"]] = row

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def save(self, row: dict) -> None:
        self.rows[row["key"]] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        pd.DataFrame(list(self.rows.values()), columns=LABEL_COLUMNS).to_csv(tmp, index=False)
        os.replace(tmp, self.path)


class AnnotationSession:
    def __init__(self, queue: pd.DataFrame, store: LabelStore, annotator: str = "") -> None:
        self.queue = queue.sort_values("order").reset_index(drop=True)
        self.store = store
        self.annotator = annotator
        self.position = self._next_unlabelled(0)

    def _next_unlabelled(self, start: int) -> int:
        for i in range(start, len(self.queue)):
            if self.queue.at[i, "key"] not in self.store:
                return i
        return len(self.queue)

    @property
    def finished(self) -> bool:
        return self.position >= len(self.queue)

    def current(self) -> pd.Series:
        return self.queue.iloc[self.position]

    def initial_labels(self) -> tuple[set[str], bool]:
        """Saved labels when revisiting a photo, otherwise the folder prefill."""
        item = self.current()
        saved = self.store.get(item["key"])
        if saved is not None:
            return {c for c in SKIN_CONDITIONS if int(saved[c]) == 1}, bool(int(saved["unsure"]))
        prefill = item["prefill"] if isinstance(item["prefill"], str) else ""
        return {c for c in prefill.split("|") if c}, False

    def submit(self, labels: set[str], unsure: bool = False, unusable: bool = False) -> None:
        unknown = set(labels) - set(SKIN_CONDITIONS)
        if unknown:
            raise ValueError(f"unknown concerns: {sorted(unknown)}")
        item = self.current()
        self.store.save({
            "key": item["key"], "dataset": item["dataset"], "split": item["split"],
            "image": item["image"], "md5": item["md5"],
            **{c: int(c in labels and not unusable) for c in SKIN_CONDITIONS},
            "unsure": int(unsure), "unusable": int(unusable),
            "annotator": self.annotator,
            "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        self.position = self._next_unlabelled(self.position + 1)

    def back(self) -> None:
        self.position = max(0, min(self.position, len(self.queue)) - 1)

    def progress(self) -> pd.DataFrame:
        q = self.queue.assign(done=self.queue["key"].isin(self.store.rows))
        return q.groupby("tier", sort=False)["done"].agg(done="sum", total="size")


def image_bytes(path: str | Path, max_side: int = 560) -> bytes:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def build_ui(session: AnnotationSession, image_root: str | Path):
    """ipywidgets interface: image, six concern toggles, Unsure, Back / Unusable / Save & next."""
    import ipywidgets as w

    image_root = Path(image_root)
    picture = w.Image(format="jpeg", layout=w.Layout(max_width="560px"))
    info = w.HTML()
    progress = w.HTML()
    toggles = {
        c: w.ToggleButton(description=CONDITION_TITLES[c], layout=w.Layout(width="150px"))
        for c in SKIN_CONDITIONS
    }
    unsure = w.Checkbox(description="Unsure (hard to judge)")
    back_btn = w.Button(description="◀ Back")
    unusable_btn = w.Button(description="Unusable", button_style="warning")
    save_btn = w.Button(description="Save & next ▶", button_style="success")

    def render() -> None:
        done = session.progress()
        progress.value = " · ".join(f"<b>{t}</b> {int(r.done)}/{int(r.total)}" for t, r in done.iterrows())
        if session.finished:
            info.value = "<h3>All photos labelled 🎉</h3>"
            picture.value = b""
            return
        item = session.current()
        labels, was_unsure = session.initial_labels()
        for c, t in toggles.items():
            t.value = c in labels
            t.button_style = "info" if t.value else ""
        unsure.value = was_unsure
        folder = item["folder_labels"].replace("|", ", ")
        info.value = (f"#{item['order']} · {item['tier']} · {item['dataset']} · folder: <i>{folder}</i>"
                      f"{' · <b>already labelled</b>' if item['key'] in session.store else ''}")
        picture.value = image_bytes(image_root / item["image"])

    def on_toggle(change) -> None:
        change["owner"].button_style = "info" if change["new"] else ""

    for t in toggles.values():
        t.observe(on_toggle, names="value")

    def chosen() -> set[str]:
        return {c for c, t in toggles.items() if t.value}

    save_btn.on_click(lambda _: (session.submit(chosen(), unsure.value), render()))
    unusable_btn.on_click(lambda _: (session.submit(set(), unsure.value, unusable=True), render()))
    back_btn.on_click(lambda _: (session.back(), render()))

    render()
    ui = w.VBox([
        progress, info, picture,
        w.HBox(list(toggles.values())[:3]), w.HBox(list(toggles.values())[3:]),
        unsure, w.HBox([back_btn, unusable_btn, save_btn]),
    ])
    ui.toggles, ui.unsure, ui.save, ui.unusable, ui.back, ui.info = (
        toggles, unsure, save_btn, unusable_btn, back_btn, info)
    return ui
