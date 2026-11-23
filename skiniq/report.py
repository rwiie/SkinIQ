"""PDF version of the results (download from the results page).

`build_pdf(result, photo)` turns the dict returned by `skiniq.pipeline.analyse` into a
PDF with the same sections as the results page. Uses ReportLab's built-in fonts, so
text is reduced to characters those fonts can show (e.g. 'α' -> 'alpha').
"""

from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#b5485a")
MUTED = colors.HexColor("#6b6b6b")
BORDER = colors.HexColor("#e3d6d3")
_REPLACE = {"α": "alpha", "β": "beta", "γ": "gamma", "Ⅲ": "III", "Ⅱ": "II", "Ⅰ": "I", "≥": ">=", "≤": "<=",
            "→": "->", "’": "'", "‘": "'", "“": '"', "”": '"'}


def _clean(text) -> str:
    """Escape for ReportLab paragraphs and keep only characters the built-in fonts can show."""
    s = "".join(_REPLACE.get(ch, ch) for ch in str(text))
    s = s.encode("cp1252", errors="replace").decode("cp1252")
    return escape(s)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13)
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=20,
                                textColor=ACCENT, alignment=0, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13.5,
                             textColor=ACCENT, spaceBefore=10, spaceAfter=4),
        "h3": ParagraphStyle("h3", parent=base["Heading3"], fontName="Helvetica-Bold", fontSize=10.5,
                             spaceBefore=6, spaceAfter=2),
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=body, fontSize=9, leading=12),
        "big": ParagraphStyle("big", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=18),
    }


def _bullets(items, style) -> ListFlowable:
    return ListFlowable([ListItem(Paragraph(_clean(t), style), leftIndent=10) for t in items],
                        bulletType="bullet", start="•", leftIndent=10, bulletFontSize=8)


def _boxed(rows, widths, header=False) -> Table:
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("BOX", (0, 0), (-1, -1), 0.6, BORDER), ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f6ecea")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


PRODUCT_PHOTO_MM = 20


def _product_photo(data: bytes | None, st: dict):
    """A product photo scaled into a square cell, or a short 'no photo' note."""
    if data:
        try:
            img = Image(io.BytesIO(data), mask="auto")  # keeps the transparent background
            scale = PRODUCT_PHOTO_MM * mm / max(img.imageWidth, img.imageHeight)
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            return img
        except Exception:  # not a readable image
            pass
    return Paragraph("No photo", st["small"])


def _routine_part(title: str, steps: list[dict], st: dict, width: float, weekly: bool = False,
                  lead: list | None = None, photos: dict[str, bytes] | None = None) -> list:
    if not steps:
        return list(lead or [])
    out = [*(lead or []), Paragraph(_clean(title), st["h3"])]
    photos = photos or {}
    rows = [[Paragraph("<b>Step</b>", st["cell"]), Paragraph("<b>Photo</b>", st["cell"]),
             Paragraph("<b>Product and why</b>", st["cell"])]]
    for step in steps:
        p = step["product"]
        parts = [f"<b>{_clean(p['product'])}</b>", _clean(p["summary"])]
        if p.get("key_ingredients"):
            parts.append("<i>Key ingredients:</i> " + _clean("; ".join(p["key_ingredients"])))
        for caution in p.get("cautions", []):
            parts.append(f"<font color='#a0522d'>Caution: {_clean(caution)}</font>")
        if weekly and step.get("how_often"):
            parts.append(_clean(f"Use {step['how_often']}."))
        if p.get("image_source"):
            parts.append(f"<font color='#6b6b6b' size='7.5'>Photo: INKEEDecoder</font>")
        rows.append([Paragraph(_clean(step["label"]), st["cell"]), _product_photo(photos.get(p["product"]), st),
                     Paragraph("<br/>".join(parts), st["cell"])])
    photo_col = (PRODUCT_PHOTO_MM + 4) * mm
    table = _boxed(rows, [28 * mm, photo_col, width - 28 * mm - photo_col], header=True)
    table.repeatRows = 1
    out.append(table)
    return [KeepTogether(out)]  # heading and table start on the same page


def build_pdf(result: dict, photo: bytes | None = None, generated: datetime | None = None,
              product_photos: dict[str, bytes] | None = None) -> bytes:
    """Render the analysis result as a PDF and return its bytes.

    `product_photos` maps a product title to its photo bytes (from INKEEDecoder); products
    without one get a "No photo" cell.
    """
    st = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="SkinIQ results", author="SkinIQ")
    width = doc.width
    generated = generated or datetime.now()
    sa = result["skin_analysis"]
    story: list = [
        Paragraph("SkinIQ: your skincare results", st["title"]),
        Paragraph(_clean(f"Personalised Korean skincare advice · generated {generated:%d %B %Y, %H:%M}"),
                  st["small"]),
        Spacer(1, 6),
    ]
    for note in result.get("notes", []):
        story.append(Paragraph(_clean(note), st["small"]))

    # Skin analysis
    story.append(Paragraph("Skin analysis", st["h2"]))
    text = [Paragraph(_clean(f"Skin type: {sa['skin_type'].title()}"), st["big"]), Spacer(1, 3),
            Paragraph(_clean(sa["skin_type_text"]), st["body"])]
    if photo:
        try:
            img = Image(io.BytesIO(photo))
            scale = 38 * mm / max(img.imageWidth, img.imageHeight)
            img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
            row = Table([[img, text]], colWidths=[42 * mm, width - 42 * mm], hAlign="LEFT")
            row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            story.append(row)
        except Exception:  # unreadable photo: report without it
            story += text
    else:
        story += text

    story.append(Paragraph("Skin concerns", st["h3"]))
    if sa["concerns"]:
        story.append(_bullets([c["text"] for c in sa["concerns"]], st["body"]))
    else:
        story.append(Paragraph(_clean(sa["no_concerns_text"]), st["body"]))

    g = sa.get("age_guidance")
    if g:
        story.append(Paragraph(_clean(f"Skin care for your age ({g['label'].replace('Ages ', '')})"), st["h3"]))
        story.append(_bullets([f"Common concerns: {g['common_concerns']}", f"Skin changes: {g['skin_changes']}",
                               f"Care focus: {g['care_focus']}"], st["body"]))

    # Location and weather
    env = result["usce"].get("environment")
    story.append(Paragraph("Your location and today's weather", st["h2"]))
    if env:
        date = (sa.get("location") or {}).get("date_label") or ""
        story.append(Paragraph(_clean(env["location"] + (f" · {date}" if date else "")), st["body"]))
        cells = [("Avg temperature", f"{env['temperature_c']:g} °C"), ("Avg humidity", f"{env['humidity_pct']:g}%")]
        if env.get("uv_index") is not None:
            cells.append(("Avg UV index", f"{env['uv_index']:g}"))
        if env.get("uv_max") is not None:
            cells.append(("Peak UV index", f"{env['uv_max']:g}"))
        w = width / len(cells)
        story.append(Spacer(1, 4))
        story.append(_boxed([[Paragraph(_clean(k), st["small"]) for k, _ in cells],
                             [Paragraph(f"<b>{_clean(v)}</b>", st["body"]) for _, v in cells]], [w] * len(cells)))
        if env.get("source"):
            story.append(Paragraph(_clean(f"Source: {env['source']}."), st["small"]))
    else:
        story.append(Paragraph(_clean(sa["environment_text"]), st["body"]))

    # Advice
    story.append(Paragraph("Advice for you", st["h2"]))
    story.append(_bullets(result["advice"], st["body"]))

    # Routine
    routine = result["routine"]
    story += _routine_part("Morning", routine["morning"], st, width, photos=product_photos,
                           lead=[Paragraph("Your routine", st["h2"])])
    story += _routine_part("Night", routine["evening"], st, width, photos=product_photos)
    story += _routine_part("Weekly", routine["weekly"], st, width, weekly=True, photos=product_photos)
    if routine["conflicts"]:
        story.append(Paragraph("Ingredient conflicts", st["h3"]))
        story.append(_bullets([f"{c['reason']}: {c['advice']} ({' + '.join(c['products'])})."
                               for c in routine["conflicts"]], st["body"]))

    story += [Spacer(1, 10), KeepTogether([Paragraph(_clean(result["disclaimer"]), st["small"])])]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(doc_.leftMargin, 9 * mm, "SkinIQ · general skincare guidance, not a medical diagnosis")
        canvas.drawRightString(A4[0] - doc_.rightMargin, 9 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
