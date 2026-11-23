"""Product photo lookup (INKEEDecoder links) and the name matching used to build it. No network."""

import sys
from pathlib import Path

import pytest

import skiniq.product_images as pi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from fetch_product_images import core, page_image, similarity, slugify  # noqa: E402


@pytest.mark.parametrize("found, wanted", [
    ("COSRX Low pH Good Morning Gel Cleanser", "COSRX Low pH Good Morning Gel Cleanser"),
    ("Hera Sun Mate Airy Sun Milk SPF50+/PA++++", "Hera Sun Mate Airy Sun Milk SPF 50+ / PA++++"),   # rating spelling
    ("innisfree Pore Clearing Facial Foam With Volcanic Clusters",
     "innisfree Volcanic Clusters Pore Clearing Facial Foam"),                                       # word order
    ("Celimax Derma Nature Fresh Blackhead Jojoba Cleansing Oil",
     "Celimax Derma Nature Fresh Blackhead Cleansing Oil"),                                          # one extra word
    ("A'pieu Deep Clean Foam Cleanser", "A'Pieu Deep Clean Foam Cleanser"),
])
def test_same_product_matches(found, wanted):
    assert similarity(found, wanted) >= 0.75


@pytest.mark.parametrize("found, wanted", [
    ("LANEIGE Moisturizing Foam Cleanser", "Laneige Moisturizing Cream Cleanser"),    # different product type
    ("innisfree Volcanic Clusters Pore Clearing Toner 2X", "innisfree Volcanic Clusters Pore Clearing Facial Foam"),
    ("COSRX Advanced Snail 96 Mucin Power Essence", "COSRX Advanced Snail 92 All In One Cream"),
])
def test_different_products_do_not_match(found, wanted):
    assert similarity(found, wanted) == 0.0


def test_names_ignore_sun_ratings_and_punctuation():
    assert core("Skinfood Sunflower Airy Sun Stick SPF 50+ PA++++") == "skinfood sunflower airy sun stick"
    assert slugify("A'Pieu Deep Clean Foam Cleanser") == "apieu-deep-clean-foam-cleanser"


def test_page_image_prefers_the_300px_photo():
    base = "https://incidecoder-content.storage.googleapis.com/x/products/p/p_front_photo"
    page = f'<img src="{base}_original.jpeg"> <source srcset="{base}_300x300@2x.jpeg 2x">'
    assert page_image(page) == f"{base}_300x300@2x.jpeg"
    assert page_image(f'<img src="{base}.png">') == f"{base}.png"
    assert page_image("<html>no photo</html>") == ""


def test_lookup_reads_the_table(tmp_path):
    csv = tmp_path / "images.csv"
    csv.write_text("brand,product,page_url,image_url,match\n"
                   "COSRX,Gel Cleanser,https://inkeedecoder.com/products/a,https://img/a.jpeg,address\n"
                   "COSRX,Missing,,,none\n", encoding="utf-8")
    rows = pi._table(str(csv))  # rows without a photo are left out
    assert rows[("COSRX", "Gel Cleanser")]["image_url"] == "https://img/a.jpeg"
    assert ("COSRX", "Missing") not in rows


def test_unknown_product_has_no_photo():
    assert pi.product_image("No Such Brand", "No Such Product") is None


# ---- tidy_photo: background removal, trim and centring ----------------------------------------

def _png(img) -> bytes:
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _tidy(img, **kw):
    import io

    import numpy as np
    from PIL import Image
    out = Image.open(io.BytesIO(pi.tidy_photo(_png(img), **kw)))
    return out, np.asarray(out)


def _bottle(body=(250, 250, 250), size=(200, 400)):
    """White photo with a product drawn in the middle: grey outline, `body` colour inside."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((60, 80, 140, 360), fill=body, outline=(150, 150, 150), width=3)
    return img


def test_white_background_becomes_transparent_and_product_is_kept():
    out, a = _tidy(_bottle(body=(40, 120, 200)))
    assert out.mode == "RGBA" and out.size == (pi.PHOTO_SIZE, pi.PHOTO_SIZE)
    assert a[0, 0, 3] == 0 and a[-1, -1, 3] == 0          # corners: background removed
    assert a[160, 160, 3] == 255                           # centre: product


def test_white_product_on_white_background_is_not_erased():
    _, a = _tidy(_bottle(body=(252, 252, 252)))
    assert a[160, 160, 3] == 255  # the outline stops the background fill


def test_product_is_trimmed_and_centred_with_even_margins():
    _, a = _tidy(_bottle(body=(40, 120, 200)))
    import numpy as np
    ys, xs = np.nonzero(a[:, :, 3] > 16)
    height = ys.max() - ys.min() + 1
    assert abs(height - round(pi.PHOTO_FILL * pi.PHOTO_SIZE)) <= 3          # longer side fills 86%
    assert abs(ys.min() - (pi.PHOTO_SIZE - 1 - ys.max())) <= 2               # centred vertically
    assert abs(xs.min() - (pi.PHOTO_SIZE - 1 - xs.max())) <= 2               # centred horizontally


def test_photo_with_a_scene_background_keeps_it():
    from PIL import Image
    import numpy as np
    rng = np.random.default_rng(0)
    scene = Image.fromarray(rng.integers(0, 200, (300, 300, 3), dtype=np.uint8))
    _, a = _tidy(scene)
    assert (a[:, :, 3] > 0).sum() > 0.7 * a.shape[0] * a.shape[1]  # nothing cut out


def test_not_an_image_raises():
    with pytest.raises(ValueError):
        pi.tidy_photo(b"<html>not an image</html>")


def test_routine_photos_skip_products_without_a_usable_photo(monkeypatch):
    import app.services as services

    services.product_photo.cache_clear()

    class Resp:
        def __init__(self, status, ctype, content):
            self.status_code, self.headers, self.content = status, {"content-type": ctype}, content

    responses = {"https://img/ok.png": Resp(200, "image/png", _png(_bottle(body=(40, 120, 200)))),
                 "https://img/404.png": Resp(404, "text/html", b""),
                 "https://img/html.png": Resp(200, "text/html", b"<html>")}
    monkeypatch.setattr(services.requests, "get", lambda url, timeout: responses[url])
    step = lambda title, url: {"product": {"product": title, "image_url": url}}
    result = {"routine": {"morning": [step("A", "https://img/ok.png"), step("B", "https://img/404.png")],
                          "evening": [step("C", "https://img/html.png"), step("D", None)], "weekly": []}}
    photos = services.routine_photos(result)
    assert list(photos) == ["A"] and photos["A"].startswith(b"\x89PNG")
    services.product_photo.cache_clear()
