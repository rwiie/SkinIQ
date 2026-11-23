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
