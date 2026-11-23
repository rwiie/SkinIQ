"""Model download on first use (mocked HTTP)."""

import hashlib

import pytest
import requests

from skiniq import pipeline
from skiniq.vision import weights
from skiniq.vision.weights import ModelUnavailable, ensure_model

PAYLOAD = b"fake model bytes" * 1000
GOOD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Stream:
    def __init__(self, status=200, body=PAYLOAD, ctype="application/octet-stream"):
        self.status_code, self.body, self.headers = status, body, {"Content-Type": ctype}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]


def test_existing_file_is_not_downloaded(tmp_path, monkeypatch):
    p = tmp_path / "model.pt"
    p.write_bytes(b"x")
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("should not download"))
    assert ensure_model(p) == p


def test_download_and_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: Stream())
    p = ensure_model(tmp_path / "m" / "model.pt", url="https://x", expected_sha256=GOOD_SHA)
    assert p.read_bytes() == PAYLOAD and not list(p.parent.glob("*.part"))


@pytest.mark.parametrize("stream, message", [
    (Stream(body=b"corrupt"), "checksum"),
    (Stream(status=404), "HTTP 404"),
    (Stream(ctype="text/html; charset=utf-8"), "web page"),
])
def test_bad_downloads_leave_no_file(tmp_path, monkeypatch, stream, message):
    monkeypatch.setattr(requests, "get", lambda *a, **k: stream)
    p = tmp_path / "model.pt"
    with pytest.raises(ModelUnavailable, match=message):
        ensure_model(p, url="https://x", expected_sha256=GOOD_SHA)
    assert not p.exists() and not list(tmp_path.glob("*.part"))


def test_offline_download_explains(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError()
    monkeypatch.setattr(requests, "get", boom)
    with pytest.raises(ModelUnavailable, match="internet"):
        ensure_model(tmp_path / "model.pt", url="https://x", expected_sha256=GOOD_SHA)


def test_pipeline_note_gives_the_reason(answers, monkeypatch, tmp_path):
    from PIL import Image

    monkeypatch.setattr(weights, "SKIN_CONCERN_MODEL", tmp_path / "missing.pt")
    monkeypatch.setattr(pipeline, "SKIN_CONCERN_MODEL", tmp_path / "missing.pt")
    monkeypatch.setattr(pipeline, "_CLASSIFIERS", {})

    def offline(path, *a, **k):
        raise ModelUnavailable("model download failed (ConnectionError); check your internet connection")
    monkeypatch.setattr(weights, "ensure_model", offline)
    assert pipeline.load_classifier(tmp_path / "missing.pt") is None
    r = pipeline.analyse(answers(), Image.new("RGB", (300, 300)))
    assert any("check your internet connection" in n for n in r["notes"])
