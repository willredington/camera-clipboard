import numpy as np
import pytest

from clipcam.camera import FakeCamera
from clipcam.config import Config
from clipcam.documents import DocumentStore
from clipcam.hardware import FakeLed
from clipcam.rectify import Rectifier
from clipcam.service import CaptureService
from clipcam.store import Store
from clipcam.web import create_app

IMG = np.full((20, 30, 3), 180, np.uint8)


class StubWatcher:
    def __init__(self, **st):
        self.st = {"live": None, "qr_visible": None, "pending": False, "last": "starting", **st}

    def status(self):
        return dict(self.st)


@pytest.fixture
def env(tmp_path):
    import cv2
    (tmp_path / "in").mkdir()
    cv2.imwrite(str(tmp_path / "in" / "a.png"), np.full((100, 200, 3), 128, np.uint8))
    cfg = Config()
    store = Store(tmp_path / "caps", cfg.output.dpi)
    svc = CaptureService(FakeCamera(tmp_path / "in", cfg.camera), FakeLed(), Rectifier(cfg.board), store)
    docs = DocumentStore(tmp_path / "pages")
    watcher = StubWatcher()
    app = create_app(svc, store, documents=docs, watcher=watcher)
    return app.test_client(), docs, watcher, (svc, store)


def test_index_is_documents_and_camera_moved(env):
    c, *_ = env
    r = c.get("/")
    assert r.status_code == 200 and b"Documents" in r.data and b"Archive" in r.data
    r = c.get("/camera")
    assert r.status_code == 200 and b"Capture" in r.data


def test_api_live_empty(env):
    c, *_ = env
    j = c.get("/api/live").get_json()
    assert j["page"] is None and j["status"]["live"] is None


def test_api_live_and_archive(env):
    c, docs, watcher, _ = env
    docs.add_revision(3, IMG, "old page")
    docs.set_live(4)
    docs.add_revision(4, IMG, "live page")
    watcher.st.update(live=4, qr_visible=True)
    live = c.get("/api/live").get_json()
    assert live["page"]["number"] == 4 and live["page"]["text"] == "live page" and live["status"]["qr_visible"] is True
    arch = c.get("/api/pages").get_json()
    assert [p["number"] for p in arch] == [3] and arch[0]["first_line"] == "old page"


def test_api_page_detail_and_404(env):
    c, docs, *_ = env
    docs.add_revision(3, IMG, "a")
    j = c.get("/api/pages/3").get_json()
    assert j["number"] == 3 and len(j["revisions"]) == 1
    assert c.get("/api/pages/99").status_code == 404
    assert c.get("/pages/abc").status_code == 404
    assert c.get("/pages/99").status_code == 404


def test_txt_download(env):
    c, docs, *_ = env
    docs.add_revision(3, IMG, "line one\nlíne two")
    r = c.get("/pages/3.txt")
    assert r.status_code == 200 and r.data.decode("utf-8") == "line one\nlíne two"
    assert "page-3.txt" in r.headers["Content-Disposition"] and r.mimetype == "text/plain"
    assert c.get("/pages/99.txt").status_code == 404


def test_image_route_and_traversal(env):
    c, docs, *_ = env
    docs.add_revision(3, IMG, "a")
    name = docs.get(3)["revisions"][0]["image"]
    r = c.get(f"/pages/3/img/{name}")
    assert r.status_code == 200 and r.mimetype == "image/jpeg"
    for bad in ("..%2Flive.json", "nope.jpg", "page.json"):
        assert c.get(f"/pages/3/img/{bad}").status_code == 404


def test_corrupt_page_does_not_break_portal(env, tmp_path):
    c, docs, *_ = env
    docs.add_revision(3, IMG, "a")
    (tmp_path / "pages" / "5").mkdir()
    (tmp_path / "pages" / "5" / "page.json").write_text("{")
    r = c.get("/api/pages")
    assert r.status_code == 200 and [p["number"] for p in r.get_json()] == [3]
    assert c.get("/api/pages/5").status_code == 404


def test_archive_list_has_no_revisions(env):
    c, docs, watcher, _ = env
    for i in range(300):
        docs.add_revision(3, IMG, f"r{i}")
    docs.set_live(4)
    docs.add_revision(4, IMG, "x")
    for i in range(5):
        docs.add_revision(4, IMG, f"y{i}")
    assert all("revisions" not in p for p in c.get("/api/pages").get_json())
    assert "revisions" not in c.get("/api/live").get_json()["page"]


def test_page_view_html(env):
    c, docs, *_ = env
    docs.add_revision(3, IMG, "a")
    r = c.get("/pages/3")
    assert r.status_code == 200 and b"/api/pages/3" in r.data


def test_no_documents_store(tmp_path):
    import cv2
    (tmp_path / "in").mkdir()
    cv2.imwrite(str(tmp_path / "in" / "a.png"), np.full((100, 200, 3), 128, np.uint8))
    cfg = Config()
    store = Store(tmp_path / "caps", cfg.output.dpi)
    svc = CaptureService(FakeCamera(tmp_path / "in", cfg.camera), FakeLed(), Rectifier(cfg.board), store)
    c = create_app(svc, store).test_client()
    r = c.get("/")
    assert r.status_code == 200 and b"/camera" in r.data
    assert c.get("/api/live").get_json() == {"status": None, "page": None}
    assert c.get("/api/pages").get_json() == []


def test_page_text_is_not_injected_as_html(env):
    """Page text only reaches the browser through JSON + textContent, never server-side into the HTML."""
    c, docs, *_ = env
    docs.add_revision(3, IMG, "<script>alert(1)</script>")
    assert b"<script>alert(1)</script>" not in c.get("/").data
    assert b"<script>alert(1)</script>" not in c.get("/pages/3").data


def test_index_has_status_bar(env):
    c, *_ = env
    html = c.get("/").data
    assert b'id="status"' in html and b"calls_last_hour" in html
