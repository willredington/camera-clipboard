"""Web app + service driven by the fake camera/LED (no picamera2, no GPIO)."""
import io
import threading
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from clipcam.camera import FakeCamera
from clipcam.config import Config, load_config
from clipcam.hardware import FakeLed
from clipcam.rectify import Rectifier
from clipcam.service import CaptureService
from clipcam.store import Store
from clipcam.web import create_app
from synth import render_scene


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_in")
    cv2.imwrite(str(d / "a_good.jpg"), render_scene(seed=1, pitch=3, yaw=-1)[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(d / "b_three.jpg"), render_scene(seed=2, pitch=3, yaw=-1, hide_markers=(1,))[0])
    cv2.imwrite(str(d / "c_blank.jpg"), np.full((1000, 1800, 3), 128, np.uint8))
    return d


@pytest.fixture
def env(images, tmp_path):
    cfg = Config()
    store = Store(tmp_path / "caps", cfg.output.dpi, keep=0)
    cam, led = FakeCamera(images, cfg.camera), FakeLed()
    svc = CaptureService(cam, led, Rectifier(cfg.board, cfg.output.dpi), store)
    return svc, store, led, create_app(svc, store).test_client()


def test_capture_gallery_png_pdf(env):
    svc, store, led, c = env
    r1 = c.post("/capture")
    assert r1.status_code == 200 and r1.json["method"] == "4 markers" and r1.json["warnings"] == []
    assert led.history == [True, False]  # on for the capture, off afterwards
    r2 = c.post("/capture")  # image with a marker hidden
    assert r2.json["method"] == "3 markers" and r2.json["warnings"]
    r3 = c.post("/capture")  # blank image: falls back to the cached homography
    assert r3.json["method"] == "cached" and r3.json["warnings"]

    names = c.get("/captures").json
    assert len(names) == 3 and names == sorted(names, reverse=True)
    png = c.get(f"/captures/{names[0]}.png")
    im = Image.open(io.BytesIO(png.data))
    assert im.size == (1700, 2200) and round(im.info["dpi"][0]) == 200
    assert c.get(f"/thumbs/{names[0]}.jpg").mimetype == "image/jpeg"

    pdf = c.get("/pdf?names=" + ",".join(names[::-1]))
    assert pdf.mimetype == "application/pdf" and pdf.data.startswith(b"%PDF")
    assert pdf.data.count(b"/Type /Page\n") + pdf.data.count(b"/Type /Page ") + pdf.data.count(b"/Type /Page/") >= 3


def test_preview_and_index(env):
    _, _, _, c = env
    assert c.get("/camera").status_code == 200 and b"Capture" in c.get("/camera").data
    p = c.get("/preview.jpg")
    assert p.mimetype == "image/jpeg" and cv2.imdecode(np.frombuffer(p.data, np.uint8), 1).shape[1] == 640


def test_failure_when_cold_saves_raw_and_reports(images, tmp_path):
    only_blank = tmp_path / "blank"
    only_blank.mkdir()
    cv2.imwrite(str(only_blank / "x.jpg"), np.full((1000, 1800, 3), 128, np.uint8))
    cfg = Config()
    store = Store(tmp_path / "caps", 200)
    svc = CaptureService(FakeCamera(only_blank), FakeLed(), Rectifier(cfg.board, 200), store)
    c = create_app(svc, store).test_client()
    r = c.post("/capture")
    assert r.status_code == 422 and "markers" in r.json["error"]
    assert list((tmp_path / "caps").glob("failed-*.jpg")) and c.get("/captures").json == []


def test_names_are_validated(env):
    _, _, _, c = env
    assert c.get("/captures/..%2Fsecret.png").status_code == 404
    assert c.get("/captures/20250101-000000.png").status_code == 404
    assert c.get("/pdf?names=../../etc/passwd").status_code == 404
    assert c.get("/pdf").status_code == 404


def test_busy_returns_409(images, tmp_path):
    cfg = Config()
    store = Store(tmp_path / "caps", 200)
    cam = FakeCamera(images)
    gate = threading.Event()
    orig = cam.capture_still
    cam.capture_still = lambda: (gate.wait(5), orig())[1]
    svc = CaptureService(cam, FakeLed(), Rectifier(cfg.board, 200), store)
    c = create_app(svc, store).test_client()
    t = threading.Thread(target=c.post, args=("/capture",))
    t.start()
    time.sleep(0.3)
    assert c.post("/capture").status_code == 409
    gate.set()
    t.join()


def test_prune_keeps_newest(tmp_path):
    s = Store(tmp_path, 200, keep=2)
    img = np.zeros((20, 10, 3), np.uint8)
    for _ in range(4):
        s.save_page(img)
    assert len(s.names()) == 2 and len(list((tmp_path / ".thumbs").glob("*.jpg"))) == 2


def test_config_toml(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[led]\npin = 12\n[button]\npin = 17\n[output]\ndpi = 150\n'
                 '[board.markers]\n0 = [10, 10]\n1 = [20, 10]\n2 = [20, 30]\n')
    cfg = load_config(f)
    assert (cfg.led.pin, cfg.button.pin, cfg.output.dpi) == (12, 17, 150)
    assert cfg.board.markers == {0: (10.0, 10.0), 1: (20.0, 10.0), 2: (20.0, 30.0)}
    f.write_text("[led]\npinn = 1\n")
    with pytest.raises(ValueError, match="led.pinn"):
        load_config(f)
    assert load_config(None).led.pin == 18 and load_config(None).button.pin is None


def test_example_config_matches_defaults():
    from pathlib import Path
    ex = load_config(Path(__file__).resolve().parent.parent / "config.example.toml")
    assert ex == Config()
