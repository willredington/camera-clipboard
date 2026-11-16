import re

import cv2
import numpy as np
import pytest

from clipcam.__main__ import main
from clipcam.pageid import parse_payload, payload, read_page_id, render_sheet, write_sheets_pdf

PX_PER_MM = 300 / 25.4


def camera_like(img, width=1080, seed=0):
    """Downscale a 300 dpi sheet to ~0.2 mm/px like the boom camera, blur and add sensor noise."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    rng = np.random.default_rng(seed)
    return np.clip(small + rng.normal(0, 3, small.shape), 0, 255).astype(np.uint8)


def test_payload_roundtrip():
    assert parse_payload(payload(17)) == 17


@pytest.mark.parametrize("s", ["17", "CCP:", "CCP:0", "CCP:-3", "CCP:x", "CCP:1.5", "https://x.y", ""])
def test_parse_rejects_foreign(s):
    assert parse_payload(s) is None


def test_payload_rejects_zero():
    with pytest.raises(ValueError):
        payload(0)


def test_render_sheet_decodes():
    img = render_sheet(42)
    assert img.shape == (3300, 2550, 3) and read_page_id(img) == 42


def test_decodes_at_camera_scale():
    assert read_page_id(camera_like(render_sheet(7))) == 7


def test_rotated_frame_decodes():
    assert read_page_id(cv2.rotate(camera_like(render_sheet(7)), cv2.ROTATE_90_CLOCKWISE)) == 7


def test_foreign_qr_ignored():
    qr = cv2.QRCodeEncoder.create().encode("hello")
    img = cv2.cvtColor(cv2.copyMakeBorder(cv2.resize(qr, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST),
                                          60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)
    assert read_page_id(img) is None


def test_no_qr_is_none():
    assert read_page_id(np.full((500, 500, 3), 255, np.uint8)) is None


def test_clip_area_left_clear():
    img = render_sheet(3)
    w = img.shape[1]
    band = img[:round(25 * PX_PER_MM), w // 2 - round(50 * PX_PER_MM): w // 2 + round(50 * PX_PER_MM)]
    assert band.min() == 255


def test_sheets_pdf(tmp_path):
    f = tmp_path / "s.pdf"
    write_sheets_pdf(5, 3, f)
    data = f.read_bytes()
    assert data.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page(?!s)", data)) == 3


def test_sheets_cli(tmp_path, capsys):
    f = tmp_path / "s.pdf"
    main(["sheets", "--start", "5", "--count", "2", str(f)])
    assert f.is_file() and "5..6" in capsys.readouterr().out


@pytest.mark.parametrize("seed", range(4))
def test_decodes_with_margin_below_camera_scale(seed):
    """The real page is ~1000 px wide in the frame; keep ~25% margin."""
    assert read_page_id(camera_like(render_sheet(123), width=750, seed=seed)) == 123
