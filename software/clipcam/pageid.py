"""Numbered sheets: QR page ids (payload, detection) and a printable PDF of blank numbered pages.

Pure OpenCV/numpy (Pillow only to save the PDF), no camera imports.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PREFIX = "CCP:"
_PAYLOAD_RE = re.compile(r"^CCP:([1-9]\d*)$")

DPI = 300
_MM = DPI / 25.4
SHEET_W, SHEET_H = 2550, 3300          # US Letter at 300 dpi
QR_PX = round(35 * _MM)                # 35 mm square: >= ~5 px per module at the camera's ~0.2 mm/px
MARGIN_PX = round(12 * _MM)            # from the top and side edges
CLIP_HALF_PX = round(50 * _MM)         # top-centre 100 mm left clear for the clip
_detector = cv2.QRCodeDetector()


def payload(n: int) -> str:
    if n < 1:
        raise ValueError(f"page number must be >= 1, got {n}")
    return f"{PREFIX}{n}"


def parse_payload(s: str) -> int | None:
    m = _PAYLOAD_RE.match(s or "")
    return int(m.group(1)) if m else None


def read_page_id(bgr: np.ndarray) -> int | None:
    """First QR in the image whose payload is a clipcam page, else None. Never raises."""
    try:
        ok, infos, _, _ = _detector.detectAndDecodeMulti(bgr)
        texts = list(infos) if ok else []
        if not texts:
            text, _, _ = _detector.detectAndDecode(bgr)
            texts = [text]
    except cv2.error as e:
        log.debug("QR detection failed: %s", e)
        return None
    for t in texts:
        n = parse_payload(t)
        if n is not None:
            return n
    return None


def _qr_image(n: int) -> np.ndarray:
    modules = cv2.QRCodeEncoder.create().encode(payload(n))  # no quiet zone included
    scale = QR_PX // modules.shape[0]
    qr = cv2.resize(modules, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    pad = QR_PX - qr.shape[0]
    return cv2.copyMakeBorder(qr, pad // 2, pad - pad // 2, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=255)


def render_sheet(n: int) -> np.ndarray:
    """A blank Letter page: 'Page n' top-left, the QR top-right, the top-centre clip area left white."""
    img = np.full((SHEET_H, SHEET_W, 3), 255, np.uint8)
    qr = _qr_image(n)
    x0 = SHEET_W - MARGIN_PX - QR_PX
    img[MARGIN_PX:MARGIN_PX + QR_PX, x0:x0 + QR_PX] = qr[:, :, None]

    text, font, thick = f"Page {n}", cv2.FONT_HERSHEY_SIMPLEX, 8
    max_w = SHEET_W // 2 - CLIP_HALF_PX - 2 * MARGIN_PX   # stay left of the clip area
    scale = 4.0
    while cv2.getTextSize(text, font, scale, thick)[0][0] > max_w and scale > 1:
        scale -= 0.25
    (_, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (MARGIN_PX, MARGIN_PX + th), font, scale, (0, 0, 0), thick, cv2.LINE_AA)
    return img


def write_sheets_pdf(start: int, count: int, path: str | Path) -> None:
    pages = [Image.fromarray(cv2.cvtColor(render_sheet(n), cv2.COLOR_BGR2RGB)) for n in range(start, start + count)]
    pages[0].save(str(path), "PDF", save_all=True, append_images=pages[1:], resolution=float(DPI))
