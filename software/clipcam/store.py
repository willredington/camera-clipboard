"""Saving captures (PNG + small JPEG thumbnail), listing them, building PDFs."""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

NAME_RE = re.compile(r"^\d{8}-\d{6}(-\d+)?$")
THUMB_W = 240


class Store:
    def __init__(self, directory: Path, dpi: int, keep: int = 0):
        self.dir = Path(directory)
        self.thumbs = self.dir / ".thumbs"
        self.thumbs.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.keep = keep

    @staticmethod
    def _unique_name(directory: Path, ext: str) -> str:
        base = time.strftime("%Y%m%d-%H%M%S")
        name, n = base, 1
        while (directory / f"{name}{ext}").exists():
            name, n = f"{base}-{n}", n + 1
        return name

    def save_page(self, bgr: np.ndarray) -> str:
        name = self._unique_name(self.dir, ".png")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(self.dir / f"{name}.png", dpi=(self.dpi, self.dpi), compress_level=3)
        h = round(bgr.shape[0] * THUMB_W / bgr.shape[1])
        cv2.imwrite(str(self.thumbs / f"{name}.jpg"), cv2.resize(bgr, (THUMB_W, h), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 75])
        self._prune()
        return name

    def save_ocr(self, bgr: np.ndarray, text: str) -> str:
        """Auto-capture result: ocr/<name>.jpg (the frame sent to the model) and ocr/<name>.txt."""
        d = self.dir / "ocr"
        d.mkdir(exist_ok=True)
        name = self._unique_name(d, ".txt")
        cv2.imwrite(str(d / f"{name}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        (d / f"{name}.txt").write_text(text, encoding="utf-8")
        return name

    def save_failed(self, bgr: np.ndarray) -> Path:
        """Keep the raw frame when rectification fails, so marker detection can be debugged offline."""
        path = self.dir / f"failed-{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path

    def names(self) -> list[str]:
        """Newest first."""
        return sorted((p.stem for p in self.dir.glob("*.png") if NAME_RE.match(p.stem)), reverse=True)

    def png_path(self, name: str) -> Path | None:
        p = self.dir / f"{name}.png"
        return p if NAME_RE.match(name) and p.is_file() else None

    def thumb_path(self, name: str) -> Path | None:
        p = self.thumbs / f"{name}.jpg"
        return p if NAME_RE.match(name) and p.is_file() else None

    def pdf(self, names: list[str]) -> bytes:
        """One page per name, in the order given, each page the size of the paper at the capture dpi."""
        paths = [self.png_path(n) for n in names]
        if not paths or any(p is None for p in paths):
            raise KeyError("unknown capture name")
        images = [Image.open(p) for p in paths]  # opened lazily; Pillow decodes one page at a time
        buf = io.BytesIO()
        images[0].save(buf, "PDF", save_all=True, append_images=images[1:], resolution=float(self.dpi))
        for im in images:
            im.close()
        return buf.getvalue()

    def _prune(self):
        if self.keep > 0:
            for name in self.names()[self.keep:]:
                (self.dir / f"{name}.png").unlink(missing_ok=True)
                (self.thumbs / f"{name}.jpg").unlink(missing_ok=True)
