"""Camera layer: Picamera2 on the Pi, FakeCamera (images from a folder) elsewhere.

Both give: capture_still() -> full-res BGR uint8 array, preview_jpeg() -> bytes.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .config import CameraCfg

log = logging.getLogger(__name__)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _jpeg(bgr: np.ndarray, quality: int = 70) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


class PiCamera:
    """Preview runs at low res; a capture switches to the full-res still mode, autofocuses, grabs one frame
    and switches back. The LED is switched by the caller around capture_still()."""

    def __init__(self, cfg: CameraCfg):
        from libcamera import controls
        from picamera2 import Picamera2

        self.cfg = cfg
        self._controls = controls
        self._lock = threading.Lock()
        self._last_preview = b""
        self.cam = Picamera2()
        self._preview_cfg = self.cam.create_preview_configuration(
            main={"size": (cfg.preview_width, cfg.preview_height), "format": "RGB888"},
            controls={"AfMode": controls.AfModeEnum.Continuous},
            buffer_count=2)
        self._still_cfg = self.cam.create_still_configuration(
            main={"size": (cfg.width, cfg.height), "format": "RGB888"},
            sensor={"output_size": (cfg.width, cfg.height), "bit_depth": 10},
            buffer_count=1)
        self.cam.configure(self._preview_cfg)
        self.cam.start()

    def preview_jpeg(self) -> bytes:
        if not self._lock.acquire(blocking=False):  # capture in progress: serve the last frame
            return self._last_preview
        try:
            # "RGB888" in libcamera/Picamera2 is B,G,R in memory, i.e. already OpenCV order.
            self._last_preview = _jpeg(self.cam.capture_array("main"))
            return self._last_preview
        finally:
            self._lock.release()

    def capture_still(self) -> np.ndarray:
        with self._lock:
            try:
                self.cam.switch_mode(self._still_cfg)
                time.sleep(self.cfg.settle_s)  # let auto-exposure settle under the LED
                rng = getattr(self._controls.AfRangeEnum, self.cfg.af_range.capitalize())
                self.cam.set_controls({"AfRange": rng})
                try:  # wait=<seconds> raises on timeout; returns False if the cycle ends unfocused
                    if not self.cam.autofocus_cycle(wait=self.cfg.af_timeout_s):
                        log.warning("autofocus did not lock; capturing anyway")
                except Exception as e:
                    log.warning("autofocus failed (%s); capturing anyway", e)
                return self.cam.capture_array("main")
            finally:
                self.cam.switch_mode(self._preview_cfg)

    def close(self):
        self.cam.stop()
        self.cam.close()


class FakeCamera:
    """Cycles through the images in a folder as 'captures'; the preview is the next one, downscaled."""

    def __init__(self, folder: str | Path, cfg: CameraCfg | None = None):
        self.files = sorted(p for p in Path(folder).expanduser().iterdir() if p.suffix.lower() in IMG_EXT)
        if not self.files:
            raise SystemExit(f"--fake: no images in {folder}")
        self.cfg = cfg or CameraCfg()
        self._i = 0
        self._lock = threading.Lock()

    def _load(self, i: int) -> np.ndarray:
        img = cv2.imread(str(self.files[i % len(self.files)]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cannot read {self.files[i % len(self.files)]}")
        return img

    def preview_jpeg(self) -> bytes:
        img = self._load(self._i)
        w = self.cfg.preview_width
        return _jpeg(cv2.resize(img, (w, round(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_AREA))

    def capture_still(self) -> np.ndarray:
        with self._lock:
            img = self._load(self._i)
            log.info("[fake camera] frame from %s", self.files[self._i % len(self.files)].name)
            self._i += 1
            return img

    def close(self):
        pass
