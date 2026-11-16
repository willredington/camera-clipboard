"""Capture service: LED on -> camera still -> rectify -> save. Hardware-agnostic (takes the layers as arguments)."""
from __future__ import annotations

import logging
import threading
import time

from .rectify import Rectifier, RectifyError
from .store import Store

log = logging.getLogger(__name__)


class Busy(RuntimeError):
    pass


class CaptureService:
    def __init__(self, camera, led, rectifier: Rectifier, store: Store):
        self.camera, self.led, self.rectifier, self.store = camera, led, rectifier, store
        self._lock = threading.Lock()

    def capture(self) -> dict:
        """Returns {name, method, markers, rms_mm, warnings, seconds}. Raises Busy or RectifyError."""
        if not self._lock.acquire(blocking=False):
            raise Busy("a capture is already in progress")
        try:
            t0 = time.monotonic()
            self.led.on()
            try:
                frame = self.camera.capture_still()
            finally:
                self.led.off()
            try:
                res = self.rectifier.process(frame)
            except RectifyError:
                path = self.store.save_failed(frame)
                log.error("rectification failed; raw frame saved to %s", path)
                raise
            name = self.store.save_page(res.image)
            out = {"name": name, "method": res.method, "markers": res.found_ids,
                   "rms_mm": None if res.rms_mm is None else round(res.rms_mm, 3),
                   "warnings": res.warnings, "seconds": round(time.monotonic() - t0, 1)}
            log.info("captured %s (%s, rms %s mm) in %.1fs", name, res.method, out["rms_mm"], out["seconds"])
            return out
        finally:
            self._lock.release()
