"""Auto-capture loop: frame -> rotate/crop -> gate -> (cooldown, hourly cap, error backoff) -> OCR -> sink."""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable

import numpy as np

from .config import AutoCfg
from .gate import Gate, prepare
from .ocr import OcrError, OcrResult

log = logging.getLogger(__name__)


class Watcher:
    def __init__(self, camera, gate: Gate, ocr, sink: Callable[[np.ndarray, OcrResult], None], cfg: AutoCfg,
                 clock: Callable[[], float] = time.monotonic):
        self.camera, self.gate, self.ocr, self.sink, self.cfg, self.clock = camera, gate, ocr, sink, cfg, clock
        self._calls: collections.deque[float] = collections.deque()  # API call times in the trailing hour
        self._next_allowed = float("-inf")                           # error backoff
        self._backoff = cfg.backoff_s

    def step(self) -> str:
        c = self.cfg
        try:
            frame = self.camera.capture_frame()
        except Exception as e:
            log.warning("camera frame failed: %s", e)
            return "camera_error"
        img = prepare(frame, c.rotate_cw, c.roi)
        d = self.gate.check(img)
        log.debug("gate %s bright=%.0f ink=%.4f motion=%s change=%s", d.reason, d.brightness, d.ink, d.motion, d.change)
        if not d.send:
            return d.reason

        now = self.clock()
        while self._calls and now - self._calls[0] >= 3600:
            self._calls.popleft()
        if now < self._next_allowed:
            return "backoff"
        if self._calls and now - self._calls[-1] < c.min_interval_s:
            return "cooldown"
        if len(self._calls) >= c.max_per_hour:
            return "rate_limited"

        if c.dry_run or self.ocr is None:
            self.gate.mark_sent()
            log.info("dry run: would send (ink=%.4f change=%s)", d.ink, d.change)
            return "dry_run"

        self._calls.append(now)  # failed calls count too: they may still be billed
        try:
            result = self.ocr.extract(img)
        except OcrError as e:
            self._next_allowed = now + self._backoff
            log.error("OCR failed (%s); retrying in %.0f s", e, self._backoff)
            self._backoff = min(self._backoff * 2, c.backoff_max_s)
            return "ocr_error"
        self._backoff = c.backoff_s
        self.gate.mark_sent()
        self.sink(img, result)
        return "sent"

    def run(self, stop: threading.Event) -> None:
        log.info("auto-capture running every %.1f s%s", self.cfg.interval_s,
                 " (dry run)" if self.cfg.dry_run or self.ocr is None else "")
        while not stop.is_set():
            try:
                self.step()
            except Exception:
                log.exception("auto-capture step failed")
            stop.wait(self.cfg.interval_s)
