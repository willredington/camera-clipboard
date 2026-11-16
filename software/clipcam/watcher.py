"""Auto-capture loop: frame -> rotate/crop -> gate -> QR page id -> (cooldown, hourly cap, error backoff) -> OCR
-> stored against that page.

A frame is only sent when the page's QR code decodes in that same frame, and the live page only changes when a
*different* page's code is read (a covered code never archives anything).
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Callable

import numpy as np

from .config import AutoCfg
from .documents import DocumentStore
from .gate import Gate, prepare
from .ocr import OcrError, OcrResult
from .pageid import read_page_id

log = logging.getLogger(__name__)
_NOT_STILL = ("dark", "blank", "moving", "settling")


class Watcher:
    def __init__(self, camera, gate: Gate, ocr, documents: DocumentStore, cfg: AutoCfg,
                 clock: Callable[[], float] = time.monotonic,
                 read_page: Callable[[np.ndarray], int | None] = read_page_id,
                 on_result: Callable[[int, OcrResult], None] | None = None):
        self.camera, self.gate, self.ocr, self.documents, self.cfg = camera, gate, ocr, documents, cfg
        self.clock, self.read_page, self.on_result = clock, read_page, on_result
        self._calls: collections.deque[float] = collections.deque()  # API call times in the trailing hour
        self._next_allowed = float("-inf")                           # error backoff
        self._backoff = cfg.backoff_s
        self._last_reason: str | None = None
        self._refs: dict[int, np.ndarray | None] = {}               # per-page "already sent" ink masks
        self._pending = False                                        # live page has changes not yet sent
        self._qr_visible: bool | None = None
        self._last = "starting"

    def status(self) -> dict:
        return {"live": self.documents.live(), "qr_visible": self._qr_visible, "pending": self._pending,
                "last": self._last}

    def step(self) -> str:
        self._last = self._step()
        return self._last

    def _step(self) -> str:
        c = self.cfg
        try:
            frame = self.camera.capture_frame()
        except Exception as e:
            log.warning("camera frame failed: %s", e)
            return "camera_error"
        img = prepare(frame, c.rotate_cw, c.roi)
        d = self.gate.check(img)
        self._log_gate(d)
        if d.reason in _NOT_STILL:
            return d.reason

        page = self.read_page(img)
        self._qr_visible = page is not None
        if page is None:
            if d.send:
                self._pending = True
                return "no_qr"
            return d.reason
        if page != self.documents.live():
            d = self._switch(page)
        if not d.send:
            return d.reason

        now = self.clock()
        while self._calls and now - self._calls[0] >= 3600:
            self._calls.popleft()
        blocked = ("backoff" if now < self._next_allowed else
                   "cooldown" if self._calls and now - self._calls[-1] < c.min_interval_s else
                   "rate_limited" if len(self._calls) >= c.max_per_hour else None)
        if blocked:
            self._pending = True
            return blocked

        if c.dry_run or self.ocr is None:
            self._sent(page)
            log.info("dry run: would send page %s (ink=%.4f change=%s)", page, d.ink, d.change)
            return "dry_run"

        self._calls.append(now)  # failed calls count too: they may still be billed
        try:
            result = self.ocr.extract(img)
        except OcrError as e:
            self._pending = True
            self._next_allowed = now + self._backoff
            log.error("OCR failed (%s); retrying in %.0f s", e, self._backoff)
            self._backoff = min(self._backoff * 2, c.backoff_max_s)
            return "ocr_error"
        self._backoff = c.backoff_s
        self._sent(page)
        self.documents.add_revision(page, img, result.text)
        if self.on_result is not None:
            self.on_result(page, result)
        return "sent"

    def _switch(self, page: int):
        old = self.documents.set_live(page)
        if old is not None:
            self._refs[old] = self.gate.reference
            if self._pending:
                self.documents.set_unsent(old, True)
        self.gate.reference = self._refs.get(page)
        self._pending = False
        log.info("page %s -> %s", old, page)
        return self.gate.recheck()

    def _sent(self, page: int) -> None:
        self.gate.mark_sent()
        self._refs[page] = self.gate.reference
        self._pending = False

    def _log_gate(self, d) -> None:
        if d.reason != self._last_reason:  # log transitions only, so tuning info doesn't flood the journal
            log.info("gate: %s (brightness %.0f, ink %.4f, motion %s, change %s)", d.reason, d.brightness, d.ink,
                     "-" if d.motion is None else f"{d.motion:.4f}", "-" if d.change is None else f"{d.change:.4f}")
            self._last_reason = d.reason

    def run(self, stop: threading.Event) -> None:
        log.info("auto-capture running every %.1f s%s", self.cfg.interval_s,
                 " (dry run)" if self.cfg.dry_run or self.ocr is None else "")
        while not stop.is_set():
            try:
                self.step()
            except Exception:
                log.exception("auto-capture step failed")
            stop.wait(self.cfg.interval_s)
