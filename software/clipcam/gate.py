"""Cheap local check deciding whether a frame is worth sending to the (paid) OCR model.

Pure OpenCV/numpy, no camera imports. Order of checks: dark -> blank -> moving -> settling -> unchanged -> changed.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class GateCfg:
    work_side: int = 400            # checks run on a copy downscaled to this long side
    min_brightness: float = 40.0    # mean grey level (0-255) below which the frame is "dark"
    min_ink_frac: float = 0.0005    # fraction of ink pixels below which the page is "blank"
    motion_thresh: float = 0.02     # mean abs difference to the previous frame (0-1) above which it is "moving"
    stable_frames: int = 2          # consecutive still frames required before sending
    min_change_frac: float = 0.0003  # ink added/removed vs the last sent frame, as a fraction of the image
    tolerance_px: int = 3           # ink this close to the reference's ink counts as unchanged


@dataclass
class Decision:
    send: bool
    reason: str                     # dark | blank | moving | settling | unchanged | changed
    brightness: float
    ink: float
    motion: float | None
    change: float | None


def prepare(bgr: np.ndarray, rotate_cw: int, roi: list[float]) -> np.ndarray:
    """Rotate clockwise by 0/90/180/270 degrees, then crop roi = [x, y, w, h] as fractions of the rotated frame."""
    if rotate_cw not in _ROTATIONS:
        raise ValueError(f"rotate_cw must be one of {sorted(_ROTATIONS)}, got {rotate_cw}")
    img = bgr if _ROTATIONS[rotate_cw] is None else cv2.rotate(bgr, _ROTATIONS[rotate_cw])
    h, w = img.shape[:2]
    x0, y0 = round(roi[0] * w), round(roi[1] * h)
    x1, y1 = min(w, round((roi[0] + roi[2]) * w)), min(h, round((roi[1] + roi[3]) * h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"empty roi {roi}")
    return img[y0:y1, x0:x1]


class Gate:
    def __init__(self, cfg: GateCfg):
        self.cfg = cfg
        self._prev: np.ndarray | None = None
        self._stable = 0
        self._mask: np.ndarray | None = None   # ink mask of the last checked frame
        self._ref: np.ndarray | None = None    # ink mask of the last sent frame

    def _gray(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        s = self.cfg.work_side / max(h, w)
        small = cv2.resize(bgr, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)

    def check(self, bgr: np.ndarray) -> Decision:
        c = self.cfg
        gray = self._gray(bgr)
        prev, self._prev = self._prev, gray
        if prev is not None and prev.shape != gray.shape:
            prev = None
        bright = float(gray.mean())
        if bright < c.min_brightness:
            self._stable = 0
            return Decision(False, "dark", bright, 0.0, None, None)

        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
        self._mask = mask
        ink = float(mask.mean())
        if ink < c.min_ink_frac:
            self._stable = 0
            return Decision(False, "blank", bright, ink, None, None)

        motion = None if prev is None else float(np.abs(gray.astype(np.int16) - prev).mean() / 255)
        if motion is not None and motion > c.motion_thresh:
            self._stable = 0
            return Decision(False, "moving", bright, ink, motion, None)
        if motion is not None:
            self._stable += 1
        if self._stable < c.stable_frames:
            return Decision(False, "settling", bright, ink, motion, None)

        change = None
        if self._ref is not None and self._ref.shape == mask.shape:
            change = self._change(mask, self._ref)
            if change < c.min_change_frac:
                return Decision(False, "unchanged", bright, ink, motion, change)
        return Decision(True, "changed", bright, ink, motion, change)

    def _change(self, mask: np.ndarray, ref: np.ndarray) -> float:
        k = np.ones((2 * self.cfg.tolerance_px + 1,) * 2, np.uint8)
        grow = lambda m: cv2.dilate(m.astype(np.uint8), k) > 0  # noqa: E731
        added = np.count_nonzero(mask & ~grow(ref))
        removed = np.count_nonzero(ref & ~grow(mask))
        return (added + removed) / mask.size

    def mark_sent(self) -> None:
        """The ink mask from the last check() becomes the reference for 'unchanged'."""
        self._ref = self._mask
