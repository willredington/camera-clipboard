"""Pure image rectification: ArUco markers -> homography -> flat top-down page.

No camera or GPIO imports here, so it runs anywhere OpenCV + numpy do.

Board coordinates: millimetres, origin at the board's top-left corner as seen
from above with the clip at the top, x to the right, y down.
Page pixels: origin at the page's top-left corner, same orientation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Corner order of a detected marker: TL, TR, BR, BL of the marker as printed,
# which (stuck on upright) is the same as board-space TL, TR, BR, BL.
_UNIT_CORNERS = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float64) / 2


class RectifyError(RuntimeError):
    pass


@dataclass
class BoardGeometry:
    board_w: float = 260.0
    board_h: float = 330.0
    page_w: float = 215.9
    page_h: float = 279.4
    marker_size: float = 18.0  # outer edge of the black square
    # marker id -> centre (x, y) in board mm
    markers: dict[int, tuple[float, float]] = field(default_factory=lambda: {
        # Top/bottom margins (25.3 mm deep, so an 18 mm marker fits with ~3.6 mm to spare), centred in
        # the margin, 45 mm in from the side edges: clear of the 24 mm corner holders and the 100 mm
        # clip base. The side margins are only 22 mm wide and would put the markers within ~7 mm of the
        # frame edge on the short image axis, which a ~1 degree boom tilt would push out of view.
        0: (45.0, 12.65),    # top-left
        1: (215.0, 12.65),   # top-right
        2: (215.0, 317.35),  # bottom-right
        3: (45.0, 317.35),   # bottom-left
    })
    dictionary: str = "DICT_4X4_50"

    @property
    def page_origin(self) -> tuple[float, float]:
        """Board mm of the page's top-left corner (page is centred)."""
        return ((self.board_w - self.page_w) / 2, (self.board_h - self.page_h) / 2)

    def marker_corners(self, marker_id: int) -> np.ndarray:
        cx, cy = self.markers[marker_id]
        return np.array([cx, cy]) + _UNIT_CORNERS * self.marker_size


@dataclass
class RectifyResult:
    image: np.ndarray            # flat page, BGR
    h_img_to_board: np.ndarray   # 3x3, image px -> board mm
    method: str                  # "4 markers" | "3 markers" | "cached"
    found_ids: list[int]
    rms_mm: float | None         # marker-corner fit residual (None if cached)
    warnings: list[str]


def detect_markers(img: np.ndarray, geom: BoardGeometry,
                   detect_max_side: int = 1600) -> dict[int, np.ndarray]:
    """Return {id: 4x2 float corners (image px)} for the configured ids only.

    Detection runs on a downscaled grey image (faster on a Pi Zero, and the
    default adaptive-threshold windows suit marker bits ~10 px wide, not 30 px).
    Corners are then refined to sub-pixel at full resolution.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, detect_max_side / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray

    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, geom.dictionary))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dic, params)
    corners, ids, _ = detector.detectMarkers(small)
    if ids is None:
        return {}

    found: dict[int, np.ndarray] = {}
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    for c, i in zip(corners, ids.ravel()):
        i = int(i)
        if i not in geom.markers or i in found:
            continue
        pts = (c.reshape(4, 1, 2) / scale).astype(np.float32)
        # refinement window ~ a third of a coarse-pixel quantisation step either side
        half = max(3, int(round(1.5 / scale)) + 2)
        pts = cv2.cornerSubPix(gray, pts, (half, half), (-1, -1), crit)
        found[i] = pts.reshape(4, 2).astype(np.float64)
    return found


def fit_homography(found: dict[int, np.ndarray], geom: BoardGeometry):
    """Least-squares homography image px -> board mm from all marker corners."""
    ids = sorted(found)
    src = np.concatenate([found[i] for i in ids])
    dst = np.concatenate([geom.marker_corners(i) for i in ids])
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise RectifyError("homography fit failed")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))
    return H, rms


def board_to_page_px(geom: BoardGeometry, dpi: float) -> np.ndarray:
    """3x3 matrix: board mm -> output page pixels."""
    s = dpi / 25.4
    ox, oy = geom.page_origin
    return np.array([[s, 0, -ox * s], [0, s, -oy * s], [0, 0, 1]], dtype=np.float64)


def page_size_px(geom: BoardGeometry, dpi: float) -> tuple[int, int]:
    return (round(geom.page_w * dpi / 25.4), round(geom.page_h * dpi / 25.4))


class Rectifier:
    """Stateful wrapper: remembers the last good homography for fallback."""

    # Residual above this suggests wrong marker positions / printed scale / a misdetection.
    WARN_RMS_MM = 0.6

    def __init__(self, geom: BoardGeometry, dpi: float = 200.0, detect_max_side: int = 1600):
        self.geom = geom
        self.dpi = dpi
        self.detect_max_side = detect_max_side
        self.last_good: np.ndarray | None = None

    def process(self, img: np.ndarray) -> RectifyResult:
        warnings: list[str] = []
        found = detect_markers(img, self.geom, self.detect_max_side)
        ids = sorted(found)
        want = sorted(self.geom.markers)
        rms = None

        if len(found) >= 3:
            H, rms = fit_homography(found, self.geom)
            method = f"{len(found)} markers"
            self.last_good = H
            if len(found) < 4:
                missing = [i for i in want if i not in found]
                warnings.append(f"only {len(found)} markers found (missing {missing}); "
                                "using the 3-marker fit, which is less accurate")
            if rms > self.WARN_RMS_MM:
                warnings.append(f"marker fit residual {rms:.2f} mm is high; check marker "
                                "positions in the config and the printed marker size")
        elif self.last_good is not None:
            H = self.last_good
            method = "cached"
            warnings.append(f"only {len(found)} marker(s) found; reusing the last good homography "
                            "(valid only if the camera and board have not moved)")
        else:
            raise RectifyError(f"only {len(found)} marker(s) found and no earlier good "
                               "homography to reuse; check the markers are visible and lit")

        for w in warnings:
            log.warning(w)
        M = board_to_page_px(self.geom, self.dpi) @ H
        out = cv2.warpPerspective(img, M, page_size_px(self.geom, self.dpi),
                                  flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        return RectifyResult(out, H, method, ids, rms, warnings)
