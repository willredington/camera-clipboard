"""Synthetic camera images of the clipboard, with exact ground truth.

Run as a script to write a few JPEGs for the --fake camera:
    python tests/synth.py /tmp/fake_captures
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clipcam.rectify import BoardGeometry  # noqa: E402

PX_PER_MM = 10  # resolution of the flat board texture

# Discs at known page positions (mm from page top-left), clear of the text block.
DISCS_PAGE_MM = [(15, 15), (200.9, 15), (200.9, 264.4), (15, 264.4), (108, 144)]
DISC_R_MM = 3.0


@dataclass
class Truth:
    h_board_to_img: np.ndarray
    geom: BoardGeometry
    discs_board_mm: np.ndarray


def render_board(geom: BoardGeometry, hide_markers=()) -> np.ndarray:
    S = PX_PER_MM
    img = np.full((round(geom.board_h * S), round(geom.board_w * S), 3), (150, 175, 195), np.uint8)  # wood-ish BGR
    ox, oy = geom.page_origin
    x0, y0 = round(ox * S), round(oy * S)
    x1, y1 = x0 + round(geom.page_w * S), y0 + round(geom.page_h * S)
    img[y0:y1, x0:x1] = 250

    # text and ruled lines, kept clear of the discs
    for k in range(12):
        y = y0 + round((40 + k * 16) * S)
        cv2.putText(img, f"Line {k + 1}: the quick brown fox jumps over", (x0 + 30 * S, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (40, 40, 40), 3, cv2.LINE_AA)
        cv2.line(img, (x0 + 25 * S, y + 25), (x1 - 25 * S, y + 25), (200, 200, 200), 2)
    for (px, py) in DISCS_PAGE_MM:
        cv2.circle(img, (x0 + round(px * S), y0 + round(py * S)), round(DISC_R_MM * S), (20, 20, 20), -1, cv2.LINE_AA)

    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, geom.dictionary))
    side = round(geom.marker_size * S)
    quiet = round(2.5 * S)
    for mid, (cx, cy) in geom.markers.items():
        if mid in hide_markers:
            continue
        m = cv2.aruco.generateImageMarker(dic, mid, side)
        mx, my = round(cx * S) - side // 2, round(cy * S) - side // 2
        img[my - quiet:my + side + quiet, mx - quiet:mx + side + quiet] = 255
        img[my:my + side, mx:mx + side] = m[..., None]
    return img


def camera_matrix(w, h, vfov_deg):
    f = (h / 2) / np.tan(np.radians(vfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])


def rot(axis, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def render_scene(geom: BoardGeometry | None = None, *, size=(4608, 2592), vfov=67.0,
                 height_mm=205.0, offset_mm=(0.0, 0.0), pitch=0.0, yaw=0.0, roll=90.0,
                 noise=4.0, blur=1.2, vignette=0.25, hide_markers=(), k1=0.0, seed=0):
    """World: board plane z=0, x right, y down, z into the board; camera above at z=-height.

    Pinhole camera with the 67 deg vertical FOV (short sensor axis) from the README, i.e. the
    tighter reading of "102 x 67". pitch = tilt about the board's x axis (short axis, shifts the
    view along the long axis: lots of slack); yaw = tilt about the board's y axis (long axis,
    shifts the view across the short axis: only ~7 mm of slack beyond the markers). roll = 90
    puts the sensor's long side along the page's long side, as the README describes.

    k1 adds radial (barrel < 0) distortion in normalised image coordinates.
    """
    geom = geom or BoardGeometry()
    W, Hh = size
    K = camera_matrix(W, Hh, vfov)
    R = rot("z", roll) @ rot("y", yaw) @ rot("x", pitch)
    C = np.array([geom.board_w / 2 + offset_mm[0], geom.board_h / 2 + offset_mm[1], -height_mm])
    t = -R @ C
    Hb2m = K @ np.column_stack([R[:, 0], R[:, 1], t])  # board mm -> image px

    board = render_board(geom, hide_markers)
    S = PX_PER_MM
    Hpx = Hb2m @ np.diag([1 / S, 1 / S, 1])
    img = cv2.warpPerspective(board, Hpx, (W, Hh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(70, 80, 85))
    if k1 != 0.0:
        # distort: output pixel (distorted) samples the ideal image at undistorted coords
        f = K[0, 0]
        yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
        xn, yn = (xx - W / 2) / f, (yy - Hh / 2) / f
        # invert xd = xu * (1 + k1 r_u^2) by fixed point
        xu, yu = xn.copy(), yn.copy()
        for _ in range(8):
            r2 = xu * xu + yu * yu
            xu, yu = xn / (1 + k1 * r2), yn / (1 + k1 * r2)
        img = cv2.remap(img, xu * f + W / 2, yu * f + Hh / 2, cv2.INTER_LINEAR)

    rng = np.random.default_rng(seed)
    out = img.astype(np.float32)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if vignette:
        yy, xx = np.mgrid[0:Hh, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - Hh / 2) / (Hh / 2)) ** 2
        out *= (1 - vignette * r2 / 2)[..., None]
    if noise:
        out += rng.normal(0, noise, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)

    ox, oy = geom.page_origin
    discs = np.array([(ox + px, oy + py) for px, py in DISCS_PAGE_MM])
    return out, Truth(Hb2m, geom, discs)


if __name__ == "__main__":
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "fake_captures")
    outdir.mkdir(parents=True, exist_ok=True)
    cases = [dict(pitch=3, yaw=-1, roll=90), dict(pitch=-4, yaw=1.5, roll=92, offset_mm=(2, -10)),
             dict(pitch=5, yaw=0.5, roll=88, height_mm=207, hide_markers=(2,))]
    for i, kw in enumerate(cases):
        im, _ = render_scene(seed=i, **kw)
        cv2.imwrite(str(outdir / f"scene{i}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote", len(cases), "images to", outdir)
