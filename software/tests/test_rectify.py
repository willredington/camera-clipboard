import cv2
import numpy as np
import pytest

from clipcam.rectify import BoardGeometry, Rectifier, RectifyError, board_to_page_px, page_size_px
from synth import DISC_R_MM, render_scene

DPI = 200
# pitch: tilt about the board's short axis; yaw: tilt about its long axis; roll 90 = sensor long side
# along the page's long side. Angles are in degrees.
CASES = {
    "nominal": dict(pitch=0, yaw=0, roll=90),
    "tilted": dict(pitch=4, yaw=-3, roll=90),
    "tilt+offset+roll": dict(pitch=-5, yaw=4, roll=93, offset_mm=(6, -15), height_mm=208),
    "strong tilt": dict(pitch=6, yaw=-3.5, roll=87, height_mm=203),
    "landscape upside down": dict(pitch=3, yaw=2, roll=-90),
}


def corner_error_mm(res, truth):
    """Page corners: true image position -> recovered board mm vs true board mm."""
    g = truth.geom
    ox, oy = g.page_origin
    P = np.array([[ox, oy], [ox + g.page_w, oy], [ox + g.page_w, oy + g.page_h], [ox, oy + g.page_h]])
    img_pts = cv2.perspectiveTransform(P.reshape(-1, 1, 2), truth.h_board_to_img)
    rec = cv2.perspectiveTransform(img_pts, res.h_img_to_board).reshape(-1, 2)
    return np.linalg.norm(rec - P, axis=1).max()


def disc_error_mm(res, truth):
    """Find each black disc in the rectified page and compare to where it should be."""
    g = truth.geom
    s = DPI / 25.4
    gray = cv2.cvtColor(res.image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, cent = cv2.connectedComponentsWithStats(bw)
    want_area = np.pi * (DISC_R_MM * s) ** 2
    discs = [c for c, st in zip(cent[1:], stats[1:]) if 0.8 * want_area < st[cv2.CC_STAT_AREA] < 1.2 * want_area]
    ox, oy = g.page_origin
    errs = []
    for bx, by in truth.discs_board_mm:
        e = np.array([(bx - ox) * s, (by - oy) * s])
        d = min((np.linalg.norm(c - e) for c in discs), default=1e9)
        errs.append(d / s)
    assert max(errs) < 50, "a disc was not found in the rectified image"
    return max(errs)


@pytest.mark.parametrize("name", CASES)
def test_four_markers(name):
    img, truth = render_scene(seed=1, **CASES[name])
    res = Rectifier(truth.geom, DPI).process(img)
    assert res.method == "4 markers" and res.found_ids == [0, 1, 2, 3]
    assert res.image.shape[:2][::-1] == page_size_px(truth.geom, DPI)
    assert res.rms_mm < 0.3 and not res.warnings
    ce, de = corner_error_mm(res, truth), disc_error_mm(res, truth)
    print(f"{name}: rms {res.rms_mm:.3f} mm, page-corner err {ce:.3f} mm, disc err {de:.3f} mm")
    assert ce < 1.0 and de < 1.0


def test_three_markers_still_accurate():
    img, truth = render_scene(seed=2, pitch=3, yaw=-2, hide_markers=(2,))
    res = Rectifier(truth.geom, DPI).process(img)
    assert res.method == "3 markers" and res.found_ids == [0, 1, 3]
    assert any("only 3 markers" in w for w in res.warnings)
    assert corner_error_mm(res, truth) < 1.0 and disc_error_mm(res, truth) < 1.0


def test_fewer_than_three_reuses_last_good_then_fails_cold():
    img, truth = render_scene(seed=3, pitch=3, yaw=-2)
    r = Rectifier(truth.geom, DPI)
    r.process(img)  # good frame
    same_pose_bad, _ = render_scene(seed=4, pitch=3, yaw=-2, hide_markers=(0, 1))
    res = r.process(same_pose_bad)
    assert res.method == "cached" and any("last good" in w for w in res.warnings)
    assert corner_error_mm(res, truth) < 1.0  # camera did not move, so still correct
    with pytest.raises(RectifyError):
        Rectifier(truth.geom, DPI).process(same_pose_bad)


def test_custom_marker_positions_and_dpi():
    g = BoardGeometry(markers={0: (30.0, 12.0), 1: (230.0, 12.0), 2: (230.0, 318.0), 3: (30.0, 318.0)})
    img, truth = render_scene(g, seed=5, pitch=2, yaw=1)
    res = Rectifier(g, dpi=150).process(img)
    assert res.image.shape[:2][::-1] == (1275, 1650)  # 215.9 x 279.4 mm at 150 dpi
    assert corner_error_mm(res, truth) < 1.0


def test_page_mapping_is_centred():
    g = BoardGeometry()
    M = board_to_page_px(g, 200)
    top_left = M @ [(260 - 215.9) / 2, (330 - 279.4) / 2, 1]
    assert np.allclose(top_left[:2], 0, atol=1e-9)
