import cv2
import numpy as np
import pytest

from clipcam.gate import Gate, GateCfg, prepare

H, W = 1296, 2304


def page(lines, blob_at=None, noise=0.0, seed=0):
    img = np.full((H, W, 3), 200, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (200, 300 + i * 250), cv2.FONT_HERSHEY_SIMPLEX, 4, (20, 20, 20), 8, cv2.LINE_AA)
    if blob_at is not None:  # a hand: big dark blob
        cv2.ellipse(img, blob_at, (350, 250), 0, 0, 360, (60, 70, 90), -1)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def dark():
    return np.full((H, W, 3), 10, np.uint8)


def run_n(gate, img, n):
    return [gate.check(img) for _ in range(n)]


def sent_gate(img):
    """A gate that has just sent `img`."""
    g = Gate(GateCfg())
    ds = run_n(g, img, 3)
    assert ds[-1].send
    g.mark_sent()
    return g


def test_dark_frame_skipped():
    assert Gate(GateCfg()).check(dark()).reason == "dark"


def test_blank_page_skipped():
    assert all(d.reason == "blank" for d in run_n(Gate(GateCfg()), page([]), 3))


def test_first_send_needs_stable_frames():
    ds = run_n(Gate(GateCfg()), page(["hello"]), 3)
    assert [d.reason for d in ds] == ["settling", "settling", "changed"]
    assert ds[-1].send and not ds[0].send


def test_moving_hand_blocks_send():
    g = Gate(GateCfg())
    frames = [page(["hello"], blob_at=(600 + 500 * (i % 2), 700)) for i in range(6)]
    ds = [g.check(f) for f in frames]
    assert not any(d.send for d in ds)
    assert all(d.reason == "moving" for d in ds[1:])


def test_unchanged_after_mark_sent():
    img = page(["hello"])
    g = sent_gate(img)
    assert [d.reason for d in run_n(g, img, 5)] == ["unchanged"] * 5


def test_new_word_triggers_change():
    g = sent_gate(page(["hello"]))
    ds = run_n(g, page(["hello world"]), 3)
    assert ds[-1].reason == "changed" and ds[-1].send
    assert ds[-1].change >= GateCfg().min_change_frac


def test_sensor_noise_is_unchanged():
    g = sent_gate(page(["hello", "second line"], noise=3, seed=0))
    ds = [g.check(page(["hello", "second line"], noise=3, seed=s)) for s in range(1, 6)]
    assert all(d.reason == "unchanged" for d in ds), [(d.reason, d.motion, d.change) for d in ds]


def test_dark_resets_stability():
    g = Gate(GateCfg())
    img = page(["hello"])
    g.check(img)
    g.check(img)          # stable_count 1
    g.check(dark())       # reset
    assert g.check(img).reason in ("moving", "settling")


def test_prepare_rotation_and_roi():
    img = np.zeros((100, 200, 3), np.uint8)
    assert prepare(img, 270, [0, 0, 1, 1]).shape == (200, 100, 3)
    assert prepare(img, 90, [0, 0, 1, 1]).shape == (200, 100, 3)
    assert prepare(img, 0, [0.5, 0, 0.5, 0.5]).shape == (50, 100, 3)
    with pytest.raises(ValueError):
        prepare(img, 45, [0, 0, 1, 1])
    with pytest.raises(ValueError):
        prepare(img, 0, [0.5, 0.5, 0, 0.5])


def test_prepare_270_puts_right_edge_on_top():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, -10:] = 255  # right edge white
    out = prepare(img, 270, [0, 0, 1, 1])
    assert out[:10].mean() == 255 and out[-10:].mean() == 0


def small_page(lines, noise=0.0, seed=0):
    """Handwriting-sized text: ~5 mm letters at the real ~0.2 mm/px scale (about 24 px tall)."""
    img = np.full((H, W, 3), 200, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(img, text, (300, 300 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    if noise:
        rng = np.random.default_rng(seed)
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def test_handwriting_size_line_is_not_blank_and_new_word_is_seen():
    first = small_page(["meeting notes for monday"], noise=3)
    g = Gate(GateCfg())
    ds = run_n(g, first, 3)
    assert ds[-1].send, [(d.reason, d.ink) for d in ds]
    g.mark_sent()
    ds = [g.check(small_page(["meeting notes for monday", "call bob"], noise=3, seed=s)) for s in range(1, 4)]
    assert ds[-1].send, [(d.reason, d.change) for d in ds]
