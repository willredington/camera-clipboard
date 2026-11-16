import threading

import cv2
import numpy as np
import pytest

from clipcam.camera import FakeCamera
from clipcam.config import AutoCfg, load_config
from clipcam.gate import Decision, Gate, GateCfg
from clipcam.ocr import OcrError, OcrResult
from clipcam.store import Store
from clipcam.watcher import Watcher
from test_gate import page


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Cam:
    def __init__(self, frame=None, fail_first=0):
        self.frame = page(["hello"]) if frame is None else frame
        self.fail_first = fail_first

    def capture_frame(self):
        if self.fail_first:
            self.fail_first -= 1
            raise RuntimeError("camera gone")
        return self.frame


class AlwaysChanged:
    """Gate stub that always wants to send."""
    def __init__(self):
        self.marked = 0

    def check(self, bgr):
        return Decision(True, "changed", 200.0, 0.01, 0.0, None)

    def mark_sent(self):
        self.marked += 1


class Ocr:
    def __init__(self, script=("ok",)):
        self.script, self.calls = list(script), 0

    def extract(self, bgr):
        self.calls += 1
        what = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if what == "err":
            raise OcrError("HTTP 500: boom", 500)
        return OcrResult("hello", "m", 5, 1)


def make(gate=None, ocr=None, cam=None, **cfg):
    clock, sink = Clock(), []
    w = Watcher(cam or Cam(), gate or Gate(GateCfg()), ocr if ocr is not None else Ocr(),
                lambda img, res: sink.append(res), AutoCfg(**cfg), clock=clock)
    return w, clock, sink


def ticks(w, clock, n, dt=2.0):
    out = []
    for _ in range(n):
        out.append(w.step())
        clock.t += dt
    return out


def test_unchanged_page_sends_once():
    w, clock, sink = make()
    out = ticks(w, clock, 20)
    assert out.count("sent") == 1 and out[-1] == "unchanged" and len(sink) == 1


def test_cooldown_between_sends():
    w, clock, _ = make(gate=AlwaysChanged())
    assert ticks(w, clock, 6) == ["sent", "cooldown", "cooldown", "cooldown", "cooldown", "sent"]


def test_hourly_cap():
    w, clock, _ = make(gate=AlwaysChanged(), max_per_hour=3, min_interval_s=0)
    out = ticks(w, clock, 10)
    assert out.count("sent") == 3 and out[3:] == ["rate_limited"] * 7
    clock.t += 3600
    assert w.step() == "sent"


def test_ocr_error_backs_off_then_retries():
    gate = Gate(GateCfg())
    ocr = Ocr(["err", "ok"])
    w, clock, sink = make(gate=gate, ocr=ocr)
    out = ticks(w, clock, 25)  # 50 s
    first_err = out.index("ocr_error")
    sent = out.index("sent")
    assert all(o == "backoff" for o in out[first_err + 1:sent])
    assert (sent - first_err) * 2 >= 30          # waited out the 30 s backoff
    assert len(sink) == 1 and ocr.calls == 2


def test_backoff_doubles_and_caps():
    w, clock, _ = make(gate=AlwaysChanged(), ocr=Ocr(["err"]), min_interval_s=0)
    call_times = []
    for _ in range(4000):
        if w.step() == "ocr_error":
            call_times.append(clock.t)
        clock.t += 1.0
    gaps = [b - a for a, b in zip(call_times, call_times[1:])]
    assert gaps[:5] == [30, 60, 120, 240, 480] and set(gaps[5:]) == {600}


def test_failed_call_does_not_mark_sent():
    gate = AlwaysChanged()
    w, clock, _ = make(gate=gate, ocr=Ocr(["err"]))
    assert w.step() == "ocr_error" and gate.marked == 0


def test_dry_run_never_calls_ocr():
    ocr = Ocr()
    w, clock, sink = make(ocr=ocr, dry_run=True)
    out = ticks(w, clock, 10)
    assert ocr.calls == 0 and not sink and out.count("dry_run") == 1 and out[-1] == "unchanged"


def test_no_ocr_behaves_as_dry_run():
    w = Watcher(Cam(), AlwaysChanged(), None, lambda *a: None, AutoCfg(), clock=Clock())
    assert w.step() == "dry_run"


def test_camera_error_does_not_crash():
    w, clock, _ = make(cam=Cam(fail_first=1), gate=AlwaysChanged())
    assert ticks(w, clock, 2) == ["camera_error", "sent"]


def test_rotation_and_roi_applied_before_ocr():
    seen = []
    w = Watcher(Cam(frame=np.zeros((100, 200, 3), np.uint8)), AlwaysChanged(), Ocr(),
                lambda img, res: seen.append(img.shape), AutoCfg(rotate_cw=270, roi=[0, 0, 1, 0.5]), clock=Clock())
    assert w.step() == "sent" and seen == [(100, 100, 3)]


def test_run_stops_and_survives_exceptions():
    class Boom(AlwaysChanged):
        n = 0

        def check(self, bgr):
            self.n += 1
            if self.n == 1:
                raise ValueError("bad frame")
            return super().check(bgr)

    gate, stop = Boom(), threading.Event()
    sink = []
    w = Watcher(Cam(), gate, Ocr(), lambda img, res: (sink.append(res), stop.set()), AutoCfg(interval_s=0.01))
    t = threading.Thread(target=w.run, args=(stop,), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive() and gate.n >= 2 and len(sink) == 1


def test_fake_camera_capture_frame(tmp_path):
    for i in range(2):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.full((10, 20, 3), i * 100, np.uint8))
    cam = FakeCamera(tmp_path)
    assert cam.capture_frame().mean() == 0 and cam.capture_frame().mean() == 100


def test_store_save_ocr(tmp_path):
    s = Store(tmp_path, 200)
    img = np.full((20, 30, 3), 128, np.uint8)
    a, b = s.save_ocr(img, "héllo\nworld"), s.save_ocr(img, "x")
    assert a != b
    assert (tmp_path / "ocr" / f"{a}.txt").read_text(encoding="utf-8") == "héllo\nworld"
    assert cv2.imread(str(tmp_path / "ocr" / f"{a}.jpg")).shape == (20, 30, 3)


def test_config_auto_and_gate_sections(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text('[auto]\nenabled = true\nrotate_cw = 270\nroi = [0.1, 0, 0.8, 1]\n[gate]\nmin_ink_frac = 0.01\n')
    cfg = load_config(f)
    assert cfg.auto.enabled and cfg.auto.rotate_cw == 270 and cfg.auto.roi == [0.1, 0, 0.8, 1]
    assert cfg.gate.min_ink_frac == 0.01 and cfg.gate.stable_frames == 2
    d = load_config(None)
    assert not d.auto.enabled and d.auto.interval_s == 2.0 and d.camera.watch_width == 2304
