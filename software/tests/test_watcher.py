import tempfile
import threading

import cv2
import numpy as np

from clipcam.camera import FakeCamera
from clipcam.config import AutoCfg, load_config
from clipcam.documents import DocumentStore
from clipcam.gate import Decision, Gate, GateCfg
from clipcam.ocr import OcrError, OcrResult
from clipcam.store import Store
from clipcam.watcher import Watcher
from test_gate import page, small_page


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
        self.reference = None

    def check(self, bgr):
        return Decision(True, "changed", 200.0, 0.01, 0.0, None)

    def recheck(self):
        return self.check(None)

    def mark_sent(self):
        self.marked += 1


class Ocr:
    def __init__(self, script=("ok",), text="hello"):
        self.script, self.calls, self.text = list(script), 0, text

    def extract(self, bgr):
        self.calls += 1
        what = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if what == "err":
            raise OcrError("HTTP 500: boom", 500)
        return OcrResult(self.text, "m", 5, 1)


class Pages:
    """Scripted QR reader: whatever page id is set (None = QR not readable)."""
    def __init__(self, n=1):
        self.n = n

    def __call__(self, img):
        return self.n


def make(gate=None, ocr=None, cam=None, pages=None, docs=None, on_result=None, **cfg):
    clock = Clock()
    docs = docs or DocumentStore(tempfile.mkdtemp())
    w = Watcher(cam or Cam(), gate or Gate(GateCfg()), ocr if ocr is not None else Ocr(), docs, AutoCfg(**cfg),
                clock=clock, read_page=pages or Pages(1), on_result=on_result)
    return w, clock, docs


def revisions(docs, n=1):
    p = docs.get(n)
    return [] if p is None else p["revisions"]


def ticks(w, clock, n, dt=2.0):
    out = []
    for _ in range(n):
        out.append(w.step())
        clock.t += dt
    return out


# --- unchanged behaviour (cost controls) ---

def test_unchanged_page_sends_once():
    w, clock, docs = make()
    out = ticks(w, clock, 20)
    assert out.count("sent") == 1 and out[-1] == "unchanged" and len(revisions(docs)) == 1


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
    ocr = Ocr(["err", "ok"])
    w, clock, docs = make(ocr=ocr)
    out = ticks(w, clock, 25)  # 50 s
    first_err = out.index("ocr_error")
    sent = out.index("sent")
    assert all(o == "backoff" for o in out[first_err + 1:sent])
    assert (sent - first_err) * 2 >= 30          # waited out the 30 s backoff
    assert len(revisions(docs)) == 1 and ocr.calls == 2


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
    w, clock, docs = make(ocr=ocr, dry_run=True)
    out = ticks(w, clock, 10)
    assert ocr.calls == 0 and not revisions(docs) and out.count("dry_run") == 1 and out[-1] == "unchanged"


def test_no_ocr_behaves_as_dry_run():
    w, _, _ = make(gate=AlwaysChanged(), ocr=None)
    w.ocr = None
    assert w.step() == "dry_run"


def test_camera_error_does_not_crash():
    w, clock, _ = make(cam=Cam(fail_first=1), gate=AlwaysChanged())
    assert ticks(w, clock, 2) == ["camera_error", "sent"]


def test_rotation_and_roi_applied_before_ocr():
    seen = []

    class SeeOcr(Ocr):
        def extract(self, bgr):
            seen.append(bgr.shape)
            return super().extract(bgr)

    w, _, _ = make(cam=Cam(frame=np.zeros((100, 200, 3), np.uint8)), gate=AlwaysChanged(), ocr=SeeOcr(),
                   rotate_cw=270, roi=[0, 0, 1, 0.5])
    assert w.step() == "sent" and seen == [(100, 100, 3)]


def test_run_stops_and_survives_exceptions():
    class Boom(AlwaysChanged):
        n = 0

        def check(self, bgr):
            self.n += 1
            if self.n == 1:
                raise ValueError("bad frame")
            return super().check(bgr)

    gate, stop, results = Boom(), threading.Event(), []
    w, _, _ = make(gate=gate, interval_s=0.01, on_result=lambda n, r: (results.append(r), stop.set()))
    w.clock = __import__("time").monotonic
    t = threading.Thread(target=w.run, args=(stop,), daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive() and gate.n >= 2 and len(results) == 1


def test_logs_gate_result_only_when_it_changes(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="clipcam.watcher")
    w, clock, _ = make(dry_run=True)
    ticks(w, clock, 8)  # settling, settling, dry_run, unchanged x5
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("gate:")]
    assert [l.split()[1] for l in lines] == ["settling", "changed", "unchanged"], lines


# --- pages (document portal Task 4) ---

def test_first_qr_makes_page_live():
    w, clock, docs = make(pages=Pages(5))
    assert "sent" in ticks(w, clock, 4)
    assert docs.live() == 5 and docs.get(5)["text"] == "hello"
    assert w.status()["live"] == 5 and w.status()["qr_visible"] is True


def test_no_qr_never_sends_or_archives():
    docs = DocumentStore(tempfile.mkdtemp())
    docs.set_live(5)
    ocr = Ocr()
    w, clock, _ = make(gate=AlwaysChanged(), ocr=ocr, pages=Pages(None), docs=docs)
    assert ticks(w, clock, 6) == ["no_qr"] * 6
    assert ocr.calls == 0 and docs.live() == 5
    st = w.status()
    assert st["qr_visible"] is False and st["pending"] is True and st["last"] == "no_qr"


def test_no_qr_on_unchanged_frame_is_just_unchanged():
    pages = Pages(1)
    w, clock, _ = make(pages=pages)
    ticks(w, clock, 4)
    pages.n = None
    assert w.step() == "unchanged" and w.status()["qr_visible"] is False


def test_switch_archives_and_marks_unsent():
    cam, pages = Cam(small_page(["page five"])), Pages(5)
    w, clock, docs = make(cam=cam, pages=pages, min_interval_s=60)
    assert "sent" in ticks(w, clock, 4)
    cam.frame = small_page(["page five", "more writing here"])
    out = ticks(w, clock, 4)
    assert "cooldown" in out and w.status()["pending"]
    cam.frame, pages.n = small_page(["a different sheet"]), 6
    ticks(w, clock, 4)
    assert docs.live() == 6
    assert docs.get(5)["unsent"] is True


def test_return_page_resumes_reference():
    a, b = small_page(["page five words"]), small_page(["six six six", "other"])
    cam, pages, ocr = Cam(a), Pages(5), Ocr()
    w, clock, docs = make(cam=cam, pages=pages, ocr=ocr, min_interval_s=0)
    assert ticks(w, clock, 4).count("sent") == 1
    cam.frame, pages.n = b, 6
    assert ticks(w, clock, 4).count("sent") == 1
    calls = ocr.calls
    cam.frame, pages.n = a, 5
    out = ticks(w, clock, 4)
    assert "sent" not in out and out[-1] == "unchanged" and ocr.calls == calls
    assert docs.live() == 5 and len(revisions(docs, 5)) == 1


def test_return_page_with_new_writing_sends():
    a = small_page(["page five words"])
    cam, pages = Cam(a), Pages(5)
    w, clock, docs = make(cam=cam, pages=pages, min_interval_s=0)
    ticks(w, clock, 4)
    cam.frame, pages.n = small_page(["six six six"]), 6
    ticks(w, clock, 4)
    docs.set_unsent(5, True)
    cam.frame, pages.n = small_page(["page five words", "added later on"]), 5
    assert "sent" in ticks(w, clock, 4)
    assert len(revisions(docs, 5)) == 2 and docs.get(5)["unsent"] is False


def test_restart_keeps_live_page():
    docs = DocumentStore(tempfile.mkdtemp())
    docs.set_live(9)
    w, clock, _ = make(docs=docs, pages=Pages(9))
    assert w.status()["live"] == 9
    assert "sent" in ticks(w, clock, 4)
    assert len(revisions(docs, 9)) == 1


def test_status_shape():
    w, _, _ = make()
    st = w.status()
    assert set(st) == {"live", "qr_visible", "pending", "last", "last_at", "last_sent_at", "calls_last_hour",
                       "dry_run"}
    assert st["qr_visible"] is None and st["last_at"] is None and st["last_sent_at"] is None
    assert st["calls_last_hour"] == 0 and st["dry_run"] is False


def test_status_tracks_checks_sends_and_calls():
    w, clock, _ = make(gate=AlwaysChanged(), min_interval_s=0)
    w.wall = lambda: 1_800_000_000.0
    ticks(w, clock, 3)
    st = w.status()
    assert st["calls_last_hour"] == 3 and st["last"] == "sent"
    import time as _t
    assert st["last_at"] == _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(1_800_000_000.0))
    assert st["last_sent_at"] == st["last_at"]
    clock.t += 3601
    assert w.status()["calls_last_hour"] == 0


def test_on_result_called_with_page():
    got = []
    w, clock, _ = make(pages=Pages(4), on_result=lambda n, r: got.append((n, r.text)))
    ticks(w, clock, 4)
    assert got == [(4, "hello")]


# --- other modules ---

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


# --- build_watcher ---
from clipcam.__main__ import build_watcher  # noqa: E402
from clipcam.config import Config  # noqa: E402
from clipcam.ocr import OpenRouterOcr  # noqa: E402


def auto_cfg(**kw):
    cfg = Config()
    cfg.auto.enabled = True
    for k, v in kw.items():
        setattr(cfg.auto, k, v)
    return cfg


def bw(cfg, tmp_path, env):
    return build_watcher(cfg, Cam(), Store(tmp_path, 200), DocumentStore(tmp_path / "pages"), env)


def test_build_watcher_disabled(tmp_path):
    assert bw(Config(), tmp_path, {"OPENROUTER_API_KEY": "k"}) is None


def test_build_watcher_missing_key_logs(tmp_path, caplog):
    assert bw(auto_cfg(), tmp_path, {}) is None
    assert "OPENROUTER_API_KEY" in caplog.text


def test_build_watcher_dry_run_without_key(tmp_path):
    w = bw(auto_cfg(dry_run=True), tmp_path, {})
    assert w is not None and w.ocr is None


def test_build_watcher_with_key(tmp_path):
    w = bw(auto_cfg(model="x/y"), tmp_path, {"OPENROUTER_API_KEY": "k"})
    assert isinstance(w.ocr, OpenRouterOcr) and w.ocr.model == "x/y" and w.ocr.api_key == "k"


def test_build_watcher_custom_env_name(tmp_path):
    w = bw(auto_cfg(api_key_env="MY_KEY"), tmp_path, {"MY_KEY": "z"})
    assert w.ocr.api_key == "z"


def test_build_watcher_logs_results(tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    w = bw(auto_cfg(), tmp_path, {"OPENROUTER_API_KEY": "k"})
    w.on_result(3, OcrResult("line1\nline2", "m", 1, 2))
    assert "ocr page 3: line1 ⏎ line2" in caplog.text


# --- final review fixes ---

def test_new_sheet_without_qr_yet_does_not_mark_old_page_unsent():
    cam, pages = Cam(small_page(["page five"])), Pages(5)
    w, clock, docs = make(cam=cam, pages=pages, min_interval_s=0)
    assert "sent" in ticks(w, clock, 4)
    cam.frame, pages.n = small_page(["a totally new sheet"]), None   # QR not read yet
    assert "no_qr" in ticks(w, clock, 3)
    pages.n = 6
    ticks(w, clock, 3)
    assert docs.live() == 6 and docs.get(5)["unsent"] is False


def test_hand_lifted_from_qr_clears_pending():
    cam, pages = Cam(small_page(["page five"])), Pages(5)
    w, clock, docs = make(cam=cam, pages=pages, min_interval_s=0)
    ticks(w, clock, 4)
    cam.frame, pages.n = small_page(["page five", "new words"]), None
    ticks(w, clock, 3)
    assert w.status()["pending"] is True
    cam.frame, pages.n = small_page(["page five"]), 5                  # writing not there after all; QR visible
    out = ticks(w, clock, 3)
    assert out[-1] == "unchanged" and w.status()["pending"] is False


def test_unsent_flag_clears_when_returning_page_is_unchanged():
    a = small_page(["page five words"])
    cam, pages = Cam(a), Pages(5)
    w, clock, docs = make(cam=cam, pages=pages, min_interval_s=0)
    ticks(w, clock, 4)
    cam.frame, pages.n = small_page(["six six"]), 6
    ticks(w, clock, 4)
    docs.set_unsent(5, True)
    cam.frame, pages.n = a, 5
    out = ticks(w, clock, 4)
    assert out[-1] == "unchanged" and docs.get(5)["unsent"] is False


def test_switch_during_cooldown_keeps_new_page_pending_without_extra_call():
    cam, pages, ocr = Cam(small_page(["page five"])), Pages(5), Ocr()
    w, clock, docs = make(cam=cam, pages=pages, ocr=ocr, min_interval_s=60)
    ticks(w, clock, 4)
    cam.frame, pages.n = small_page(["sheet six text"]), 6
    out = ticks(w, clock, 4)
    assert "cooldown" in out and ocr.calls == 1 and w.status()["pending"] is True and docs.live() == 6
