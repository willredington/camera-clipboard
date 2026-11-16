# Auto-capture + OCR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every ~2 s grab a frame from the clipboard camera, run cheap local checks on it, and only when the page has new, settled writing send it to an OpenRouter vision model that returns the text.

**Architecture:** A background `Watcher` thread inside the existing `clipcam serve` process (one process must own the camera). Each tick: grab a frame from the running video stream (no mode switch, no LED) → rotate + crop to a configured page rectangle → `Gate` (pure OpenCV) decides send / skip → if send, and cooldown and hourly cap allow it, `OpenRouterOcr` sends a downscaled JPEG and gets text back → a `sink` callable stores it. Where the text goes later is out of scope; the sink is the seam for that.

**Tech Stack:** Python 3.11 (Pi OS Bookworm 64-bit), picamera2 (apt), OpenCV + numpy (pip, as already pinned), requests (apt `python3-requests`), pytest.

**Spec:** No design doc. The user waived design questions; requirements come from chat: "take a photo every couple seconds, pass the image to a thing that extracts the text, check the image before passing it to a vision model (OpenRouter) so as not to overcharge." Decisions made with the user: reflash the Pi to Bookworm; no ArUco markers, so checks run on a fixed crop of the raw frame.

## Global Constraints

- Target: Raspberry Pi 4 B, **Pi OS Lite 64-bit Bookworm** (Imager lists it under "Raspberry Pi OS (Legacy)"), hostname `raspberrypi`, user `will`, key `~/.ssh/id_ed25519`. Do not use Trixie: its apt numpy is 2.x and breaks the `numpy<2` pin in `requirements-pi.txt`.
- Camera: Camera Module 3 Wide (`imx708_wide`). Watch stream: 2304x1296 (its 56 fps full-FOV binned mode).
- The camera is mounted so the page appears rotated: page top is at the image's right edge, so `rotate_cw = 270` (verify in Task 6).
- No markers: do not call `Rectifier` from the watcher. The existing manual Capture button still requires markers; leave it unchanged.
- API key comes only from the environment variable named by `auto.api_key_env` (default `OPENROUTER_API_KEY`), never from `config.toml`. On the Pi it lives in `software/clipcam.env` (mode 600).
- OpenRouter endpoint: `POST https://openrouter.ai/api/v1/chat/completions`. Default model `google/gemini-2.5-flash` (config key; check the current id on openrouter.ai/models before deploying).
- Defaults: `interval_s = 2.0`, `min_interval_s = 10.0` between API calls, `max_per_hour = 60`, image sent at long side `max_side = 1600`, JPEG quality 85.
- The new code must not break existing tests (`tests/test_app.py`, `tests/test_rectify.py`). `test_example_config_matches_defaults` means every new default must also appear in `config.example.toml`.

## Review Focus

1. **Blank page or lights off**: the gate must say `blank` / `dark` and never call the API. Tested in Task 2.
2. **A hand moving over the page while writing**: no call until two consecutive still frames. Tested in Task 2.
3. **Page left untouched for hours**: exactly one call, then `unchanged` forever. Tested in Tasks 2 and 4.
4. **OpenRouter down, bad key (401), rate limit (429), or timeout**: the watcher logs it, backs off (30 s doubling to 600 s), does not crash, and does not mark the page as sent, so it retries after the backoff. Tested in Tasks 3 and 4.
5. **`OPENROUTER_API_KEY` missing**: the server still starts and serves the web page, logs one clear error, and the watcher stays off (unless `dry_run`). Tested in Task 5.

---

### Task 1: Reflash the Pi, init git, baseline green

Mostly manual (the user does the flash). The deliverable is a Bookworm Pi with the existing app installed and the laptop test suite passing under git.

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: User flashes the SD card.** Raspberry Pi Imager → Raspberry Pi 4 → Raspberry Pi OS (Legacy) **Lite 64-bit (Bookworm)**. In OS customisation: hostname `raspberrypi`, user `will`, Wi-Fi, enable SSH with public key = contents of `~/.ssh/id_ed25519.pub`.
- [ ] **Step 2: Verify the OS from the laptop.**
  Run: `ssh-keygen -R raspberrypi.local; ssh -o StrictHostKeyChecking=accept-new raspberrypi.local 'grep VERSION_CODENAME /etc/os-release; getconf LONG_BIT; python3 --version; rpicam-hello --list-cameras | head -3'`
  Expected: `VERSION_CODENAME=bookworm`, `64`, `Python 3.11.x`, and a line containing `imx708_wide`.
- [ ] **Step 3: Init git in `software/` and add `.gitignore`** with `.venv/`, `__pycache__/`, `.pytest_cache/`, `config.toml`, `clipcam.env`.
  Run: `git init && git add -A && git commit -m "chore: import existing clipcam"`
- [ ] **Step 4: Laptop baseline.** Add `requests` to `requirements-dev.txt`, then run the README's test commands.
  Run: `.venv/bin/python -m pytest -q`
  Expected: all existing tests pass.
- [ ] **Step 5: Commit.** `git add requirements-dev.txt .gitignore && git commit -m "chore: dev deps for OCR client"`

---

### Task 2: Gate (local "is it worth sending?" check)

**Files:**
- Create: `clipcam/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Produces:
  - `@dataclass GateCfg`: `work_side: int = 400`, `min_brightness: float = 40.0`, `min_ink_frac: float = 0.002`, `motion_thresh: float = 0.02`, `stable_frames: int = 2`, `min_change_frac: float = 0.0015`, `tolerance_px: int = 3`
  - `@dataclass Decision`: `send: bool`, `reason: str` (one of `"dark" "blank" "moving" "settling" "unchanged" "changed"`), `brightness: float`, `ink: float`, `motion: float | None`, `change: float | None`
  - `class Gate(cfg: GateCfg)` with `check(bgr: np.ndarray) -> Decision` (updates the previous-frame state) and `mark_sent() -> None` (the ink mask from the last `check` becomes the reference)
  - `prepare(bgr: np.ndarray, rotate_cw: int, roi: list[float]) -> np.ndarray`: rotate by 0/90/180/270 degrees clockwise (`cv2.rotate`), then crop `roi = [x, y, w, h]` given as fractions of the rotated frame. Raises `ValueError` on another angle or an empty ROI.

Algorithm (not determined by the signatures):

```
gray   = downscale(bgr to long side work_side), to gray, GaussianBlur 5x5
bright = mean(gray)                                    -> "dark" if < min_brightness
mask   = adaptiveThreshold(gray, MEAN_C, BINARY_INV, block 25, C 15), then morph-open 2x2
ink    = mean(mask > 0)                                -> "blank" if < min_ink_frac
motion = mean(|gray - prev_gray|) / 255 (None on first frame); prev_gray = gray
         motion > motion_thresh -> stable_count = 0, "moving"
         else stable_count += 1; stable_count < stable_frames -> "settling"
change = None if no reference, else
         (count(mask & ~dilate(ref, tolerance_px)) + count(ref & ~dilate(mask, tolerance_px))) / mask.size
         reference exists and change < min_change_frac -> "unchanged"
otherwise -> send=True, "changed"
```

Run the checks in that order. `dark` and `blank` also reset `stable_count`. The first frame never counts as stable (`motion` is None, so it reports `settling`).

- [ ] **Step 1: Write failing tests.** Build synthetic pages with numpy/cv2: `page(text_lines)` gives a 1296x2304 light-grey (200) image with black `cv2.putText` lines; `dark()` gives all 10s.

```python
def test_dark_frame_skipped():            assert run(dark()).reason == "dark"
def test_blank_page_skipped():            # page with no text, checked 3x
    assert all(d.reason == "blank" for d in run_n(page([]), 3))
def test_first_send_needs_stable_frames():
    ds = run_n(page(["hello"]), 3); assert [d.reason for d in ds] == ["settling", "settling", "changed"]
def test_moving_hand_blocks_send():       # alternate page / page with a big dark blob (hand) at a different spot
    assert not any(d.send for d in alternating_frames(6))
def test_unchanged_after_mark_sent():     # send, mark_sent, then 5 more identical frames
    assert [d.reason for d in after_send(page(["hello"]), 5)] == ["unchanged"] * 5
def test_new_word_triggers_change():      # sent "hello"; then "hello world" checked 3x
    assert ds[-1].reason == "changed" and ds[-1].change >= GateCfg().min_change_frac
def test_sensor_noise_is_unchanged():     # same page + gaussian noise sigma 3 per frame, after mark_sent
    assert all(d.reason == "unchanged" for d in ds)
def test_prepare_rotation_and_roi():
    img = np.zeros((100, 200, 3), np.uint8)
    assert prepare(img, 270, [0, 0, 1, 1]).shape == (200, 100, 3)
    assert prepare(img, 0, [0.5, 0, 0.5, 0.5]).shape == (50, 100, 3)
    with pytest.raises(ValueError): prepare(img, 45, [0, 0, 1, 1])
```

- [ ] **Step 2: Run them to see them fail.** `.venv/bin/python -m pytest tests/test_gate.py -v`. Expected: ImportError for `clipcam.gate`.
- [ ] **Step 3: Implement `clipcam/gate.py`** as specified. Pure OpenCV/numpy, no picamera2 import (same rule as `rectify.py`).
- [ ] **Step 4: Run the tests.** `.venv/bin/python -m pytest tests/test_gate.py -v`. Expected: all pass. If the noise or new-word tests fight each other, tune the `GateCfg` defaults (not the tests), and keep the noise test passing with sigma 3.
- [ ] **Step 5: Commit.** `git add clipcam/gate.py tests/test_gate.py && git commit -m "feat: local gate deciding when a frame is worth OCR"`

---

### Task 3: OpenRouter OCR client

**Files:**
- Create: `clipcam/ocr.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Produces:
  - `class OcrError(RuntimeError)` with attribute `status: int | None`
  - `@dataclass OcrResult`: `text: str`, `model: str`, `prompt_tokens: int | None`, `completion_tokens: int | None`
  - `class OpenRouterOcr(api_key: str, model: str, max_side: int = 1600, timeout_s: float = 60.0, session: requests.Session | None = None)` with `extract(bgr: np.ndarray) -> OcrResult`
  - `PROMPT` (module constant, exact copy): `"Transcribe all text on this page exactly as written, preserving line breaks. Output only the transcribed text, with no commentary. If there is no text, output exactly NO_TEXT."`

Request: `session.post(URL, json=body, headers={"Authorization": f"Bearer {api_key}", "X-Title": "clipcam"}, timeout=timeout_s)`. The body has `model`, `temperature: 0`, and one user message whose content is `[{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]`. Downscale the image so its long side is at most `max_side` (never upscale) and encode as JPEG quality 85. Text = `choices[0].message.content.strip()`, with `"NO_TEXT"` mapped to `""`. Tokens come from `usage` when present.

- [ ] **Step 1: Write failing tests** using a fake session object that records the call and returns a stub response (`status_code`, `json()`, `text`).

```python
def test_request_shape():          # 3000x2000 input
    # url == URL; headers Authorization == "Bearer k"; body["model"] == "m"
    # decoded data-URL JPEG has max(h, w) == 1600
def test_parses_text_and_usage():  # content "  hi\nthere  ", usage {prompt_tokens: 10, completion_tokens: 3}
    assert r.text == "hi\nthere" and r.prompt_tokens == 10 and r.completion_tokens == 3
def test_no_text_sentinel():       assert ocr.extract(img).text == ""
def test_small_image_not_upscaled(): # 800x600 in -> 800x600 sent
def test_http_error_raises():      # 401 / 429 / 500
    with pytest.raises(OcrError) as e: ...; assert e.value.status == 429
def test_timeout_raises():         # session.post raises requests.Timeout
    with pytest.raises(OcrError) as e: ...; assert e.value.status is None
def test_malformed_body_raises():  # 200 with {"choices": []}
```

- [ ] **Step 2: Run them to see them fail.** `.venv/bin/python -m pytest tests/test_ocr.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement `clipcam/ocr.py`.** Wrap `requests.RequestException`, non-2xx status, and `KeyError`/`IndexError`/`ValueError` from parsing in `OcrError`. Include the first 200 characters of the response body in the message.
- [ ] **Step 4: Run the tests.** Expected: all pass.
- [ ] **Step 5: Commit.** `git add clipcam/ocr.py tests/test_ocr.py && git commit -m "feat: OpenRouter vision OCR client"`

---

### Task 4: Config, camera frame grab, OCR storage, Watcher loop

**Files:**
- Modify: `clipcam/config.py` (add `AutoCfg`, embed `GateCfg`)
- Modify: `config.example.toml` (new `[auto]` and `[gate]` sections with every default)
- Modify: `clipcam/camera.py` (watch stream + `capture_frame`)
- Modify: `clipcam/store.py` (`save_ocr`)
- Create: `clipcam/watcher.py`
- Test: `tests/test_watcher.py`; extend `tests/test_app.py` only if a config test needs it

**Interfaces:**
- Consumes: `Gate`, `GateCfg`, `Decision`, `prepare` (Task 2); `OpenRouterOcr`, `OcrResult`, `OcrError` (Task 3)
- Produces:
  - `@dataclass AutoCfg` in `config.py`: `enabled: bool = False`, `dry_run: bool = False`, `interval_s: float = 2.0`, `rotate_cw: int = 0`, `roi: list[float] = [0.0, 0.0, 1.0, 1.0]` (use `field(default_factory=...)`), `min_interval_s: float = 10.0`, `max_per_hour: int = 60`, `model: str = "google/gemini-2.5-flash"`, `api_key_env: str = "OPENROUTER_API_KEY"`, `max_side: int = 1600`, `backoff_s: float = 30.0`, `backoff_max_s: float = 600.0`
  - `Config.auto: AutoCfg`, `Config.gate: GateCfg`
  - `CameraCfg.watch_width: int = 2304`, `CameraCfg.watch_height: int = 1296`. The PiCamera preview configuration's main stream uses this size; `preview_jpeg()` downscales to `preview_width` before encoding.
  - `PiCamera.capture_frame() -> np.ndarray` and `FakeCamera.capture_frame() -> np.ndarray`. On the Pi it returns the current main-stream frame under the existing lock (blocking: it waits while a still capture runs). The fake returns the current image and advances, like `capture_still`.
  - `Store.save_ocr(bgr: np.ndarray, text: str) -> str` writes `<dir>/ocr/<name>.jpg` (quality 85) and `<dir>/ocr/<name>.txt` (UTF-8), with the name made the same way as `save_page` (timestamp, plus a `-n` suffix on collision), and returns the name.
  - `class Watcher(camera, gate: Gate, ocr: OpenRouterOcr | None, sink: Callable[[np.ndarray, OcrResult], None], cfg: AutoCfg, clock: Callable[[], float] = time.monotonic)` with
    - `step() -> str`: one tick; returns one of the gate reasons (`dark blank moving settling unchanged`) or `"cooldown" "rate_limited" "backoff" "dry_run" "sent" "ocr_error" "camera_error"`
    - `run(stop: threading.Event) -> None`: calls `step()` and then `stop.wait(cfg.interval_s)` until `stop` is set. It never lets an exception escape: catch, log, and continue.

`step()` order:
1. `capture_frame()`; on exception return `camera_error`.
2. `prepare(frame, cfg.rotate_cw, cfg.roi)`, then `gate.check`. If not send, return the reason.
3. If in error backoff (now < `next_allowed`), return `backoff`. If `now - last_call < min_interval_s`, return `cooldown`. If there are `max_per_hour` calls in the trailing 3600 s, return `rate_limited`.
4. If `dry_run` (or `ocr is None`), call `gate.mark_sent()` and return `dry_run`. Log the Decision so thresholds can be tuned without spending money.
5. Call `ocr.extract`. On `OcrError`: set `next_allowed = now + backoff`, double the backoff up to `backoff_max_s`, do **not** call `mark_sent`, log, and return `ocr_error`. On success: reset the backoff, `gate.mark_sent()`, call `sink(prepared, result)`, record the call time, and return `sent`.

Failed calls count toward the cooldown and the hourly cap, because they may still be billed.

- [ ] **Step 1: Write failing tests** with a scripted fake camera (a list of frames), a stub gate (or the real Gate on Task 2's synthetic pages), a fake OCR (returns or raises from a script), a list-appending sink, and a fake clock you advance by hand.

```python
def test_unchanged_page_sends_once():      # same page for 20 ticks, clock += 2 per tick
    assert outcomes.count("sent") == 1 and outcomes[-1] == "unchanged"
def test_cooldown_between_sends():          # gate always says changed; ticks 2 s apart
    assert outcomes[:3] == ["sent", "cooldown", "cooldown"]   # min_interval_s 10
def test_hourly_cap():                       # max_per_hour=3, min_interval_s=0, gate always changed
    assert outcomes.count("sent") == 3 and "rate_limited" in outcomes
def test_ocr_error_backs_off_then_retries(): # OCR raises once, then succeeds; same changed page
    # first call "ocr_error", then "backoff" until clock passes +30 s, then "sent"; sink called once
def test_backoff_doubles_and_caps():         # OCR always raises: gaps 30, 60, 120, ... capped at 600
def test_dry_run_never_calls_ocr():          # dry_run=True: fake OCR call count 0; outcome "dry_run" once, then "unchanged"
def test_camera_error_does_not_crash():      # capture_frame raises -> "camera_error"; next tick works
def test_run_stops_and_survives_exceptions(): # gate.check raises on tick 1; run() keeps going; stop.set() ends it
def test_store_save_ocr(tmp_path):           # writes ocr/<name>.txt with the exact text and ocr/<name>.jpg
def test_config_auto_and_gate_sections(tmp_path): # TOML [auto] rotate_cw=270, roi=[0.1,0,0.8,1], [gate] min_ink_frac=0.01 load
```

- [ ] **Step 2: Run them to see them fail.** `.venv/bin/python -m pytest tests/test_watcher.py -v`. Expected: ImportError.
- [ ] **Step 3: Implement the config, camera, store, and watcher changes.** In `config.example.toml`, set `rotate_cw = 0` with a comment saying this mount needs 270, so the defaults test keeps passing. `_apply` handles nested dataclasses already. Check that a TOML array lands as `list[float]`.
- [ ] **Step 4: Run the whole suite.** `.venv/bin/python -m pytest -q`. Expected: all pass, including `test_example_config_matches_defaults`.
- [ ] **Step 5: Commit.** `git add -A clipcam tests config.example.toml && git commit -m "feat: auto-capture watcher with cooldown, hourly cap and error backoff"`

---

### Task 5: Wire into `serve`, install, docs

**Files:**
- Modify: `clipcam/__main__.py`
- Modify: `install.sh`, `clipcam.service`, `README.md`
- Test: `tests/test_watcher.py` (`build_watcher` tests)

**Interfaces:**
- Consumes: everything from Task 4
- Produces: `build_watcher(cfg: Config, camera, store: Store, env: Mapping[str, str]) -> Watcher | None` in `__main__.py`

Behaviour:
- `auto.enabled` false → `None`.
- Key missing and not `dry_run` → log `ERROR auto-capture disabled: set <api_key_env> (see clipcam.env)` and return `None`.
- Key missing and `dry_run` → Watcher with `ocr=None`.
- Otherwise a Watcher with `OpenRouterOcr(key, cfg.auto.model, cfg.auto.max_side)`.
- The default sink calls `store.save_ocr` and logs `INFO ocr <name>: <first 80 chars of text, newlines as ⏎>` plus the token counts.
- `cmd_serve` starts `watcher.run(stop)` in a daemon thread before `app.run`, and calls `stop.set()` in the existing `finally` before closing the camera. It works with `--fake` too.

- [ ] **Step 1: Write failing tests.**

```python
def test_build_watcher_disabled():          assert build_watcher(cfg_disabled, cam, store, {}) is None
def test_build_watcher_missing_key_logs(caplog):  # enabled, not dry_run, env {}
    assert build_watcher(...) is None and "OPENROUTER_API_KEY" in caplog.text
def test_build_watcher_dry_run_without_key():     assert build_watcher(...).ocr is None
def test_build_watcher_with_key():                assert isinstance(build_watcher(..., {"OPENROUTER_API_KEY": "k"}).ocr, OpenRouterOcr)
```

- [ ] **Step 2: Run them to see them fail.** Expected: ImportError for `build_watcher`.
- [ ] **Step 3: Implement** `build_watcher` and the thread start/stop in `cmd_serve`.
- [ ] **Step 4: Install bits.**
  - `install.sh`: add `python3-requests` to the apt line. If `clipcam.env` is missing, create it with the line `OPENROUTER_API_KEY=` and `chmod 600`.
  - `clipcam.service`: add `EnvironmentFile=-@DIR@/clipcam.env` under `[Service]`.
  - `README.md`: an "Auto-capture + OCR" section covering what the gate checks, every `[auto]`/`[gate]` key, tuning with `dry_run`, where results land (`~/captures/ocr/`), cost controls, and the note that the manual Capture button still needs markers.
- [ ] **Step 5: Run the suite.** `.venv/bin/python -m pytest -q`. Expected: all pass.
- [ ] **Step 6: Commit.** `git add -A && git commit -m "feat: run auto-capture watcher inside clipcam serve"`

---

### Task 6: Deploy and tune on the Pi

Manual verification on real hardware. Code changes here are config only, unless tuning shows a gate default is wrong; then fix the default in `gate.py` and the example config, re-run the tests, and commit.

- [ ] **Step 1: Copy and install.** `rsync -a --exclude .venv --exclude .git --exclude .pytest_cache ./ raspberrypi.local:~/clipcam/ && ssh raspberrypi.local 'cd ~/clipcam && ./install.sh'`. Expected: ends with `Installed.` and `systemctl is-active clipcam` prints `active`.
- [ ] **Step 2: Find the rotation and ROI.** Take a still from the web preview, or stop the service and run `rpicam-still`. Confirm `rotate_cw = 270` makes the text upright. Set `roi` to the paper's rectangle in the rotated frame, with about 3% margin. Restart the service.
- [ ] **Step 3: Dry run.** Set `[auto] enabled = true, dry_run = true`, restart, and run `journalctl -u clipcam -f`. Check: a blank page logs `blank`; writing with your hand moving logs `moving`; lifting your hand gives one `dry_run` within about 4 s; then `unchanged` repeats; adding one word gives another `dry_run`; room lights off logs `dark`. Adjust `[gate]` thresholds until this holds.
- [ ] **Step 4: Live.** Put the key in `~/clipcam/clipcam.env`, set `dry_run = false`, restart. Write "hello world" and expect `ocr <name>: hello world` in the log, with `~/captures/ocr/<name>.txt` containing it. Leave the page for 5 minutes: no further `ocr` lines.
- [ ] **Step 5: Commit tuned defaults, if any.** `git commit -am "chore: tune gate defaults on real hardware"`
