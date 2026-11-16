# Numbered Pages + Document Portal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pre-printed sheets carry a QR page number. The watcher reads the QR code locally to know which page is live, files each OCR result against that page, and a debug web portal shows the live page (auto-updating) and the archive.

**Architecture:**
- **New modules:** `pageid.py` (QR payload, detection, printable sheets), `documents.py` (per-page store on disk, atomic writes) and a portal in `web.py` (JSON API plus two HTML pages).
- **Gate:** gains a swappable per-page "already sent" reference.
- **Watcher:** reads the QR code on every still frame, switches the live page on a new code, sends only when the code is readable in that frame, and marks pages that leave with unsent changes.

**Tech Stack:** Python 3.11, OpenCV 4.9 (`cv2.QRCodeEncoder`, `cv2.QRCodeDetector`), Pillow (PDF only; the Pi has 9.4, so no Pillow ≥10 APIs), Flask, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-document-portal-design.md`

## Global Constraints

- No new pip/apt dependencies. Draw text on the sheets with `cv2.putText`; Pillow only saves the PDF (Pillow 9.4 compatible).
- QR payload: `CCP:<n>`, where `n` is a positive integer. Anything else decodes to "not a page".
- Sheets are US Letter at 300 dpi (2550x3300 px). The QR code is 25 mm square (295 px) in the top-right corner, 12 mm from the top and right edges. "Page <n>" is printed large in the top-left, also 12 mm in. The top-centre 100 mm stays empty for the clip.
- A frame is sent for OCR **only if a page QR decodes in that same frame**.
- A page is never archived because its QR code isn't seen. Only a *different* decoded page changes the live page.
- Documents live under `<output.dir>/pages/`. They are the source of truth; the watcher no longer writes `ocr/`.
- Writes to JSON files are atomic: write `*.tmp`, then `os.replace`.
- `/` becomes the Documents portal and the existing camera page moves to `/camera`.
- The OCR prompt gains: `Ignore the printed page number and the QR code at the top of the sheet.`
- Existing behaviour stays: the gate checks, cooldown, hourly cap, backoff, dry run, and manual capture. All existing tests keep passing, updated only where an interface in this plan changes them.

## Review Focus

1. **A hand resting on the QR code while the writing is still and changed**: no OCR call, no archive, status `qr_visible: false`, and the page is marked pending. Tested in Task 4.
2. **Another QR code on the desk, or a sheet from a different system**: ignored as "not a page". Tested in Task 1.
3. **Power loss mid-write, or a corrupt `page.json`**: the portal lists the remaining pages and logs the bad one, instead of returning a 500. A restart restores the live page. Tested in Tasks 2 and 5.
4. **Odd portal URLs** (unknown page, non-numeric page, `../` in an image name): 404, never a file outside the page folder. Tested in Task 5.
5. **A page with hundreds of revisions**: the archive list and `/api/live` don't carry revision lists, so they stay small. Only the per-page endpoint does. Tested in Tasks 2 and 5.

---

### Task 1: QR page ids and printable sheets

**Files:**
- Create: `clipcam/pageid.py`
- Modify: `clipcam/__main__.py` (add the `sheets` subcommand)
- Test: `tests/test_pageid.py`

**Interfaces:**
- Produces:
  - `PREFIX = "CCP:"`
  - `payload(n: int) -> str`, which raises `ValueError` for `n < 1`
  - `parse_payload(s: str) -> int | None`
  - `read_page_id(bgr: np.ndarray) -> int | None`: uses `cv2.QRCodeDetector().detectAndDecodeMulti`, returns the first payload that parses, otherwise `None`, and never raises (catch `cv2.error`)
  - `render_sheet(n: int) -> np.ndarray`: a 3300x2550x3 uint8 white page laid out per Global Constraints. Build the QR with `cv2.QRCodeEncoder`, scale it with `INTER_NEAREST` to 295 px, and keep a white quiet zone.
  - `write_sheets_pdf(start: int, count: int, path: str | Path) -> None`: Pillow multi-page PDF at `resolution=300`
  - CLI: `python -m clipcam sheets --start N [--count 10] OUT.pdf`, printing `wrote pages N..N+count-1 to OUT.pdf`

- [ ] **Step 1: Write failing tests.**

```python
def test_payload_roundtrip():          assert parse_payload(payload(17)) == 17
def test_parse_rejects_foreign():      # "17", "CCP:", "CCP:0", "CCP:-3", "CCP:x", "https://x.y", "" -> None
def test_payload_rejects_zero():       with pytest.raises(ValueError): payload(0)
def test_render_sheet_decodes():       assert read_page_id(render_sheet(42)) == 42 and render_sheet(42).shape == (3300, 2550, 3)
def test_decodes_at_camera_scale():    # render_sheet(7) -> resize to 1080 px wide (~0.2 mm/px), GaussianBlur sigma 1, noise sigma 3
    assert read_page_id(img) == 7
def test_rotated_frame_decodes():      # same image rotated 90 deg (camera mount) still decodes 7
def test_foreign_qr_ignored():         # an image containing a QR of "hello" -> None
def test_no_qr_is_none():              assert read_page_id(np.full((500, 500, 3), 255, np.uint8)) is None
def test_clip_area_left_clear():       # render_sheet(3): the top-centre 100 mm x 25 mm band is all white
def test_sheets_pdf(tmp_path):         # write_sheets_pdf(5, 3, f): PIL opens it with n_frames == 3
def test_sheets_cli(tmp_path, capsys): # main(["sheets", "--start", "5", "--count", "2", str(f)]) -> file exists, output says "5..6"
```

- [ ] **Step 2: Run them to see them fail.** `.venv/bin/python -m pytest tests/test_pageid.py -q`. Expected: ImportError.
- [ ] **Step 3: Implement** `pageid.py` and the `sheets` subcommand. Wire the subcommand like `rectify`, but without loading the config.
- [ ] **Step 4: Run them.** Expected: all pass. If the camera-scale test fails, enlarge the QR's module scale (not the test), staying within 25 mm, or lower the error-correction level.
- [ ] **Step 5: Commit.** `feat: QR page ids and printable numbered sheets`

---

### Task 2: Document store

**Files:**
- Create: `clipcam/documents.py`
- Test: `tests/test_documents.py`

**Interfaces:**
- Produces `class DocumentStore(directory: Path, now: Callable[[], float] = time.time)` with:
  - `live() -> int | None`
  - `set_live(n: int) -> int | None`: makes `n` live and returns the previous live page. Creates the page (`first_seen`) if new. Persists to `live.json`.
  - `add_revision(n: int, bgr: np.ndarray, text: str) -> dict`: saves `<n>/<stamp>.jpg` (quality 85), appends `{"time", "image", "text"}` to `revisions`, sets `text` and `last_updated`, and clears `unsent`. Returns the summary.
  - `set_unsent(n: int, value: bool) -> None`
  - `get(n: int) -> dict | None`: the full page, including `revisions`
  - `summaries() -> list[dict]`: every page as `{"number", "first_line", "text", "first_seen", "last_updated", "unsent", "image"}`, newest `last_updated` first, **without** `revisions`. `image` is the latest revision's file name or `None`. `first_line` is the first non-empty line of `text`, up to 80 characters.
  - `image_path(n: int, name: str) -> Path | None`: `None` unless `name` is a bare `*.jpg` listed in that page's revisions
- On disk: `<dir>/<n>/page.json` and `<dir>/live.json`. Times are ISO-8601 local strings with seconds. The stamp format is `%Y%m%d-%H%M%S`, plus a `-k` suffix on collision.
- An unreadable `page.json` is logged as a warning and skipped by `summaries()`; `get()` returns `None` for it. An unreadable `live.json` reads as no live page.

- [ ] **Step 1: Write failing tests.**

```python
def test_new_store_empty(tmp_path):        # live() is None, summaries() == []
def test_set_live_returns_previous():      # set_live(3) -> None; set_live(4) -> 3; live() == 4
def test_set_live_persists():              # new DocumentStore(same dir).live() == 4
def test_add_revision_replaces_text():     # two revisions -> text == second; len(get(n)["revisions"]) == 2; both jpgs exist
def test_first_line_and_order():           # pages updated at t=1, t=2 (fake clock) -> summaries order [newest, older]; first_line skips blank lines
def test_summaries_exclude_revisions():    # 300 revisions -> "revisions" not in summaries()[0]
def test_unsent_flag():                    # set_unsent(3, True) -> summary unsent True; add_revision clears it
def test_corrupt_page_json_skipped(caplog):  # write "{" into 5/page.json -> summaries lists the others; get(5) is None; warning logged
def test_corrupt_live_json():              # live.json = "garbage" -> live() is None
def test_atomic_write_leaves_no_tmp():     # after operations, no *.tmp files remain
def test_image_path_validation():          # listed name -> Path; "../live.json", "x.png", unknown name, unknown page -> None
```

- [ ] **Step 2: Run them to see them fail.** Expected: ImportError.
- [ ] **Step 3: Implement** `documents.py`. Use a `threading.Lock` around writes, because the watcher thread writes while Flask threads read.
- [ ] **Step 4: Run them.** Expected: all pass.
- [ ] **Step 5: Commit.** `feat: per-page document store`

---

### Task 3: Swappable gate reference

**Files:**
- Modify: `clipcam/gate.py`
- Test: `tests/test_gate.py`

**Interfaces:**
- Produces:
  - `Gate.reference` (property, get and set): the "already sent" ink mask (`np.ndarray | None`). `mark_sent()` sets it from the last checked frame, as now.
  - `Gate.recheck() -> Decision`: re-evaluates the last `check()`'s frame against the current `reference`, with no change to motion or stability state. If the last check didn't reach the change comparison (dark, blank, moving, settling), it returns that same decision.

- [ ] **Step 1: Write failing tests.**

```python
def test_reference_swap_and_recheck():
    # send page A (mark_sent), save refA = g.reference; settle on page B -> "changed"
    # g.reference = None; recheck() -> "changed" (no reference)
    # mark_sent on B; check A again until still; g.reference = refA; recheck() -> "unchanged"
def test_recheck_keeps_moving_decision():  # last check was "moving" -> recheck().reason == "moving"
def test_reference_setter_none_means_send(): # after mark_sent, g.reference = None; next still frame -> "changed"
```

- [ ] **Step 2: Run them to see them fail.** Expected: AttributeError.
- [ ] **Step 3: Implement.** Keep the last `Decision` and mask; `recheck` reuses the change logic from `check`.
- [ ] **Step 4: Run** `tests/test_gate.py`. Expected: all pass, the old tests included.
- [ ] **Step 5: Commit.** `feat: per-page gate reference`

---

### Task 4: Page-aware watcher

**Files:**
- Modify: `clipcam/watcher.py`, `clipcam/ocr.py` (prompt line), `clipcam/__main__.py` (`build_watcher`; `cmd_serve` creates `DocumentStore(cfg.output_dir / "pages")` and passes it in)
- Test: `tests/test_watcher.py` (update the existing tests for the new constructor and sink, then add the new ones), `tests/test_ocr.py` (prompt assertion)

**Interfaces:**
- Consumes: `read_page_id` (Task 1), `DocumentStore` (Task 2), `Gate.reference` / `Gate.recheck` (Task 3)
- Produces:
  - `Watcher(camera, gate, ocr, documents: DocumentStore, cfg: AutoCfg, clock=time.monotonic, read_page=read_page_id, on_result=None)`
    - `on_result(page: int, result: OcrResult)` is an optional hook, called after the revision is stored. `build_watcher` uses it to log.
    - The sink is now `documents.add_revision(page, img, result.text)` inside the watcher.
  - `Watcher.status() -> dict`: `{"live": int | None, "qr_visible": bool | None, "pending": bool, "last": str}`. `qr_visible` is `None` until a still frame has been checked. `last` is the last `step()` outcome.
  - New `step()` outcomes: `"no_qr"` (still, changed frame without a readable page QR) and `"switched"` (logged, never returned; the step continues after a switch).
  - `build_watcher(cfg, camera, store, documents, env)`, which gains the `documents` parameter.
- Prompt in `ocr.py`: append ` Ignore the printed page number and the QR code at the top of the sheet.` to `PROMPT`.

`step()` order:

```
frame -> prepare -> d = gate.check(img)
if d.reason in (dark, blank, moving, settling): return d.reason
page = read_page(img); qr_visible = page is not None
if page is None:
    if d.send: pending = True; return "no_qr"
    return d.reason
if page != documents.live():
    old = documents.set_live(page)            # also creates the page
    if old is not None:
        refs[old] = gate.reference            # keep old page's reference
        if pending: documents.set_unsent(old, True)
    gate.reference = refs.get(page); pending = False
    d = gate.recheck(); log "page <old> -> <page>"
if not d.send: return d.reason
(backoff / cooldown / rate_limited as today) -> on block: pending = True; return the reason
dry run -> gate.mark_sent(); refs[page] = gate.reference; pending = False; return "dry_run"
extract -> OcrError: pending = True; (backoff as today); return "ocr_error"
success -> gate.mark_sent(); refs[page] = gate.reference; pending = False
           documents.add_revision(page, img, result.text); on_result(page, result); return "sent"
```

On construction, `refs` is empty. The live page comes from `documents.live()`, so a restart keeps it live, and its first still frame sends once.

- [ ] **Step 1: Update the existing watcher tests** to the new constructor (`documents=DocumentStore(tmp_path)`, `read_page=lambda img: 1`) and check results through `documents.get(1)` instead of the sink list. Then add the new failing tests:

```python
def test_first_qr_makes_page_live():          # read_page -> 5; after send: documents.live() == 5, get(5)["text"] == "hello"
def test_no_qr_never_sends_or_archives():     # live 5; read_page -> None for 6 still changed ticks: outcomes all "no_qr",
                                              # ocr.calls == 0, live() still 5, status()["qr_visible"] is False, ["pending"] True
def test_switch_archives_and_marks_unsent():  # page 5 sent; new ink blocked by cooldown (pending); read_page -> 6:
                                              # live() == 6, get(5)["unsent"] True, page 6 gets its own revision
def test_return_page_resumes_reference():     # 5 sent, switch to 6 and send, back to 5 with the SAME frame as 5's last send:
                                              # outcome "unchanged", no new call; live() == 5
def test_return_page_with_new_writing_sends():# back to 5 with extra writing -> "sent", unsent cleared
def test_restart_keeps_live_page():           # documents.set_live(9) before constructing; read_page -> 9; first still frame -> "sent" to page 9
def test_status_shape():                      # keys {"live", "qr_visible", "pending", "last"}
def test_prompt_mentions_page_number():       # in test_ocr.py: "Ignore the printed page number" in PROMPT
```

- [ ] **Step 2: Run them to see the new ones fail.** `.venv/bin/python -m pytest tests/test_watcher.py tests/test_ocr.py -q`. Expected: the new tests fail; the updated old tests fail only on the constructor.
- [ ] **Step 3: Implement** the watcher changes, the prompt line and the `build_watcher` change. `build_watcher`'s `on_result` logs `ocr page <n>: <first 80 chars, newlines as ⏎> (tokens in=… out=…)`.
- [ ] **Step 4: Run the full suite.** `.venv/bin/python -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Commit.** `feat: watcher files OCR results by QR page and tracks the live page`

---

### Task 5: Document portal

**Files:**
- Modify: `clipcam/web.py`, `clipcam/__main__.py` (pass the `DocumentStore` from Task 4 and the watcher to `create_app`)
- Modify: `tests/test_app.py` (the camera page is now at `/camera`)
- Test: `tests/test_portal.py`
- Modify: `README.md` (a "Numbered pages and the document portal" section: printing sheets, the start number rule, what live/archived mean, the "QR not visible" and "unsent changes" markers)

**Interfaces:**
- Consumes: `DocumentStore` (Task 2), `Watcher.status()` (Task 4)
- Produces: `create_app(service, store, gallery_size=24, documents: DocumentStore | None = None, watcher=None)` with these routes:
  - `GET /`: Documents page (layout A). If `documents is None`, it shows a short notice and a link to `/camera`.
  - `GET /camera`: the existing camera page (unchanged HTML, except the nav)
  - `GET /api/live`: `{"status": watcher.status() or null, "page": summary-of-live or null}`
  - `GET /api/pages`: `[summary, ...]` excluding the live page (the archive), newest first
  - `GET /api/pages/<int:n>`: full page including `revisions`, or 404
  - `GET /pages/<int:n>`: per-page HTML view (fetches `/api/pages/<n>`, shows full text, the latest photo, and the revision list with times and links to each image)
  - `GET /pages/<int:n>.txt`: `text/plain; charset=utf-8` attachment `page-<n>.txt`, or 404
  - `GET /pages/<int:n>/img/<name>`: image via `documents.image_path`, or 404
- Documents page behaviour (layout A from the spec):
  - **Nav:** "Documents | Camera".
  - **Live box:** green border, `LIVE` tag, `Page <n>`, "updated <relative time> · started <time>", text in a `<pre>`, a thumbnail of the latest image (links to the full image), and **Copy text** and **Download .txt** buttons. Show **QR not visible** (amber) when `status.qr_visible === false`, and **unsent changes** when the page's `unsent` flag is set.
  - **Archive:** a grid of cards (`<n>`, first line, last updated, an "unsent changes" tag if set), each linking to `/pages/<n>`.
  - Poll `/api/live` every 2 s and `/api/pages` every 10 s.
  - **Copy:** use `navigator.clipboard` when `window.isSecureContext`, else a hidden textarea plus `document.execCommand("copy")`, because the portal is plain http on the LAN.
  - **Escaping:** all text is set via `textContent`, never `innerHTML` with page text.

- [ ] **Step 1: Write failing tests** (Flask test client, a real `DocumentStore` in `tmp_path`, and a stub watcher with `status()`).

```python
def test_index_is_documents_and_camera_moved(): # "/" contains "Documents" and "Archive"; "/camera" contains "Capture"
def test_api_live_empty():                       # no pages -> {"status": {...}, "page": None}
def test_api_live_and_archive():                 # pages 3 (archived) and 4 (live) -> /api/live page.number == 4; /api/pages == [page 3 summary]
def test_api_page_detail_and_404():              # /api/pages/3 has revisions; /api/pages/99 -> 404; /pages/abc -> 404
def test_txt_download():                         # /pages/3.txt -> body == text, Content-Disposition has page-3.txt
def test_image_route_and_traversal():            # listed image -> 200 image/jpeg; /pages/3/img/..%2Flive.json and unknown name -> 404
def test_corrupt_page_does_not_break_portal():   # corrupt 5/page.json -> /api/pages 200 and lists 3
def test_archive_list_has_no_revisions():        # page with 300 revisions -> no "revisions" key in /api/pages or /api/live
def test_page_view_html():                       # /pages/3 -> 200 and contains "/api/pages/3"
def test_no_documents_store():                   # create_app(..., documents=None): "/" 200 with a link to /camera; /api/live page None
```

- [ ] **Step 2: Run them to see them fail.** `.venv/bin/python -m pytest tests/test_portal.py -q`. Expected: failures (routes missing).
- [ ] **Step 3: Implement** the routes and pages in `web.py`, the `/camera` move, and the `__main__` wiring. Fix `test_app.py`'s index assertion to `/camera`.
- [ ] **Step 4: Run the full suite.** `.venv/bin/python -m pytest -q`. Expected: all pass.
- [ ] **Step 5: Smoke test** with `--fake` and dry run: a folder holding `render_sheet(3)` with some `cv2.putText` writing. Then `curl /api/live` shows page 3 as live with `qr_visible: true`.
- [ ] **Step 6: Commit.** `feat: document portal with live page and archive`

---

### Task 6: Print, deploy and verify on the Pi

Manual, on real hardware. Code changes only if a test here exposes a bug; then reproduce it with a failing test first.

- [ ] **Step 1: Print sheets.** `python -m clipcam sheets --start 1 --count 10 sheets.pdf`. The user prints at 100% scale on Letter.
- [ ] **Step 2: Deploy.** `rsync -a --exclude .venv --exclude .git --exclude .pytest_cache --exclude .superpowers --exclude __pycache__ --exclude config.toml --exclude clipcam.env ./ raspberrypi.local:~/clipcam/`. Then on the Pi, set `min_interval_s = 3` in `config.toml` (anchored `sed` on `^min_interval_s`) and `sudo systemctl restart clipcam`.
- [ ] **Step 3: Check the QR reads in the real frame.** Put page 1 on the board. Expected log: `page None -> 1`, and `/api/live` shows `qr_visible: true`. If it doesn't read, check whether the QR is inside `roi` and in focus, and adjust `roi` before touching code.
- [ ] **Step 4: Walk the behaviours**, with the user writing and swapping sheets while you watch the portal (`http://raspberrypi.local:8080/`) and the journal:
  - Write on page 1, lift your hand: page 1's text appears within about 10 s.
  - Swap to page 2: page 1 moves to the archive and page 2 is live (empty text or a fresh revision).
  - Rest a hand over page 2's QR while still: "QR not visible" shows, and no call happens.
  - Put page 1 back: it becomes live again, its text is intact, and nothing is re-sent unless new writing was added.
- [ ] **Step 5: Commit any fixes** (each with its test) and `git commit -am "chore: ..."` for README or config tweaks found here.
