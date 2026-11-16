# Numbered pages + document portal: design

## Intent

The clipboard already auto-captures new writing and transcribes it with a vision model (`clipcam/watcher.py`). The results currently land as loose timestamped text files in `~/captures/ocr/`. The user wants to **see the output as documents**:

- the **live document**: the sheet currently on the clipboard
- the **archived documents**: sheets that were on it earlier

The portal is a **debugging tool**, not the production UI. It should make it easy to see what the system thinks is on the page and why: which page it believes is live, what text it holds, and the photo each transcription came from. A later production version will drop things like the original photo, so don't polish this beyond a clear, working debug view.

## Decisions made with the user

- **One live document at a time.** It is the sheet physically on the clipboard.
- **Writing is in pen, so changes on a sheet are always additive.**
- **Pages are pre-printed and numbered.** Each sheet carries a printed page number plus a **QR code** encoding that number. The Pi reads the QR code locally with OpenCV to know which page is on the board. The QR guarantees uniqueness, so no ink-vanishing heuristics are needed to detect a new page.
- **Old pages can become live again.** Putting page 12 back on the clipboard makes it live again and its text keeps updating. Whatever page was live becomes archived.
- **Portal layout "A"** (chosen from mockups): one scrolling page.
  - **Top:** the live page in a large box: page number, "LIVE" tag, last-updated and first-seen times, its current text, a thumbnail of the photo the text came from, and **Copy text** and **Download .txt** buttons. It updates on its own without a reload.
  - **Below:** an archive grid of cards, one per archived page: page number, first line of text, last-updated date. Clicking a card opens that page's own view (full text and photo).
  - **Nav:** "Documents" (this page) and "Camera" (the existing preview/capture page).

## Goals

1. Print numbered QR sheets from a command.
2. The watcher knows which page is live, from the QR code, without any API call. That includes a freshly placed blank page.
3. Each OCR result is stored against the page it came from. A page's current text is its latest transcription. Each call transcribes the whole sheet, so the latest result replaces the text; it doesn't append.
4. The portal shows the live page (auto-updating) and the archive (layout A), plus a per-page view.
5. It all survives a service restart: documents persist on disk, and the last live page is remembered.

## Non-goals

- Production polish, authentication, multi-user. It stays on the home network like the existing page, with no login.
- Editing text in the portal, search, tags, export beyond a per-page .txt download.
- Handling sheets without a QR code (unnumbered paper). See "No QR visible" below for what happens.
- Using the ArUco board markers or page rectification.

## Anti-goals (would make this a failure even if it "works")

- Extra paid or slow API calls just to work out *which* page is on the board. Page identity must come from the QR code locally.
- A page being archived because a hand or pen briefly covered its QR code.
- The portal making the watcher slower, or holding the camera lock.
- Losing documents on a restart, or a restart corrupting the store.

## Behaviour

**Page identity:**
- On each tick, after the gate's motion and settling checks have passed (the frame is still), look for a QR code in the cropped, rotated frame at full stream resolution (2304x1296 before the crop), not the gate's 400 px working copy.
- A decoded code in the clipcam format gives the page number.
- **Different from the current live page:** the new page becomes live, the previous one is archived, and the gate's "already sent" reference switches to the new page's last-sent reference (or none if it has none in this run).
- **Same page:** carry on as now.
- **No QR visible** (covered, out of focus, or a sheet without one): keep the current live page. Never archive just because the code isn't seen.
- **No live page yet** (first boot with no saved state, or no QR ever seen): nothing is live and nothing is sent. See the next section.

**Only send when the QR is readable in that frame (user decision):**
- A frame is sent for OCR only if the page's QR decodes in **that same frame**. If the gate says "changed" but no QR decodes, don't send. Wait for a frame where it does.
- The portal shows a visible "QR not visible" status on the live page while that's happening, so a smudged or out-of-focus code is obvious.
- Nothing is lost, because the next frame with a readable code carries all the writing. Accepted cost: a page whose code never reads is never transcribed until the code is fixed.
- This replaces the "unnumbered" document idea above. With this rule, OCR text always belongs to a numbered page, so drop the unnumbered document.

**Writing not yet sent when the page is swapped (user decision, adjusted):**
- The system can't foresee a swap, so there's no "last frame" of the outgoing page to send.
- Unsent writing is sent on the first still, QR-readable frame that the cooldown, cap and backoff allow. That's the current behaviour.
- If a page leaves the board while it has changes the gate saw but that weren't sent, it's marked **"unsent changes"**. The portal shows that on its card and page view. When it becomes live again, the normal gate comparison against its last-sent reference sends those changes.
- The per-page "already sent" references are kept in memory and lost on restart, so after a restart a returning page is simply sent once.
- To shrink the window, the user's Pi config sets `min_interval_s = 3` (their OpenAI BYOK makes calls effectively free). The code default stays 10.

**OCR on a fresh page:**
- A newly placed sheet already has printed ink (the number and the QR), so the gate will usually send it once. That's acceptable.
- The prompt must tell the model to ignore the printed page number and QR code, so a fresh page transcribes to empty text instead of "Page 17".

**Storage:**
- One directory per page under the output folder. It holds the page's metadata (number, first seen, last updated, current text) and every transcription revision, each with the image that was sent.
- Keep all revisions: this is a debugging tool and the history is useful.
- The live page number is persisted so a restart comes back with the same live page.
- Writes must not leave a half-written file that breaks the portal or the next start.

**Portal:** served by the existing Flask app on the same port. It is read-only apart from the buttons above. It polls for the live page every couple of seconds.

**Printing sheets:**
- A CLI subcommand writes a PDF of numbered US Letter pages (the board is designed around Letter) starting at a given number.
- Each page has the big human-readable number and the QR code, in the top margin, clear of the clip (centre top, ~100 mm wide).
- The QR code is about 25 mm square. At the camera's ~0.2 mm/px it's ~120 px across in the stream, comfortably decodable.

## Constraints

- Runs on the existing Pi setup: Pi 4, Bookworm 64-bit, Python 3.11, apt picamera2, pip OpenCV 4.9 (`cv2.QRCodeDetector` and `cv2.QRCodeEncoder` are available; check on the laptop: encode then decode round-trips). No new pip dependencies unless unavoidable; Pillow is available for the PDF.
- Must keep the existing behaviour and tests: the gate, cooldown, hourly cap, backoff, dry run, and the existing camera/capture page.
- The current mount needs `rotate_cw = 270`, and the crop `roi = [0.14, 0.23, 0.79, 0.55]` covers just the paper. The QR code must sit inside that crop.
- The existing timestamped `~/captures/ocr/` output can stop or stay; the builder decides. Documents are the new source of truth.

## Calls made by the designer (not the user)

- QR rather than having the vision model read the printed number.
- Keep every revision and its photo (debugging).
- Letter paper, QR ~25 mm in the top margin.
- Portal polls every couple of seconds (no websockets).
- "unsent changes" marker instead of trying to send before a swap (a swap can't be foreseen).

## Left to the builder

- Exact QR payload format (short, prefixed so a random QR on the desk isn't mistaken for a page), error-correction level, and layout dimensions on the sheet.
- On-disk file format and names. The JSON metadata and atomic write approach.
- How the per-page in-memory "already sent" references are kept, and how the portal gets live data (JSON endpoints plus a small script, or similar).
- How print numbering avoids reusing numbers. At minimum, the CLI requires an explicit start number and the README explains it.
- Visual details of the portal, within layout A.
