# Camera clipboard: capture software

Captures a full-res still under the LED, finds the four ArUco markers, and warps the page to a flat top-down PNG
(optionally a multi-page PDF). Triggered from a web page on the Pi or an optional push button.

```
clipcam/rectify.py   pure OpenCV/numpy: marker detection, homography, warp (no picamera2/GPIO imports)
clipcam/camera.py    PiCamera (picamera2) and FakeCamera (images from a folder)
clipcam/hardware.py  Led / Button (gpiozero) and FakeLed
clipcam/service.py   LED on -> still -> rectify -> save, with a lock
clipcam/store.py     PNG + thumbnail, listing, PDF
clipcam/web.py       Flask app;  clipcam/__main__.py  CLI
tests/               pytest, synthetic board renderer (tests/synth.py)
```

## Setup on the Pi (Pi OS Lite 64-bit, Bookworm)

1. Enable nothing special: the camera is auto-detected on Bookworm. Copy this `software/` folder to the Pi.
2. `./install.sh`. It apt-installs `python3-picamera2 python3-flask python3-gpiozero python3-pil python3-numpy`,
   creates `.venv` with `--system-site-packages`, pip-installs `opencv-contrib-python-headless` (4.7 to 4.9) and `numpy<2`,
   copies `config.example.toml` to `config.toml`, installs and starts the `clipcam` systemd service.
   Bookworm's own `python3-opencv` is 4.6 and has no `cv2.aruco.ArucoDetector`, hence pip. The `<4.10` and `numpy<2`
   pins keep the venv on the same numpy ABI as the apt-built picamera2 dependencies.
3. Open `http://<pi-hostname>.local:8080/` from a phone or laptop on the same Wi-Fi. Logs: `journalctl -u clipcam -f`.
   There is **no authentication**: anyone on the network can trigger captures and download pages.

## Config (`config.toml`)

See `config.example.toml`; every key is optional. Most relevant:

| Key | Default | |
|---|---|---|
| `led.pin` | 18 | BCM number of the MOSFET gate (driven with gpiozero/lgpio) |
| `button.pin` | unset | BCM number; wire a push button from the pin to GND (internal pull-up) |
| `output.dpi` | 200 | output resolution (1700 x 2200 px for Letter) |
| `output.dir` | `~/captures` | PNGs, `.thumbs/`, and `failed-*.jpg` raw frames when rectification fails |
| `board.markers` | see below | marker centres in board mm |

After editing: `sudo systemctl restart clipcam`.

## Measuring the marker positions

Board coordinates: **origin at the board's top-left corner as seen from above with the clip at the top, x to the
right, y down, millimetres.** For each marker give the position of the **centre of the black square**.

1. Stick the markers flat, upright (as printed, not rotated), ids 0 TL, 1 TR, 2 BR, 3 BL. Print at 100% and check the
   black square measures 18.0 mm with calipers (if not, set `board.marker_size`).
2. With a steel rule or calipers measure from the board's left edge to the marker's left and right edges (centre x =
   the mean) and from the top edge to its top and bottom edges (centre y = the mean). Aim for +/-0.3 mm.
3. Put the values in `[board.markers]`. The defaults are the centres of the 25.3 mm top and bottom margins, 45 mm in
   from the side edges (clear of the 24 mm corner holders and the 100 mm clip base).
4. Check them: take a photo, then
   `.venv/bin/python -m clipcam --config config.toml rectify photo.jpg flat.png`. It prints the fit residual. Below about
   0.1 mm the positions are consistent; above 0.3 mm suggests a mismeasured marker, a wrong printed size, or lens
   distortion (see below). The web UI warns above 0.6 mm.

Fallbacks: 4 markers found = normal; 3 = still works, with a warning; fewer = reuses the last good homography with a
warning (only valid if the camera and board haven't moved) and fails with an error if there isn't one yet. A failed
frame is saved as `failed-*.jpg` in the output folder.

## Usage

* Web: **Capture** (about 10 s on a Zero 2 W, expected not measured) saves a PNG; tick captures in the gallery and
  **PDF** to download them as a multi-page PDF in tick order. PNGs carry the DPI so they print at page size.
* Button: set `button.pin`; a press does the same as Capture.
* Laptop, no hardware: `python -m clipcam --fake some_folder_of_photos`. It cycles through the images as "captures" and
  skips the camera, LED and button. `python tests/synth.py DIR` writes synthetic test photos.

## Tests (laptop)

```
uv venv --python 3.11 .venv      # or: python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt     # or .venv/bin/pip install -r ...
.venv/bin/python -m pytest -q -s
```

`tests/synth.py` renders the board (markers, text, ruled lines, reference discs) through a pinhole camera model (67 deg
vertical FOV, 205 mm, tilt, roll, offset, blur, noise, vignette) and `test_rectify.py` checks the recovered page geometry
against ground truth.

## Known limits

* **No lens-distortion correction.** The rectification is a homography, which cannot model the Wide lens's barrel
  distortion. `tests/distortion_sweep.py` quantifies the effect on synthetic images: about 0.8% edge shrink already puts
  the page corners ~1.6 mm off (interior ~0.25 mm); 1.6% gives 3.4 mm. The corners suffer most because the page extends
  beyond the marker rectangle. If the fit residual on real photos is above ~0.2 mm, calibrate the camera and undistort first.
* **Tight field of view.** At 205 mm the short sensor axis sees about +/-135 mm, so markers 85 mm from the board centre
  leave ~50 mm of slack, but markers in the 22 mm side margins would leave only ~7 mm (about 2 degrees of boom tilt).
