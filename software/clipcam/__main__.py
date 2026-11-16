"""python -m clipcam [--config FILE] [--fake DIR] [--host H] [--port N]
   python -m clipcam [--config FILE] rectify IMAGE OUT.png   # offline check of marker positions
"""
from __future__ import annotations

import argparse
import logging
import sys

import cv2

from .camera import FakeCamera
from .config import load_config
from .hardware import FakeLed
from .rectify import Rectifier
from .service import Busy, CaptureService
from .store import Store
from .web import create_app

log = logging.getLogger("clipcam")


def cmd_rectify(args, cfg):
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"cannot read {args.image}")
    res = Rectifier(cfg.board, cfg.output.dpi, cfg.camera.detect_max_side).process(img)
    cv2.imwrite(args.out, res.image)
    print(f"{res.method}; ids {res.found_ids}; fit residual "
          f"{'n/a' if res.rms_mm is None else format(res.rms_mm, '.3f') + ' mm'}; wrote {args.out}")
    for w in res.warnings:
        print("warning:", w)


def cmd_serve(args, cfg):
    store = Store(cfg.output_dir, cfg.output.dpi, cfg.output.keep)
    rect = Rectifier(cfg.board, cfg.output.dpi, cfg.camera.detect_max_side)
    button = None
    if args.fake:
        camera, led = FakeCamera(args.fake, cfg.camera), FakeLed()
    else:
        from .camera import PiCamera
        from .hardware import Led
        camera, led = PiCamera(cfg.camera), Led(cfg.led.pin, cfg.led.active_high)
    service = CaptureService(camera, led, rect, store)

    if cfg.button.pin is not None and not args.fake:
        from .hardware import Button

        def pressed():
            try:
                service.capture()
            except Busy:
                log.info("button ignored: capture in progress")
            except Exception as e:
                log.error("button capture failed: %s", e)
        button = Button(cfg.button.pin, pressed)
        log.info("button on GPIO%d", cfg.button.pin)

    app = create_app(service, store, cfg.output.gallery)
    try:
        app.run(host=cfg.server.host, port=cfg.server.port, threaded=True)
    finally:
        for c in (button, camera, led):
            if c is not None:
                c.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="clipcam")
    ap.add_argument("--config")
    ap.add_argument("--fake", metavar="DIR", help="use images from DIR instead of the camera/GPIO")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("rectify", help="rectify one image file and report the marker fit")
    r.add_argument("image")
    r.add_argument("out")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.host:
        cfg.server.host = args.host
    if args.port:
        cfg.server.port = args.port
    (cmd_rectify if args.cmd == "rectify" else cmd_serve)(args, cfg)


if __name__ == "__main__":
    main()
