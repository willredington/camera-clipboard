"""Configuration: dataclasses with defaults, overridden from a TOML file."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path

from .rectify import BoardGeometry


@dataclass
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8080


@dataclass
class CameraCfg:
    width: int = 4608            # IMX708 full resolution
    height: int = 2592
    preview_width: int = 640
    preview_height: int = 360
    settle_s: float = 1.0        # after switching to still mode with the LED on, before autofocus
    af_timeout_s: float = 6.0
    af_range: str = "full"       # "normal" | "macro" | "full"
    detect_max_side: int = 1600  # marker detection runs on an image downscaled to this long side


@dataclass
class LedCfg:
    pin: int = 18                # BCM number; drives the MOSFET gate
    active_high: bool = True


@dataclass
class ButtonCfg:
    pin: int | None = None       # BCM number; None disables the physical button (wired to GND, pull-up on)


@dataclass
class OutputCfg:
    dir: str = "~/captures"
    dpi: int = 200
    keep: int = 200              # oldest captures are deleted beyond this many (0 = keep all)
    gallery: int = 24            # how many recent captures the web page lists


@dataclass
class Config:
    server: ServerCfg = field(default_factory=ServerCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    led: LedCfg = field(default_factory=LedCfg)
    button: ButtonCfg = field(default_factory=ButtonCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    board: BoardGeometry = field(default_factory=BoardGeometry)

    @property
    def output_dir(self) -> Path:
        return Path(self.output.dir).expanduser()


def _apply(obj, data: dict, path: str):
    names = {f.name: f for f in fields(obj)}
    for key, val in data.items():
        if key not in names:
            raise ValueError(f"unknown config key: {path}{key}")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            _apply(cur, val, f"{path}{key}.")
        else:
            setattr(obj, key, val)


def load_config(path: str | Path | None) -> Config:
    cfg = Config()
    if path is None:
        return cfg
    with open(path, "rb") as f:
        data = tomllib.load(f)
    # [board.markers] has integer ids as keys; TOML keys are strings
    markers = data.get("board", {}).pop("markers", None)
    _apply(cfg, data, "")
    if markers is not None:
        cfg.board.markers = {int(k): (float(v[0]), float(v[1])) for k, v in markers.items()}
    cfg.board.markers = {int(k): (float(v[0]), float(v[1])) for k, v in cfg.board.markers.items()}
    if len(cfg.board.markers) < 3:
        raise ValueError("board.markers needs at least 3 entries")
    return cfg
