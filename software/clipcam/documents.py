"""Per-page document store: one folder per printed page number, with every OCR revision and its image.

<dir>/<n>/page.json   {"number", "first_seen", "last_updated", "text", "unsent", "revisions": [{"time", "image", "text"}]}
<dir>/<n>/summary.json the same without "revisions" (what the portal polls, so its cost doesn't grow with history)
<dir>/<n>/<stamp>.jpg the image each revision was transcribed from
<dir>/live.json       {"live": n | null}

JSON writes are atomic (temp file + os.replace) so a power cut can't leave a half-written file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)
_IMAGE_RE = re.compile(r"^\d{8}-\d{6}(-\d+)?\.jpg$")


class DocumentStore:
    def __init__(self, directory: Path, now: Callable[[], float] = time.time):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.now = now
        self._lock = threading.Lock()

    # --- helpers ---
    def _iso(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.now()))

    @staticmethod
    def _write_json(path: Path, data) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _page_file(self, n: int) -> Path:
        return self.dir / str(n) / "page.json"

    def _load(self, n: int) -> dict | None:
        path = self._page_file(n)
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                page = json.load(f)
            if not isinstance(page, dict) or page.get("number") != n:
                raise ValueError("not a page record")
            return page
        except (OSError, ValueError) as e:
            log.warning("skipping unreadable page %s (%s): %s", n, path, e)
            return None

    def _load_or_new(self, n: int) -> dict:
        page = self._load(n)
        path = self._page_file(n)
        if page is None and path.exists():
            # Unreadable record: keep it aside for recovery instead of silently replacing the page's history.
            aside = path.with_name(f"page.json.corrupt-{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.now()))}")
            os.replace(path, aside)
            log.warning("page %s: corrupt page.json moved to %s; starting a new record", n, aside.name)
        if page is None:
            t = self._iso()
            page = {"number": n, "first_seen": t, "last_updated": t, "text": "", "unsent": False, "revisions": []}
        return page

    def _save(self, page: dict) -> None:
        d = self.dir / str(page["number"])
        d.mkdir(exist_ok=True)
        self._write_json(d / "page.json", page)
        self._write_json(d / "summary.json", self._summary(page))

    @staticmethod
    def _summary(page: dict) -> dict:
        lines = [ln.strip() for ln in page.get("text", "").splitlines() if ln.strip()]
        revs = page.get("revisions") or []
        return {"number": page["number"], "first_line": lines[0][:80] if lines else "", "text": page.get("text", ""),
                "first_seen": page.get("first_seen"), "last_updated": page.get("last_updated"),
                "unsent": bool(page.get("unsent")), "image": revs[-1]["image"] if revs else None}

    # --- live page ---
    def live(self) -> int | None:
        try:
            with open(self.dir / "live.json", encoding="utf-8") as f:
                n = json.load(f).get("live")
            return n if isinstance(n, int) else None
        except (OSError, ValueError, AttributeError):
            return None

    def set_live(self, n: int) -> int | None:
        with self._lock:
            prev = self.live()
            page = self._load(n)
            if page is None:
                self._save(self._load_or_new(n))
            self._write_json(self.dir / "live.json", {"live": n})
            return prev

    # --- pages ---
    def add_revision(self, n: int, bgr: np.ndarray, text: str) -> dict:
        with self._lock:
            page = self._load_or_new(n)
            d = self.dir / str(n)
            d.mkdir(exist_ok=True)
            base = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.now()))
            name, k = f"{base}.jpg", 1
            while (d / name).exists():
                name, k = f"{base}-{k}.jpg", k + 1
            cv2.imwrite(str(d / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            t = self._iso()
            page["revisions"].append({"time": t, "image": name, "text": text})
            page.update(text=text, last_updated=t, unsent=False)
            self._save(page)
            return self._summary(page)

    def set_unsent(self, n: int, value: bool) -> None:
        with self._lock:
            page = self._load_or_new(n)
            page["unsent"] = bool(value)
            self._save(page)

    def get(self, n: int) -> dict | None:
        return self._load(n)

    def summary(self, n: int) -> dict | None:
        """One page without its revisions; reads the small summary.json, falling back to page.json."""
        try:
            with open(self.dir / str(n) / "summary.json", encoding="utf-8") as f:
                summ = json.load(f)
            if isinstance(summ, dict) and summ.get("number") == n:
                return summ
        except (OSError, ValueError):
            pass
        page = self._load(n)
        return None if page is None else self._summary(page)

    def summaries(self) -> list[dict]:
        pages = []
        for d in self.dir.iterdir():
            if d.is_dir() and d.name.isdigit():
                summ = self.summary(int(d.name))
                if summ is not None:
                    pages.append(summ)
        return sorted(pages, key=lambda p: (p["last_updated"] or "", p["number"]), reverse=True)

    def image_path(self, n: int, name: str) -> Path | None:
        if not _IMAGE_RE.match(name or ""):
            return None
        page = self._load(n)
        if page is None or name not in {r["image"] for r in page.get("revisions", [])}:
            return None
        p = self.dir / str(n) / name
        return p if p.is_file() else None
