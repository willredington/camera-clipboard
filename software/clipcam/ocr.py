"""Text extraction with a vision model on OpenRouter (OpenAI-compatible chat completions)."""
from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2
import numpy as np
import requests

URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT = ("Transcribe all text on this page exactly as written, preserving line breaks. Output only the "
          "transcribed text, with no commentary. If there is no text, output exactly NO_TEXT.")
NO_TEXT = "NO_TEXT"


class OcrError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class OcrResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


class OpenRouterOcr:
    def __init__(self, api_key: str, model: str, max_side: int = 1600, timeout_s: float = 60.0,
                 session: requests.Session | None = None):
        self.api_key, self.model, self.max_side, self.timeout_s = api_key, model, max_side, timeout_s
        self.session = session or requests.Session()

    def _jpeg_b64(self, bgr: np.ndarray) -> str:
        h, w = bgr.shape[:2]
        s = self.max_side / max(h, w)
        if s < 1:
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise OcrError("JPEG encoding failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def extract(self, bgr: np.ndarray) -> OcrResult:
        body = {
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + self._jpeg_b64(bgr)}},
            ]}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "X-Title": "clipcam"}
        try:
            r = self.session.post(URL, json=body, headers=headers, timeout=self.timeout_s)
        except requests.RequestException as e:
            raise OcrError(f"request failed: {e}") from e
        if not 200 <= r.status_code < 300:
            raise OcrError(f"HTTP {r.status_code}: {r.text[:200]}", r.status_code)
        try:
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            raise OcrError(f"unexpected response: {r.text[:200]}", r.status_code) from e
        usage = data.get("usage") or {}
        return OcrResult("" if text == NO_TEXT else text, self.model,
                         usage.get("prompt_tokens"), usage.get("completion_tokens"))
