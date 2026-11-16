import base64

import cv2
import numpy as np
import pytest
import requests

from clipcam.ocr import PROMPT, URL, OcrError, OpenRouterOcr


class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text or str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class Session:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.calls = resp, exc, []

    def post(self, url, **kw):
        self.calls.append((url, kw))
        if self.exc:
            raise self.exc
        return self.resp


def ok(content, usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = usage
    return Resp(200, body)


def sent_image(session):
    url = session.calls[0][1]["json"]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    data = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def img(h, w):
    return np.full((h, w, 3), 200, np.uint8)


def test_request_shape():
    s = Session(ok("hi"))
    OpenRouterOcr("k", "m", session=s).extract(img(2000, 3000))
    url, kw = s.calls[0]
    assert url == URL == "https://openrouter.ai/api/v1/chat/completions"
    assert kw["headers"]["Authorization"] == "Bearer k"
    assert kw["timeout"] == 60.0
    body = kw["json"]
    assert body["model"] == "m" and body["temperature"] == 0
    assert body["messages"][0]["content"][0] == {"type": "text", "text": PROMPT}
    assert max(sent_image(s).shape[:2]) == 1600


def test_parses_text_and_usage():
    s = Session(ok("  hi\nthere  ", {"prompt_tokens": 10, "completion_tokens": 3}))
    r = OpenRouterOcr("k", "m", session=s).extract(img(100, 100))
    assert (r.text, r.model, r.prompt_tokens, r.completion_tokens) == ("hi\nthere", "m", 10, 3)


def test_missing_usage_is_none():
    r = OpenRouterOcr("k", "m", session=Session(ok("x"))).extract(img(100, 100))
    assert r.prompt_tokens is None and r.completion_tokens is None


def test_no_text_sentinel():
    assert OpenRouterOcr("k", "m", session=Session(ok(" NO_TEXT\n"))).extract(img(100, 100)).text == ""


def test_small_image_not_upscaled():
    s = Session(ok("x"))
    OpenRouterOcr("k", "m", session=s).extract(img(600, 800))
    assert sent_image(s).shape[:2] == (600, 800)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_error_raises(status):
    s = Session(Resp(status, {"error": {"message": "nope"}}))
    with pytest.raises(OcrError) as e:
        OpenRouterOcr("k", "m", session=s).extract(img(100, 100))
    assert e.value.status == status and "nope" in str(e.value)


def test_timeout_raises():
    with pytest.raises(OcrError) as e:
        OpenRouterOcr("k", "m", session=Session(exc=requests.Timeout("slow"))).extract(img(100, 100))
    assert e.value.status is None


def test_malformed_body_raises():
    with pytest.raises(OcrError):
        OpenRouterOcr("k", "m", session=Session(Resp(200, {"choices": []}))).extract(img(100, 100))


def test_non_json_body_raises():
    with pytest.raises(OcrError):
        OpenRouterOcr("k", "m", session=Session(Resp(200, ValueError("bad json"), "<html>"))).extract(img(100, 100))
