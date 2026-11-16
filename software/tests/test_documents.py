import logging

import numpy as np

from clipcam.documents import DocumentStore

IMG = np.full((20, 30, 3), 180, np.uint8)


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def store(tmp_path, clock=None):
    return DocumentStore(tmp_path, now=clock or Clock())


def test_new_store_empty(tmp_path):
    s = store(tmp_path)
    assert s.live() is None and s.summaries() == []


def test_set_live_returns_previous(tmp_path):
    s = store(tmp_path)
    assert s.set_live(3) is None
    assert s.set_live(4) == 3
    assert s.live() == 4
    assert {p["number"] for p in s.summaries()} == {3, 4}


def test_set_live_persists(tmp_path):
    store(tmp_path).set_live(4)
    assert DocumentStore(tmp_path).live() == 4


def test_add_revision_replaces_text(tmp_path):
    c = Clock()
    s = store(tmp_path, c)
    s.set_live(3)
    s.add_revision(3, IMG, "hello")
    c.t += 1
    summary = s.add_revision(3, IMG, "hello\nworld")
    page = s.get(3)
    assert page["text"] == "hello\nworld" and summary["text"] == "hello\nworld"
    assert len(page["revisions"]) == 2
    assert all((tmp_path / "3" / r["image"]).is_file() for r in page["revisions"])
    assert page["revisions"][0]["text"] == "hello"


def test_same_second_revisions_get_unique_images(tmp_path):
    s = store(tmp_path)
    s.add_revision(3, IMG, "a")
    s.add_revision(3, IMG, "b")
    names = [r["image"] for r in s.get(3)["revisions"]]
    assert len(set(names)) == 2


def test_first_line_and_order(tmp_path):
    c = Clock()
    s = store(tmp_path, c)
    s.add_revision(1, IMG, "\n\n  first real line  \nsecond")
    c.t += 5
    s.add_revision(2, IMG, "x" * 200)
    sums = s.summaries()
    assert [p["number"] for p in sums] == [2, 1]
    assert sums[1]["first_line"] == "first real line"
    assert len(sums[0]["first_line"]) == 80
    assert sums[0]["image"] == s.get(2)["revisions"][-1]["image"]


def test_page_without_revisions_summary(tmp_path):
    s = store(tmp_path)
    s.set_live(8)
    (p,) = s.summaries()
    assert p["text"] == "" and p["first_line"] == "" and p["image"] is None and p["unsent"] is False


def test_summaries_exclude_revisions(tmp_path):
    s = store(tmp_path)
    for i in range(300):
        s.add_revision(5, IMG, f"rev {i}")
    assert "revisions" not in s.summaries()[0]
    assert len(s.get(5)["revisions"]) == 300


def test_unsent_flag(tmp_path):
    s = store(tmp_path)
    s.set_live(3)
    s.set_unsent(3, True)
    assert s.summaries()[0]["unsent"] is True
    s.add_revision(3, IMG, "t")
    assert s.summaries()[0]["unsent"] is False


def test_corrupt_page_json_skipped(tmp_path, caplog):
    s = store(tmp_path)
    s.add_revision(3, IMG, "ok")
    (tmp_path / "5").mkdir()
    (tmp_path / "5" / "page.json").write_text("{")
    with caplog.at_level(logging.WARNING):
        assert [p["number"] for p in s.summaries()] == [3]
    assert s.get(5) is None
    assert "5" in caplog.text


def test_corrupt_live_json(tmp_path):
    (tmp_path / "live.json").write_text("garbage")
    assert DocumentStore(tmp_path).live() is None


def test_atomic_write_leaves_no_tmp(tmp_path):
    s = store(tmp_path)
    s.set_live(1)
    s.add_revision(1, IMG, "a")
    s.set_unsent(1, True)
    s.set_live(2)
    assert not list(tmp_path.rglob("*.tmp"))


def test_image_path_validation(tmp_path):
    s = store(tmp_path)
    s.add_revision(3, IMG, "a")
    name = s.get(3)["revisions"][0]["image"]
    assert s.image_path(3, name) == tmp_path / "3" / name
    for bad in ("../live.json", "x.png", "nope.jpg", "page.json"):
        assert s.image_path(3, bad) is None
    assert s.image_path(99, name) is None


# --- final review fixes ---

def test_summary_of_one_page(tmp_path):
    s = store(tmp_path)
    s.add_revision(3, IMG, "hello\nthere")
    assert s.summary(3)["first_line"] == "hello" and "revisions" not in s.summary(3)
    assert s.summary(99) is None


def test_summaries_do_not_read_page_json(tmp_path, monkeypatch):
    s = store(tmp_path)
    for n in (1, 2):
        s.add_revision(n, IMG, f"page {n}")
    import json as _json
    real = _json.load
    opened = []

    def spy(f, *a, **k):
        opened.append(getattr(f, "name", ""))
        return real(f, *a, **k)
    monkeypatch.setattr("clipcam.documents.json.load", spy)
    assert [p["number"] for p in s.summaries()] == [2, 1] or [p["number"] for p in s.summaries()] == [1, 2]
    assert opened and not any(name.endswith("page.json") for name in opened), opened


def test_corrupt_page_json_is_kept_aside_not_overwritten(tmp_path, caplog):
    s = store(tmp_path)
    s.add_revision(7, IMG, "precious")
    (tmp_path / "7" / "page.json").write_text("{broken")
    s.set_live(7)
    kept = list((tmp_path / "7").glob("page.json.corrupt-*"))
    assert len(kept) == 1 and kept[0].read_text() == "{broken"
    assert "corrupt" in caplog.text
