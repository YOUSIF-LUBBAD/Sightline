# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Exercise the public image-series command through its normal argument parser."""

import json

from PIL import Image

from examples import observe_changes


def test_independent_image_paths_share_one_sequence(tmp_path, monkeypatch, capsys) -> None:
    files = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in files:
        Image.new("RGB", (101, 90)).save(path)
    calls = []

    class Reader:
        def __init__(self, config):
            pass

        def prepare(self):
            pass

        def read(self, frame):
            calls.append(frame.pixels.shape)
            return ()

    monkeypatch.setattr(observe_changes, "EasyOCREngine", Reader)
    assert observe_changes.main(["--images", *(str(p) for p in files)]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["read"]["mode"] for row in rows] == ["full", "unchanged"]
    assert len(calls) == 1
    assert rows[-1]["timings"]["capture"]["calls"] == 2
