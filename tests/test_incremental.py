# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Use visible marker pixels to check complete snapshots after partial updates."""

import numpy as np
import pytest

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.incremental import IncrementalOCR
from sightline.ocr import TextDetection
from sightline.planning import RegionPolicy


class Markers:
    def __init__(self):
        self.calls = []
        self.fail = False

    def read(self, frame):
        self.calls.append((frame.width, frame.height))
        if self.fail:
            raise RuntimeError("temporary")
        found = []
        for value, text in [(90, "Left"), (200, "Right")]:
            ys, xs = np.nonzero(frame.pixels[:, :, 0] == value)
            if len(xs):
                found.append(
                    TextDetection(text, Box(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), 1)
                )
        return tuple(found)


def screen(*, left=True, right=False, source="desktop", width=140):
    pixels = np.zeros((82, width, 3), dtype=np.uint8)
    if left:
        pixels[21:27, 13:22] = 90
    if right:
        pixels[48:52, 91:99] = 200
    return Frame(pixels, source=source)


def test_moves_removals_unchanged_and_coordinate_reuse() -> None:
    model = Markers()
    fast = IncrementalOCR(model, policy=RegionPolicy(tile_size=8, padding=5))
    old = fast.read(screen())
    assert fast.read(screen()) == old
    assert fast.last_read.mode == "unchanged" and model.calls == [(140, 82)]
    added = fast.read(screen(right=True))
    assert added == Markers().read(screen(right=True))
    assert fast.last_read.mode == "regions" and model.calls[-1][0] < 140
    assert fast.read(screen(left=False, right=True)) == (added[-1],)
    assert fast.read(screen(left=False)) == ()


def test_failed_crop_preserves_last_successful_baseline() -> None:
    model = Markers()
    fast = IncrementalOCR(model, policy=RegionPolicy(tile_size=8, padding=5))
    baseline = fast.read(screen())
    model.fail = True
    with pytest.raises(RuntimeError):
        fast.read(screen(right=True))
    assert fast.last_read.reason == "first frame"
    assert fast.read(screen()) == baseline
    model.fail = False
    assert fast.read(screen(right=True)) == Markers().read(screen(right=True))


def test_source_resize_and_periodic_refresh() -> None:
    fast = IncrementalOCR(Markers(), policy=RegionPolicy(refresh_interval=2))
    for frame, reason in [
        (screen(), "first frame"),
        (screen(), "same pixels"),
        (screen(), "periodic refresh"),
        (screen(source="other"), "capture changed"),
        (screen(source="other", width=150), "capture changed"),
    ]:
        fast.read(frame)
        assert fast.last_read.reason == reason
