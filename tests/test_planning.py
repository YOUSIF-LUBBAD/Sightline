# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Crops include complete old labels, and broad changes revert to a full read."""

import numpy as np

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import TextDetection
from sightline.planning import RegionPolicy, plan_regions


def test_old_label_bridges_crops_and_edges_are_clipped() -> None:
    frame = Frame(np.zeros((80, 103, 3), dtype=np.uint8))
    old = (TextDetection("Settings", Box(12.5, 3, 43.5, 11), 1),)
    policy = RegionPolicy(tile_size=10, padding=3, max_area_ratio=1)
    assert plan_regions(frame, [(0, 0, 1, 1), (4, 0, 5, 1), (10, 7, 11, 8)], old, policy) == (
        Box(0, 0, 53, 14),
        Box(97, 67, 103, 80),
    )


def test_area_and_fragmentation_return_full_read() -> None:
    frame = Frame(np.zeros((100, 100, 3), dtype=np.uint8))
    policy = RegionPolicy(tile_size=10, padding=0, max_regions=1)
    assert plan_regions(frame, [(0, 0, 10, 6)], (), policy) is None
    assert plan_regions(frame, [(0, 0, 1, 1), (9, 9, 10, 10)], (), policy) is None
    assert plan_regions(frame, [], (), policy) == ()
