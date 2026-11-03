# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Merged rectangles must cover every changed tile exactly once."""

import numpy as np

from sightline.regions import merge_tiles


def test_equal_runs_join_without_bridging_empty_tiles() -> None:
    mask = np.array([[1, 1, 0, 1], [1, 1, 0, 1], [0, 1, 1, 1]], dtype=bool)
    assert merge_tiles(mask) == [(0, 0, 2, 2), (3, 0, 4, 2), (1, 2, 4, 3)]


def test_random_masks_round_trip_without_overlaps() -> None:
    rng = np.random.default_rng(681)
    for density in (0.0, 0.1, 0.5, 1.0):
        mask = rng.random((17, 31)) < density
        covered = np.zeros(mask.shape, dtype=int)
        for left, top, right, bottom in merge_tiles(mask):
            covered[top:bottom, left:right] += 1
        np.testing.assert_array_equal(covered, mask)
