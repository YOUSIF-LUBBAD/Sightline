# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Check the tile grid against individual RGB changes at its boundaries."""

import numpy as np
import pytest

from sightline.diff import changed_tiles
from sightline.frame import Frame


def test_difference_in_partial_tile_and_unsigned_threshold() -> None:
    a = np.zeros((5, 7, 3), dtype=np.uint8)
    b = a.copy()
    b[4, 6, 2] = 255
    b[0, 0, 0] = 12
    expected = [[False, False], [False, True]]
    assert changed_tiles(Frame(a), Frame(b), 4, 12).tolist() == expected
    assert changed_tiles(Frame(b), Frame(a), 4, 12).tolist() == expected
    assert not changed_tiles(Frame(a), Frame(b), 4, 255).any()
    assert changed_tiles(Frame(a), Frame(b), 32).tolist() == [[True]]


@pytest.mark.parametrize("tile,threshold", [(0, 0), (True, 0), (1, -1), (1, 256)])
def test_invalid_grid_parameters(tile, threshold) -> None:
    frame = Frame(np.zeros((1, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        changed_tiles(frame, frame, tile, threshold)


def test_resize_requires_new_baseline() -> None:
    a = Frame(np.zeros((3, 4, 3), dtype=np.uint8))
    b = Frame(np.zeros((4, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="sizes"):
        changed_tiles(a, b)
