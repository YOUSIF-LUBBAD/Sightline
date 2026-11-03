# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Find tiles with pixel changes. True means at least one RGB channel changed."""

import numpy as np
from numpy.typing import NDArray

from sightline.frame import Frame


def check_diff(previous: Frame, current: Frame, tile_size: int, threshold: int) -> None:
    if previous.pixels.shape != current.pixels.shape:
        raise ValueError("Frame sizes must match for a pixel comparison")
    if type(tile_size) is not int or not 1 <= tile_size <= 65535:
        raise ValueError("Tile size must be an integer from 1 to 65535")
    if type(threshold) is not int or not 0 <= threshold <= 255:
        raise ValueError("Threshold must be an integer from 0 to 255")


def changed_tiles(
    previous: Frame, current: Frame, tile_size: int = 32, threshold: int = 0
) -> NDArray[np.bool_]:
    """Compare absolute channel differences, including partial tiles at the edges."""
    check_diff(previous, current, tile_size, threshold)
    if threshold == 0:
        changed = np.any(previous.pixels != current.pixels, axis=2)
    else:
        # Widen before subtraction so a black-to-white change cannot wrap around.
        delta = np.subtract(previous.pixels, current.pixels, dtype=np.int16)
        changed = np.any(np.abs(delta) > threshold, axis=2)
    rows = np.logical_or.reduceat(changed, np.arange(0, current.height, tile_size), axis=0)
    return np.logical_or.reduceat(rows, np.arange(0, current.width, tile_size), axis=1)
