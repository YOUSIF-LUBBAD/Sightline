# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Find tiles with pixel changes. True means at least one RGB channel changed."""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from sightline.frame import Frame


def checked_pixels(
    previous: Frame, current: Frame, tile_size: int, threshold: int
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    # Views keep their own shape and strides while the immutable pixel bytes are shared.
    before, after = previous.pixels.view(), current.pixels.view()
    for pixels in (before, after):
        if (
            pixels.dtype != np.uint8
            or pixels.ndim != 3
            or pixels.shape[2] != 3
            or not pixels.size
            or not pixels.flags.c_contiguous
        ):
            raise ValueError("Frame diff requires a non-empty packed uint8 RGB image")
    if before.shape != after.shape:
        raise ValueError("Frame sizes must match for a pixel comparison")
    if type(tile_size) is not int or not 1 <= tile_size <= 65535:
        raise ValueError("Tile size must be an integer from 1 to 65535")
    if type(threshold) is not int or not 0 <= threshold <= 255:
        raise ValueError("Threshold must be an integer from 0 to 255")
    return before, after


def check_diff(previous: Frame, current: Frame, tile_size: int, threshold: int) -> None:
    """Check inputs without running either pixel kernel."""
    checked_pixels(previous, current, tile_size, threshold)


def changed_tiles(
    previous: Frame, current: Frame, tile_size: int = 32, threshold: int = 0
) -> NDArray[np.bool_]:
    """Compare absolute channel differences, including partial tiles at the edges."""
    before, after = checked_pixels(previous, current, tile_size, threshold)
    height, width, _ = after.shape
    if threshold == 0:
        changed = before != after
    else:
        # Widen before subtraction so a black-to-white change cannot wrap around.
        delta = np.subtract(before, after, dtype=np.int16)
        changed = np.abs(delta) > threshold
    # Group adjacent RGB bytes before combining rows. This avoids one reduction per pixel.
    scanlines = changed.reshape(height, -1)
    starts = np.arange(0, width * 3, tile_size * 3)
    runs = np.logical_or.reduceat(scanlines, starts, axis=1)
    mask = np.logical_or.reduceat(runs, np.arange(0, height, tile_size), axis=0)
    return cast(NDArray[np.bool_], mask)
