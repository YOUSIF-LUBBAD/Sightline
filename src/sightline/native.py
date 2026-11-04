# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Use an explicitly built Sightline shared library through ctypes."""

import ctypes as ct
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sightline.diff import check_diff
from sightline.frame import Frame
from sightline.regions import GridRect, check_mask


class NativeBackend:
    def __init__(self, library: Path) -> None:
        try:
            lib = ct.CDLL(str(library.resolve()))
            lib.sl_abi_version.argtypes = []
            lib.sl_abi_version.restype = ct.c_uint
            if lib.sl_abi_version() != 1:
                raise ValueError("Unsupported Sightline native ABI version")
            self._diff = lib.sl_diff
            self._merge = lib.sl_merge
        except (OSError, AttributeError) as error:
            raise ValueError(f"Cannot load the Sightline native backend at {library}") from error
        ptr, size = ct.c_void_p, ct.c_size_t
        self._diff.argtypes = [ptr, ptr, size, size, size, ct.c_uint8, ptr, size]
        self._diff.restype = ct.c_int
        self._merge.argtypes = [ptr, size, size, ptr, size, ct.POINTER(size)]
        self._merge.restype = ct.c_int

    def changed_tiles(
        self, previous: Frame, current: Frame, tile_size: int = 32, threshold: int = 0
    ) -> NDArray[np.bool_]:
        check_diff(previous, current, tile_size, threshold)
        rows = (current.height + tile_size - 1) // tile_size
        cols = (current.width + tile_size - 1) // tile_size
        mask = np.empty((rows, cols), dtype=bool)
        status = self._diff(
            previous.pixels.ctypes.data,
            current.pixels.ctypes.data,
            current.width,
            current.height,
            tile_size,
            threshold,
            mask.ctypes.data,
            mask.size,
        )
        if status:
            raise ValueError("C frame diff rejected the input dimensions")
        return mask

    def merge_tiles(self, mask: NDArray[np.bool_]) -> list[GridRect]:
        check_mask(mask)
        grid = np.ascontiguousarray(mask)
        rows, cols = grid.shape
        count = ct.c_size_t()
        args = (grid.ctypes.data, rows, cols)
        if self._merge(*args, None, 0, ct.byref(count)):
            raise ValueError("C region merge rejected the tile grid")
        if not count.value:
            return []
        boxes = np.empty((count.value, 4), dtype=np.uintp)
        if self._merge(*args, boxes.ctypes.data, len(boxes), ct.byref(count)):
            raise ValueError("C region merge rejected the output buffer")
        return [
            (int(left), int(top), int(right), int(bottom)) for left, top, right, bottom in boxes
        ]
