# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Turn changed tiles into rectangles without filling the gaps between them."""

import numpy as np
from numpy.typing import NDArray

GridRect = tuple[int, int, int, int]


def check_mask(mask: NDArray[np.bool_]) -> None:
    if mask.dtype != np.bool_ or mask.ndim != 2 or not mask.size:
        raise ValueError("Expected a non-empty two-dimensional boolean tile mask")


def merge_tiles(mask: NDArray[np.bool_]) -> list[GridRect]:
    """Join equal horizontal runs on adjacent rows, in top-to-bottom order."""
    check_mask(mask)
    padded = np.pad(mask, ((0, 0), (1, 1)))
    edges = padded[:, 1:] != padded[:, :-1]
    result: list[GridRect] = []
    previous: dict[tuple[int, int], int] = {}
    for y, row in enumerate(edges):
        current: dict[tuple[int, int], int] = {}
        for start, end in np.flatnonzero(row).reshape(-1, 2):
            span = (int(start), int(end))
            index = previous.get(span)
            if index is None:
                index = len(result)
                result.append((span[0], y, span[1], y + 1))
            else:
                left, top, right, _ = result[index]
                result[index] = (left, top, right, y + 1)
            current[span] = index
        previous = current
    return result
