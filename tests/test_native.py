# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Compile the C routines and compare their actual outputs with the reference."""

import shutil

import numpy as np
import pytest

from sightline.diff import changed_tiles
from sightline.frame import Frame
from sightline.native import NativeBackend
from sightline.native_build import build_native
from sightline.regions import merge_tiles


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    if not any(shutil.which(name) for name in ("cc", "clang", "cl")):
        pytest.skip("Native parity tests need a C compiler")
    return NativeBackend(build_native(tmp_path_factory.mktemp("native")))


@pytest.mark.parametrize("shape", [(1, 1), (7, 1), (1, 19), (33, 67), (147, 193)])
def test_diff_all_channels_strided_input_and_thresholds(native, shape) -> None:
    rng = np.random.default_rng(81)
    raw = rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
    before = Frame(raw[:, ::-1])
    changed = before.pixels.copy()
    changed[::5, ::7] ^= 160
    after = Frame(changed)
    for tile in (1, 7, 32, 65535):
        for threshold in (0, 1, 127, 254, 255):
            np.testing.assert_array_equal(
                native.changed_tiles(before, after, tile, threshold),
                changed_tiles(before, after, tile, threshold),
            )


def test_every_three_by_three_mask_and_strided_random_views(native) -> None:
    for bits in range(512):
        grid = ((bits >> np.arange(9)) & 1).reshape(3, 3).astype(bool)
        assert native.merge_tiles(grid) == merge_tiles(grid)
    rng = np.random.default_rng(512)
    for density in (0.05, 0.5, 0.95):
        grid = (rng.random((101, 77)) < density)[::-1, ::2]
        assert native.merge_tiles(grid) == merge_tiles(grid)


def test_parallel_backends_reject_bad_inputs_before_calling_c(native) -> None:
    frame = Frame(np.ones((1, 1, 3), dtype=np.uint8))
    for diff in (native.changed_tiles, changed_tiles):
        with pytest.raises(ValueError):
            diff(frame, frame, 0, 0)
    for merge in (native.merge_tiles, merge_tiles):
        with pytest.raises(ValueError):
            merge(np.zeros((0, 1), dtype=bool))
        with pytest.raises(ValueError):
            merge(np.ones((3, 3), dtype=np.uint8))
