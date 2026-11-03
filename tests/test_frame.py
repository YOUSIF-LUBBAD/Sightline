# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Catch coordinate and mutation mistakes before they can move a mouse."""

import numpy as np
import pytest

from sightline.frame import Frame
from sightline.geometry import Box


def test_frame_owns_an_immutable_copy() -> None:
    pixels = np.zeros((20, 40, 3), dtype=np.uint8)
    frame = Frame(pixels)
    pixels[:] = 255
    assert not frame.pixels.any()
    with pytest.raises(ValueError):
        frame.pixels.setflags(write=True)


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 2), (2, 2, 4)])
def test_invalid_pixel_shape(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        Frame(np.zeros(shape, dtype=np.uint8))


def test_float_pixels_are_not_silently_reinterpreted() -> None:
    with pytest.raises(ValueError, match="uint8"):
        Frame(np.zeros((2, 2, 3), dtype=np.float32))


def test_scaled_and_offset_coordinates() -> None:
    frame = Frame(np.zeros((200, 400, 3), dtype=np.uint8), Box(-100, 50, 100, 150))
    assert frame.to_desktop(200, 100) == (0, 100)
    assert frame.to_desktop(399, 199) == (99, 149)


@pytest.mark.parametrize("point", [(-1, 0), (40, 0), (0, 20), (float("nan"), 0)])
def test_outside_point_is_rejected(point: tuple[float, float]) -> None:
    frame = Frame(np.zeros((20, 40, 3), dtype=np.uint8), Box(0, 0, 40, 20))
    with pytest.raises(ValueError):
        frame.to_desktop(*point)


def test_saved_image_cannot_be_clicked_without_mapping() -> None:
    with pytest.raises(ValueError, match="no desktop mapping"):
        Frame(np.zeros((2, 2, 3), dtype=np.uint8)).to_desktop(1, 1)


@pytest.mark.parametrize("values", [(0, 0, 0, 1), (2, 0, 1, 1), (0, 0, 1, float("inf"))])
def test_invalid_boxes(values: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        Box(*values)


def test_box_clipping() -> None:
    assert Box(-5, -2, 20, 30).clipped_to(Box(0, 0, 10, 10)) == Box(0, 0, 10, 10)
    assert Box(20, 20, 30, 30).clipped_to(Box(0, 0, 10, 10)) is None
