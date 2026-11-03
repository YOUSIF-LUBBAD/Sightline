# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""A captured RGB image and its optional desktop coordinate mapping."""

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sightline.geometry import Box


@dataclass(frozen=True, eq=False)
class Frame:
    """Pixels never change after capture, even if the caller edits its array."""

    pixels: NDArray[np.uint8]
    desktop_bounds: Box | None = None
    captured_at: float = field(default_factory=time.monotonic)
    source: str = "image"

    def __post_init__(self) -> None:
        if self.pixels.dtype != np.uint8:
            raise ValueError("Frame pixels must use uint8 values")
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError("Frame pixels must have shape (height, width, 3) in RGB order")
        if self.pixels.shape[0] == 0 or self.pixels.shape[1] == 0:
            raise ValueError("A frame cannot be empty")
        if not math.isfinite(self.captured_at) or self.captured_at < 0:
            raise ValueError("Capture time must be a non-negative monotonic timestamp")
        if not self.source.strip():
            raise ValueError("A frame needs a source name")
        # Bytes make the array read-only; callers cannot turn writes back on.
        frozen = np.frombuffer(self.pixels.tobytes(), dtype=np.uint8).reshape(self.pixels.shape)
        object.__setattr__(self, "pixels", frozen)

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def pixel_bounds(self) -> Box:
        return Box(0, 0, self.width, self.height)

    def to_desktop(self, x: float, y: float) -> tuple[int, int]:
        """Map image pixels to mouse coordinates, including display scaling."""
        bounds = self.desktop_bounds
        if bounds is None:
            raise ValueError("This image has no desktop mapping; it cannot drive a mouse")
        if not self.pixel_bounds.contains(x, y):
            raise ValueError("The point is outside the captured image")
        return (
            math.floor(bounds.left + x * bounds.width / self.width),
            math.floor(bounds.top + y * bounds.height / self.height),
        )
