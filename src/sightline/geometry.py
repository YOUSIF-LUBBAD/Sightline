# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Small, checked rectangles shared by capture, OCR, and mouse input."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """A rectangle with an exclusive right and bottom edge."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in (self.left, self.top, self.right, self.bottom)):
            raise ValueError("Box coordinates must be finite")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError("A box must have positive width and height")

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2, self.top + self.height / 2)

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def clipped_to(self, bounds: "Box") -> "Box | None":
        """Keep only the shared area; return None when there is none."""
        left, top = max(self.left, bounds.left), max(self.top, bounds.top)
        right, bottom = min(self.right, bounds.right), min(self.bottom, bounds.bottom)
        return Box(left, top, right, bottom) if right > left and bottom > top else None
