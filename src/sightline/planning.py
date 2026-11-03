# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Choose complete OCR crops and bound the cost of fragmented changes."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import TextDetection
from sightline.regions import GridRect


@dataclass(frozen=True)
class RegionPolicy:
    tile_size: int = 32
    padding: int = 24
    max_regions: int = 12
    max_area_ratio: float = 0.45
    refresh_interval: int = 30

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("tile size", self.tile_size, 1),
            ("padding", self.padding, 0),
            ("region limit", self.max_regions, 1),
            ("refresh interval", self.refresh_interval, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"Invalid {name}")
        if self.tile_size > 65535 or not 0 < self.max_area_ratio <= 1:
            raise ValueError("Invalid tile size or maximum area ratio")


def _union(a: Box, b: Box) -> Box:
    return Box(
        min(a.left, b.left), min(a.top, b.top), max(a.right, b.right), max(a.bottom, b.bottom)
    )


def _padded(box: Box, frame: Frame, amount: int) -> Box:
    return Box(
        max(0, math.floor(box.left) - amount),
        max(0, math.floor(box.top) - amount),
        min(frame.width, math.ceil(box.right) + amount),
        min(frame.height, math.ceil(box.bottom) + amount),
    )


def plan_regions(
    frame: Frame, tiles: Sequence[GridRect], known: Sequence[TextDetection], policy: RegionPolicy
) -> tuple[Box, ...] | None:
    """Return disjoint crops, or None when one whole-screen read is cheaper."""
    if len(tiles) > policy.max_regions * 8:
        return None
    regions: list[Box] = []
    for tile in tiles:
        seed = Box(*(v * policy.tile_size for v in tile))
        area = _padded(seed, frame, policy.padding)
        # Follow chains of labels and crops until nothing crosses a crop boundary.
        while True:
            before = area
            for detection in known:
                if area.clipped_to(detection.box) is not None:
                    area = _union(area, _padded(detection.box, frame, policy.padding))
            separate = []
            for other in regions:
                if area.clipped_to(other) is None:
                    separate.append(other)
                else:
                    area = _union(area, other)
            regions = separate
            if area == before:
                break
        regions.append(area)
    pixels = sum(region.width * region.height for region in regions)
    if (
        len(regions) > policy.max_regions
        or pixels > frame.width * frame.height * policy.max_area_ratio
    ):
        return None
    return tuple(regions)
