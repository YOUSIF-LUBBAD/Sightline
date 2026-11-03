# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Reuse OCR on stable pixels and refresh the affected labels after a change."""

import threading
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sightline.cache import CachedOCR
from sightline.diff import changed_tiles
from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import OcrEngine, TextDetection
from sightline.planning import RegionPolicy, plan_regions
from sightline.regions import GridRect, merge_tiles
from sightline.roi import read_region
from sightline.timing import Timings, timed


class PixelBackend(Protocol):
    def changed_tiles(
        self, previous: Frame, current: Frame, tile_size: int, threshold: int
    ) -> NDArray[np.bool_]: ...
    def merge_tiles(self, mask: NDArray[np.bool_]) -> list[GridRect]: ...


class NumpyBackend:
    changed_tiles = staticmethod(changed_tiles)
    merge_tiles = staticmethod(merge_tiles)


@dataclass(frozen=True)
class ReadReport:
    mode: str
    reason: str
    regions: tuple[Box, ...] = ()


def _touches_cut(box: Box, crop: Box, frame: Frame) -> bool:
    # Text near an internal crop edge may continue into pixels we did not read.
    return (
        (crop.left > 0 and box.left <= crop.left + 1)
        or (crop.top > 0 and box.top <= crop.top + 1)
        or (crop.right < frame.width and box.right >= crop.right - 1)
        or (crop.bottom < frame.height and box.bottom >= crop.bottom - 1)
    )


class IncrementalOCR:
    def __init__(
        self,
        engine: OcrEngine,
        *,
        policy: RegionPolicy | None = None,
        backend: PixelBackend | None = None,
        cache_size: int = 128,
        timings: Timings | None = None,
    ) -> None:
        self.policy = policy or RegionPolicy()
        self.backend: PixelBackend = backend if backend is not None else NumpyBackend()
        self.timings = timings
        self.cache = CachedOCR(engine, cache_size, timings=timings)
        self.last_read: ReadReport | None = None
        self._previous: Frame | None = None
        self._detections: tuple[TextDetection, ...] = ()
        self._since_full = 0
        self._lock = threading.Lock()

    def _changes(self, frame: Frame) -> ReadReport:
        previous = self._previous
        if previous is None:
            return ReadReport("full", "first frame")
        if (
            previous.source != frame.source
            or previous.pixels.shape != frame.pixels.shape
            or previous.desktop_bounds != frame.desktop_bounds
        ):
            return ReadReport("full", "capture changed")
        if self._since_full >= self.policy.refresh_interval - 1:
            return ReadReport("full", "periodic refresh")
        with timed(self.timings, "diff"):
            mask = self.backend.changed_tiles(previous, frame, self.policy.tile_size, 0)
            dirty = int(np.count_nonzero(mask))
        if dirty == 0:
            return ReadReport("unchanged", "same pixels")
        if dirty > mask.size * self.policy.max_area_ratio:
            return ReadReport("full", "large change")
        with timed(self.timings, "merge"):
            tiles = self.backend.merge_tiles(mask)
        with timed(self.timings, "roi.plan"):
            regions = plan_regions(frame, tiles, self._detections, self.policy)
        if regions is None:
            return ReadReport("full", "crop budget exceeded")
        return ReadReport("regions", "local change", regions)

    def read(self, frame: Frame) -> tuple[TextDetection, ...]:
        with self._lock:
            report = self._changes(frame)
            detections = self._detections
            if report.mode == "regions":
                new: list[TextDetection] = []
                for region in report.regions:
                    batch = read_region(self.cache, frame, region, self.timings)
                    if any(_touches_cut(item.box, region, frame) for item in batch):
                        report = ReadReport("full", "label at crop edge")
                        break
                    new.extend(batch)
                else:
                    with timed(self.timings, "roi.reconcile"):
                        kept = [
                            item
                            for item in detections
                            if all(item.box.clipped_to(region) is None for region in report.regions)
                        ]
                        detections = tuple(
                            sorted(
                                [*kept, *new],
                                key=lambda item: (item.box.top, item.box.left, item.text),
                            )
                        )
            if report.mode == "full":
                detections = self.cache.read(frame)
            # Move the baseline only after all requested reads succeed.
            self._previous = frame
            self._detections = detections
            self._since_full = 0 if report.mode == "full" else self._since_full + 1
            self.last_read = report
            return detections
