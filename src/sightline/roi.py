# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Read one crop and move its detections back into the full image."""

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import OcrEngine, OcrError, TextDetection
from sightline.timing import Timings, timed


def read_region(
    engine: OcrEngine, frame: Frame, region: Box, timings: Timings | None = None
) -> tuple[TextDetection, ...]:
    with timed(timings, "roi.crop"):
        coords = (region.left, region.top, region.right, region.bottom)
        left, top, right, bottom = map(int, coords)
        if (left, top, right, bottom) != coords or region.clipped_to(frame.pixel_bounds) != region:
            raise ValueError("An OCR region must use whole pixels inside the image")
        # A crop has no mouse mapping. Only the complete fresh frame can drive input.
        crop = Frame(
            frame.pixels[top:bottom, left:right], captured_at=frame.captured_at, source=frame.source
        )
    detections = engine.read(crop)
    with timed(timings, "roi.translate"):
        result = []
        for item in detections:
            box = item.box
            if box.clipped_to(crop.pixel_bounds) != box:
                raise OcrError("OCR returned a detection outside its crop")
            full = Box(left + box.left, top + box.top, left + box.right, top + box.bottom)
            result.append(TextDetection(item.text, full, item.confidence))
        return tuple(result)
