# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Join perception and input without hidden clicks or retries."""

import math
import time
from dataclasses import dataclass

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.locator import TextLocator
from sightline.mouse import InputError, MouseController
from sightline.ocr import OcrEngine, OcrError, TextDetection
from sightline.sources import FrameSource
from sightline.timing import Timings, timed


@dataclass(frozen=True)
class Observation:
    frame: Frame
    detections: tuple[TextDetection, ...]


@dataclass(frozen=True)
class LocatedText:
    frame: Frame
    detection: TextDetection


@dataclass(frozen=True)
class ClickResult:
    target: LocatedText
    point: tuple[int, int]
    executed: bool


class Sightline:
    """Read pixels and click a unique label only when explicitly requested.

    The caller owns the frame source and must close it when finished. Prepare
    expensive OCR models before using this class so model loading does not age
    the screenshot. A fresh screenshot still cannot guarantee an unchanged UI.
    """

    def __init__(
        self,
        source: FrameSource,
        ocr: OcrEngine,
        mouse: MouseController | None = None,
        *,
        max_frame_age: float = 5.0,
        timings: Timings | None = None,
    ) -> None:
        if not math.isfinite(max_frame_age) or max_frame_age <= 0:
            raise ValueError("Maximum frame age must be a positive number of seconds")
        self.source = source
        self.ocr = ocr
        self.mouse = mouse
        self.max_frame_age = max_frame_age
        self.timings = timings

    def look(self) -> Observation:
        with timed(self.timings, "capture"):
            frame = self.source.capture()
        with timed(self.timings, "ocr.read"):
            return Observation(frame, self.ocr.read(frame))

    def find_text(self, text: str | TextLocator) -> LocatedText:
        locator = TextLocator(text) if isinstance(text, str) else text
        observation = self.look()
        with timed(self.timings, "locate"):
            detection = locator.one(observation.detections)
        # A custom OCR engine must obey the same image bounds as EasyOCR.
        if detection.box.clipped_to(observation.frame.pixel_bounds) != detection.box:
            raise OcrError("The matched text box is outside the captured image")
        return LocatedText(observation.frame, detection)

    def click_text(self, text: str | TextLocator, *, execute: bool = False) -> ClickResult:
        """Find once, then return a proposed click or send exactly one left click."""
        target = self.find_text(text)
        try:
            with timed(self.timings, "input.map"):
                point = target.frame.to_desktop(*target.detection.box.center)
        except ValueError as error:
            raise InputError(str(error)) from error
        if execute:
            if self.mouse is None:
                raise InputError("A mouse controller is required to execute a click")
            with timed(self.timings, "input.validate"):
                width, height = self.mouse.desktop_size()
                if target.frame.desktop_bounds != Box(0, 0, width, height):
                    raise InputError(
                        "Desktop size changed or the frame is not from the primary screen"
                    )
                age = time.monotonic() - target.frame.captured_at
                if age < 0 or age > self.max_frame_age:
                    raise InputError("The frame is too old to click; capture a fresh screen")
            with timed(self.timings, "input.click"):
                self.mouse.click(*point)
        return ClickResult(target, point, execute)
