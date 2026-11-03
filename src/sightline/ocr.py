# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""OCR results stay independent of the recognition library."""

import math
from dataclasses import dataclass
from typing import Protocol

from sightline.frame import Frame
from sightline.geometry import Box


class OcrError(RuntimeError):
    """Text recognition could not finish or returned invalid data."""


@dataclass(frozen=True)
class TextDetection:
    """One OCR label with an image-local box and confidence from zero to one."""

    text: str
    box: Box
    confidence: float

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("An OCR label cannot be empty")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("OCR confidence must be between zero and one")


class OcrEngine(Protocol):
    def read(self, frame: Frame) -> tuple[TextDetection, ...]:
        """Read labels using only the frame's pixels."""
        ...
