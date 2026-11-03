# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Lazy EasyOCR adapter. Screenshots remain local; model downloads are opt-in."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import OcrError, TextDetection

RawDetection = tuple[Sequence[Sequence[float]], str, float]


class Reader(Protocol):
    def readtext(
        self, image: NDArray[np.uint8], *, detail: int, paragraph: bool
    ) -> list[RawDetection]: ...


@dataclass(frozen=True)
class EasyOCRConfig:
    languages: tuple[str, ...] = ("en",)
    gpu: bool = False
    model_directory: Path | None = None
    download_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.languages or any(not code.strip() for code in self.languages):
            raise ValueError("Choose at least one non-empty EasyOCR language code")
        if len(set(self.languages)) != len(self.languages):
            raise ValueError("Language codes must not be repeated")


class EasyOCREngine:
    """Create one Reader when needed, then reuse it for later frames."""

    def __init__(
        self,
        config: EasyOCRConfig | None = None,
        *,
        reader_factory: Callable[[], Reader] | None = None,
    ) -> None:
        self.config = config or EasyOCRConfig()
        self._factory = reader_factory
        self._reader: Reader | None = None

    def prepare(self) -> None:
        """Load models before taking a screenshot that will be used for a click."""
        if self._reader is not None:
            return
        try:
            if self._factory is not None:
                self._reader = self._factory()
                return
            import easyocr

            directory = self.config.model_directory
            self._reader = cast(
                Reader,
                easyocr.Reader(
                    list(self.config.languages),
                    gpu=self.config.gpu,
                    model_storage_directory=str(directory) if directory is not None else None,
                    download_enabled=self.config.download_enabled,
                    verbose=False,
                ),
            )
        except ImportError as error:
            raise OcrError("Install Sightline's ocr extra to use EasyOCR") from error
        except Exception as error:
            raise OcrError(
                "Cannot load EasyOCR models; check the model folder or explicitly enable downloads"
            ) from error

    def read(self, frame: Frame) -> tuple[TextDetection, ...]:
        self.prepare()
        assert self._reader is not None
        try:
            # EasyOCR treats NumPy color images as OpenCV BGR, while our frames are RGB.
            image = np.ascontiguousarray(frame.pixels[:, :, ::-1])
            raw = self._reader.readtext(image, detail=1, paragraph=False)
            detections: list[TextDetection] = []
            for polygon, text, confidence in raw:
                if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
                    raise ValueError("Expected four points per OCR polygon")
                xs, ys = [float(p[0]) for p in polygon], [float(p[1]) for p in polygon]
                if not all(math.isfinite(v) for v in (*xs, *ys)):
                    raise ValueError("OCR coordinates must be finite")
                box = Box(min(xs), min(ys), max(xs), max(ys)).clipped_to(frame.pixel_bounds)
                if box is not None and text.strip():
                    detections.append(TextDetection(text, box, float(confidence)))
            return tuple(detections)
        except Exception as error:
            raise OcrError(
                "EasyOCR could not read this frame or returned invalid detections"
            ) from error
