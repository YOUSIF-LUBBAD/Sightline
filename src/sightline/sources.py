# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Frame sources share one small interface; the core never sees an app tree."""

from pathlib import Path
from types import TracebackType
from typing import Protocol, Self

import numpy as np
from PIL import Image, ImageOps

from sightline.frame import Frame


class CaptureError(RuntimeError):
    """The requested pixels could not be captured."""


class FrameSource(Protocol):
    def capture(self) -> Frame:
        """Return one fresh RGB frame."""
        ...

    def close(self) -> None:
        """Release capture resources."""
        ...


class ImageFrameSource:
    """Repeat a local image for offline inspection and deterministic tests."""

    def __init__(self, frame: Frame) -> None:
        self._frame = frame
        self._closed = False

    @classmethod
    def from_file(cls, path: str | Path) -> Self:
        try:
            with Image.open(path) as image:
                # Respect camera rotation, then use the same RGB format everywhere.
                pixels = np.array(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as error:
            raise CaptureError(f"Cannot load image: {path}") from error
        return cls(Frame(pixels, source=str(path)))

    def capture(self) -> Frame:
        if self._closed:
            raise CaptureError("The image source is closed")
        return Frame(self._frame.pixels, self._frame.desktop_bounds, source=self._frame.source)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
