# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Capture the primary desktop through MSS, without inspecting UI controls."""

import time
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self, cast

import numpy as np

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.sources import CaptureError


class ScreenShot(Protocol):
    width: int
    height: int

    @property
    def rgb(self) -> bytes: ...


class CaptureBackend(Protocol):
    @property
    def monitors(self) -> list[dict[str, int]]: ...

    def grab(self, monitor: dict[str, int]) -> ScreenShot: ...

    def close(self) -> None: ...


def _open_mss() -> CaptureBackend:
    try:
        import mss
    except ImportError as error:
        raise CaptureError("Install Sightline's desktop extra to capture a screen") from error
    try:
        return cast(CaptureBackend, mss.mss())
    except Exception as error:
        raise CaptureError(
            "Cannot open screen capture; check display access and permissions"
        ) from error


class MSSFrameSource:
    """Primary-monitor capture. Pass the mouse backend's size for DPI mapping."""

    def __init__(
        self,
        input_size: tuple[int, int] | None = None,
        *,
        backend_factory: Callable[[], CaptureBackend] = _open_mss,
    ) -> None:
        if input_size is not None and any(v <= 0 for v in input_size):
            raise ValueError("Input width and height must be positive")
        self._input_size = input_size
        self._factory = backend_factory
        self._backend: CaptureBackend | None = None
        self._closed = False

    def capture(self) -> Frame:
        if self._closed:
            raise CaptureError("The desktop source is closed")
        try:
            if self._backend is None:
                self._backend = self._factory()
            monitors = self._backend.monitors
            if len(monitors) < 2:
                raise CaptureError("No monitor is available")
            monitor = monitors[1]
            # Phase 1 only supports the primary screen, where mouse input is reliable.
            if monitor["left"] != 0 or monitor["top"] != 0:
                raise CaptureError(
                    "The first monitor is not primary; multi-monitor input is not ready"
                )
            captured_at = time.monotonic()
            shot = self._backend.grab(monitor)
            pixels = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)
            bounds = None
            if self._input_size is not None:
                width, height = self._input_size
                bounds = Box(0, 0, width, height)
            return Frame(pixels, bounds, captured_at, "desktop:primary")
        except CaptureError:
            raise
        except Exception as error:
            raise CaptureError("Screen capture failed; check display permissions") from error

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None
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
