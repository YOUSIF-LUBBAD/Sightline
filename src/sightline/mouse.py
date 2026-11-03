# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""A small mouse interface with bounds checks and the native fail-safe intact."""

from collections.abc import Callable
from typing import Protocol, cast


class InputError(RuntimeError):
    """Mouse input was rejected or stopped."""


class MouseController(Protocol):
    def desktop_size(self) -> tuple[int, int]: ...

    def click(self, x: int, y: int) -> None: ...


class MouseBackend(Protocol):
    FAILSAFE: bool

    def size(self) -> tuple[int, int]: ...

    def click(self, x: int, y: int, *, button: str) -> None: ...


def _open_pyautogui() -> MouseBackend:
    try:
        import pyautogui
    except ImportError as error:
        raise InputError("Install Sightline's desktop extra to control the mouse") from error
    except Exception as error:
        raise InputError("Cannot open mouse input; check display access and permissions") from error
    return cast(MouseBackend, pyautogui)


class PyAutoGUIMouse:
    """Primary-screen mouse input, loaded only when the caller needs it."""

    def __init__(self, *, backend_factory: Callable[[], MouseBackend] = _open_pyautogui) -> None:
        self._factory = backend_factory
        self._backend: MouseBackend | None = None

    def _get_backend(self) -> MouseBackend:
        if self._backend is None:
            self._backend = self._factory()
        # Moving the cursor to a primary-screen corner must always remain an escape.
        self._backend.FAILSAFE = True
        return self._backend

    def desktop_size(self) -> tuple[int, int]:
        try:
            width, height = self._get_backend().size()
            if width <= 0 or height <= 0:
                raise ValueError("Invalid desktop size")
            return int(width), int(height)
        except InputError:
            raise
        except Exception as error:
            raise InputError("Cannot read the primary desktop size") from error

    def click(self, x: int, y: int) -> None:
        if type(x) is not int or type(y) is not int:
            raise InputError("Mouse coordinates must be integers")
        width, height = self.desktop_size()
        if not 0 <= x < width or not 0 <= y < height:
            raise InputError("Refusing to click outside the primary desktop")
        try:
            self._get_backend().click(x, y, button="left")
        except Exception as error:
            raise InputError(
                "Mouse input stopped; check the corner fail-safe and permissions"
            ) from error
