# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""No unit test sends real mouse events."""

import pytest

from sightline.mouse import InputError, PyAutoGUIMouse


class Backend:
    FAILSAFE = False

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []
        self.fail = False

    def size(self) -> tuple[int, int]:
        return 800, 600

    def click(self, x: int, y: int, *, button: str) -> None:
        assert self.FAILSAFE is True
        if self.fail:
            raise RuntimeError("corner fail-safe")
        self.calls.append((x, y, button))


def test_one_left_click_with_failsafe_enabled() -> None:
    backend = Backend()
    mouse = PyAutoGUIMouse(backend_factory=lambda: backend)
    mouse.click(20, 30)
    assert backend.calls == [(20, 30, "left")]
    assert backend.FAILSAFE is True


@pytest.mark.parametrize("point", [(-1, 20), (800, 20), (20, 600), (True, 10), (1.5, 10)])
def test_invalid_coordinates_never_reach_backend(point: tuple[int, int]) -> None:
    backend = Backend()
    with pytest.raises(InputError):
        PyAutoGUIMouse(backend_factory=lambda: backend).click(*point)
    assert backend.calls == []


def test_failsafe_error_is_preserved() -> None:
    backend = Backend()
    backend.fail = True
    with pytest.raises(InputError, match="fail-safe") as caught:
        PyAutoGUIMouse(backend_factory=lambda: backend).click(20, 30)
    assert str(caught.value.__cause__) == "corner fail-safe"
    assert backend.calls == []
