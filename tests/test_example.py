# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Check the example's public arguments without opening a display or downloading models."""

import os
import subprocess
import sys

import pytest


def run_example(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    return subprocess.run(
        [sys.executable, "-m", "examples.find_and_click", *arguments],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
        timeout=10,
    )


def test_help_works_without_a_desktop() -> None:
    result = run_example("--help")
    assert result.returncode == 0
    assert "--click" in result.stdout
    assert "--download-models" in result.stdout


def test_image_cannot_drive_mouse() -> None:
    result = run_example("--image", "missing.png", "--click")
    assert result.returncode == 2
    assert "cannot be used together" in result.stderr


@pytest.mark.parametrize("arguments", [("",), ("--min-confidence", "2"), ("--max-frame-age", "0")])
def test_invalid_locator_fails_before_model_loading(arguments: tuple[str, ...]) -> None:
    result = run_example(*arguments)
    assert result.returncode == 2
    assert "Sightline:" in result.stderr
    assert "EasyOCR" not in result.stderr
