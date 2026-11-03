# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""The base package must work without a desktop or heavy model imports."""

import os
import subprocess
import sys
from importlib.metadata import version

import sightline


def test_installed_version_matches_public_version() -> None:
    assert version("sightline-ui") == sightline.__version__


def test_import_does_not_load_desktop_or_models() -> None:
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sightline, sys; "
            "assert not {'easyocr', 'torch', 'pyautogui', 'mss'} & sys.modules.keys()",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
