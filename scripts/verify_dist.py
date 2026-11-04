# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Check distribution contents, then install the wheel in a clean temporary environment."""

import argparse
import os
import subprocess
import tarfile
import tempfile
import venv
import zipfile
from email.parser import Parser
from pathlib import Path


def verify(directory: Path, *, native: bool = False) -> None:
    wheels = list(directory.glob("*.whl"))
    archives = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(archives) != 1:
        raise ValueError("Build exactly one wheel and one source archive first")
    wheel = wheels[0].resolve()
    with zipfile.ZipFile(wheel) as package:
        names = set(package.namelist())
        required = {
            "sightline/session.py",
            "sightline/desktop.py",
            "sightline/py.typed",
            "sightline/native_src/pixels.c",
            "sightline/incremental.py",
            "sightline/bench.py",
            "sightline/worker.py",
        }
        if not required <= names:
            raise ValueError("The wheel is missing the public engine or its type marker")
        for notice in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
            if not any(name.endswith(f"/licenses/{notice}") for name in names):
                raise ValueError(f"The wheel is missing {notice}")
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(package.read(metadata_path).decode())
        if metadata["License-Expression"] != "LicenseRef-Proprietary":
            raise ValueError("The wheel has the wrong license metadata")
        if metadata["Author"] != "Yousif Lubbad":
            raise ValueError("The wheel has the wrong author metadata")
    with tarfile.open(archives[0], "r:gz") as archive:
        source_names = {name.split("/", 1)[-1] for name in archive.getnames()}
        required_source = {
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "examples/demo_app.py",
            "tests/test_live_demo.py",
            "scripts/verify_dist.py",
            "src/sightline/native_src/pixels.c",
            "examples/observe_changes.py",
            "examples/benchmark_ocr.py",
        }
        if not required_source <= source_names:
            raise ValueError("The source archive is missing notices, examples, or validation")
        if any(name.startswith((".models/", ".venv/", "artifacts/")) for name in source_names):
            raise ValueError("Local weights, environments, and test output must not be packaged")
    # Run away from the checkout so an editable install cannot hide a broken wheel.
    with tempfile.TemporaryDirectory(prefix="sightline-wheel-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [str(executable), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
            check=True,
        )
        smoke = """
import sys
from pathlib import Path
import numpy as np
import sightline
from sightline import Frame, Sightline, TextLocator
assert str(Path(sightline.__file__).resolve()).startswith(sys.prefix)
assert Frame(np.zeros((2, 3, 3), dtype=np.uint8)).width == 3
assert TextLocator('Start').text == 'Start'
assert Sightline.__module__ == 'sightline.session'
assert not {'torch', 'easyocr', 'mss', 'pyautogui'} & sys.modules.keys()
print('Clean wheel installation and headless public API passed')
"""
        if native:
            smoke += """
from sightline import NativeBackend
from sightline.diff import changed_tiles
from sightline.native_build import build_native
kernel = NativeBackend(build_native(Path.cwd() / "compiled"))
before = Frame(np.zeros((5, 11, 3), dtype=np.uint8))
pixels = before.pixels.copy()
pixels[4,10] = 255
after = Frame(pixels)
mask = kernel.changed_tiles(before, after, 4)
np.testing.assert_array_equal(mask, changed_tiles(before, after, 4))
assert kernel.merge_tiles(mask) == [(2,1,3,2)]
print('Installed C source builds and both native kernels pass')
"""
        subprocess.run([str(executable), "-I", "-c", smoke], cwd=temporary, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    parser.add_argument(
        "--native", action="store_true", help="Build and check the installed C source"
    )
    args = parser.parse_args()
    verify(args.directory, native=args.native)
