# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Build the optional C backend from the source shipped in the package."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def build_native(directory: Path, compiler: str | None = None) -> Path:
    cc = compiler or shutil.which("cc") or shutil.which("clang") or shutil.which("cl")
    if not cc:
        raise RuntimeError("Install a C compiler or pass --cc with its path")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    target = directory / ("sightline_pixels" + suffix)
    source = Path(__file__).resolve().parent / "native_src" / "pixels.c"
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        built = Path(temporary) / target.name
        if Path(cc).stem.lower() == "cl":
            command = [cc, "/nologo", "/O2", "/LD", str(source), "/link", f"/OUT:{built}"]
        else:
            link = "-dynamiclib" if sys.platform == "darwin" else "-shared"
            command = [
                cc,
                "-std=c99",
                "-O3",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-fPIC",
                link,
                str(source),
                "-o",
                str(built),
            ]
        subprocess.run(command, check=True, cwd=temporary)
        # A failed build leaves the previous working library in place.
        os.replace(built, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory for the compiled library")
    parser.add_argument("--cc", help="C compiler executable")
    args = parser.parse_args()
    print(build_native(args.directory, args.cc))


if __name__ == "__main__":
    main()
