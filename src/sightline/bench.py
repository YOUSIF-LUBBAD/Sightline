# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Benchmark the Python-facing NumPy and C kernels on the same in-memory frames."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np

from sightline.frame import Frame
from sightline.incremental import NumpyBackend
from sightline.native import NativeBackend


def summarize(samples: list[int]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples_ns": samples,
        "median_ns": statistics.median(ordered),
        "p95_ns": ordered[math.ceil(0.95 * len(ordered)) - 1],
    }


def _pair(
    reference: Callable[[], object], native: Callable[[], object], warmups: int, repeats: int
) -> dict[str, object]:
    samples: dict[str, list[int]] = {"numpy": [], "c": []}
    functions = [("numpy", reference), ("c", native)]
    for i in range(warmups + repeats):
        for name, function in functions if i % 2 == 0 else reversed(functions):
            start = time.perf_counter_ns()
            function()
            elapsed = time.perf_counter_ns() - start
            if i >= warmups:
                samples[name].append(elapsed)
    return {
        "numpy": summarize(samples["numpy"]),
        "c": summarize(samples["c"]),
        "numpy_over_c": statistics.median(samples["numpy"]) / statistics.median(samples["c"]),
    }


def _size(value: str) -> tuple[int, int]:
    try:
        width, height = map(int, value.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a size like 1280x800") from error
    if min(width, height) < 1 or width * height > 16_777_216:
        raise argparse.ArgumentTypeError("Choose a positive size up to 16 megapixels")
    return width, height


def benchmark(
    library: Path,
    sizes: list[tuple[int, int]],
    *,
    tile: int,
    threshold: int,
    warmups: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    if repeats < 1 or warmups < 0 or not 1 <= tile <= 65535 or not 0 <= threshold <= 255:
        raise ValueError("Invalid sample count, warmup count, tile size, or threshold")
    native, reference = NativeBackend(library), NumpyBackend()
    rows = []
    for width, height in sizes:
        rng = np.random.default_rng(seed)
        rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        before = Frame(rgb)
        for case in ("static", "local", "fragmented", "full"):
            other = rgb.copy()
            if case == "local":
                y, x = height // 3, width // 3
                other[y : y + 24, x : x + 80] ^= 255
            elif case == "fragmented":
                other[:: tile * 2, :: tile * 2] ^= 255
                other[tile :: tile * 2, tile :: tile * 2] ^= 255
            elif case == "full":
                np.bitwise_xor(other, 255, out=other)
            after = Frame(other)
            diff_np = partial(reference.changed_tiles, before, after, tile, threshold)
            diff_c = partial(native.changed_tiles, before, after, tile, threshold)
            grid = diff_np()
            np.testing.assert_array_equal(grid, diff_c())
            merge_np, merge_c = (
                partial(reference.merge_tiles, grid),
                partial(native.merge_tiles, grid),
            )
            expected_boxes = merge_np()
            if expected_boxes != merge_c():
                raise AssertionError("C and NumPy region outputs differ")
            rows.append(
                {
                    "size": f"{width}x{height}",
                    "case": case,
                    "changed_tiles": int(grid.sum()),
                    "rectangles": len(expected_boxes),
                    "diff": _pair(diff_np, diff_c, warmups, repeats),
                    "merge": _pair(merge_np, merge_c, warmups, repeats),
                }
            )
    return {
        "schema": 1,
        "author": "Yousif Lubbad",
        "kind": "pixel kernels",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.system(),
        "machine": platform.machine(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "parameters": {
            "tile": tile,
            "threshold": threshold,
            "warmups": warmups,
            "repeats": repeats,
            "seed": seed,
        },
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--size", action="append", type=_size, help="Repeat to compare resolutions")
    parser.add_argument("--tile", type=int, default=32)
    parser.add_argument("--threshold", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = benchmark(
            args.native,
            args.size or [(1280, 800), (1920, 1080)],
            tile=args.tile,
            threshold=args.threshold,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed,
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
