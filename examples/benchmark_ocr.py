# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Measure full and incremental OCR on the same app-style pixel sequence."""

import argparse
import json
import platform
import statistics
import time
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine
from sightline.frame import Frame
from sightline.incremental import IncrementalOCR
from sightline.locator import TextLocator
from sightline.native import NativeBackend
from sightline.ocr import OcrEngine, TextDetection
from sightline.timing import Timings
from sightline.worker import ProcessOCR


def sequence() -> list[tuple[Frame, str]]:
    font = ImageFont.load_default(size=40)
    frames = []
    for status, dark in [
        ("Ready", False),
        ("Ready", False),
        ("Done", False),
        ("Done", False),
        ("Ready", False),
        ("Done", False),
        ("Done", True),
    ]:
        image = Image.new("RGB", (800, 480), "#172130" if dark else "#f0f4fa")
        draw = ImageDraw.Draw(image)
        for x, label in ((180, "Start"), (620, status)):
            draw.rounded_rectangle((x - 105, 196, x + 105, 284), radius=9, fill="#164e8a")
            draw.text((x, 240), label, font=font, fill="white", anchor="mm")
        frames.append((Frame(np.asarray(image), source="owned sequence"), status))
    return frames


def check_labels(result: tuple[TextDetection, ...], status: str) -> None:
    if len(result) != 2:
        raise AssertionError(f"Expected two labels, found {[item.text for item in result]}")
    for label, x in (("Start", 180), (status, 620)):
        if not TextLocator(label).one(result).box.contains(x, 240):
            raise AssertionError(f"{label} missed its known screen location")


def measure(engine: OcrEngine, native: NativeBackend | None, rounds: int) -> dict:
    frames = sequence()
    check_labels(engine.read(frames[0][0]), frames[0][1])
    runs = []
    totals: dict[str, list[int]] = {"full": [], "incremental": []}
    for n in range(rounds):
        order = ("full", "incremental") if n % 2 == 0 else ("incremental", "full")
        for name in order:
            timing = Timings()
            fast = (
                IncrementalOCR(engine, backend=native, timings=timing)
                if name == "incremental"
                else None
            )
            reader = fast if fast is not None else engine
            steps = []
            for frame, status in frames:
                start = time.perf_counter_ns()
                with timing.measure("ocr.read"):
                    found = reader.read(frame)
                elapsed = time.perf_counter_ns() - start
                check_labels(found, status)
                steps.append(
                    {
                        "expected": ["Start", status],
                        "elapsed_ns": elapsed,
                        "read": asdict(fast.last_read)
                        if fast and fast.last_read
                        else {"mode": "full"},
                    }
                )
            total = sum(step["elapsed_ns"] for step in steps)
            totals[name].append(total)
            runs.append(
                {
                    "round": n + 1,
                    "method": name,
                    "sequence_ns": total,
                    "frames": steps,
                    "stages": {k: asdict(v) for k, v in timing.snapshot().items()},
                    "cache": asdict(fast.cache.info()) if fast else None,
                }
            )
    medians = {name: statistics.median(values) for name, values in totals.items()}
    return {
        "schema": 1,
        "author": "Yousif Lubbad",
        "kind": "owned OCR sequence",
        "size": "800x480",
        "rounds": rounds,
        "python": platform.python_version(),
        "easyocr": version("easyocr"),
        "torch": version("torch"),
        "numpy": np.__version__,
        "pixel_backend": "c" if native else "numpy",
        "median_sequence_ns": medians,
        "full_over_incremental": medians["full"] / medians["incremental"],
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--native", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path(".models"))
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    native = NativeBackend(args.native) if args.native else None
    config = EasyOCRConfig(model_directory=args.model_dir, download_enabled=args.download_models)
    with ExitStack() as stack:
        engine: OcrEngine
        if args.worker:
            engine = stack.enter_context(ProcessOCR(partial(EasyOCREngine, config)))
        else:
            model = EasyOCREngine(config)
            model.prepare()
            engine = model
        result = measure(engine, native, args.rounds)
    result["worker"] = args.worker
    data = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data, encoding="utf-8")
    else:
        print(data, end="")


if __name__ == "__main__":
    main()
