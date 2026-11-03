# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Find a label in your own local app. Actual clicking requires --click."""

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

from sightline.desktop import MSSFrameSource
from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine
from sightline.locator import AmbiguousTarget, TargetNotFound, TextLocator
from sightline.mouse import InputError, PyAutoGUIMouse
from sightline.ocr import OcrError
from sightline.session import Sightline
from sightline.sources import CaptureError, ImageFrameSource


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="?", default="Start", help="Unique visible text to find")
    parser.add_argument("--click", action="store_true", help="Send one left click to the match")
    parser.add_argument("--image", type=Path, help="Read a saved image without accessing a desktop")
    parser.add_argument("--model-dir", type=Path, default=Path(".models"))
    parser.add_argument(
        "--download-models", action="store_true", help="Allow EasyOCR to download missing weights"
    )
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument(
        "--max-frame-age",
        type=float,
        default=5.0,
        help="Refuse clicks after this many seconds; increase only for a static local demo",
    )
    args = parser.parse_args(argv)
    if args.image is not None and args.click:
        parser.error("--image and --click cannot be used together")
    try:
        locator = TextLocator(args.text, min_confidence=args.min_confidence)
        if not math.isfinite(args.max_frame_age) or args.max_frame_age <= 0:
            raise ValueError("Maximum frame age must be a positive number of seconds")
        engine = EasyOCREngine(
            EasyOCRConfig(model_directory=args.model_dir, download_enabled=args.download_models)
        )
        # Model loading happens before capture, so it cannot make the frame stale.
        engine.prepare()
        if args.image is not None:
            with ImageFrameSource.from_file(args.image) as source:
                found = Sightline(source, engine).find_text(locator)
                payload = {
                    "text": found.detection.text,
                    "confidence": found.detection.confidence,
                    "pixel_box": asdict(found.detection.box),
                    "desktop_point": None,
                    "executed": False,
                }
        else:
            mouse = PyAutoGUIMouse()
            with MSSFrameSource(input_size=mouse.desktop_size()) as desktop:
                session = Sightline(desktop, engine, mouse, max_frame_age=args.max_frame_age)
                result = session.click_text(locator, execute=args.click)
                payload = {
                    "text": result.target.detection.text,
                    "confidence": result.target.detection.confidence,
                    "pixel_box": asdict(result.target.detection.box),
                    "desktop_point": result.point,
                    "executed": result.executed,
                }
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except (
        CaptureError,
        OcrError,
        InputError,
        TargetNotFound,
        AmbiguousTarget,
        ValueError,
    ) as error:
        print(f"Sightline: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Sightline: stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
