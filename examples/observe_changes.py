# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Read successive images or primary-screen frames and print one JSON record per observation."""

import argparse
import json
import sys
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from pathlib import Path

from sightline.desktop import MSSFrameSource
from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine
from sightline.incremental import IncrementalOCR
from sightline.native import NativeBackend
from sightline.ocr import OcrEngine, OcrError
from sightline.session import Sightline
from sightline.sources import CaptureError, ImageFrameSource
from sightline.timing import Timings
from sightline.worker import ProcessOCR


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path, nargs="+", help="Read these images in order")
    inputs.add_argument("--live", type=int, metavar="COUNT", help="Read this many desktop frames")
    parser.add_argument("--native", type=Path, help="Compiled Sightline shared library")
    parser.add_argument("--worker", action="store_true", help="Run recognition in a child process")
    parser.add_argument("--model-dir", type=Path, default=Path(".models"))
    parser.add_argument("--download-models", action="store_true")
    args = parser.parse_args(argv)
    if args.live is not None and args.live < 1:
        parser.error("--live requires a positive count")
    timings = Timings()
    try:
        backend = NativeBackend(args.native) if args.native else None
        config = EasyOCRConfig(
            model_directory=args.model_dir, download_enabled=args.download_models
        )
        with ExitStack() as stack:
            engine: OcrEngine
            if args.worker:
                engine = stack.enter_context(
                    ProcessOCR(partial(EasyOCREngine, config), timings=timings)
                )
            else:
                local = EasyOCREngine(config)
                with timings.measure("model.prepare"):
                    local.prepare()
                engine = local
            fast = IncrementalOCR(engine, backend=backend, timings=timings)

            def emit(session: Sightline, index: int) -> None:
                observation = session.look()
                assert fast.last_read is not None
                print(
                    json.dumps(
                        {
                            "frame": index,
                            "read": asdict(fast.last_read),
                            "detections": [asdict(d) for d in observation.detections],
                            "cache": asdict(fast.cache.info()),
                            "timings": {k: asdict(v) for k, v in timings.snapshot().items()},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            if args.images:
                for index, path in enumerate(args.images):
                    # All files in this explicit sequence share one screen identity.
                    with timings.measure("image.load"):
                        source = ImageFrameSource.from_file(path, source_name="image sequence")
                    with source:
                        emit(Sightline(source, fast, timings=timings), index)
            else:
                source_live = stack.enter_context(MSSFrameSource())
                session = Sightline(source_live, fast, timings=timings)
                for index in range(args.live):
                    emit(session, index)
        return 0
    except (ValueError, OcrError, CaptureError, OSError) as error:
        print(f"Sightline: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
