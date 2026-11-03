# Phase 1 architecture

Yousif Lubbad's Sightline starts with explicit interfaces so future performance
work can replace one stage without changing what a frame or detection means.

| Module | Contract |
| --- | --- |
| `geometry` | A finite positive-area `Box`, with exclusive right and bottom edges |
| `frame` | Immutable `uint8` RGB pixels, a monotonic timestamp, and optional mouse-space bounds |
| `sources` | `FrameSource.capture()` returns a `Frame`; `close()` releases resources |
| `desktop` | `MSSFrameSource` captures the primary display; input size maps pixels to mouse coordinates |
| `ocr` | `OcrEngine.read(frame)` returns text, boxes, and confidence values |
| `easyocr_engine` | Loads one reader lazily, converts RGB to BGR, and clips valid boxes |
| `locator` | Normalizes Unicode, case, and spaces; filters confidence; requires one match |
| `mouse` | Sends bounded primary-screen left clicks with the fail-safe enabled |
| `session` | Joins these contracts and executes only explicitly requested clicks |

## Small API example

```python
from pathlib import Path

from sightline import Sightline, TextLocator
from sightline.desktop import MSSFrameSource
from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine
from sightline.mouse import PyAutoGUIMouse

# Download missing weights separately, or explicitly enable downloads.
engine = EasyOCREngine(EasyOCRConfig(model_directory=Path(".models")))
engine.prepare()
mouse = PyAutoGUIMouse()
with MSSFrameSource(input_size=mouse.desktop_size()) as source:
    session = Sightline(source, engine, mouse)
    proposed = session.click_text(TextLocator("Start", min_confidence=0.5))
    print(proposed.point, proposed.executed)  # A preview: executed is False.
    # To act on your own controlled app, call again with execute=True.
```

`look()` returns an `Observation` containing the frame and all detections.
`find_text()` returns a `LocatedText` with its originating frame.
`click_text()` performs a fresh capture and match each time, then returns a
`ClickResult`. A preview is not a stored click that can be executed later.

The library borrows the source; the caller closes it, normally with `with`.
Keep desktop capture on the creating thread. Importing the public package does
not load EasyOCR, PyTorch, MSS, or PyAutoGUI. Optional adapters import backends
when needed, so headless users can inspect images and run unit tests.

## Failure behavior

No automatic retries send extra clicks. Missing and ambiguous labels raise
`TargetNotFound` and `AmbiguousTarget`. Backend failures use `CaptureError`,
`OcrError`, and `InputError`, preserving the underlying cause. Bad configuration
raises `ValueError`.

Input requires a mapped frame, a matching primary desktop size, and a timestamp
within the configured age limit. Image-only sources have no mapping. Coordinate
scaling has unit tests; it is not verified on every DPI setup.

OCR identifies text rectangles, not interactive semantics. Clicking a label's
center is the limited Phase 1 strategy. Stable-screen waits, tracking, richer
locators, post-action verification, and recovery are later work.
