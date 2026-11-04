# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Run only on an isolated desktop: this test sends an actual mouse click."""

import os
import time
from functools import partial
from pathlib import Path

import pytest
from PIL import Image

from sightline.desktop import MSSFrameSource
from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine
from sightline.incremental import IncrementalOCR
from sightline.locator import TextLocator
from sightline.mouse import PyAutoGUIMouse
from sightline.native import NativeBackend
from sightline.native_build import build_native
from sightline.session import Sightline
from sightline.worker import ProcessOCR


@pytest.fixture(scope="module")
def native_worker(request, tmp_path_factory):
    config = EasyOCRConfig(
        model_directory=request.config.getoption("--model-dir"),
        download_enabled=request.config.getoption("--download-models"),
    )
    backend = NativeBackend(build_native(tmp_path_factory.mktemp("native-worker")))
    with ProcessOCR(partial(EasyOCREngine, config)) as worker:
        yield worker, backend


@pytest.mark.ocr
@pytest.mark.live
@pytest.mark.parametrize("dark, position", [(False, (80, 80)), (True, (250, 170))])
@pytest.mark.parametrize("mode", ["full", "incremental"])
def test_pixels_to_click_changes_the_native_app(request, dark, position, mode) -> None:
    import tkinter as tk

    from examples.demo_app import DemoApp

    root = tk.Tk()
    try:
        app = DemoApp(root, dark=dark, x=position[0], y=position[1])
        root.update()
        root.lift()
        root.update()
        mouse = PyAutoGUIMouse()
        with MSSFrameSource(input_size=mouse.desktop_size()) as source:
            # This app stays still until clicked. Full-screen CPU OCR can exceed
            # the normal five-second limit, so this static fixture opts into 30.
            if mode == "incremental":
                worker, backend = request.getfixturevalue("native_worker")
                engine = IncrementalOCR(worker, backend=backend)
            else:
                engine = request.getfixturevalue("real_ocr")
            session = Sightline(source, engine, mouse, max_frame_age=30)
            preview = session.click_text("Start")
            root.update()
            assert preview.executed is False
            assert app.completed is False

            result = session.click_text("Start", execute=True)
            assert result.executed is True
            if mode == "incremental":
                assert engine.last_read.mode == "unchanged"
            deadline = time.monotonic() + 2
            while not app.completed and time.monotonic() < deadline:
                root.update()
                time.sleep(0.01)
            # The app state is a test oracle. Sightline only received screen pixels.
            assert app.completed is True
            root.update()
            after = session.look()
            assert TextLocator("Complete").one(after.detections)
            assert TextLocator("Start").find_all(after.detections) == ()

            output = os.environ.get("SIGHTLINE_ARTIFACT_DIR")
            if output:
                directory = Path(output)
                directory.mkdir(parents=True, exist_ok=True)
                theme = "dark" if dark else "light"
                Image.fromarray(result.target.frame.pixels).save(
                    directory / f"{theme}-{mode}-before.png"
                )
                Image.fromarray(after.frame.pixels).save(directory / f"{theme}-{mode}-after.png")
    finally:
        root.destroy()
