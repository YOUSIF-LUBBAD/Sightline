# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Run only on an isolated desktop: this test sends an actual mouse click."""

import os
import time
from pathlib import Path

import pytest
from PIL import Image

from sightline.desktop import MSSFrameSource
from sightline.locator import TextLocator
from sightline.mouse import PyAutoGUIMouse
from sightline.session import Sightline


@pytest.mark.ocr
@pytest.mark.live
@pytest.mark.parametrize("dark, position", [(False, (80, 80)), (True, (250, 170))])
def test_pixels_to_click_changes_the_native_app(real_ocr, dark, position) -> None:
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
            session = Sightline(source, real_ocr, mouse, max_frame_age=30)
            preview = session.click_text("Start")
            root.update()
            assert preview.executed is False
            assert app.completed is False

            result = session.click_text("Start", execute=True)
            assert result.executed is True
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
                Image.fromarray(result.target.frame.pixels).save(directory / f"{theme}-before.png")
                Image.fromarray(after.frame.pixels).save(directory / f"{theme}-after.png")
    finally:
        root.destroy()
