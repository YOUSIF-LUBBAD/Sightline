# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Real model inference on images drawn here, with no borrowed screenshots."""

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from sightline.frame import Frame
from sightline.locator import TextLocator


@pytest.mark.ocr
@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_real_ocr_finds_generated_label(real_ocr, dark: bool) -> None:
    image = Image.new("RGB", (640, 360), "#14202e" if dark else "#f4f6fa")
    draw = ImageDraw.Draw(image)
    draw.rectangle((200, 170, 440, 250), fill="#145db5")
    draw.text((320, 210), "Start", font=ImageFont.load_default(size=48), fill="white", anchor="mm")
    frame = Frame(np.asarray(image))
    found = TextLocator("Start").one(real_ocr.read(frame))
    assert found.box.contains(320, 210)
    assert found.confidence >= 0.5


@pytest.mark.ocr
def test_real_ocr_does_not_invent_text_on_blank_image(real_ocr) -> None:
    frame = Frame(np.full((200, 400, 3), 255, dtype=np.uint8))
    assert real_ocr.read(frame) == ()
