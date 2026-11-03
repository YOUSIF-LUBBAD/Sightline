# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Sightline: read and operate interfaces from pixels."""

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.locator import AmbiguousTarget, TargetNotFound, TextLocator
from sightline.ocr import TextDetection
from sightline.session import ClickResult, LocatedText, Observation, Sightline

__all__ = [
    "AmbiguousTarget",
    "Box",
    "ClickResult",
    "Frame",
    "LocatedText",
    "Observation",
    "Sightline",
    "TargetNotFound",
    "TextDetection",
    "TextLocator",
]

__version__ = "0.1.0.dev1"
__author__ = "Yousif Lubbad"
