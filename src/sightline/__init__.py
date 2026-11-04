# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Sightline: read and operate interfaces from pixels."""

from sightline.cache import CachedOCR
from sightline.frame import Frame
from sightline.geometry import Box
from sightline.incremental import IncrementalOCR, ReadReport
from sightline.locator import AmbiguousTarget, TargetNotFound, TextLocator
from sightline.native import NativeBackend
from sightline.ocr import TextDetection
from sightline.planning import RegionPolicy
from sightline.session import ClickResult, LocatedText, Observation, Sightline
from sightline.timing import Timings
from sightline.worker import ProcessOCR

__all__ = [
    "AmbiguousTarget",
    "Box",
    "CachedOCR",
    "ClickResult",
    "Frame",
    "IncrementalOCR",
    "LocatedText",
    "NativeBackend",
    "Observation",
    "ProcessOCR",
    "ReadReport",
    "RegionPolicy",
    "Sightline",
    "TargetNotFound",
    "TextDetection",
    "TextLocator",
    "Timings",
]

__version__ = "0.2.0.dev1"
__author__ = "Yousif Lubbad"
