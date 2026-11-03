# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Find visible text without guessing when more than one label matches."""

import math
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from sightline.ocr import TextDetection


class TargetNotFound(LookupError):
    """No visible label met the locator rules."""


class AmbiguousTarget(LookupError):
    """More than one label matched; clicking one would be a guess."""


def normalize_text(text: str) -> str:
    """Ignore case, repeated whitespace, and equivalent Unicode forms."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


@dataclass(frozen=True)
class TextLocator:
    text: str
    mode: Literal["exact", "contains"] = "exact"
    min_confidence: float = 0.5

    def __post_init__(self) -> None:
        if not normalize_text(self.text):
            raise ValueError("Search text cannot be empty")
        if self.mode not in ("exact", "contains"):
            raise ValueError("Choose exact or contains matching")
        if not math.isfinite(self.min_confidence) or not 0 <= self.min_confidence <= 1:
            raise ValueError("Minimum confidence must be between zero and one")

    def find_all(self, detections: Iterable[TextDetection]) -> tuple[TextDetection, ...]:
        target = normalize_text(self.text)
        matches = []
        for detection in detections:
            if detection.confidence < self.min_confidence:
                continue
            label = normalize_text(detection.text)
            if (self.mode == "exact" and target == label) or (
                self.mode == "contains" and target in label
            ):
                matches.append(detection)
        return tuple(matches)

    def one(self, detections: Iterable[TextDetection]) -> TextDetection:
        matches = self.find_all(detections)
        if not matches:
            raise TargetNotFound(f"No visible match for {self.text!r}")
        if len(matches) > 1:
            raise AmbiguousTarget(
                f"Found {len(matches)} matches for {self.text!r}; no click was made"
            )
        return matches[0]
