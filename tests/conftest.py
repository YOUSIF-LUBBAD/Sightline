# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Keep normal tests independent of a desktop and model downloads."""

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-ocr", action="store_true", help="Run real model tests")
    parser.addoption("--run-live", action="store_true", help="Click the local demo app")
    parser.addoption(
        "--model-dir", type=Path, default=Path(".models"), help="EasyOCR weight folder"
    )
    parser.addoption("--download-models", action="store_true", help="Allow missing model downloads")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        # A normal test run must never move the user's mouse.
        for marker, option in (("ocr", "--run-ocr"), ("live", "--run-live")):
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"Opt in with {option}"))


@pytest.fixture(scope="session")
def real_ocr(request: pytest.FixtureRequest):
    # Nothing in the normal unit suite needs to import PyTorch or download weights.
    from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine

    engine = EasyOCREngine(
        EasyOCRConfig(
            model_directory=request.config.getoption("--model-dir"),
            download_enabled=request.config.getoption("--download-models"),
        )
    )
    engine.prepare()
    return engine
