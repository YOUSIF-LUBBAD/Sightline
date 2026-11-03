# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Keep normal tests independent of a desktop and model downloads."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-ocr", action="store_true", help="Run real model tests")
    parser.addoption("--run-live", action="store_true", help="Click the local demo app")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        # A normal test run must never move the user's mouse.
        for marker, option in (("ocr", "--run-ocr"), ("live", "--run-live")):
            if marker in item.keywords and not config.getoption(option):
                item.add_marker(pytest.mark.skip(reason=f"Opt in with {option}"))
