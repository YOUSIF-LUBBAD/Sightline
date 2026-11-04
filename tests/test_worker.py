# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Start a real child to check isolation, reuse, timeout recovery and cleanup."""

import multiprocessing as mp
import os
import signal
import threading
import time

import numpy as np
import pytest

from sightline.frame import Frame
from sightline.geometry import Box
from sightline.ocr import OcrError, TextDetection
from sightline.timing import Timings
from sightline.worker import ProcessOCR


class TestReader:
    __test__ = False

    def read(self, frame):
        assert not frame.pixels.flags.writeable
        flag = int(frame.pixels[0, 0, 0])
        if flag == 1:
            time.sleep(6)
        if flag == 2:
            raise ValueError("bad input")
        if flag == 3:
            os._exit(4)
        return (TextDetection(str(os.getpid()), Box(0, 0, 1, 1), 1),)


def slow_factory():
    time.sleep(10)
    return TestReader()


def broken_factory():
    raise RuntimeError("model unavailable")


def frame(flag=0):
    return Frame(np.full((1, 1, 3), flag, dtype=np.uint8))


def test_child_is_reused_and_closes_idempotently() -> None:
    timings = Timings()
    with ProcessOCR(TestReader, timings=timings) as worker:
        first = worker.read(frame())
        assert int(first[0].text) != os.getpid()
        assert worker.read(frame()) == first
    assert timings.snapshot()["worker.infer"].calls == 2
    assert timings.snapshot()["worker.start"].calls == 1
    worker.close()
    with pytest.raises(OcrError, match="closed"):
        worker.read(frame())


@pytest.mark.parametrize(
    "flag,message", [(1, "timed out"), (2, "bad input"), (3, "Lost connection")]
)
def test_failed_worker_cannot_return_a_stale_reply(flag, message) -> None:
    with ProcessOCR(TestReader, timeout=1) as worker:
        first = worker.read(frame())
        with pytest.raises(OcrError, match=message):
            worker.read(frame(flag))
        assert worker.read(frame()) != first


def test_startup_error_reaches_parent() -> None:
    with pytest.raises(OcrError, match="model unavailable"):
        with ProcessOCR(broken_factory):
            pass


@pytest.mark.skipif(os.name != "posix", reason="Stopping a child this way requires POSIX")
def test_timeout_includes_sending_pixels_to_a_stalled_child() -> None:
    with ProcessOCR(TestReader, timeout=0.5) as worker:
        first = worker.read(frame())
        pid = int(first[0].text)
        large = Frame(np.zeros((512, 1024, 3), dtype=np.uint8))
        resumed = threading.Event()

        def rescue() -> None:
            # Keep the regression test bounded even if the deadline is broken.
            resumed.set()
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass

        watchdog = threading.Timer(4, rescue)
        watchdog.start()
        try:
            os.kill(pid, signal.SIGSTOP)
            with pytest.raises(OcrError, match="timed out"):
                worker.read(large)
            assert not resumed.is_set(), "Sending the frame outlasted the worker timeout"
            assert worker.read(frame()) != first
        finally:
            watchdog.cancel()
            watchdog.join()


def test_stalled_startup_is_reaped() -> None:
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(OcrError, match="timed out waiting for ready"):
        with ProcessOCR(slow_factory, startup_timeout=0.4):
            pass
    assert {p.pid for p in mp.active_children()} <= before
