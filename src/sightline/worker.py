# Copyright (c) 2026 Yousif Lubbad. See LICENSE.
"""Keep one recognizer in a spawned child process. One request is in flight at a time."""

import math
import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from typing import Self, cast

from sightline.frame import Frame
from sightline.ocr import OcrEngine, OcrError, TextDetection
from sightline.timing import Timings, timed


def _serve(connection: Connection, factory: Callable[[], OcrEngine]) -> None:
    try:
        engine = factory()
        prepare = getattr(engine, "prepare", None)
        if prepare is not None:
            prepare()
        connection.send(("ready", None, 0))
        while (request := connection.recv()) is not None:
            # Unpickling a NumPy array can make it writable. Restore the Frame contract.
            frame = Frame(
                request.pixels, request.desktop_bounds, request.captured_at, request.source
            )
            start = time.perf_counter_ns()
            result = engine.read(frame)
            connection.send(("result", result, time.perf_counter_ns() - start))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        with suppress(OSError):
            connection.send(("error", f"{type(error).__name__}: {error}", 0))
    finally:
        connection.close()


@contextmanager
def _deadline(process: BaseProcess, seconds: float, expected: str) -> Iterator[None]:
    # A child can stall before reading a large request. Killing it also wakes a
    # parent blocked in send(), or midway through receiving a partial reply.
    end = time.monotonic() + seconds
    expired = threading.Event()

    def interrupt() -> None:
        expired.set()
        with suppress(ProcessLookupError):
            process.kill()

    timer = threading.Timer(seconds, interrupt)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()
        timer.join()
        if expired.is_set() or time.monotonic() >= end:
            raise OcrError(f"OCR worker timed out waiting for {expected}")


class ProcessOCR:
    """Use a top-level factory or picklable partial. Close the worker after use."""

    def __init__(
        self,
        factory: Callable[[], OcrEngine],
        *,
        timeout: float = 45,
        startup_timeout: float = 120,
        timings: Timings | None = None,
    ) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (timeout, startup_timeout)):
            raise ValueError("Worker timeouts must be positive finite seconds")
        self.factory = factory
        self.timeout = timeout
        self.startup_timeout = startup_timeout
        self.timings = timings
        self._lock = threading.Lock()
        self._process: BaseProcess | None = None
        self._connection: Connection | None = None
        self._closed = False

    def _stop(self, *, force: bool) -> None:
        connection, process = self._connection, self._process
        self._connection = self._process = None
        if connection is not None:
            if not force:
                with suppress(OSError):
                    connection.send(None)
            connection.close()
        if process is not None and process.pid is not None:
            if not force:
                process.join(1)
            if process.is_alive():
                process.terminate()
                process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()

    def _receive(self, expected: str) -> tuple[object, int]:
        assert self._connection is not None
        tag, payload, elapsed = self._connection.recv()
        if tag == "error":
            raise OcrError(f"OCR worker failed: {payload}")
        if tag != expected:
            raise OcrError(f"Unexpected OCR worker reply: {tag}")
        return payload, int(elapsed)

    def _ensure_ready(self) -> None:
        if self._closed:
            raise OcrError("OCR worker is closed")
        if self._connection is not None:
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=_serve, args=(child, self.factory), daemon=True)
        self._process, self._connection = process, parent
        try:
            with timed(self.timings, "worker.start"):
                process.start()
                child.close()
                with _deadline(process, self.startup_timeout, "ready"):
                    self._receive("ready")
        except BaseException as error:
            self._stop(force=True)
            if isinstance(error, Exception) and not isinstance(error, OcrError):
                raise OcrError("Could not start the OCR worker; use a picklable factory") from error
            raise
        finally:
            child.close()

    def prepare(self) -> None:
        with self._lock:
            self._ensure_ready()

    def read(self, frame: Frame) -> tuple[TextDetection, ...]:
        with self._lock:
            self._ensure_ready()
            assert self._connection is not None and self._process is not None
            try:
                with timed(self.timings, "worker.roundtrip"):
                    with _deadline(self._process, self.timeout, "result"):
                        self._connection.send(frame)
                        payload, elapsed = self._receive("result")
                if self.timings is not None:
                    self.timings.record("worker.infer", elapsed)
                return cast(tuple[TextDetection, ...], payload)
            except BaseException as error:
                # Discard this connection so a late reply cannot satisfy a later request.
                self._stop(force=True)
                if isinstance(error, (EOFError, OSError)):
                    raise OcrError("Lost connection to the OCR worker") from error
                raise

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._stop(force=False)

    def __enter__(self) -> Self:
        self.prepare()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
