# Phase 2 · Faster repeated screen reads

Yousif Lubbad · validated on November 3, 2026

On seven original 800×480 frames, full-screen EasyOCR took **17.73 s** per sequence
and incremental OCR took **5.73 s**. These are medians from five rounds per method
with the same warmed CPU model in a persistent child process. The speedup was
**3.10×**. All 70 observations returned exactly two expected labels at the expected
screen positions.

## What changed on each frame

| Frame | What changed | Full read (ms) | Incremental read (ms) | Decision |
|---:|---|---:|---:|---|
| 1 | Initial screen | 2473.15 | 2504.15 | full |
| 2 | Same screen | 2549.51 | 0.99 | unchanged |
| 3 | Ready changed to Done | 2807.30 | 197.92 | regions |
| 4 | Same screen | 2587.24 | 0.63 | unchanged |
| 5 | Done changed to Ready | 2799.43 | 177.34 | regions |
| 6 | Ready changed to Done again | 2470.36 | 0.95 | regions |
| 7 | Full theme change | 2491.87 | 2981.18 | full |

Both approaches paid for full recognition on the first frame. The incremental path
was slower there and on the full theme change. Its gains came from identical frames
and small text edits. In each round it ran the model four times, served one crop
from the image cache, and reused the entire result twice. The full reader ran
the model seven times. The sequence median is computed from each round's total.

[Raw OCR timings, crop decisions, cache statistics, and stage counters](benchmarks/phase2-xeon-ocr.json).

## C and NumPy pixel kernels

Both backends compare the same immutable RGB arrays. A tile changes if any channel
difference exceeds the threshold. The reference uses NumPy to reduce byte runs and
find tile boundaries, then Python to join matching runs across rows. The C version
stops scanning a tile as soon as it finds a changed byte. Both emit identical,
non-overlapping rectangles in the same order.

| Size | Change | NumPy diff (ms) | C diff (ms) | NumPy/C | NumPy merge (ms) | C merge (ms) | NumPy/C |
|---|---|---:|---:|---:|---:|---:|---:|
| 1280x800 | static | 2.124 | 0.400 | 5.31× | 0.072 | 0.005 | 14.93× |
| 1280x800 | local | 1.439 | 0.249 | 5.78× | 0.073 | 0.010 | 7.06× |
| 1280x800 | fragmented | 1.597 | 0.241 | 6.63× | 0.750 | 0.582 | 1.29× |
| 1280x800 | full | 0.901 | 0.024 | 37.84× | 0.102 | 0.013 | 7.91× |
| 1920x1080 | static | 3.904 | 0.742 | 5.26× | 0.082 | 0.005 | 14.96× |
| 1920x1080 | local | 2.944 | 0.570 | 5.16× | 0.087 | 0.010 | 8.69× |
| 1920x1080 | fragmented | 3.228 | 0.483 | 6.69× | 1.356 | 1.091 | 1.24× |
| 1920x1080 | full | 1.578 | 0.036 | 43.90× | 0.184 | 0.019 | 9.76× |

Each cell contains a median. The harness used 32-pixel tiles, threshold zero, seed
2026, five warmups and 31 measured pairs. It alternated execution order and checked
outputs before timing. Times include the Python-facing call, allocation and result
conversion. The merge receives the same precomputed mask on both paths. On a fully
changed image C can exit every tile early, while the OCR planner still requests a
whole-screen read. On the fragmented 1920×1080 grid, emitting many Python rectangles
reduced the C merge advantage to 1.24×.

[Raw kernel samples, median, p95, workload counts, and binary hash](benchmarks/phase2-xeon-kernels.json).

## Reproduce the measurements

The tables above are from Linux x86-64 on an Intel Xeon Platinum 8573C, GCC 13.3.0 at
`-O3`, Python 3.12.14, NumPy 2.5.4, Pillow 13.0.0, EasyOCR 1.7.2, and CPU
PyTorch 2.15.0. The engine used for these tables is recorded in
[this commit](https://github.com/YOUSIF-LUBBAD/Sightline/commit/8978c3a5b811257ee06cba9aba47c17f025e01a0).

An earlier run on an AMD EPYC 9V74, before the buffer and worker deadline
checks, measured 15.10 s for full OCR and 4.60 s for incremental OCR (3.28×).
Each ratio compares methods on the same host. The machines and revisions differ
between runs. [Earlier OCR samples](benchmarks/phase2-ocr.json) and
[earlier kernel samples](benchmarks/phase2-kernels.json).

From the repository root, install the dependencies in the
[setup guide](setup.md). Build the library and use the path the command prints.
These commands show the Linux filename.

```bash
python -m sightline.native_build build/native
python -m sightline.bench --native build/native/sightline_pixels.so --output artifacts/kernels.json
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m examples.benchmark_ocr \
  --native build/native/sightline_pixels.so --worker --rounds 5 \
  --download-models --output artifacts/ocr.json
```

The OCR benchmark draws the frames before timing, warms the model once, alternates
method order, and starts each incremental round with an empty result cache. It times
reading already-captured pixels, including worker transport. Model loading, image
generation, validation, desktop capture, and mouse input are outside that timing.
The kernel harness also supports `--size`, `--tile`, `--threshold`, `--seed`,
`--warmups`, and `--repeats`. It checks parity before reporting latency.

## Use the incremental reader

Pass `IncrementalOCR` anywhere a session accepts an OCR engine. Its default backend is
NumPy. Supply `NativeBackend(path)` to use the compiled C library. The wheel ships
the C source. Building it is an explicit step. A C compiler is needed for that step.

```python
from functools import partial
from pathlib import Path

from sightline import IncrementalOCR, ProcessOCR, Sightline, Timings
from sightline.desktop import MSSFrameSource
from sightline.easyocr_engine import EasyOCRConfig, EasyOCREngine


def main():
    times = Timings()
    config = EasyOCRConfig(model_directory=Path(".models"))
    factory = partial(EasyOCREngine, config)
    with ProcessOCR(factory, timings=times) as worker, MSSFrameSource() as screen:
        reader = IncrementalOCR(worker, timings=times)
        session = Sightline(screen, reader, timings=times)
        for _ in range(5):
            observation = session.look()
            print(reader.last_read, observation.detections)
        print(times.snapshot())


if __name__ == "__main__":
    main()
```

The worker starts with Python's `spawn` method, loads one model, and handles one request
at a time. Place process startup under the main guard as shown above. The default
model startup budget is 120 seconds. Once a request owns the worker, its default
45-second result budget covers sending pixels, inference, and receiving the reply.
A stalled child is terminated even if the parent is blocked writing a large frame.
A crash, error, or timeout discards that worker and its connection. The caller's next read starts a new
one. The context manager closes the child. Sending work to a process adds transfer
overhead and the calling thread waits for its result.

## How a read is chosen

1. The first read, a source or size change, and a scheduled refresh look up the
   complete image. A desktop coordinate mapping change also resets the comparison.
2. Otherwise the reader compares 32×32 tiles. It checks every RGB change.
   Identical pixels reuse the previous detections with the current capture.
3. For a local edit it adds 24 pixels of context, includes complete old labels
   that intersect a crop, and combines overlapping crops. New detections move
   back into full-frame coordinates. Old labels inside a reread region are replaced.
4. A crop touching a detected label at an internal cut edge triggers a complete
   lookup. Broad changes, excessive fragmentation, more than 12 crops, or crops
   covering more than 45% of the frame also use a full lookup. The default
   refresh interval is 30 observations. `RegionPolicy` configures these bounds.
5. Each reader has a bounded LRU cache for 128 images. The key includes the two
   dimensions and a 256-bit BLAKE2b digest of immutable RGB bytes. Empty valid
   reads can hit the cache. Errors and invalid boxes cannot. Use a fresh reader
   after changing the recognizer configuration.

A successful read updates the baseline only after all crops complete. A failed
read leaves the previous successful state available. OCR can interpret the same
word differently with less surrounding context. For an independent whole-frame
reading, pass the underlying recognizer directly to a session.

## Timing and validation

Share one `Timings` instance across the session, reader, and worker. Snapshots
record call counts, total nanoseconds and the last duration. Stages cover capture,
diff, tile merge, region planning, cropping, lookup, OCR, coordinate translation,
result reconciliation, text location, and mouse validation/input. Worker startup,
round trip, and child inference have separate counters. An outer stage includes its
inner work, so adding all stage totals would count some time twice.

The offline suite passed **115 tests** with **83% combined statement/branch coverage**
in the recorded environment. Compiled parity tests cover all 512 possible 3×3 masks,
partial edge tiles, threshold extremes, and strided inputs. The C boundary rejects
invalid array layouts before passing a pointer to native code. A moving-label test
compares 80 incremental updates with independent whole-frame reads. Worker tests
cover stalled startup, a stopped child during frame transfer, cleanup, and recovery. At each of 1280×800 and 1920×1080,
**eight actual OCR/native cases passed** on isolated Linux X11 displays, including
light and dark clicks through both recognition paths. A clean wheel installation
also compiled and exercised both C kernels outside the checkout.
