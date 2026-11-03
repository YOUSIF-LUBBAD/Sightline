# Setup and troubleshooting

Maintained by Yousif Lubbad. Use is subject to the repository LICENSE.

## Development environment

Use Python 3.12 to reproduce the validated OCR environment. The base package
supports Python 3.11+. Start in the repository root, create a virtual environment,
and install the extras shown in the README. On Windows, activate it with
`.venv\Scripts\Activate.ps1` in PowerShell. On macOS/Linux, use
`source .venv/bin/activate`. Windows and macOS live behavior is not yet verified.

On Ubuntu, the native demo and headless test need system libraries:

```bash
sudo apt-get update
sudo apt-get install -y python3-tk xvfb xauth fonts-dejavu-core
```

Your chosen Python interpreter must include Tk support. Check it with
`python -m tkinter`. If that fails, install Tk for that interpreter before running
the demo. `.[dev]` is sufficient for the offline unit suite; `.[desktop,ocr]`
adds real screen capture, mouse input, and OCR. PyTorch and model weights are
large separate downloads. The pinned CPU PyTorch command in the README avoids
installing GPU packages for the tested Linux setup.

## Models and private pixels

`EasyOCRConfig` defaults to CPU English OCR and disables model downloads. The
example uses `.models/` in the checkout. Pass `--download-models` the first time
you want EasyOCR to fetch missing weights, then omit it for later offline runs.
Use `--model-dir /your/model/folder` to choose another location. EasyOCR verifies
its expected downloaded files. No model weights are committed or packaged.

Sightline's code processes screenshots locally. It does not send captured pixels
to an OCR service or save them automatically. Optional test screenshots are
written only when `SIGHTLINE_ARTIFACT_DIR` is set; CI captures only the
project-owned demo in Xvfb. Do not run that artifact-producing job on a personal
desktop that contains private windows.

## Display and click checks

| Symptom | What to check |
| --- | --- |
| Cannot open screen capture or mouse input | Use a working graphical session and check OS capture/input permissions. On Linux, verify `DISPLAY` points to your X11 desktop. |
| Cannot load EasyOCR models | Install the OCR extra, check the model folder, and explicitly allow a first download if needed. The original exception is preserved for debugging. |
| No visible match | Keep the whole label visible, use exact text, and inspect confidence with `look()`. |
| More than one match | Remove duplicate visible labels or use a more specific label. Phase 1 refuses to choose one. |
| Frame is too old | CPU inference may take several seconds. Use a static local fixture before considering a larger `--max-frame-age`. |
| Mouse input stopped | Check the corner fail-safe and OS permissions. The library never turns the fail-safe off. |
| Display size changed | Keep resolution and scaling fixed during one run. Capture a new frame with a newly constructed desktop source. |

macOS normally requires Screen Recording and Accessibility permission for the
terminal/interpreter. Windows scaling and secure/elevated windows need validation
before use. Wayland support is not claimed. Only the primary screen at desktop
origin `(0, 0)` is accepted in Phase 1.

## Isolated real tests

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 xvfb-run -a -s '-screen 0 1280x800x24 -nolisten tcp' \
  python -m pytest --run-ocr --run-live --download-models -m 'ocr or live'
```

Repeat with `1920x1080x24` for the second configured resolution. Both flags are
needed for native tests. `--run-ocr` alone runs real model checks without mouse
input. Omitting `--download-models` prevents fetching missing weights. The static
native fixture sets a 30-second frame-age budget and keeps the default five-second
policy covered by unit tests. Model loading happens before screen capture.

PyTorch/EasyOCR currently emit upstream quantization and CPU pin-memory warnings;
MSS 10.2 also warns about its compatibility constructor. The recorded run kept
these warnings visible. They did not fail the tests, but should be revisited when
dependencies change.
