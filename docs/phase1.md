# Phase 1 validation record

Owner: **Yousif Lubbad**. Recorded locally on **2026-11-02 UTC**.
This record describes the first working slice, not a v1.0 release or a broad
performance benchmark. Later work must update results when behavior changes.

## Results

| Check | Recorded result |
| --- | --- |
| Ruff format and lint | Passed |
| Strict mypy | Passed, 10 source files |
| Default pytest suite | 77 passed; 5 real OCR/native cases skipped by design |
| Default suite coverage | 93% combined statement/branch coverage across `src/sightline` |
| Real model/native tests, 1280×800 | 5 passed |
| Real model/native tests, 1920×1080 | 5 passed |
| Source archive and wheel | Both built successfully with Hatchling |
| Clean wheel environment | Installed and imported the public API without desktop/OCR extras |
| Distribution ownership | Author, proprietary license metadata, notices, and type marker checked |
| GitHub Actions | Python 3.11–3.13 and both native resolutions; [current run results](https://github.com/YOUSIF-LUBBAD/Sightline/actions/workflows/ci.yml) |

The five real cases are two generated-image OCR cases (light and dark), one blank
image case, and two live native-app cases (light and dark at different positions).
Each live case verifies that a preview does nothing, a real mouse click completes
the app, and a new OCR pass reads `Complete` and no longer finds `Start`.
Tests obtain screen pixels through MSS and input through PyAutoGUI. The engine
does not inspect the Tk widgets, app state, DOM, or accessibility tree.

The real runs use an isolated Xvfb display and CPU inference. The development
workspace required an authenticated local TCP display transport; normal Linux CI
uses `xvfb-run` with TCP disabled. That environment adjustment did not replace
capture, OCR, or mouse input with mocks. The README screenshot comes from the
successful dark-theme test at 1280×800 and has not been retouched.

## What failed and what changed

The first CI run exposed a test that assumed at least 60 seconds of machine
uptime. On a fresh runner, its fake screenshot received a negative timestamp.
The age-rejection test now uses a fixed clock so stale and future frames are
checked consistently on every runner. The production age checks are unchanged.

Full-screen CPU inference exceeded the default five-second capture-to-click
budget. Sightline correctly refused those clicks. The stationary demo now opts
into a 30-second limit, while the library and example keep a five-second default.
There is no automatic retry or silent limit increase. Dynamic UI work needs the
speed and stability features planned for later phases.

The native fixture proves a controlled end-to-end path, not that a text box is
always clickable or that OCR cannot be wrong. HiDPI hardware, Windows/macOS live
input, Wayland, Arabic layouts, and arbitrary web apps have not been validated.
Upstream model and MSS warnings remain visible. No latency, CER, precision, recall,
or general task-success benchmark is claimed.

## Reproduction environment

| Component | Version used locally |
| --- | --- |
| Python | 3.12.14 |
| NumPy / Pillow | 2.5.4 / 13.0.0 |
| EasyOCR | 1.7.2, English, CRAFT detector and English generation-2 recognizer |
| PyTorch / torchvision | 2.15.0+cpu / 0.29.2+cpu |
| MSS / PyAutoGUI | 10.2.0 / 0.9.54 |
| pytest / Ruff / mypy | 9.2.0 / 0.17.3 / 2.4.1 |
| Build backend | Hatchling 1.33.0 |

Inference used `OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2`. Dependencies are
constrained in `pyproject.toml`; this table records an observed environment, not
a complete lockfile. See [setup.md](setup.md) and the CI workflow for commands.

## The 18 meaningful commits

The license is the first commit. Each step adds a distinct part of Phase 1; no
empty commits or invented historical dates are needed.

| # | Change |
| --- | --- |
| 1 | Establish ownership and reuse terms |
| 2 | Scaffold the package and dependency extras |
| 3 | Isolate pytest and check headless imports |
| 4 | Enforce formatting, lint, and strict types |
| 5 | Model immutable RGB frames and desktop coordinate scaling |
| 6 | Define frame sources and load offline images |
| 7 | Capture primary-screen pixels through MSS |
| 8 | Define typed OCR detections and an engine contract |
| 9 | Add lazy EasyOCR loading and validate boxes |
| 10 | Locate text and reject ambiguous matches |
| 11 | Add bounded mouse clicks with the corner fail-safe |
| 12 | Connect capture and OCR to explicit validated clicks |
| 13 | Build the project-owned native demo |
| 14 | Add the runnable image/preview/click example |
| 15 | Prove real OCR and native clicks end to end |
| 16 | Configure offline and native GitHub Actions jobs |
| 17 | Export a typed public API and verify clean wheel installation |
| 18 | Document setup, ownership, roadmap, and measured validation |
