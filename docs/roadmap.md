# Sightline v2 roadmap

Owner: Yousif Lubbad. This is a plan, not a list of shipped capabilities.
The ten-phase plan and commit budget can change with measured results.
Every commit should represent a useful change; there are no empty-history targets.

| Phase | Scope | State |
| --- | --- | --- |
| 1 | Package, tests, lint/types, CI, frame sources, EasyOCR, unique text click, native demo | Implemented; [validation record](phase1.md) |
| 2 | NumPy frame diff, changed regions, ROI OCR/cache/worker, C hot path, timing and benchmarks | Implemented; [measurements and validation](phase2.md) |
| 3 | Toggleable preprocessing, template/feature matching, controls, video and webcam sources | Planned |
| 4 | Scene elements, spatial relations, RTL-aware relations, tracking, serialization | Planned |
| 5 | Chained/fuzzy locators, recovery logs, mouse paths, typing, stable waits, verification | Planned |
| 6 | pytest integration, HTML reports, CLI commands, configuration and structured logs | Planned |
| 7 | Local dashboard, live overlays, pause/step/scrub, record and replay | Planned |
| 8 | Safe app exploration, graph/test export, searchable video text, Arabic and bilingual fixtures | Planned |
| 9 | Playwright UI generator, labeled benchmark, robustness metrics, accessibility checks | Planned |
| 10 | Optional MCP bridge, docs site, package/container publication, demos and a tested release | Planned |

Use original project fixtures and apps the owner is authorized to automate.
Publish failures alongside successes; do not extrapolate a demo into accuracy or
speed claims. The learned
detector, MCP bridge, and Android/scrcpy control are optional cuts if time is short.

Game bots for owned games such as JOJO and bowlrot, reinforcement learning, and
phone-game automation remain future plans. No game integrations are part of Phase 1.
