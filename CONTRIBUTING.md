# Contributing to BioSphere AI

Thanks for taking a look. BioSphere is a personal project published as open source, so the
process is light: open an issue to talk about anything bigger than a small fix, then send a
pull request. Issues and pull requests in **English or Turkish** are both fine.

## Ground rules

- **Measure, don't guess.** This codebase is built around measurements: most constants carry
  a comment with the number that put them there, and several modules record approaches that
  were tried and rejected *with the numbers*. A change to the pipeline should come with a
  before/after measurement (the synthetic quality gate, a bench script, or images you looked
  at), not only "looks better".
- **Explain why in the code.** Comments say *why* a thing is the way it is, especially when it
  looks wrong. Match the density and tone of the file you are editing.
- **Keep the tests green** and add tests for behaviour you change.
- **Never commit captures of real homes, people or personal data.** Test with the synthetic
  rooms in `engine/biosphere/synth`.
- Be kind — see the [Code of Conduct](CODE_OF_CONDUCT.md).

## Development setup

See the [Quick start](README.md#quick-start). In short:

```bash
cd engine && uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
cd web && npm install
```

## Running the checks

```bash
# engine — fast suite (~560 tests); the synthetic end-to-end gate is marked `slow`
cd engine
.venv/bin/python -m pytest -m "not slow"
.venv/bin/python -m pytest tests/test_pipeline_quality.py -q     # slow gate, minutes

# web
cd web
npx tsc --noEmit
npm test
```

Tests never call paid APIs: the listing and investment-assistant tests patch the key out and
use fake clients. Keep it that way.

## Good first issues

- `ruff check biosphere tests` reports ~50 findings (unused imports, `zip()` without `strict=`,
  `str` enums) — tidy a module at a time.
- `npm run lint` has a few pre-existing `react-hooks/set-state-in-effect` findings
  (e.g. `components/player/Player.tsx`, `components/dashboard/UploadDropzone.tsx`).
- Linux/macOS versions of `scripts/setup.ps1` and `scripts/dev.ps1`.
- A Dockerfile for the engine.

## Bigger things that would help

- **Furniture removal and virtual staging** with stronger models (see the parked status in the
  README); anything that runs on an 8 GB GPU and does not invent objects is interesting.
- **VR on real devices** — report how the WebXR and Cardboard modes behave on a Quest, a Vision
  Pro, Android and iPhone.
- **3D walkthrough capture guidance** — the reconstruction breaks on fast, blurry walks; better
  in-app guidance would fix more than any algorithm.
- **The slow realistic three-row gate** currently links only 29 of 93 keyframes; finding out
  why (and whether an older commit does the same) is a well-defined investigation.

## Pull requests

- One topic per pull request; describe what changed, why, and how you measured it.
- Screenshots for UI changes (no real homes in them).
- By contributing you agree that your contribution is licensed under the [MIT License](LICENSE).
