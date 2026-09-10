# Tasks

## Current task

M2.1 — Edge TTS

## Milestones

- [x] M0.1 Repository Bootstrap
- [x] M0.2 Agent / Spec / Tasks
- [x] M1.1 Gradio Shell
- [x] M1.2 Validation
- [ ] M2.1 Edge TTS
- [ ] M2.2 Audio Preview
- [ ] M3.1 FFmpeg Audio Normalize
- [ ] M3.2 Doctor
- [ ] M4.1 Wav2Lip Wrapper
- [ ] M4.2 End-to-End Talking Photo
- [ ] M5.1 Colab Notebook
- [ ] M5.2 Colab Documentation
- [ ] M6.1 Windows Setup
- [ ] M6.2 GTX 1050 Low VRAM
- [ ] M7 UX Polish

## Completed

### M0.1 — Repository Bootstrap
- Created the initial Python package and repository directory structure.
- Added baseline tests and project metadata.
- Verified `python -m pytest` passes locally: 2 passed.

### M0.2 — Agent / Spec / Tasks
- Added `AGENTS.md`, `docs/SPEC.md`, and this task tracker.
- Established GitHub as the project source of truth.

### M1.1 — Gradio Shell
- Added a Traditional Chinese single-page Gradio interface.
- Added portrait, script, voice, speed, audio preview, and video preview components.
- Generate action explicitly returns `Pipeline 尚未啟用` until the real pipeline is connected.
- Verified `python -m pytest`: 4 passed.

### M1.2 — Validation
- Added image validation for missing, unreadable, unsupported, and undersized images.
- Added script validation for blank and over-3000-character input.
- Added script normalization for line endings, whitespace, and Markdown bold markers.
- Connected validation to the Gradio generate action before pipeline execution.
- Verified `python -m pytest`: 15 passed.
