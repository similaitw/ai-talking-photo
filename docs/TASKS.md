# Tasks

## Current task

M3.2 — Doctor

## Milestones

- [x] M0.1 Repository Bootstrap
- [x] M0.2 Agent / Spec / Tasks
- [x] M1.1 Gradio Shell
- [x] M1.2 Validation
- [x] M2.1 Edge TTS
- [x] M2.2 Audio Preview
- [x] M3.1 FFmpeg Audio Normalize
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

### M2.1 — Edge TTS
- Added Taiwan Mandarin Edge TTS voice mapping for female and male voices.
- Added 0.8–1.2 speed multiplier conversion to Edge TTS rate syntax.
- Added script normalization, output directory creation, output verification, and user-friendly TTS errors.
- Kept `edge-tts` as a lazy runtime dependency so unit tests do not require network access.
- Verified `python -m pytest`: 26 passed.

### M2.2 — Audio Preview
- Added a dedicated `產生語音預覽` action to the Gradio UI.
- Audio preview can be generated from script, voice, and rate without requiring a portrait first.
- Generated preview audio is written to a unique temporary MP3 path and returned to `gr.Audio` for playback/download.
- Kept the video-generation action separate and explicitly unavailable until the video pipeline milestone.
- Added user-facing validation and TTS failure messages.
- Verified `python -m pytest`: 29 passed.

### M3.1 — FFmpeg Audio Normalize
- Added `normalize_audio(input_path, output_path)` using real FFmpeg subprocess execution to produce 16 kHz mono signed 16-bit PCM WAV.
- Added input/output validation, automatic output directory creation, absolute output paths, and Traditional Chinese conversion errors.
- Verify WAV output before replacing the destination; failed conversions preserve existing output and clean temporary files.
- Added 14 media tests covering invalid paths, unavailable FFmpeg, failed/invalid output, real WAV/MP3 conversion, and corrupt input. Real FFmpeg tests skip with an explicit reason when FFmpeg is unavailable.
- Documented the FFmpeg prerequisite and standalone API in README; pipeline integration remains in its planned milestone.
- Verified `python -m pytest` on Windows with FFmpeg installed: 43 passed, no skipped tests.
