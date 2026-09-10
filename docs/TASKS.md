# Tasks

## Current task

M4.1 — Wav2Lip Wrapper

## Milestones

- [x] M0.1 Repository Bootstrap
- [x] M0.2 Agent / Spec / Tasks
- [x] M1.1 Gradio Shell
- [x] M1.2 Validation
- [x] M2.1 Edge TTS
- [x] M2.2 Audio Preview
- [x] M3.1 FFmpeg Audio Normalize
- [x] M3.2 Doctor
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

### M3.2 — Doctor
- 新增 `scripts/doctor.py`，以繁體中文檢查 Python、PyTorch、CUDA、GPU／顯示記憶體、FFmpeg、edge-tts 與模型檔，並提供建議模式。
- 實作延遲載入 PyTorch 的 `get_device_info()`；CPU 與不超過 2.5 GiB 的 GPU 優先建議 Colab，較大容量 GPU 可使用本機 CUDA。
- 支援自訂模型路徑、缺項後繼續診斷、結束代碼與 Windows UTF-8 輸出；不下載或載入權重、不執行推論或網路語音請求。
- README 補充使用方式、預設模型路徑、硬體建議與基本檢查的限制。
- 新增 17 項測試，涵蓋 CPU、2GB／2.5GB／12GB GPU、PyTorch／CUDA 錯誤、FFmpeg 失敗與逾時、模型缺漏及跨工作目錄執行。
- 完整測試 `python -m pytest`：60 passed，無略過；實機診斷正確回報目前缺少 PyTorch 與模型檔，FFmpeg 與 edge-tts 可使用。
