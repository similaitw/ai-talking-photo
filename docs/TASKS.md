# Tasks

## Current task

M6.3 — MuseTalk 1.5 High Quality Backend

## Milestones

- [x] M0.1 Repository Bootstrap
- [x] M0.2 Agent / Spec / Tasks
- [x] M1.1 Gradio Shell
- [x] M1.2 Validation
- [x] M2.1 Edge TTS
- [x] M2.2 Audio Preview
- [x] M3.1 FFmpeg Audio Normalize
- [x] M3.2 Doctor
- [x] M4.1 Wav2Lip Wrapper
- [x] M4.2 End-to-End Talking Photo
- [x] M5.1 Colab Notebook
- [x] M5.2 Colab Documentation
- [x] M6.1 Windows Setup
- [x] M6.2 GTX 1050 Low VRAM
- [ ] M6.3 MuseTalk 1.5 High Quality Backend
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

### M4.1 — Wav2Lip Wrapper
- 新增 `generate_lip_sync(image_path, audio_path, output_path, device, low_vram=False)`，以 subprocess 呼叫外部 Wav2Lip `inference.py`，不把第三方原始碼或 checkpoint 提交進本 repository。
- 預設使用 `vendor/Wav2Lip` 與 `models/wav2lip_gan.pth`，亦可透過 `WAV2LIP_DIR`、`WAV2LIP_CHECKPOINT` 自訂。
- 支援 `cpu`、`cuda`、`cuda:N`；CPU 模式隱藏 CUDA，CUDA 模式先確認裝置可用，並拒絕 Wav2Lip 靜默改用 CPU。
- GTX 1050 類低顯示記憶體模式使用 `face_det_batch_size=1`、`wav2lip_batch_size=1`、`resize_factor=2`。
- CUDA OOM、人臉偵測失敗、模型／inference.py 缺漏與空白 MP4 都會回傳繁體中文錯誤；CUDA OOM 不會自動 fallback 至 CPU。
- 推論先寫入暫存 MP4，成功後才原子取代正式輸出，失敗時保留既有影片並清除暫存檔。
- 新增 16 項 wrapper focused tests；不下載模型、不執行 GPU inference。
- README 補充 Wav2Lip 外部安裝方式、環境變數、低顯示記憶體行為與第三方非商業使用限制。

### M4.2 — End-to-End Talking Photo
- 實作 `generate_talking_video()`：驗證 → UUID 工作資料夾 → Edge TTS MP3 → 16 kHz 單聲道 PCM WAV → 既有 Wav2Lip wrapper → H.264／AAC MP4，提供繁體中文階段進度。
- Gradio「產生影片」已接入真實流程，成功回傳語音與影片預覽；錯誤清除舊預覽並顯示原因。
- 每次工作及 wrapper 的官方暫存 AVI 皆隔離；成功清除中間檔、保留 MP3／MP4，失敗清除該 UUID 工作，不覆蓋使用者原圖。
- GTX 1050 2GB 使用 `low_vram=True`、batch size 1、照片最長邊 512 px；CUDA OOM 提示 Colab／降低解析度，不自動切換 CPU。可透過環境變數明確指定 CPU 備援。
- 準備官方 Wav2Lip commit `bac9a81e63ecc153202353372e5724b83d9e6322`，固定 Python 3.11／PyTorch 2.5.1+cu118 與相容音訊套件；新增相依版本清單、官方權重驗證／格式轉換腳本及 README 重現步驟。
- 真實推論最初卡在官方 GAN TorchScript 與 inference.py 預期格式不一致；已將同一官方權重轉成 state_dict 並以官方模型 strict=True 驗證，沒有換用其他來源模型。
- 真實 E2E（2026-09-10）：AI 生成的虛構成人人像、短繁體中文講稿、真實 Edge TTS／FFmpeg／Wav2Lip，在 NVIDIA GTX 1050 2GB、CUDA、low_vram=True 成功產出 7.872 秒 H.264／AAC MP4，通過 ffprobe 與完整解碼。沒有使用 CPU 備援。
- 真實 Gradio 瀏覽器驗證：上傳人像、輸入講稿、按「產生影片」，預設語速 0.95 成功產出 8.32 秒影片，成功狀態、語音播放及影片播放均正常。
- 完整 `python -m pytest`：Python 3.11 推論環境與系統 Python 3.14 皆為 86 passed，無略過。CI 仍僅跑測試、不下載模型或執行 GPU inference。
- Wav2Lip checkout、權重、測試圖片、影音與暫存檔皆未納入 Git；未開始 M5.1 的實作。

### M5.1 — Colab Notebook
- 新增 `notebooks/AI_Talking_Photo_Colab.ipynb`，可由乾淨 Colab GPU 工作階段依序 clone 專案、建立獨立 Python 3.11 環境、安裝已驗證的 PyTorch 2.5.1+cu118 與 Wav2Lip 相依套件、準備官方模型、執行 Doctor，最後以 Gradio `share=True` 提供公開網址。
- Notebook 不依賴 Colab 預裝 Python 版本，降低預設 runtime 從 Python 3.12 升級至 3.13 後的相容性風險。
- Wav2Lip 固定官方 commit `bac9a81e63ecc153202353372e5724b83d9e6322`，沿用官方 GAN／S3FD 下載來源與既有 SHA-256 驗證／格式轉換流程；不提交第三方原始碼、模型或輸出影音。
- 新增 3 項 notebook 靜態測試，驗證合法 nbformat、GPU metadata、乾淨輸出、完整 Run All bootstrap 關鍵步驟，以及未嵌入 token／API key／密碼。
- 本地 focused tests：3 passed；GitHub Actions 驗收為 86 passed、3 skipped。M5.2 補齊 Colab 使用說明與入口文件。

### M5.2 — Colab Documentation
- README 最上方新增 Open in Colab badge，並提供 GPU 執行階段、Run All、Gradio 啟動與輸出保存的最短操作流程。
- 新增 `docs/COLAB.md`，完整說明第一次執行、輸出位置、重跑方式，以及 GPU、Doctor、模型 SHA、Edge TTS、人臉偵測、CUDA OOM、Gradio share 等常見問題排除。
- 明確記錄 Colab 免費資源、GPU 型號與執行時間不保證，並依 Google 官方 FAQ 說明免費受管 runtime 以 Web UI 為主要互動時可能提早終止；Gradio share 僅定位為短時間測試入口，不作正式部署。
- 補充公開 `gradio.live` 網址的隱私風險、暫存檔保存提醒與 Wav2Lip 非商業使用限制。
- 新增 3 項文件測試，固定 badge、notebook 路徑、Run All 與關鍵限制說明。
- 本地文件 focused tests：3 passed；GitHub Actions：89 passed、3 skipped。

### M6.1 — Windows Setup
- 新增 `scripts/setup_windows.ps1`，從腳本位置自動解析專案根目錄，支援任意磁碟機、空白與中文路徑。
- 檢查 Git、Python 3.11、FFmpeg 與 NVIDIA GPU／驅動資訊；不會自動安裝或更新 NVIDIA 驅動程式。
- 建立或安全沿用 Python 3.11 `.venv`，安裝已驗證的 PyTorch 2.5.1+cu118 與固定 Wav2Lip 相依套件。
- 可重複準備官方 Wav2Lip checkout、GAN 與 S3FD 權重，並執行 `prepare_wav2lip.py` 與 Doctor；模型與第三方程式仍不納入 Git。
- 新增 6 項 Windows setup 測試，涵蓋路徑無關、固定版本、重跑安全性、繁中提示與 PowerShell AST 語法解析。
- GitHub Actions：95 passed、3 skipped。
- Windows 實機驗證：`H:\AI_Project\ai-talking-photo` 的 Python 3.11 `.venv` 可成功啟動 Gradio；因本機 7860 已被占用，Gradio 自動改用 `http://127.0.0.1:7861`，屬正常行為。

### M6.2 — GTX 1050 Low VRAM
- 針對 NVIDIA GTX 1050 2GB 實作低顯示記憶體清晰模式：不再把整張照片固定縮成 512 px，輸出最長邊提高至 1280 px。
- 低顯存模式先用 512 px 預覽做一次 CPU 人臉定位，再把座標映射回較高解析度照片，以固定臉框跳過 Wav2Lip 的 GPU 人臉偵測；若定位失敗則回到相容模式。
- Wav2Lip wrapper 支援固定臉框與嘴周柔和融合；相容性修正可重複套用，不需重新下載模型。
- 影片封裝採 H.264 CRF 18，降低二次壓縮造成的模糊。
- GitHub Actions：106 passed、3 skipped。
- Windows GTX 1050 2GB 實機可成功產生影片；清晰度路徑已改善原先整體模糊問題，但同一張人物照片實測嘴型仍明顯不自然。
- 結論：2GB 顯存與 Wav2Lip 路線保留為「快速／低顯存」模式，不再投入更多參數微調；高品質嘴型改由 M6.3 MuseTalk 1.5 在 Colab／較大顯存 GPU 路線處理。
