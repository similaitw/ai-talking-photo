# AI Talking Photo — Codex 專案開發規格

> 專案目標：使用「單張人物照片 + 中文講稿」產生人物說話 MP4。
>
> 開發原則：免費、開源、可在 GitHub 維護；目前硬體以 GTX 1050 2GB 為基準，因此主要運算模式採 Google Colab 免費 GPU，本機提供 CPU / 低顯存備援。未來可無痛升級至 RTX 3060 12GB。

## 工作規則

1. 每次開始先讀 `AGENTS.md`，再讀 `docs/TASKS.md`；只有需要完整產品或架構資訊時才讀本文件。
2. `docs/TASKS.md` 是目前工作狀態的唯一依據，每次只完成 `Current task`。
3. 每個 task 完成前必須實作、測試、修正錯誤、更新文件與 `docs/TASKS.md`。
4. 禁止假功能、假按鈕、硬編碼成功結果。
5. 不提交模型權重、大型影片、API Key、`.env`、私人照片、私人音訊或暫存檔。
6. UI、TTS、模型推論、FFmpeg 必須模組化。

## 1. 專案定位

顯示名稱：**AI Talking Photo**  
Repository：`ai-talking-photo`  
Python package：`talking_photo`

使用者流程：

1. 上傳一張人物照片。
2. 貼上繁體中文講稿。
3. 選擇台灣中文男聲或女聲。
4. 產生 TTS 語音。
5. 使用 Talking Face 模型對嘴。
6. 輸出、預覽並下載 MP4。

主要用途：教師致詞、頒獎感言、活動開場、校園影片、人物介紹、教學素材。

## 2. 硬體與執行環境

目前本機：NVIDIA GTX 1050 2GB、DDR4 32GB、Windows。

主要模式：Google Colab 免費 GPU。  
備援模式：Windows CPU / GTX 1050 低顯存。  
未來升級目標：RTX 3060 12GB，可擴充 MuseTalk / LivePortrait 等 backend，但 MVP 不實作重型模型。

## 3. MVP 技術選型

- Web UI：Gradio
- TTS：`edge-tts`
- 台灣女聲：`zh-TW-HsiaoChenNeural`
- 台灣男聲：`zh-TW-YunJheNeural`
- Talking Face：Wav2Lip
- 影音處理：FFmpeg（由 Python `subprocess` 呼叫）
- 測試：pytest
- Python：3.10+

不要以 MoviePy 作為核心依賴。

## 4. 系統流程

```text
人物照片 + 中文講稿 + 聲音/語速
        ↓
輸入驗證 / 文稿正規化
        ↓
Edge TTS → speech.mp3
        ↓
FFmpeg → speech.wav (16 kHz mono PCM)
        ↓
Wav2Lip
        ↓
FFmpeg finalize
        ↓
output.mp4
        ↓
Gradio 預覽 / 下載
```

## 5. Repository 結構

```text
ai-talking-photo/
├─ AGENTS.md
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ pyproject.toml
├─ app.py
├─ talking_photo/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ models.py
│  ├─ pipeline.py
│  ├─ tts.py
│  ├─ wav2lip.py
│  ├─ media.py
│  ├─ validation.py
│  ├─ device.py
│  └─ utils.py
├─ scripts/
│  ├─ setup_windows.ps1
│  ├─ setup_linux.sh
│  ├─ download_models.py
│  └─ doctor.py
├─ notebooks/
│  └─ AI_Talking_Photo_Colab.ipynb
├─ docs/
│  ├─ SPEC.md
│  ├─ TASKS.md
│  ├─ ARCHITECTURE.md
│  ├─ COLAB.md
│  ├─ WINDOWS.md
│  └─ TROUBLESHOOTING.md
├─ tests/
├─ models/.gitkeep
├─ temp/.gitkeep
└─ output/.gitkeep
```

## 6. UI 規格

Gradio 單頁式介面。

Header：

```text
AI Talking Photo
單張照片 + 中文講稿 → AI 說話影片
```

提示：`建議使用正面、清楚、嘴部未遮擋的人像照片。`

輸入：

- `gr.Image`：人物照片（jpg/jpeg/png/webp）
- `gr.Textbox(lines=12)`：講稿
- `gr.Dropdown`：台灣女聲 / 台灣男聲，預設女聲
- `gr.Slider`：語速 0.8～1.2，預設 0.95，step 0.05
- `產生影片` 按鈕

輸出：

- `gr.Audio` TTS 預覽
- `gr.Video` MP4 預覽與下載

M1.1 尚未接 pipeline 時，按鈕必須明確顯示 `Pipeline 尚未啟用`，不可產生假影片。

## 7. 輸入驗證

圖片拒絕：無圖片、無法解碼、長寬小於 256 px、異常檔案。建議 512×512 以上，但不強制正方形。

文字拒絕空白；MVP 單次最大 3000 字元。超過顯示：

`講稿過長，請分段產生。建議每段 10～30 秒，效果通常較自然。`

預留 `split_script()`，未來依 `。！？、換行` 切段，目標每段 20～80 中文字，不可在單字中間切斷。

## 8. TTS

`taking_photo/tts.py`：

```python
def synthesize_speech(text: str, voice: str, rate: float, output_path: str) -> str:
    ...
```

```python
VOICE_MAP = {
    "台灣女聲": "zh-TW-HsiaoChenNeural",
    "台灣男聲": "zh-TW-YunJheNeural",
}
```

需清理空白、保留標點、正確處理 UTF-8、捕捉 edge-tts 錯誤並輸出可理解訊息。

## 9. 文稿預處理

提供 `normalize_script(text)`，處理 CRLF、多重空白、多個空行與 Markdown 粗體符號，例如 `**不輕易放棄任何一個孩子。**` 送 TTS 前變成一般文字，不能朗讀星號。

## 10. Media / FFmpeg

`taking_photo/media.py`：

```python
def normalize_audio(input_path: str, output_path: str) -> str:
    ...
```

Wav2Lip 音訊統一為 WAV / 16 kHz / mono / PCM。

```bash
ffmpeg -y -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 11. Device Detection

`taking_photo/device.py` 提供 `get_device_info()`，回傳 device、GPU 名稱、VRAM、low_vram。

規則：

- CUDA 不存在 → CPU
- VRAM <= 2.5GB → low_vram
- VRAM > 2.5GB → normal CUDA

GTX 1050 模式：batch size 1、低解析度、釋放不必要 tensor、可用時 `torch.cuda.empty_cache()`。

CUDA OOM 不得無提示轉跑長時間 CPU；顯示建議使用 Colab、切短講稿或降低解析度。

## 12. Wav2Lip Wrapper

`taking_photo/wav2lip.py`：

```python
def generate_lip_sync(image_path: str, audio_path: str, output_path: str, device: str, low_vram: bool = False) -> str:
    ...
```

Wrapper 負責 inference 參數、工作目錄、輸出驗證與錯誤處理；UI 不直接依賴 Wav2Lip 細節。

## 13. Pipeline

`taking_photo/pipeline.py`：

```python
def generate_talking_video(image_path: str, text: str, voice: str, rate: float, progress_callback=None) -> dict:
    ...
```

流程：validate → job folder → TTS → normalize audio → Wav2Lip → finalize MP4 → cleanup → return。

每次工作使用 UUID：

```text
temp/<job-id>/input.png
temp/<job-id>/speech.mp3
temp/<job-id>/speech.wav
temp/<job-id>/raw.mp4
output/<job-id>.mp4
```

不得所有任務共用同一個 `output.mp4`。

預留 `cleanup_old_jobs()`，預設清除 24 小時前的 temp/output；本機可關閉。

## 14. Config

`config.py` 至少包含：APP_NAME、MAX_TEXT_LENGTH=3000、MIN_IMAGE_SIZE=256、DEFAULT_VOICE、DEFAULT_RATE=0.95、TEMP_DIR、OUTPUT_DIR、MODEL_DIR、LOW_VRAM_THRESHOLD_GB=2.5。

## 15. Google Colab

建立 `notebooks/AI_Talking_Photo_Colab.ipynb`，可 Run All：clone repo → 安裝依賴 → 下載模型 → 啟動 Gradio → 顯示 public share URL。

README 最上方未來提供 Open in Colab badge。GTX 1050 2GB 使用者以 Colab 為主。

## 16. Windows

`scripts/setup_windows.ps1`：建立 `.venv`、安裝 requirements、檢查 FFmpeg、顯示下一步；不得自動安裝 NVIDIA driver。

啟動：

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

預設 `http://127.0.0.1:7860`。

## 17. Doctor

`scripts/doctor.py` 檢查 Python、PyTorch、CUDA、GPU/VRAM、FFmpeg、model、edge-tts，並給 Recommended mode。GTX 1050 2GB 建議 Colab；RTX 3060 12GB 可建議 Local CUDA。

## 18. 模型下載

`scripts/download_models.py`：模型不存在才下載、顯示進度、可重試、檢查檔案，若有 hash 就驗證，失敗清 incomplete file。Checkpoint 不進 Git。

## 19. 安全與肖像

UI 顯示：

`請只使用你有權使用的人像與聲音。產生的內容應避免冒充、誤導或未經本人同意的公開發布。`

MVP 不做 voice cloning，不提供名人模板，不內建人物照片或他人聲音，不自動發布社群。

## 20. README / 語言

README 必須說明功能、Colab、Windows、硬體建議。GTX 1050 2GB 明確建議 Colab；RTX 3060 12GB 可本機 CUDA。

所有使用者介面與文件使用繁體中文、台灣用語：使用「影片、資料夾、預設、資訊」，避免「視頻、文件夾、默認、信息」。

## 21. Dependencies / 品質

`requirements.txt` MVP：gradio、edge-tts、numpy、opencv-python、Pillow、soundfile、pytest。PyTorch 由環境安裝腳本處理，避免 Windows/Colab CUDA wheel 衝突。

程式優先 type hints、pathlib、dataclass、logging；避免大量 global state、裸 except、正式環境用 print 當 logger。

Log 可記 job id、device、stage、duration、error；不得記完整講稿、照片內容或個資。

## 22. 測試與 CI

pytest unit tests：validation、TTS（mock edge-tts）、media（無 FFmpeg 時 skip with reason）、device（mock CPU/2GB/12GB）、pipeline（mock TTS/Wav2Lip）。GitHub Actions 在 push / PR 執行 pytest，不跑完整 GPU inference。

## 23. 里程碑

- M0.1 Repository Bootstrap
- M0.2 Agent / Spec / Tasks
- M1.1 Gradio Shell
- M1.2 Validation
- M2.1 Edge TTS
- M2.2 Audio Preview
- M3.1 FFmpeg Audio Normalize
- M3.2 Doctor
- M4.1 Wav2Lip Wrapper
- M4.2 End-to-End Talking Photo
- M5.1 Colab Notebook
- M5.2 Colab Documentation
- M6.1 Windows Setup
- M6.2 GTX 1050 Low VRAM
- M7 UX Polish
- M8 Long Script（Future）
- M9 RTX 3060 Backend Upgrade（Future）

M8：長稿自動斷句 → 逐段 TTS → 逐段 talking face → FFmpeg concatenate；建議 10～30 秒/segment。

M9：抽象 `TalkingFaceBackend`，現有 Wav2LipBackend，未來可新增 MuseTalkBackend / LivePortraitBackend；自動模式依 VRAM 選 backend。

## 24. MVP 不做

Voice cloning、唱歌、即時 webcam avatar、多人照片、3D avatar、SaaS 登入、資料庫、帳號、付費、Vercel AI inference、GitHub Pages AI inference、手機原生 App。

GitHub 只放 source code/docs/notebook；正式 AI inference 在 Colab 或 Local PC。

## 25. Definition of Done

Task 只有在「功能已實作 + 測試已通過 + 無假資料 + 無未處理 exception + 文件與行為一致 + TASKS 已更新」時才完成。

## 26. MVP 最終驗收

使用者可在 Colab Run All，開 Gradio，上傳 JPG/PNG、貼繁體中文講稿、選台灣女聲、產生語音與嘴型同步 MP4、預覽與下載。不用付費 API、OpenAI API、HeyGen 或雲端訂閱。

GitHub repository 不包含人物照片、語音、影片、模型權重或 secrets。

## 27. 開發優先順序

```text
Repository → UI → Validation → TTS → FFmpeg → Wav2Lip → Colab → Windows low VRAM → UX → Long script → RTX 3060 backend
```

不要一開始同時整合 Wav2Lip + MuseTalk + LivePortrait + EchoMimic。MVP 先確保「照片 + 中文 → 穩定產生 MP4」。

## 28. Post-MVP M6.3 / M6.4 品質擴充

MVP 穩定後，依 GTX 1050 實機品質驗收新增兩段非破壞式擴充：

### M6.3 — Wav2Lip + GFPGAN HD Post-processing

- 保留 Wav2Lip 為 GTX 1050 2GB 快速／低顯存 backend。
- GFPGAN 為可選後處理，只改善中心人臉清晰度，不宣稱重新計算或修正 lip-sync。
- GFPGAN 使用獨立 `.venv-gfpgan` 與外部 `vendor/GFPGAN`，不得污染主 `.venv`。
- GTX 1050 預設 `upscale=1`、中心人臉、不使用 Real-ESRGAN 背景放大。

### M6.4 — MuseTalk 1.5 High Quality Backend

- Gradio 新增 `嘴型引擎`：`Wav2Lip（快速／低顯存）` 與 `MuseTalk 1.5（高品質／建議 Colab）`。
- MuseTalk 1.5 是獨立 backend；選 MuseTalk 時不得呼叫 Wav2Lip。
- MuseTalk 使用外部 `vendor/MuseTalk` 與獨立 `.venv-musetalk`，固定官方 commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`。
- MuseTalk 環境固定 Python 3.10、PyTorch 2.0.1 / torchvision 0.15.2 / torchaudio 2.0.2 + CUDA 11.8，依官方建議安裝 MMLab 套件。
- 推論固定 `version=v15`、`use_float16=True`、25fps；初始保守 `batch_size=4`。
- 本專案 MuseTalk 執行門檻為 CUDA 且 VRAM >= 4GB。GTX 1050 2GB 必須在 TTS／推論前停止並提示使用 Colab；不得靜默 CPU fallback。
- 單張人物照片送入 MuseTalk 前最長邊限制 1280 px；MuseTalk 本身處理 256×256 臉部區域。
- 官方 inference 對 task 內部例外可能自行捕捉，因此 wrapper 除了檢查 exit code，必須驗證預期 MP4 真正存在且非空白。
- 建立 `notebooks/AI_Talking_Photo_MuseTalk_Colab.ipynb`，可 Run All：檢查 GPU → 更新專案 → 建主 Python 3.11 UI/TTS 環境 → 建 MuseTalk Python 3.10 環境 → 下載模型 → 啟動 `share=True` Gradio。
- MuseTalk notebook 預設嘴型引擎為 MuseTalk；第一次 A/B 測試 GFPGAN 預設關閉，以隔離嘴型引擎差異。
- `.venv-musetalk`、MuseTalk checkout、模型、使用者照片、語音、影片與暫存資料不得進 Git。
- CI 只做 wrapper、參數、錯誤處理、notebook / setup 靜態驗證，不下載模型、不假裝完成 GPU inference。
- M6.4 只有在真正 Colab GPU 完成至少一支 MuseTalk 1.5 端到端影片後，才可在 `docs/TASKS.md` 勾選完成。
