# AI Talking Photo — Codex 專案開發規格

> 專案目標：使用「人物照片或影片 + 中文講稿」產生人物說話 MP4。
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

1. 上傳一張人物照片或一段人物影片。
2. 貼上繁體中文講稿。
3. 選擇台灣中文男聲或女聲。
4. 選擇嘴型引擎：Wav2Lip 或 MuseTalk 1.5。
5. 產生 TTS 語音。
6. 使用 Talking Face 模型對嘴；影片輸入保留來源的眨眼、頭部與身體微動。
7. 輸出、預覽並下載 MP4。

主要用途：教師致詞、頒獎感言、活動開場、校園影片、人物介紹、教學素材。

## 2. 硬體與執行環境

目前本機：NVIDIA GTX 1050 2GB、DDR4 32GB、Windows。

主要高品質模式：Google Colab GPU。  
備援／快速模式：Windows CPU / GTX 1050 低顯存 Wav2Lip。  
未來升級目標：RTX 3060 12GB，可再驗證 MuseTalk 本機推論與其他 backend。

## 3. 技術選型

- Web UI：Gradio
- TTS：`edge-tts`
- 台灣女聲：`zh-TW-HsiaoChenNeural`
- 台灣男聲：`zh-TW-YunJheNeural`
- 快速 Talking Face：Wav2Lip
- 高品質 Talking Face：MuseTalk 1.5
- 可選臉部後處理：GFPGAN V1.3
- 影音處理：FFmpeg / FFprobe（由 Python `subprocess` 呼叫）
- 測試：pytest
- 主程式 Python：3.11；MuseTalk 獨立 Python 3.10

不要以 MoviePy 作為主程式核心依賴。

## 4. 系統流程

```text
人物照片或影片 + 中文講稿 + 聲音/語速
        ↓
輸入驗證 / 文稿正規化
        ↓
Edge TTS → speech.mp3
        ↓
FFmpeg → speech.wav (16 kHz mono PCM)
        ↓
選擇嘴型引擎
   ├─ Wav2Lip（快速／低顯存）
   └─ MuseTalk 1.5（高品質／Colab）
        ↓
可選 GFPGAN 後處理
        ↓
FFmpeg finalize (H.264 / AAC)
        ↓
output.mp4
        ↓
Gradio 預覽 / 下載
```

影片輸入的原始音軌不作為成品聲音；成品使用 Edge TTS 產生的新語音。

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
│  ├─ pipeline.py
│  ├─ tts.py
│  ├─ wav2lip.py
│  ├─ musetalk.py
│  ├─ gfpgan.py
│  ├─ media.py
│  ├─ validation.py
│  ├─ quality.py
│  └─ device.py
├─ scripts/
│  ├─ setup_windows.ps1
│  ├─ setup_gfpgan.ps1
│  ├─ setup_musetalk_colab.sh
│  ├─ download_musetalk_models.py
│  ├─ prepare_wav2lip.py
│  └─ doctor.py
├─ notebooks/
│  ├─ AI_Talking_Photo_Colab.ipynb
│  └─ AI_Talking_Photo_MuseTalk_Colab.ipynb
├─ docs/
│  ├─ SPEC.md
│  ├─ TASKS.md
│  ├─ COLAB.md
│  └─ MUSETALK.md
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
人物照片或影片 + 中文講稿 → AI 說話影片
```

提示：建議使用正面、清楚、嘴部未遮擋的人像。影片若有自然眨眼、微點頭或呼吸感，成品通常比單張照片更自然。

輸入：

- `gr.File(type="filepath", file_types=["image", "video"])`：人物素材（照片或影片）
  - 圖片：JPG / JPEG / PNG / WebP
  - 影片：MP4 / MOV / WebM / MKV
- `gr.Textbox(lines=12)`：講稿
- `gr.Dropdown`：台灣女聲 / 台灣男聲，預設女聲
- `gr.Slider`：語速 0.8～1.2，預設 0.95，step 0.05
- `gr.Dropdown`：嘴型引擎，Wav2Lip / MuseTalk 1.5
- `gr.Dropdown`：畫質後處理，關閉 / GFPGAN
- `產生語音預覽`、`產生影片` 按鈕

輸出：

- `gr.Audio` TTS 預覽
- `gr.Video` MP4 預覽與下載

## 7. 輸入驗證

圖片拒絕：無檔案、無法解碼、不支援格式、長寬小於 256 px、異常檔案。建議 512×512 以上，但不強制正方形。

影片拒絕：無檔案、空白檔案、不支援容器、FFprobe 無可用視訊軌、寬或高小於 256 px。建議使用 720p 以上、正面清楚人物影片。

影片自然度建議：5～10 秒自然待機，正面或小角度、光線穩定、嘴部未遮擋，僅輕微眨眼／點頭／呼吸；避免大幅轉頭與劇烈動作。

文字拒絕空白；單次最大 3000 字元。超過顯示：

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

嘴型引擎音訊統一為 WAV / 16 kHz / mono / PCM。

```bash
ffmpeg -y -i input.mp3 -ar 16000 -ac 1 output.wav
```

影片輸入使用 FFprobe 驗證第一視訊軌尺寸。最終 MP4 以 H.264 CRF 18、AAC、`yuv420p`、`+faststart` 封裝。

## 11. Device Detection

`taking_photo/device.py` 提供 `get_device_info()`，回傳 device、GPU 名稱、VRAM、low_vram。

規則：

- CUDA 不存在 → Wav2Lip 可明確使用 CPU 備援
- VRAM <= 2.5GB → Wav2Lip low_vram
- MuseTalk → 必須 CUDA 且 VRAM >= 4GB

CUDA OOM 不得無提示轉跑長時間 CPU；顯示建議使用 Colab、切短講稿或降低人物素材解析度。

## 12. Wav2Lip Wrapper

`taking_photo/wav2lip.py`：

```python
def generate_lip_sync(
    source_path: str,
    audio_path: str,
    output_path: str,
    device: str,
    low_vram: bool = False,
    face_box=None,
    soft_blend: bool = False,
    static: bool = True,
) -> str:
    ...
```

照片模式保留 `static=True`，低顯存時可使用 CPU 預覽人臉框、固定 `--box` 與嘴周柔和融合。

影片模式必須 `static=False`，不得加固定照片臉框或靜態嘴周融合；讓官方 Wav2Lip 逐幀處理來源影片。若 TTS 比來源影片長，官方 inference 依 `i % len(frames)` 循環使用來源幀。

Wrapper 負責 inference 參數、工作目錄、輸出驗證與錯誤處理；UI 不直接依賴 Wav2Lip 細節。

## 13. MuseTalk 1.5 Wrapper

`taking_photo/musetalk.py` 使用外部 `vendor/MuseTalk` 與獨立 `.venv-musetalk`。

固定官方 commit：`0a89dec45a0192b824e3cf4daf96c239440c5ed8`。

核心規則：

- `version=v15`
- `use_float16=True`
- `batch_size=4`
- 照片模式長邊最多 1280 px、預設 25fps
- 影片模式保留原始副檔名交給官方 `scripts.inference`，由官方讀取來源 FPS
- CUDA 且 VRAM >= 4GB；不得靜默 CPU fallback
- 除了 subprocess exit code，必須驗證預期 MP4 存在且非空白

官方 normal inference 支援 video / image / directory of images。影片若比音訊短，官方流程使用 `frame_list + frame_list[::-1]` 的幀循環來源，因此建議人物動作小而自然。

## 14. Pipeline

`taking_photo/pipeline.py`：

```python
def generate_talking_video(
    media_path: str,
    text: str,
    voice: str,
    rate: float,
    progress_callback=None,
    *,
    enhancement="關閉（較快）",
    backend="Wav2Lip（快速／低顯存）",
) -> dict:
    ...
```

流程：validate media → job folder → TTS → normalize audio → selected backend → finalize MP4 → optional GFPGAN → cleanup → return。

每次工作使用 UUID，媒體副檔名依輸入類型保留：

```text
temp/<job-id>/input.png            # 圖片範例
temp/<job-id>/input.mp4            # 影片範例
temp/<job-id>/speech.mp3
temp/<job-id>/speech.wav
temp/<job-id>/raw.mp4
output/<job-id>.mp4
```

不得所有任務共用同一個 `output.mp4`。不得修改或刪除使用者原始上傳檔。

## 15. Config

`config.py` 至少包含：APP_NAME、MAX_TEXT_LENGTH=3000、MIN_IMAGE_SIZE=256、DEFAULT_VOICE、DEFAULT_RATE=0.95、TEMP_DIR、OUTPUT_DIR、MODEL_DIR、LOW_VRAM_THRESHOLD_GB=2.5。

## 16. Google Colab

- `notebooks/AI_Talking_Photo_Colab.ipynb`：Wav2Lip 快速模式。
- `notebooks/AI_Talking_Photo_MuseTalk_Colab.ipynb`：MuseTalk 1.5 高品質模式。

兩者可 Run All：clone / 更新 repo → 安裝對應環境 → 下載模型 → 啟動 Gradio → 顯示 public share URL。

MuseTalk notebook 預設嘴型引擎為 MuseTalk 1.5，第一次高品質實測優先使用 5～10 秒自然待機人物影片，GFPGAN 關閉以隔離嘴型引擎差異。

## 17. Windows

`scripts/setup_windows.ps1`：建立 `.venv`、安裝 requirements、檢查 FFmpeg、顯示下一步；不得自動安裝 NVIDIA driver。

`scripts/setup_gfpgan.ps1`：建立獨立 `.venv-gfpgan`，不得污染主 Wav2Lip 環境。

啟動：

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

預設 `http://127.0.0.1:7860`；若占用由 Gradio 使用下一個可用連接埠。

## 18. Doctor

`scripts/doctor.py` 檢查 Python、PyTorch、CUDA、GPU/VRAM、FFmpeg、model、edge-tts，並給 Recommended mode。GTX 1050 2GB 建議 Wav2Lip / Colab；MuseTalk 由專用 setup 驗證 CUDA 與 >=4GB VRAM。

## 19. 模型下載

模型不存在才下載、顯示進度、可重試、檢查檔案，若有 hash 就驗證，失敗清 incomplete file。Checkpoint 不進 Git。

MuseTalk 只下載 v1.5 推論所需 UNet、VAE、Whisper、DWPose、Face Parse 等模型；第三方 checkout 與模型皆在 `.gitignore`。

## 20. 安全與肖像

UI 顯示：

`請只使用你有權使用的人像與聲音。產生的內容應避免冒充、誤導或未經本人同意的公開發布。`

不做 voice cloning，不提供名人模板，不內建人物照片或他人聲音，不自動發布社群。

## 21. README / 語言

README 必須說明功能、照片／影片輸入、Colab、Windows、硬體建議。GTX 1050 2GB 明確定位 Wav2Lip；MuseTalk 高品質建議 Colab。

所有使用者介面與文件使用繁體中文、台灣用語：使用「影片、資料夾、預設、資訊」，避免「視頻、文件夾、默認、信息」。

## 22. Dependencies / 品質

主 `requirements.txt`：gradio、edge-tts、numpy、opencv-python、Pillow、soundfile、pytest。PyTorch 由環境安裝腳本處理，避免 Windows/Colab CUDA wheel 衝突。

MuseTalk 使用獨立 Python 3.10 環境，固定 PyTorch 2.0.1 / torchvision 0.15.2 / torchaudio 2.0.2 cu118，並為 OpenMIM / MMCV 相容性固定 `pip==24.0`、`setuptools==69.5.1`。

程式優先 type hints、pathlib、dataclass、logging；避免大量 global state、裸 except。Log 可記 job id、device、stage、duration、error；不得記完整講稿、照片／影片內容或個資。

## 23. 測試與 CI

pytest unit tests：validation、TTS、media、device、Wav2Lip、MuseTalk、GFPGAN、pipeline、UI、setup scripts、Colab notebooks。

影片輸入至少驗證：

- Wav2Lip video path 不得強制 `--static`。
- Wav2Lip video pipeline 不得套照片專用 face box / soft blend。
- MuseTalk wrapper 必須保留來源影片副檔名，讓官方 inference 正確判定 video。
- MuseTalk backend 不得呼叫 Wav2Lip。
- CI 不下載重型模型、不假裝完成 GPU inference。

GitHub Actions 在 push / PR 執行 pytest；真正 GPU E2E 需人工在目標硬體／Colab 驗證。

## 24. 里程碑

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
- M6.3 Wav2Lip + GFPGAN HD Post-processing
- M6.4 Photo / Video Input + MuseTalk 1.5 High Quality Backend
- M7 UX Polish
- M8 Long Script（Future）
- M9 RTX 3060 Backend Upgrade（Future）

M8：長稿自動斷句 → 逐段 TTS → 逐段 talking face → FFmpeg concatenate；建議 10～30 秒/segment。

M9：進一步抽象 `TalkingFaceBackend`，並在 RTX 3060 12GB 實機驗證高品質 backend 與自動模式。

## 25. 不做

Voice cloning、唱歌、即時 webcam avatar、多人照片／多人影片、3D avatar、SaaS 登入、資料庫、帳號、付費、Vercel AI inference、GitHub Pages AI inference、手機原生 App。

GitHub 只放 source code/docs/notebook；正式 AI inference 在 Colab 或 Local PC。

## 26. Definition of Done

Task 只有在「功能已實作 + 測試已通過 + 無假資料 + 無未處理 exception + 文件與行為一致 + TASKS 已更新」時才完成。

M6.4 額外要求：必須在真正 Colab GPU 使用 MuseTalk 1.5 完成至少一支端到端影片；目前以**影片輸入**作為優先實機驗收路線。在此之前不得勾選完成。

## 27. 目前端到端驗收目標

使用者可在 Colab Run All，開 Gradio，上傳 JPG/PNG/WebP 或 MP4/MOV/WebM/MKV 人物素材、貼繁體中文講稿、選台灣中文聲音、選嘴型引擎、產生語音與嘴型同步 MP4、預覽與下載。不用付費 API、OpenAI API、HeyGen 或雲端訂閱。

GitHub repository 不包含人物照片、語音、影片、模型權重或 secrets。

## 28. 開發優先順序

```text
Repository → UI → Validation → TTS → FFmpeg → Wav2Lip → Colab → Windows low VRAM → GFPGAN → MuseTalk → Photo/Video input → UX → Long script → RTX 3060
```

目前以穩定的 Wav2Lip 路線作快速／低顯存基線，MuseTalk 1.5 作高品質路線；不在 M6.4 同時加入其他重型 Talking Face 模型。

## 29. M6.3 / M6.4 品質擴充

### M6.3 — Wav2Lip + GFPGAN HD Post-processing

- 保留 Wav2Lip 為 GTX 1050 2GB 快速／低顯存 backend。
- GFPGAN 為可選後處理，只改善中心人臉清晰度，不宣稱重新計算或修正 lip-sync。
- GFPGAN 使用獨立 `.venv-gfpgan` 與外部 `vendor/GFPGAN`，不得污染主 `.venv`。
- GTX 1050 預設 `upscale=1`、中心人臉、不使用 Real-ESRGAN 背景放大。

### M6.4 — Photo / Video Input + MuseTalk 1.5 High Quality Backend

- Gradio `人物素材（照片或影片）` 統一接受 image / video。
- Gradio `嘴型引擎`：`Wav2Lip（快速／低顯存）` 與 `MuseTalk 1.5（高品質／建議 Colab）`。
- Wav2Lip 與 MuseTalk 都必須支援照片與影片；選 MuseTalk 時不得呼叫 Wav2Lip。
- 影片原始音軌由新 Edge TTS 取代。
- Wav2Lip 影片模式 `static=False`，保留逐幀動作；不得套照片固定 face box / soft blend。
- MuseTalk 影片模式保留來源副檔名與來源 FPS；若音訊較長，沿用官方正放＋倒放幀循環。
- MuseTalk 使用外部 `vendor/MuseTalk` 與獨立 `.venv-musetalk`，固定官方 commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`。
- MuseTalk 環境固定 Python 3.10、PyTorch 2.0.1 / torchvision 0.15.2 / torchaudio 2.0.2 + CUDA 11.8；OpenMIM 相容工具固定 `pip==24.0`、`setuptools==69.5.1`。
- 推論固定 `version=v15`、`use_float16=True`；初始保守 `batch_size=4`。照片模式 25fps，影片模式由官方讀來源 FPS。
- 本專案 MuseTalk 執行門檻為 CUDA 且 VRAM >= 4GB。GTX 1050 2GB 必須在 TTS／推論前停止並提示使用 Colab；不得靜默 CPU fallback。
- 單張人物照片送入 MuseTalk 前最長邊限制 1280 px；MuseTalk 本身處理 256×256 臉部區域。
- 官方 inference 對 task 內部例外可能自行捕捉，因此 wrapper 除了檢查 exit code，必須驗證預期 MP4 真正存在且非空白。
- `notebooks/AI_Talking_Photo_MuseTalk_Colab.ipynb` 可 Run All：檢查 GPU → 更新專案 → 建主 Python 3.11 UI/TTS 環境 → 建 MuseTalk Python 3.10 環境 → 下載模型 → 啟動 `share=True` Gradio。
- 第一次高品質 A/B 測試使用同一支 5～10 秒自然待機人物影片、相同講稿／聲音／語速；GFPGAN 關閉。
- `.venv-musetalk`、MuseTalk checkout、模型、使用者照片／影片、語音、輸出與暫存資料不得進 Git。
- CI 只做 wrapper、參數、影片路由、錯誤處理、notebook / setup 靜態驗證，不下載模型、不假裝完成 GPU inference。
- M6.4 只有在真正 Colab GPU 完成至少一支 MuseTalk 1.5 端到端影片後，才可在 `docs/TASKS.md` 勾選完成。
