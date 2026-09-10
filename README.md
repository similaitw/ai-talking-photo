# AI Talking Photo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/similaitw/ai-talking-photo/blob/main/notebooks/AI_Talking_Photo_Colab.ipynb)
[![MuseTalk 1.5 High Quality Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/similaitw/ai-talking-photo/blob/main/notebooks/AI_Talking_Photo_MuseTalk_Colab.ipynb)

單張人物照片 + 繁體中文講稿 → AI 說話影片。

目前提供兩種嘴型引擎與一種可選畫質後處理：

| 模式 | 建議環境 | 定位 |
| --- | --- | --- |
| **Wav2Lip** | GTX 1050 2GB / Windows / Colab | 快速、低顯存 |
| **Wav2Lip + GFPGAN** | GTX 1050 2GB / Windows | Wav2Lip 後再修復臉部清晰度 |
| **MuseTalk 1.5** | Google Colab GPU / >= 4GB VRAM | 嘴型自然度與高品質優先 |

GTX 1050 2GB 本機請使用 Wav2Lip；MuseTalk 1.5 高品質模式不會靜默改跑 CPU，本專案建議使用 Google Colab GPU。

## 最快開始：Google Colab

### Wav2Lip 快速模式

點第一個 **Open In Colab** badge，然後：

1. 在 Colab 選擇「執行階段 → 變更執行階段類型 → GPU」。
2. 選擇「執行階段 → 全部執行」。
3. 等待環境、Wav2Lip 與模型準備完成，Doctor 通過後會啟動 Gradio。
4. 開啟輸出中的 `gradio.live` 網址，上傳有權使用的人像、輸入繁體中文講稿，再按「產生影片」。
5. 影片完成後請下載保存；Colab 工作階段結束後暫存檔可能消失。

完整 Wav2Lip Colab 操作與排錯：[`docs/COLAB.md`](docs/COLAB.md)。

### MuseTalk 1.5 高品質模式

點第二個 **MuseTalk 1.5 High Quality Colab** badge，執行：

`notebooks/AI_Talking_Photo_MuseTalk_Colab.ipynb`

專用 notebook 會建立主程式 Python 3.11 環境，以及獨立的 MuseTalk Python 3.10 `.venv-musetalk`，固定官方 MuseTalk 版本、安裝相依套件與下載 1.5 推論模型，最後以 `share=True` 啟動 Gradio，預設選擇 MuseTalk 1.5。

第一次 A/B 測試建議把「畫質後處理」保持 **關閉（較快）**，只比較 MuseTalk 與 Wav2Lip 的嘴型差異。

完整 MuseTalk 操作、硬體、模型與授權說明：[`docs/MUSETALK.md`](docs/MUSETALK.md)。

> Colab 免費 GPU、GPU 型號與執行時間都不是保證資源。Gradio 公開網址只適合短時間互動測試，不應視為長期部署服務，也不要用公開 `gradio.live` 處理敏感素材。

## Windows 一鍵設定

Windows 第一次使用 Wav2Lip 前請先自行準備：

- Git for Windows
- 64 位元 Python 3.11
- FFmpeg，且 `ffmpeg` 已加入 PATH
- 若要用 NVIDIA CUDA：已安裝可用的 NVIDIA 驅動程式

本專案**不會自動安裝或更新 NVIDIA 驅動程式**。

```powershell
cd H:\AI_Project\ai-talking-photo
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

腳本會檢查 Git、Python 3.11、FFmpeg、NVIDIA GPU，建立或沿用 `.venv`，安裝已驗證的 PyTorch 2.5.1+cu118 / Wav2Lip 相依套件，取得固定官方 Wav2Lip commit，準備 GAN / S3FD 權重，執行 `prepare_wav2lip.py` 與 Doctor。第三方 checkout、模型與私人媒體都不會納入 Git。

設定完成後：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\doctor.py
python app.py
```

啟動後通常使用 `http://127.0.0.1:7860`；若該連接埠已被占用，Gradio 會自動改用下一個可用連接埠。

## GTX 1050 2GB 清晰模式

Wav2Lip 的低顯存路徑已針對 GTX 1050 2GB 調整：

1. 人物照片最多保留到長邊 **1280 px**，不再整張強制縮成 512 px。
2. 用低解析度 CPU 預覽做人臉定位，再把臉框映射回較高清照片。
3. 使用固定 `--box` 避免大圖 GPU 人臉偵測。
4. `wav2lip_batch_size=1`、`face_det_batch_size=1`。
5. 可選嘴周柔和融合，保留眼睛、額頭與大部分原始臉部細節。
6. 最終 MP4 採 H.264 CRF 18，減少二次壓縮損失。

如果預覽找不到臉，會退回 512 px 相容模式；CUDA OOM 不會無提示轉成長時間 CPU 推論。

舊版環境更新後可重新套用 patch：

```powershell
cd H:\AI_Project\ai-talking-photo
git pull --ff-only origin main
.\.venv\Scripts\python scripts\prepare_wav2lip.py
.\.venv\Scripts\python app.py
```

Wav2Lip 本身仍有嘴型自然度與細節上限；若清晰度已改善但嘴型仍明顯不自然，請改用 MuseTalk 1.5 高品質模式。

## GFPGAN 高清修復（實驗）

GFPGAN 是嘴型引擎之後的臉部修復，主要改善模糊感；它**不會重新計算 lip-sync**。

為避免污染 Wav2Lip `.venv`，GFPGAN 使用獨立 `.venv-gfpgan`：

```powershell
cd H:\AI_Project\ai-talking-photo
git pull --ff-only origin main
powershell -ExecutionPolicy Bypass -File .\scripts\setup_gfpgan.ps1
```

安裝完成後仍用主 `.venv` 啟動：

```powershell
.\.venv\Scripts\python app.py
```

在 Gradio 將「畫質後處理」選成 **GFPGAN 高清修復（實驗）**。GTX 1050 2GB 預設使用 GFPGAN V1.3、`upscale=1`、`only_center_face`、`bg_upsampler=none`、`weight=0.4`，避免額外 Real-ESRGAN 背景放大造成 2GB VRAM 壓力。

## MuseTalk 1.5 高品質 Backend

MuseTalk 與主程式完全隔離：

```text
.venv-musetalk/
vendor/MuseTalk/
```

本專案固定官方 `TMElyralab/MuseTalk` commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`。預設推論為：

- MuseTalk `v15`
- `use_float16=True`
- `fps=25`
- `batch_size=4`
- 人物照片長邊最多 1280 px
- 需求至少 4GB VRAM；GTX 1050 2GB 直接提示改用 Colab
- 不允許自動 CPU fallback
- wrapper 除了 subprocess exit code，還會驗證真正的 MP4 是否存在且非空白

官方 MuseTalk 1.5 的臉部工作區為 256×256，支援中文音訊。官方曾以 RTX 3050 Ti Laptop 4GB + fp16 測試 Gradio；這是官方實測案例，不代表所有 4GB GPU 都保證完成推論。

完整說明：[`docs/MUSETALK.md`](docs/MUSETALK.md)。

## 目前可用功能

- 上傳人物照片與繁體中文講稿
- 圖片與講稿輸入驗證
- 台灣中文 Edge TTS
  - 台灣女聲：`zh-TW-HsiaoChenNeural`
  - 台灣男聲：`zh-TW-YunJheNeural`
- 語速 0.8～1.2
- 獨立語音預覽
- FFmpeg 16 kHz / mono / PCM 音訊正規化
- 「嘴型引擎」切換：Wav2Lip / MuseTalk 1.5
- GTX 1050 2GB Wav2Lip 清晰／低顯存模式
- 可選 GFPGAN V1.3 中心人臉高清後處理
- Wav2Lip Run All Colab notebook
- MuseTalk 1.5 高品質 Run All Colab notebook
- Windows Wav2Lip / GFPGAN 可重複設定腳本

## 一般開發與測試

```bash
python -m pip install -r requirements.txt
python -m pytest
```

詳細規格請見 `docs/SPEC.md`，目前工作請見 `docs/TASKS.md`。

## 環境診斷

```bash
python scripts/doctor.py
python scripts/doctor.py --model-path "models/自訂模型.pth"
```

Doctor 主要診斷主 Wav2Lip 環境。MuseTalk 高品質模式由專用 Colab setup 另外檢查 CUDA 與至少 4GB VRAM。

## 第三方模型與授權

### Wav2Lip

本 repository 不內建 Wav2Lip 原始碼或模型權重。官方 Wav2Lip 公開 pretrained models 的授權／使用條款以官方 repository 為準；目前公開說明限制於個人、研究／學術、非商業用途，商業用途需另外確認。

### GFPGAN

官方 GFPGAN 程式採 Apache License 2.0；第三方依賴與模型仍需依各自條款使用。

### MuseTalk

依官方 MuseTalk README：程式碼採 MIT License，官方 trained model 說明可供任何用途，包括商業用途。但 MuseTalk 所使用的 Whisper、VAE、DWPose、Face Parse 等其他開源模型仍需各自遵守其授權條款；本 repository 的 LICENSE 不會取代第三方條款。

## 已驗證的本機路線

2026-09-10～11 已在 Windows、GTX 1050 2GB 完成 Wav2Lip 清晰模式與 GFPGAN V1.3 端到端影片實機驗證。MuseTalk 1.5 的程式、wrapper、VRAM guard、Colab setup 與自動測試已加入，但在真正 Colab GPU 的端到端影片驗證完成前，不宣稱 M6.4 已實機驗收。

## 安全與隱私

請只使用你有權使用的人像、聲音與內容。不要用生成結果冒充本人、誤導他人，或在未取得適當同意時公開發布。Colab 的 Gradio `share=True` 會建立可從網際網路存取的臨時網址，因此不要在公開連結中處理敏感或不應外流的素材。
