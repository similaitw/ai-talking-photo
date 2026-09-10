# AI Talking Photo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/similaitw/ai-talking-photo/blob/main/notebooks/AI_Talking_Photo_Colab.ipynb)

單張人物照片 + 中文講稿 → AI 說話影片。

主要推論環境規劃為 Google Colab GPU；GTX 1050 2GB 本機環境作為低顯存備援。

## 最快開始：Google Colab

1. 點上方 **Open in Colab**。
2. 在 Colab 選擇「執行階段 → 變更執行階段類型 → GPU」。
3. 選擇「執行階段 → 全部執行」。
4. 等待環境、Wav2Lip 與模型準備完成，Doctor 通過後會啟動 Gradio。
5. 開啟輸出中的 `gradio.live` 網址，上傳有權使用的人像、輸入繁體中文講稿，再按「產生影片」。
6. 影片完成後請下載保存；Colab 工作階段結束後，暫存檔可能消失。

完整操作、限制與排錯請看 [`docs/COLAB.md`](docs/COLAB.md)。

> Colab 的免費 GPU、GPU 型號與執行時間都不保證，資源與限制會動態調整。Gradio 公開網址只適合短時間互動測試，不應視為長期部署服務。

## Windows 一鍵設定

Windows 第一次使用前，請先自行準備：

- Git for Windows
- 64 位元 Python 3.11
- FFmpeg，且 `ffmpeg` 已加入 PATH
- 若要用 NVIDIA CUDA：已安裝可用的 NVIDIA 驅動程式

本專案**不會自動安裝或更新 NVIDIA 驅動程式**。

在專案根目錄開啟 PowerShell，例如目前專案放在 `H:\AI_Project\ai-talking-photo` 時：

```powershell
cd H:\AI_Project\ai-talking-photo
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

`setup_windows.ps1` 會：

1. 檢查 Git、Python 3.11、FFmpeg 與 NVIDIA GPU／驅動資訊。
2. 建立或沿用 Python 3.11 的 `.venv`。
3. 安裝已驗證的 PyTorch 2.5.1+cu118 與 Wav2Lip 相依套件。
4. 下載並固定官方 Wav2Lip commit `bac9a81e63ecc153202353372e5724b83d9e6322`。
5. 第一次執行時下載官方 GAN 與 S3FD 模型；已存在時不重複下載。
6. 執行 `prepare_wav2lip.py` 驗證來源、模型格式與本專案的相容性／品質修正。
7. 執行 Doctor；通過後顯示啟動指令。

腳本以 `$PSScriptRoot` 找專案根目錄，不依賴固定磁碟機或資料夾名稱；重跑時會沿用正確的 `.venv`、Wav2Lip 與模型，不會覆蓋成其他來源版本。若既有 `.venv` 不是 Python 3.11，或 `vendor\Wav2Lip` 不是 Git repository，腳本會停止並要求人工處理，以免破壞既有資料。

設定完成後：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\doctor.py
python app.py
```

啟動後通常使用 `http://127.0.0.1:7860`；若該連接埠已被占用，Gradio 會自動改用 7861、7862 等下一個可用連接埠。

## GTX 1050 2GB 清晰模式（M6.2）

早期低顯存路徑會先把整張人物照片縮到 512 px，雖然穩定，但會讓背景、頭髮、衣服與臉部一起變模糊。新版低顯存清晰模式改為：

1. 人物照片最多保留到長邊 **1280 px**；原圖本來小於 1280 px 時不縮小。
2. 只建立最長邊 512 px 的 CPU 預覽來做人臉定位，不用 GTX 1050 執行大圖人臉偵測。
3. 將預覽偵測到的臉框映射回較高解析度照片，使用 Wav2Lip `--box` 跳過 GPU 人臉偵測。
4. Wav2Lip 仍維持 `wav2lip_batch_size=1` 與 `face_det_batch_size=1`。
5. 只把生成結果以柔和遮罩融合到下半臉／嘴周，保留眼睛、額頭與大部分原始臉部細節。
6. 最後 MP4 使用 H.264 CRF 18 重新封裝，降低二次壓縮造成的可見畫質損失。

若 512 px 預覽找不到正面人臉，系統會自動退回已驗證的 512 px 相容模式，而不是冒險讓 GTX 1050 在大圖人臉偵測時 OOM。介面完成後會顯示實際使用「低顯示記憶體清晰模式」或「低顯示記憶體相容模式」。

已完成過舊版 Windows 設定的使用者，更新 M6.2 後不需重裝環境，只要：

```powershell
cd H:\AI_Project\ai-talking-photo
git pull --ff-only origin main
.\.venv\Scripts\python scripts\prepare_wav2lip.py
.\.venv\Scripts\python app.py
```

`prepare_wav2lip.py` 的嘴周融合修正可重複執行，不會重複插入程式碼。模型仍是同一份官方 Wav2Lip GAN 權重，沒有換模型或重新訓練。

> Wav2Lip 本身仍以 96×96 臉部模型輸入進行嘴型生成，因此新版可以明顯改善「整體影片被縮糊」與「整塊臉被替換」的問題，但不能完全消除 Wav2Lip 模型本身的嘴型與細節上限。

## GFPGAN 高清修復（M6.3，實驗）

GFPGAN 是 **Wav2Lip 之後的臉部修復**，用途是改善臉部與嘴周的模糊感；它不會重新計算 lip-sync，因此如果 Wav2Lip 的嘴型時序或形狀本身不自然，GFPGAN 只能修復畫質，不能保證修正嘴型。

為避免污染已驗證成功的 Wav2Lip `.venv`，GFPGAN 使用獨立 `.venv-gfpgan`。Windows 第一次使用請執行：

```powershell
cd H:\AI_Project\ai-talking-photo
git pull --ff-only origin main
powershell -ExecutionPolicy Bypass -File .\scripts\setup_gfpgan.ps1
```

腳本會固定官方 `TencentARC/GFPGAN` commit `7552a7791caad982045a7bbe5634bbf1cd5c8679`，建立 Python 3.11 的 `.venv-gfpgan`，安裝 PyTorch 2.1.2 + torchvision 0.16.2（CUDA 11.8），並下載官方 GFPGAN V1.3 權重。GFPGAN 原始碼、虛擬環境與權重都不會 commit 到本 repository。

安裝完成後仍用原本的 `.venv` 啟動主程式：

```powershell
.\.venv\Scripts\python app.py
```

在 Gradio 將「畫質後處理」選成 **GFPGAN 高清修復（實驗）**。GTX 1050 2GB 預設設定：

- GFPGAN V1.3：官方描述偏向較自然的修復結果。
- `upscale=1`：維持原影片尺寸，不做 2× 放大。
- `only_center_face`：只修主要人物臉部。
- `bg_upsampler=none`：不啟用 Real-ESRGAN 背景放大，降低 2GB VRAM 壓力。
- `weight=0.4`：降低過度重建造成的身份改變風險。
- 先拆幀修復，再以原幀率重新合成 H.264/AAC MP4 並保留音訊。

官方 GFPGAN：<https://github.com/TencentARC/GFPGAN>，採 Apache License 2.0。GFPGAN V1.3 官方也提醒可能有輕微 identity change，因此本專案預設採較保守的修復強度。

## 目前可用功能

- 上傳人物照片與輸入繁體中文講稿
- 圖片與講稿輸入驗證
- 台灣中文 Edge TTS
  - 台灣女聲：`zh-TW-HsiaoChenNeural`
  - 台灣男聲：`zh-TW-YunJheNeural`
- 語速 0.8～1.2
- 獨立「產生語音預覽」按鈕
- 在 Gradio 直接播放或下載產生的 MP3
- FFmpeg 轉成 16 kHz、單聲道、16 位元 PCM WAV
- Wav2Lip subprocess wrapper：支援 CPU、CUDA、固定臉框與低顯示記憶體參數
- 真實端到端流程：Edge TTS → FFmpeg → Wav2Lip → H.264／AAC MP4
- GTX 1050 低顯存清晰模式與 512 px 安全 fallback
- 可選 GFPGAN V1.3 中心人臉高清後處理
- Google Colab Run All notebook
- 可重複執行的 Windows 設定腳本

## 一般開發與測試

一般 Python 相依套件：

```bash
python -m pip install -r requirements.txt
```

測試：

```bash
python -m pytest
```

詳細規格請見 `docs/SPEC.md`，目前工作請見 `docs/TASKS.md`。

## 環境診斷

```bash
python scripts/doctor.py
python scripts/doctor.py --model-path "models/自訂模型.pth"
```

Doctor 會檢查 Python、PyTorch、CUDA、GPU 與顯示記憶體、FFmpeg、edge-tts，以及模型檔。GTX 1050 2GB（顯示記憶體不超過 2.5 GiB）建議優先使用 Google Colab GPU；RTX 3060 12GB 可使用本機 CUDA。無可用 CUDA 時，本機 CPU 僅作為備援。

## Wav2Lip 與模型

本 repository **不內建 Wav2Lip 原始碼或模型權重**。預設外部位置：

```text
vendor/Wav2Lip/inference.py
models/wav2lip_gan.pth
```

也可使用環境變數 `WAV2LIP_DIR` 與 `WAV2LIP_CHECKPOINT` 自訂。

低顯示記憶體模式固定 batch size 1。靜態照片的清晰路徑會先在 CPU 小預覽定位臉部，再用 `--box` 對較高解析度照片直接推論；這可避開大圖 S3FD 人臉偵測造成的 2GB VRAM 壓力。CUDA OOM 仍會顯示繁體中文建議，不會自動轉跑 CPU。

官方 Wav2Lip：<https://github.com/Rudrabha/Wav2Lip>

> 注意：官方 Wav2Lip 開源程式與公開 pretrained models 的使用條款限定個人、研究／學術、非商業用途。若有商業用途，請依官方 repository 的授權說明另外處理。本專案本身的 LICENSE 不會取代第三方 Wav2Lip 的授權條款。

## 已驗證的 Windows 端到端環境

2026-09-10 已驗證：Windows、Python 3.11.9、PyTorch 2.5.1+cu118、NVIDIA 驅動程式 560.94、GTX 1050 2GB、FFmpeg 8.1.2。

模型來源使用官方 Wav2Lip README 指定來源。下載檔 SHA-256：

| 檔案 | SHA-256 |
| --- | --- |
| 官方 GAN 下載檔 | `180cfd49d31d47f195d5bfc62830ffe2c40724b7bbbce6faadc73ac6ab1a3b8e` |
| S3FD | `619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543` |

`prepare_wav2lip.py` 會核對來源雜湊與 checkout 版本，將官方 GAN TorchScript 轉成官方 `inference.py` 可載入的 state_dict，並以官方模型 `strict=True` 驗證。沒有重新訓練或替換模型。

若要明確選擇 CPU 備援：

```powershell
$env:TALKING_PHOTO_DEVICE = "cpu"
.\.venv\Scripts\python app.py
Remove-Item Env:TALKING_PHOTO_DEVICE
```

## 真實短片驗證

```powershell
.\.venv\Scripts\python scripts/smoke_test.py "temp/synthetic-portrait.png"
.\.venv\Scripts\python -m pytest
```

請自行準備非私人人像，或指定有權使用的圖片；測試圖片與產物不納入 Git。2026-09-10 已在 GTX 1050 2GB／CUDA／`low_vram=True` 真實產出 **7.872 秒**影片，另以真實 Gradio 操作成功產出 **8.32 秒**影片。M6.2 清晰路徑已完成實機驗證；GFPGAN M6.3 仍需用同一張照片做 A/B 品質驗收。

## 安全與隱私

請只使用你有權使用的人像、聲音與內容。不要用生成結果冒充本人、誤導他人或在未取得適當同意時公開發布。Colab 的 Gradio `share=True` 會建立可從網際網路存取的臨時網址，因此不要在公開連結中處理敏感或不應外流的素材。
