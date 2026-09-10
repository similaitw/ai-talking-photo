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
6. 執行 `prepare_wav2lip.py` 驗證來源與模型格式。
7. 執行 Doctor；通過後顯示啟動指令。

腳本以 `$PSScriptRoot` 找專案根目錄，不依賴固定磁碟機或資料夾名稱；重跑時會沿用正確的 `.venv`、Wav2Lip 與模型，不會覆蓋成其他來源版本。若既有 `.venv` 不是 Python 3.11，或 `vendor\Wav2Lip` 不是 Git repository，腳本會停止並要求人工處理，以免破壞既有資料。

設定完成後：

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\doctor.py
python app.py
```

啟動後預設網址：`http://127.0.0.1:7860`。

> GTX 1050 2GB 已能真實產出短片，但仍建議優先使用 Colab。M6.2 會針對 2GB 顯示記憶體再做專門驗證與最佳化。

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
- Wav2Lip subprocess wrapper：支援 CPU、CUDA 與低顯示記憶體參數
- 真實端到端流程：Edge TTS → FFmpeg → Wav2Lip → H.264／AAC MP4
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

低顯示記憶體模式會使用 `face_det_batch_size=1`、`wav2lip_batch_size=1` 與 `resize_factor=2`。如果指定 CUDA，wrapper 會先確認 CUDA 可用，並拒絕 Wav2Lip 靜默改用 CPU；CUDA OOM 會顯示繁體中文建議，不會自動轉跑 CPU。

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

GTX 1050 2GB 會使用 `low_vram=True`、batch size 1，pipeline 並將人物圖片等比例縮至最長邊不超過 512 px。CUDA OOM 時會直接提示改用 Colab 或降低圖片解析度，不會自動重試 CPU。

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

請自行準備非私人人像，或指定有權使用的圖片；測試圖片與產物不納入 Git。2026-09-10 已在 GTX 1050 2GB／CUDA／`low_vram=True` 真實產出 **7.872 秒**影片，另以真實 Gradio 操作成功產出 **8.32 秒**影片。

## 安全與隱私

請只使用你有權使用的人像、聲音與內容。不要用生成結果冒充本人、誤導他人或在未取得適當同意時公開發布。Colab 的 Gradio `share=True` 會建立可從網際網路存取的臨時網址，因此不要在公開連結中處理敏感或不應外流的素材。
