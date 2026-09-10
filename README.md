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

## 本機執行

一般 Python 相依套件：

```bash
python -m pip install -r requirements.txt
python app.py
```

測試：

```bash
python -m pytest
```

詳細規格請見 `docs/SPEC.md`，目前工作請見 `docs/TASKS.md`。

## 音訊格式轉換

請另行安裝 FFmpeg，並確認終端機可執行 `ffmpeg -version`；Python 相依套件不包含 FFmpeg 執行檔。

```python
from talking_photo.media import normalize_audio

wav_path = normalize_audio("temp/speech.mp3", "temp/speech.wav")
```

函式會建立輸出資料夾並回傳 WAV 的絕對路徑，供後續 Wav2Lip 使用。輸入與輸出必須是不同檔案，輸出副檔名須為 `.wav`；成功時取代既有輸出，失敗時保留既有輸出並提供繁體中文錯誤訊息。

## 環境診斷

```bash
python scripts/doctor.py
python scripts/doctor.py --model-path "models/自訂模型.pth"
```

Doctor 會檢查 Python、PyTorch、CUDA、目前 CUDA 裝置的 GPU 與顯示記憶體總容量、FFmpeg、edge-tts，以及模型檔。預設模型路徑為 `models/wav2lip_gan.pth`；可用 `--model-path` 指定其他 checkpoint。

GTX 1050 2GB（顯示記憶體不超過 2.5 GiB）建議優先使用 Google Colab GPU，本機低顯示記憶體模式僅作為備援；RTX 3060 12GB 可使用本機 CUDA。無可用 CUDA 時建議 Colab，本機 CPU 僅作為備援。

## Wav2Lip Wrapper

本 repository **不內建 Wav2Lip 原始碼或模型權重**。`talking_photo.wav2lip.generate_lip_sync()` 會呼叫外部 Wav2Lip checkout 的 `inference.py`。

預設位置：

```text
vendor/Wav2Lip/inference.py
models/wav2lip_gan.pth
```

也可使用環境變數自訂：

```text
WAV2LIP_DIR
WAV2LIP_CHECKPOINT
```

範例：

```python
from talking_photo.wav2lip import generate_lip_sync

video_path = generate_lip_sync(
    image_path="temp/portrait.png",
    audio_path="temp/speech.wav",
    output_path="output/result.mp4",
    device="cpu",          # 或 cuda / cuda:0
    low_vram=True,
)
```

低顯示記憶體模式會使用 `face_det_batch_size=1`、`wav2lip_batch_size=1` 與 `resize_factor=2`。如果指定 CUDA，wrapper 會先確認 CUDA 可用，並拒絕 Wav2Lip 靜默改用 CPU；CUDA OOM 會顯示繁體中文建議，不會自動轉跑 CPU。

官方 Wav2Lip：<https://github.com/Rudrabha/Wav2Lip>

> 注意：官方 Wav2Lip 開源程式與公開 pretrained models 的使用條款限定個人、研究／學術、非商業用途。若有商業用途，請依官方 repository 的授權說明另外處理。本專案本身的 LICENSE 不會取代第三方 Wav2Lip 的授權條款。

## 已驗證的 Windows 端到端環境

已驗證環境：Windows、Python 3.11.9、PyTorch 2.5.1+cu118、NVIDIA 驅動程式 560.94、GTX 1050 2GB、FFmpeg 8.1.2。使用專案獨立 `.venv`，不要使用系統 Python 3.14 直接執行舊版 Wav2Lip。

在專案根目錄執行 PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
.\.venv\Scripts\python -m pip install -r requirements-wav2lip-lock.txt
git clone https://github.com/Rudrabha/Wav2Lip.git vendor/Wav2Lip
git -C vendor/Wav2Lip checkout bac9a81e63ecc153202353372e5724b83d9e6322
.\.venv\Scripts\python -m gdown 15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk -O models/wav2lip_gan.pth
curl.exe -L --fail https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth -o vendor/Wav2Lip/face_detection/detection/sfd/s3fd.pth
.\.venv\Scripts\python scripts/prepare_wav2lip.py
.\.venv\Scripts\python scripts/doctor.py
.\.venv\Scripts\python app.py
```

`requirements-wav2lip-lock.txt` 保存成功環境的完整套件版本（PyTorch 另由官方 CUDA 索引安裝）。官方原始 requirements 面向舊 Python，請勿覆蓋安裝。

模型來源使用官方 Wav2Lip README 指定來源。下載檔 SHA-256：

| 檔案 | SHA-256 |
| --- | --- |
| 官方 GAN 下載檔 | `180cfd49d31d47f195d5bfc62830ffe2c40724b7bbbce6faadc73ac6ab1a3b8e` |
| S3FD | `619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543` |

`prepare_wav2lip.py` 會核對來源雜湊與 checkout 版本，將官方 GAN TorchScript 轉成官方 `inference.py` 可載入的 state_dict，並以官方模型 `strict=True` 驗證。沒有重新訓練或替換模型。

每次工作使用 `temp/<UUID>/`，成功後保留 `speech.mp3` 供預覽，以及 `output/<UUID>.mp4`；中間圖片、WAV 與 raw MP4 會清除，失敗工作會移除。

GTX 1050 2GB 會使用 `low_vram=True`、batch size 1，pipeline 並將人物圖片等比例縮至最長邊不超過 512 px。CUDA OOM 時會直接提示改用 Colab 或降低圖片解析度，不會自動重試 CPU。

若要**明確選擇 CPU 備援**：

```powershell
$env:TALKING_PHOTO_DEVICE = "cpu"
.\.venv\Scripts\python app.py

# 恢復自動偵測
Remove-Item Env:TALKING_PHOTO_DEVICE
```

## 真實短片驗證

```powershell
.\.venv\Scripts\python scripts/smoke_test.py "temp/synthetic-portrait.png"
.\.venv\Scripts\python -m pytest
```

請自行準備非私人人像，或指定有權使用的圖片；測試圖片與產物不納入 Git。此腳本會真的連線 Edge TTS、執行 FFmpeg 與 Wav2Lip，檢查影片長度、H.264／AAC 串流及完整解碼。

2026-09-10 實測使用 AI 生成的虛構成人人像、短繁體中文講稿，在 NVIDIA GTX 1050 2GB／CUDA／`low_vram=True` 成功產出 **7.872 秒**影片；另以真實 Gradio 上傳與「產生影片」按鈕測試，預設語速 0.95 成功產出 **8.32 秒**影片。未使用 CPU 備援，未以 mock 代替真實推論。

## 安全與隱私

請只使用你有權使用的人像、聲音與內容。不要用生成結果冒充本人、誤導他人或在未取得適當同意時公開發布。Colab 的 Gradio `share=True` 會建立可從網際網路存取的臨時網址，因此不要在公開連結中處理敏感或不應外流的素材。
