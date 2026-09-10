# AI Talking Photo

單張人物照片 + 中文講稿 → AI 說話影片。

主要推論環境規劃為 Google Colab 免費 GPU；GTX 1050 2GB 本機環境作為低顯存備援。

## 目前可用功能

- 上傳人物照片與輸入繁體中文講稿
- 圖片與講稿輸入驗證
- 台灣中文 Edge TTS
  - 台灣女聲：`zh-TW-HsiaoChenNeural`
  - 台灣男聲：`zh-TW-YunJheNeural`
- 語速 0.8～1.2
- 獨立「產生語音預覽」按鈕
- 在 Gradio 直接播放或下載產生的 MP3
- 音訊處理模組：透過 FFmpeg 轉成 16 kHz、單聲道、16 位元 PCM WAV
- Wav2Lip subprocess wrapper：支援 CPU、CUDA 與低顯示記憶體參數

「產生影片」已串接真實 Edge TTS → MP3 → 16 kHz 單聲道 PCM WAV → Wav2Lip → H.264／AAC MP4。成功後可在 Gradio 播放或下載語音與影片，並顯示繁體中文進度及成功狀態。首次執行影片推論，請先依下方「本機端到端執行環境」準備官方程式與模型。

## 執行

安裝相依套件：

```bash
python -m pip install -r requirements.txt
```

啟動：

```bash
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

音訊預覽提供 MP3；影片流程會另外正規化成 WAV。測試包含實際 FFmpeg 轉檔，未安裝 FFmpeg 時會標示略過原因，其餘測試照常執行。

## 環境診斷

```bash
python scripts/doctor.py
python scripts/doctor.py --model-path "models/自訂模型.pth"
```

診斷會檢查 Python、PyTorch、CUDA、目前 CUDA 裝置的 GPU 與顯示記憶體總容量、FFmpeg、edge-tts，以及模型檔。預設模型路徑為專案內的 `models/wav2lip_gan.pth`；可用 `--model-path` 指定其他 checkpoint，不會下載或載入模型。

GTX 1050 2GB（顯示記憶體不超過 2.5 GiB）建議優先使用 Google Colab GPU，本機低顯示記憶體模式僅作為備援；RTX 3060 12GB 可使用本機 CUDA。無可用 CUDA 時建議 Colab，本機 CPU 僅作為備援。這是硬體建議，實際推論仍需後續驗證。

PyTorch 需依執行環境另行安裝，不包含在 `requirements.txt`。缺少套件或檢查失敗時，工具仍會繼續列出其他診斷結果。結束代碼 `0` 表示 Python 版本、PyTorch、FFmpeg、edge-tts 與非空白且可讀取的模型檔通過基本檢查，`1` 表示有缺項或錯誤；CPU 環境本身不視為檢查失敗。此工具不驗證模型權重相容性、網路語音服務或完整影片流程。

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

推論會先寫入暫存 MP4，成功且檔案非空白後才取代正式輸出，因此失敗時不會破壞既有影片。模型缺漏、人臉偵測失敗、CUDA OOM 與無有效輸出都會提供可理解的繁體中文錯誤訊息。

官方 Wav2Lip：<https://github.com/Rudrabha/Wav2Lip>

> 注意：官方 Wav2Lip 開源程式與公開 pretrained models 的使用條款限定個人、研究／學術、非商業用途。若有商業用途，請依官方 repository 的授權說明另外處理。本專案本身的 LICENSE 不會取代第三方 Wav2Lip 的授權條款。

## 本機端到端執行環境

已驗證環境：Windows、Python 3.11.9、PyTorch 2.5.1+cu118、NVIDIA 驅動程式 560.94、GTX 1050 2GB、FFmpeg 8.1.2。使用專案獨立 `.venv`，應用程式與現有 wrapper 共用此 Python 執行檔；不要使用系統 Python 3.14 直接執行舊版 Wav2Lip。

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

`requirements-wav2lip-lock.txt` 保存這次成功環境的完整套件版本（PyTorch 另由官方 CUDA 索引安裝）；`requirements-wav2lip.txt` 說明主要相容性版本。官方原始 requirements 面向舊 Python，請勿覆蓋安裝；librosa 0.9.2 保留官方音訊程式的舊呼叫介面，NumPy 1.26.4、Numba 0.60.0 與 OpenCV 4.10.0.84 避免 NumPy 2.x／舊音訊套件衝突。不需另外安裝完整 CUDA Toolkit。

模型來源均取自上述官方 commit 的 README：GAN 為官方 Google Drive 連結，人臉偵測權重為官方指定的 Adrian Bulat 連結。下載檔 SHA-256：

| 檔案 | SHA-256 |
| --- | --- |
| 官方 GAN 下載檔 | `180cfd49d31d47f195d5bfc62830ffe2c40724b7bbbce6faadc73ac6ab1a3b8e` |
| S3FD | `619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543` |

官方 GAN 下載檔是 TorchScript，官方 `inference.py` 卻預期 `state_dict`。`prepare_wav2lip.py` 核對來源檔雜湊與 checkout 版本，保留原檔為 `models/wav2lip_gan.official.torchscript`，抽出同一份權重並以官方模型 `strict=True` 驗證後寫入 `models/wav2lip_gan.pth`。沒有重新訓練或替換模型。另對外部 checkout 做兩項小幅相容性修正：副檔名辨識支援路徑中的句點；FFmpeg 改用參數陣列並檢查結束代碼，支援中文／空白路徑。修改可重複執行，第三方原始碼仍不納入 Git。

每次工作使用 `temp/<UUID>/`，成功後保留 `speech.mp3` 供預覽，以及 `output/<UUID>.mp4`；中間圖片、WAV 與 raw MP4 會清除，失敗工作會移除。本機成功產物可在不再播放或下載後自行清理。wrapper 的官方 `temp/result.avi` 也在每次呼叫的獨立暫存工作目錄中，結束即清除。

GTX 1050 2GB 會設定 `low_vram=True`、兩個 batch size 都為 1，pipeline 將人物圖片等比例縮至最長邊不超過 512 px。這是必要的，因為官方照片分支不套用 `resize_factor`。CUDA OOM 時直接回報顯示記憶體不足、建議下一階段使用 Colab 或降低圖片解析度，不會重試 CPU。

若要**明確選擇 CPU 備援**，可在啟動前設定；這不是 OOM 自動切換：

```powershell
$env:TALKING_PHOTO_DEVICE = "cpu"
.\.venv\Scripts\python app.py
# 恢復自動偵測：
Remove-Item Env:TALKING_PHOTO_DEVICE
```

## 真實短片驗證（M4.2）

```powershell
.\.venv\Scripts\python scripts/smoke_test.py "temp/synthetic-portrait.png"
.\.venv\Scripts\python -m pytest
```

請自行準備非私人人像，或指定有權使用的圖片；測試圖片與產物不納入 Git。此腳本會真的連線 Edge TTS、執行 FFmpeg 與 Wav2Lip，檢查影片長度 5～10 秒、H.264／AAC 串流及完整解碼，結果寫入忽略追蹤的 `temp/smoke-result.json`。GitHub Actions 僅執行 pytest，不下載模型或執行推論。

2026-09-10 實測使用 AI 生成的虛構成人人像，講稿為「大家好，這是一段說話照片測試。祝大家今天心情愉快，事事順利。」台灣女聲、語速 1.0，GTX 1050 2GB／CUDA／`low_vram=True` 成功產出 **7.872 秒**影片，通過格式與解碼檢查。另以真實 Gradio 上傳及「產生影片」按鈕測試，預設語速 0.95 成功產出 **8.32 秒**影片，語音與影片皆可播放。未使用 CPU 備援，未以 mock 代替真實推論。
