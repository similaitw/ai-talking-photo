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

完整影片 pipeline 尚未接入；「產生影片」目前仍會明確顯示 `Pipeline 尚未啟用`。下一階段 M4.2 才會串接 TTS、FFmpeg 與 Wav2Lip 產生 MP4。

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

音訊預覽仍提供 MP3；音訊轉換將於後續影片流程接入。測試包含實際 FFmpeg 轉檔，未安裝 FFmpeg 時會標示略過原因，其餘測試照常執行。

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
