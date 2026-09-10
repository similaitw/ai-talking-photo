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

影片嘴型同步尚未接入；「產生影片」目前會明確顯示 `Pipeline 尚未啟用`。

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
