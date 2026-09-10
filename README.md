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
