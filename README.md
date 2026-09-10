# AI Talking Photo

單張人物照片 + 中文講稿 → AI 說話影片。

目前專案處於初始化階段。主要推論環境規劃為 Google Colab 免費 GPU；GTX 1050 2GB 本機環境作為低顯存備援。

## Development

```bash
python -m pytest
```

詳細規格請見 `docs/SPEC.md`，目前工作請見 `docs/TASKS.md`。

## 目前進度

M1.1 Gradio Shell 與 M1.2 輸入驗證已完成。可執行：

```bash
python app.py
```

目前介面可操作並會檢查圖片與講稿；AI 產生 pipeline 尚未接入。下一階段為 Edge TTS。
