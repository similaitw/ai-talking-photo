# Google Colab 使用說明

這份說明對應 `notebooks/AI_Talking_Photo_Colab.ipynb`。目標是讓沒有合適本機 GPU 的使用者，在 Google Colab GPU 工作階段完成 AI Talking Photo 的短片推論。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/similaitw/ai-talking-photo/blob/main/notebooks/AI_Talking_Photo_Colab.ipynb)

## 使用前準備

你需要：

- Google 帳號與可使用的 Colab 工作階段。
- 一張你有權使用、正面清楚且嘴部未遮擋的人像照片。
- 一段繁體中文講稿；建議第一次先用 5～10 秒短句測試。
- 網路連線，因為 notebook 需要 clone GitHub、下載相依套件與官方模型，Edge TTS 也需要連線。

官方 Wav2Lip 公開程式與 pretrained models 限個人、研究／學術、非商業用途；商業用途需另行確認第三方授權。

## 第一次執行

### 1. 開啟 GPU 執行階段

在 Colab 選：

`執行階段 → 變更執行階段類型 → GPU`

GPU 型號由 Colab 分配，不保證每次相同。免費方案的 GPU、執行時間與用量限制都可能變動。

### 2. 全部執行

選：

`執行階段 → 全部執行`

Notebook 會依序：

1. 用 `nvidia-smi` 確認 GPU。
2. 從 GitHub `main` clone 或更新 `similaitw/ai-talking-photo`。
3. 用 `uv` 建立獨立 Python 3.11 環境，不依賴 Colab 當下的系統 Python。
4. 安裝已驗證的 PyTorch 2.5.1 + CUDA 11.8 與 Wav2Lip 相容套件。
5. clone 官方 `Rudrabha/Wav2Lip`，固定到專案已驗證 commit。
6. 下載官方 GAN 與 S3FD 權重。
7. 執行 `scripts/prepare_wav2lip.py` 驗證 SHA-256 並準備 checkpoint。
8. 執行 `scripts/doctor.py`。
9. 啟動 Gradio `share=True`。

第一次執行下載較多套件與模型；同一個 Colab 工作階段重跑時，已存在的資料會盡量重用。

### 3. 開啟 Gradio

最後一個 cell 會顯示 `https://....gradio.live` 類型的網址。

開啟後：

1. 上傳人物照片。
2. 輸入講稿。
3. 選台灣女聲或台灣男聲。
4. 調整語速。
5. 可先按「產生語音預覽」。
6. 按「產生影片」。
7. 等待完成後播放或下載 MP4。

首次測試建議使用短講稿，確認整條 pipeline 正常後再增加長度。

## 輸出檔案

成功後主要保留：

```text
temp/<UUID>/speech.mp3
output/<UUID>.mp4
```

WAV、縮圖與 raw MP4 等中間檔會自動清理。

Colab 的本機磁碟是暫時性的。工作階段結束、重設或被回收後，檔案可能消失，所以需要保留的 MP3／MP4 請在工作階段結束前下載到自己的裝置或另行備份。

## 重要限制

### Colab 資源不是保證資源

Google 官方說明，Colab 的免費資源不保證也不提供無限量使用；GPU 型號、用量限制、閒置逾時與 VM 生命週期都可能動態改變。因此：

- notebook 能啟動不代表每次都有 GPU。
- 同一份 notebook 不保證每次拿到相同 GPU。
- 長時間推論可能因工作階段終止而中斷。
- 不要把 Colab 當成常駐服務或正式部署平台。

官方 FAQ：
<https://research.google.com/colaboratory/faq.html>

### Gradio 公開網址是臨時測試入口

`share=True` 會建立公開可連線的臨時網址。知道網址的人可能可以開啟介面，因此：

- 不要上傳敏感、機密或未取得權利的人像／內容。
- 不要把網址公開貼到社群。
- 用完後停止最後一個 cell 或結束 Colab 工作階段。
- 這個網址不是固定網址，也不是正式網站部署。

Google 也明確說明，免費受管 Colab 對「繞過 notebook UI、主要透過 Web UI 互動」的使用有限制，這類工作階段可能提早終止。因此本專案的 Gradio share 適合短時間測試，不保證可長時間持續運作。

## 常見問題

### 顯示「未偵測到 NVIDIA GPU」

確認 Colab 已選 GPU 執行階段，再從第一個 cell 重新全部執行。如果 GPU 選項暫時不可用，可能是當下帳號／資源限制，稍後再試或改用本機環境。

### PyTorch 顯示 CUDA 不可用

不要只重跑最後一格。先：

1. `執行階段 → 重新啟動工作階段`
2. 確認硬體加速器仍是 GPU
3. 再「全部執行」

若仍失敗，查看 PyTorch 安裝 cell 與 `nvidia-smi` 的輸出。

### Doctor 結束代碼不是 0

往上找 Doctor 列出的缺項。常見原因：

- PyTorch／CUDA 無法使用。
- FFmpeg 無法執行。
- edge-tts 套件未完成安裝。
- Wav2Lip checkpoint 或 S3FD 權重缺失／下載不完整。

不要略過 Doctor 後強制啟動影片流程。

### 模型 SHA-256 不符

代表模型檔可能下載不完整、來源內容改變或工作階段中留下損壞檔案。請刪除 Colab 工作階段中的相關模型檔後重新執行下載 cell；不要關閉雜湊檢查來繞過錯誤。

### Edge TTS 失敗

Edge TTS 需要網路。先確認 Colab 可正常連外，然後重試短講稿。如果服務暫時無法連線，稍後重新產生。

### 找不到人臉

改用：

- 正面或接近正面
- 臉部清楚
- 嘴巴沒有被手、口罩或物品遮擋
- 不要裁切掉下巴
- 避免人物太小

### CUDA OOM／顯示記憶體不足

先使用短講稿與較簡單照片重試。本專案會限制靜態圖片最大邊長並使用低 batch size，但 Colab 分配的 GPU 仍可能因當下記憶體狀態而不足。若持續 OOM，重新啟動工作階段釋放 GPU 記憶體。

系統不會在 CUDA OOM 後偷偷改用 CPU。

### Gradio share URL 沒出現或中途失效

確認最後一個 cell 還在執行。`gradio.live` 是臨時 share，不保證長時間存在；Colab 也可能因資源或政策限制終止工作階段。必要時重新啟動工作階段並 Run All。

## 重跑與更新

Notebook 每次會從 GitHub `main` 取得最新版。如果同一工作階段已有 repository，會 fetch 並 reset 到 `origin/main`。

若環境變得混亂，最乾淨的處理方式通常是：

`執行階段 → 中斷連線並刪除執行階段`

再重新選 GPU 並「全部執行」。

## 隱私與使用責任

AI Talking Photo 是人物合成工具。請只使用有權使用的人像、聲音與內容，不要用來冒充本人、欺騙他人，或在未取得適當同意的情況下公開發布。

本 notebook 不會把你的照片、音訊或生成影片 commit 回 GitHub；但 Colab、Edge TTS、Gradio share 與其他第三方服務各自有其服務與隱私政策，使用前應自行確認。
