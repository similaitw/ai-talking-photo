# MuseTalk 1.5 高品質模式

MuseTalk 1.5 是 AI Talking Photo 的高品質嘴型 backend。它和原本的 Wav2Lip / GFPGAN 並存，不會取代已經可在 GTX 1050 2GB 執行的快速模式。人物素材現在可使用**照片或影片**；若目標是整體自然度，優先建議短人物影片。

## 模式怎麼選

| 模式 | 建議硬體 | 用途 |
| --- | --- | --- |
| Wav2Lip | GTX 1050 2GB / CPU 備援 | 快速、低顯存；照片或影片 |
| Wav2Lip + GFPGAN | GTX 1050 2GB | 在 Wav2Lip 後改善臉部清晰度 |
| MuseTalk 1.5 | Google Colab GPU / >= 4GB VRAM | 嘴型自然度與臉部品質優先；影片輸入優先推薦 |

GFPGAN 是畫質後處理，不會重新計算嘴型。若主要問題是 Wav2Lip 嘴型不自然，應改用 MuseTalk，而不是繼續提高 GFPGAN 強度。

## 為什麼影片通常更自然

MuseTalk 官方 normal inference 的 `video_path` 支援影片、圖片或圖片資料夾。影片輸入會先拆成逐幀影像並讀取來源 FPS，因此原素材中的眨眼、頭部微動、肩膀與身體動作可以保留下來；MuseTalk 主要重新產生與音訊相關的嘴型區域。

建議素材：

- 5～10 秒自然待機人物影片。
- 正面或小角度，避免大幅轉頭。
- 光線穩定，臉與嘴巴清楚，不被手、麥克風或頭髮遮住。
- 可以自然眨眼、微點頭或有輕微呼吸感。
- 最好不要在來源影片裡明顯講另一段話，讓嘴部初始動作保持簡單。

若新 TTS 音訊比來源影片長，MuseTalk 官方 normal inference 會建立「正放 + 倒放」的幀序列並持續取用，因此動作越小，循環接點越不明顯。來源影片原本的聲音不會成為成品聲音，本專案最後使用新產生的 Edge TTS。

## 為什麼 MuseTalk 走 Colab

官方 MuseTalk 1.5 的臉部生成區域為 256×256，支援中文音訊；官方 Gradio 文件曾以 RTX 3050 Ti Laptop 4GB、fp16 測試。這是官方曾驗證過的最低硬體案例，不代表所有 4GB GPU 都保證成功。

本專案因此採較保守的規則：

- MuseTalk 模式必須有 CUDA GPU。
- 顯示記憶體低於 4GB 時直接停止。
- GTX 1050 2GB 不嘗試 MuseTalk。
- 不會因 CUDA 不可用或 OOM 而靜默改成 CPU。
- 優先建議 Google Colab GPU；未來 RTX 3060 12GB 可再做本機實機驗證。

## 最快開始：專用 Colab Notebook

開啟：

`notebooks/AI_Talking_Photo_MuseTalk_Colab.ipynb`

Notebook 的流程是：

1. 檢查 Colab 是否有 NVIDIA GPU。
2. 從 GitHub `main` 取得最新 AI Talking Photo。
3. 建立主程式 Python 3.11 `.colab-venv`。
4. 執行 `scripts/setup_musetalk_colab.sh`。
5. 建立獨立 MuseTalk Python 3.10 `.venv-musetalk`。
6. 固定官方 `TMElyralab/MuseTalk` commit `0a89dec45a0192b824e3cf4daf96c239440c5ed8`。
7. 安裝官方建議的 PyTorch 2.0.1 / torchvision 0.15.2 / torchaudio 2.0.2 CUDA 11.8 與 MMLab 相依套件。
8. 下載 MuseTalk 1.5 推論所需模型。
9. 啟動 Gradio `share=True`，並預設選擇 MuseTalk 1.5 高品質模式。

第一次執行需要下載多個模型與編譯／安裝相依套件，因此會比 Wav2Lip Colab 花更久。Colab 免費 GPU 型號、可用時間與資源都不保證；若工作階段被回收，未保存的暫存檔會消失。

安裝腳本固定 MuseTalk 專用 Python 的 `pip==24.0` 與 `setuptools==69.5.1`，避開 OpenMIM / MMCV 與較新版打包工具的已知相容問題。失敗時會保留 `temp/musetalk_setup.log`，notebook 會顯示最後 120 行以便排錯。

## 第一次 A/B 測試

為了只比較嘴型引擎與整體自然度，第一次建議：

- 使用同一支 5～10 秒自然待機人物影片。
- 使用相同講稿、聲音與語速。
- 先用 **Wav2Lip（快速／低顯存）** 產生一支。
- 再用 **MuseTalk 1.5（高品質／建議 Colab）** 產生一支。
- `畫質後處理` 都先選 **關閉（較快）**。

比較四件事：嘴型自然度、嘴唇／牙齒細節、頭部與眨眼的連續感，以及來源影片循環接點是否明顯。先不要混入 GFPGAN，否則無法判斷改善來自哪一段流程。

## 環境與模型位置

MuseTalk 不安裝到主 `.venv`：

```text
.venv-musetalk/
vendor/MuseTalk/
```

兩者都列在 `.gitignore`，模型權重、第三方 checkout 與生成影片不會提交到 GitHub。

預設 wrapper 會依序尋找：

- `MUSETALK_PYTHON` 環境變數；否則使用專案的 `.venv-musetalk`。
- `MUSETALK_DIR` 環境變數；否則使用 `vendor/MuseTalk`。

正常使用專用 Colab notebook 不需要設定這些環境變數。

## 推論設定

本專案 MuseTalk 1.5 預設：

- `version=v15`
- `use_float16=True`
- `batch_size=4`
- 照片模式：長邊最多 1280 px，`fps=25`
- 影片模式：保留原始影片副檔名交給官方 inference，使用來源影片 FPS
- MuseTalk 結果再以 H.264 CRF 18 / AAC 封裝

25fps 是照片模式的預設；影片模式會由官方 inference 讀取來源影片 FPS。`batch_size=4` 是本專案為降低 Colab GPU 顯存尖峰採用的較保守值；若未來有充分實機測試，再考慮依 GPU VRAM 自動調整。

## Wav2Lip 影片模式差異

前端同一個「人物素材（照片或影片）」也支援 Wav2Lip。影片輸入時本專案不傳 `--static`，讓官方 Wav2Lip 逐幀使用來源影片；如果新語音比來源影片長，Wav2Lip 依來源幀數循環取幀。照片專用的固定人臉框與靜態嘴周柔和融合不套用到影片，避免把動態人臉鎖在單一座標。

## 錯誤處理

官方 normal inference 對個別 task 會捕捉例外，因此單看 subprocess exit code 不足以確認成功。本專案 wrapper 還會檢查預期 MP4 是否真的存在且非空白，避免顯示假成功。

常見訊息：

- `MuseTalk 1.5 高品質模式需要 CUDA GPU`：Colab 沒有選 GPU，或本機沒有可用 CUDA。
- `至少 4GB 顯示記憶體`：硬體不符合本專案 MuseTalk 門檻；GTX 1050 2GB 請使用 Wav2Lip 或改用 Colab。
- `顯示記憶體不足`：目前 Colab GPU 在該輸入下 OOM；先縮短講稿或降低人物素材解析度，再考慮更大 GPU。
- `找不到可用的人臉`：改用正面、清楚、嘴部未遮擋、臉部比例較大的照片或影片。
- `模型尚未完整下載`：重新執行 `scripts/setup_musetalk_colab.sh`。

## 授權

依官方 MuseTalk README：

- MuseTalk 程式碼採 MIT License，官方說明可供學術與商業使用。
- 官方 MuseTalk trained model 說明為可供任何用途，包括商業用途。
- 但 MuseTalk 使用的其他開源模型仍各自受自己的授權條款約束，例如 Whisper、VAE、DWPose、Face Parse 等，使用前仍需逐一確認。
- 官方測試資料另有非商業研究用途限制；本專案不提交官方測試資料。

本 repository 的 LICENSE 不會取代任何第三方模型或套件的授權條款。

## 隱私

Colab notebook 會用 Gradio `share=True` 建立暫時性的公開 `gradio.live` 網址。不要把敏感照片或影片、未取得適當同意的人像或不應外流的講稿交給公開 `gradio.live` 連結。成品產生後請自行保存；Colab 工作階段結束後暫存內容可能消失。
