#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-musetalk"
MUSETALK="$ROOT/vendor/MuseTalk"
REVISION="0a89dec45a0192b824e3cf4daf96c239440c5ed8"

step() {
  printf '\n==> %s\n' "$1"
}

cd "$ROOT"

step "檢查 Colab GPU、Git 與 FFmpeg"
command -v git >/dev/null || { echo "找不到 Git。" >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "找不到 FFmpeg。請先安裝 FFmpeg。" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "找不到 NVIDIA GPU。請在 Colab 選擇 GPU 執行階段。" >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

step "準備 uv 與 Python 3.10"
if ! command -v uv >/dev/null; then
  python3 -m pip install -q uv
fi
UV="$(command -v uv)"
"$UV" python install 3.10
if [[ ! -x "$VENV/bin/python" ]]; then
  "$UV" venv --python 3.10 "$VENV"
fi
PY="$VENV/bin/python"

step "取得官方 TMElyralab/MuseTalk"
mkdir -p "$ROOT/vendor"
if [[ ! -e "$MUSETALK" ]]; then
  git clone https://github.com/TMElyralab/MuseTalk.git "$MUSETALK"
elif [[ ! -d "$MUSETALK/.git" ]]; then
  echo "vendor/MuseTalk 已存在但不是 Git repository；為避免覆蓋資料，請先移走。" >&2
  exit 1
fi
git -C "$MUSETALK" fetch --depth 1 origin "$REVISION"
git -C "$MUSETALK" checkout --force "$REVISION"

step "安裝 MuseTalk 專用 PyTorch 2.0.1 + CUDA 11.8"
"$PY" -m pip install --upgrade pip
"$PY" -m pip install \
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu118

step "安裝 MuseTalk 官方相依套件"
"$PY" -m pip install -r "$MUSETALK/requirements.txt"
"$PY" -m pip install --upgrade openmim
"$PY" -m mim install mmengine
"$PY" -m mim install "mmcv==2.0.1"
"$PY" -m mim install "mmdet==3.1.0"
"$PY" -m mim install "mmpose==1.1.0"

step "下載 MuseTalk 1.5 推論模型"
"$PY" "$ROOT/scripts/download_musetalk_models.py" --target "$MUSETALK/models"

step "驗證 MuseTalk CUDA 環境"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("MuseTalk 專用環境看不到 CUDA GPU；請確認 Colab 已選 GPU 執行階段。")
props = torch.cuda.get_device_properties(0)
vram = props.total_memory / 1024**3
print(f"GPU: {props.name} | VRAM: {vram:.1f} GB | PyTorch: {torch.__version__}")
if vram < 4.0:
    raise SystemExit("MuseTalk 1.5 本專案要求至少 4GB 顯示記憶體；請改用較大的 Colab GPU。")
PY

cat <<EOF

MuseTalk 1.5 高品質 backend 設定完成。
MuseTalk checkout: $REVISION
專用 Python: $PY

請用主程式的 Python 啟動 app.py，並在「嘴型引擎」選擇：
MuseTalk 1.5（高品質／建議 Colab）
EOF
