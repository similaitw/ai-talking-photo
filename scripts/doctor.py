"""Read-only environment diagnostics; run with python scripts/doctor.py."""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import platform
import subprocess
import sys

# Support direct execution from any working directory without package installation.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from talking_photo.device import DeviceInfo, get_device_info


def recommended_mode(info: DeviceInfo) -> str:
    if info.error or info.device == "cpu":
        return "優先使用 Google Colab GPU；本機 CPU 僅作為備援，推論可能耗時很久。"
    if info.low_vram:
        return "優先使用 Google Colab GPU；本機 CUDA 低顯示記憶體模式僅作為備援。"
    return "可使用本機 CUDA；仍需備妥相依套件與模型，實際可用性須經推論驗證。"


def diagnose(model_path: Path) -> tuple[list[str], bool]:
    """Return human-readable checks and whether prerequisites were detected."""
    python_ok = sys.version_info >= (3, 10)
    lines = [
        "AI Talking Photo 環境診斷",
        f"Python：{platform.python_version()}（{'符合需求' if python_ok else '需要 3.10 以上版本'}）",
    ]
    info = get_device_info()
    lines.extend([
        f"PyTorch：{info.torch_version or '無法使用'}",
        f"PyTorch CUDA 建置版本：{info.cuda_version or '無'}",
        f"CUDA：{'可使用' if info.device.startswith('cuda') else '無法使用'}",
        f"GPU：{info.gpu_name or '未偵測到可用的 CUDA GPU'}",
        f"顯示記憶體：{info.vram_gb:.2f} GiB（裝置總容量，非目前可用容量）",
    ])
    if info.error:
        lines.append(f"裝置診斷：{info.error}")

    ffmpeg_ok = False
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], check=True, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=10,
        )
        ffmpeg_ok = result.stdout.startswith("ffmpeg version")
        lines.append("FFmpeg：" + (result.stdout.splitlines()[0] if ffmpeg_ok else "版本資訊無法辨識"))
    except (OSError, subprocess.SubprocessError):
        lines.append("FFmpeg：無法執行，請確認已安裝並加入 PATH。")

    edge_ok = False
    try:
        edge = import_module("edge_tts")
        edge_ok = callable(getattr(edge, "Communicate", None))
        lines.append("edge-tts：" + ("可載入（未測試網路語音服務）" if edge_ok else "套件不完整，請重新安裝。"))
    except Exception:
        lines.append("edge-tts：未安裝或無法載入，請安裝專案相依套件。")

    try:
        model_ok = model_path.is_file() and model_path.stat().st_size > 0
        if model_ok:
            with model_path.open("rb") as model:
                model_ok = bool(model.read(1))
    except OSError:
        model_ok = False
    lines.append(f"模型檔：{model_path}")
    lines.append("模型狀態：" + (
        "檔案存在且非空白（未驗證權重內容與相容性）。" if model_ok
        else "不存在、空白或無法讀取；請備妥 Wav2Lip checkpoint。"
    ))
    lines.append("建議模式：" + recommended_mode(info))
    lines.append("此診斷不執行模型推論，也不代表影片流程已可使用。")
    return lines, python_ok and info.torch_version is not None and info.error is None and ffmpeg_ok and edge_ok and model_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="檢查 AI Talking Photo 執行環境", add_help=False,
        usage="python scripts/doctor.py [-h] [--model-path 模型路徑]",
    )
    parser._optionals.title = "選項"
    parser.add_argument("-h", "--help", action="help", help="顯示使用說明並結束")
    parser.add_argument("--model-path", type=Path, metavar="模型路徑", default=ROOT / "models" / "wav2lip_gan.pth", help="指定要檢查的 Wav2Lip 模型檔路徑")
    args = parser.parse_args(argv)
    lines, ready = diagnose(args.model_path.expanduser().resolve())
    sys.stdout.write("\n".join(lines) + "\n")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
