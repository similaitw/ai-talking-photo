"""MuseTalk 1.5 high-quality lip-sync backend.

MuseTalk stays in an external checkout and a dedicated Python environment. This
module only prepares one isolated task, runs the pinned upstream inference
entrypoint, validates the real MP4 output, and atomically publishes it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MUSETALK_REVISION = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
MIN_MUSETALK_VRAM_GB = 4.0

_REQUIRED_MODELS = (
    "models/musetalkV15/musetalk.json",
    "models/musetalkV15/unet.pth",
    "models/sd-vae/config.json",
    "models/sd-vae/diffusion_pytorch_model.bin",
    "models/whisper/config.json",
    "models/whisper/pytorch_model.bin",
    "models/whisper/preprocessor_config.json",
    "models/dwpose/dw-ll_ucoco_384.pth",
    "models/face-parse-bisent/79999_iter.pth",
    "models/face-parse-bisent/resnet18-5c106cde.pth",
)


class MuseTalkError(RuntimeError):
    """Raised when the MuseTalk backend cannot produce a usable result."""


def _default_python() -> str:
    configured = os.environ.get("MUSETALK_PYTHON")
    if configured:
        return configured
    windows = PROJECT_ROOT / ".venv-musetalk" / "Scripts" / "python.exe"
    posix = PROJECT_ROOT / ".venv-musetalk" / "bin" / "python"
    if windows.is_file():
        return str(windows)
    if posix.is_file():
        return str(posix)
    return sys.executable


def _default_root() -> Path:
    return Path(
        os.environ.get("MUSETALK_DIR", str(PROJECT_ROOT / "vendor" / "MuseTalk"))
    ).expanduser().resolve()


def _decode(data: bytes | None) -> str:
    return data.decode("utf-8", errors="ignore") if data else ""


def _run(
    command: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True)
    except FileNotFoundError as exc:
        raise MuseTalkError("找不到 MuseTalk Python 或 FFmpeg，請先完成 MuseTalk Colab 設定。") from exc
    if check and result.returncode != 0:
        text = (_decode(result.stdout) + "\n" + _decode(result.stderr)).lower()
        if "out of memory" in text or ("cuda" in text and "memory" in text):
            raise MuseTalkError("MuseTalk 1.5 顯示記憶體不足。請改用較大 Colab GPU、縮短講稿或降低人物素材解析度。")
        raise MuseTalkError("MuseTalk 1.5 執行失敗，請確認 Colab 環境、模型與 FFmpeg 都已準備完成。")
    return result


def get_musetalk_runtime_info(
    *,
    python_executable: str | None = None,
    musetalk_dir: str | None = None,
) -> dict:
    """Probe the dedicated MuseTalk environment without silently using CPU."""
    root = Path(musetalk_dir).expanduser().resolve() if musetalk_dir else _default_root()
    python = python_executable or _default_python()
    code = (
        "import json, torch; "
        "ok=torch.cuda.is_available(); "
        "i=0; "
        "p=torch.cuda.get_device_properties(i) if ok else None; "
        "print(json.dumps({'cuda':ok,'name':p.name if p else None,"
        "'vram_gb':(p.total_memory/1024**3) if p else 0.0}))"
    )
    try:
        result = subprocess.run(
            [python, "-c", code], cwd=root if root.is_dir() else PROJECT_ROOT,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise MuseTalkError("找不到 MuseTalk 專用 Python。請先執行 MuseTalk Colab 設定。") from exc
    if result.returncode != 0:
        raise MuseTalkError("MuseTalk Python 環境無法載入 PyTorch，請重新執行 MuseTalk Colab 設定。")
    lines = [line.strip() for line in _decode(result.stdout).splitlines() if line.strip()]
    payload = next((line for line in reversed(lines) if line.startswith("{") and line.endswith("}")), None)
    if payload is None:
        raise MuseTalkError("無法讀取 MuseTalk CUDA 環境資訊。")
    try:
        info = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise MuseTalkError("無法解析 MuseTalk CUDA 環境資訊。") from exc
    if not info.get("cuda"):
        raise MuseTalkError("MuseTalk 1.5 高品質模式需要 CUDA GPU，不會自動改用 CPU。請使用 Google Colab GPU。")
    if float(info.get("vram_gb") or 0) < MIN_MUSETALK_VRAM_GB:
        raise MuseTalkError(
            f"MuseTalk 1.5 高品質模式在本專案要求至少 {MIN_MUSETALK_VRAM_GB:g}GB 顯示記憶體；"
            "GTX 1050 2GB 請改用 Google Colab GPU。"
        )
    return info


def _validate_install(root: Path) -> None:
    inference = root / "scripts" / "inference.py"
    if not inference.is_file():
        raise MuseTalkError("找不到 MuseTalk 1.5。請先執行 scripts/setup_musetalk_colab.sh。")
    missing = [relative for relative in _REQUIRED_MODELS if not (root / relative).is_file()]
    if missing:
        raise MuseTalkError("MuseTalk 1.5 模型尚未完整下載。請重新執行 scripts/setup_musetalk_colab.sh。")

    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise MuseTalkError("找不到 Git，無法確認 MuseTalk 版本。") from exc
    if revision.returncode != 0 or _decode(revision.stdout).strip() != MUSETALK_REVISION:
        raise MuseTalkError("MuseTalk checkout 版本與專案鎖定版本不符，請重新執行 MuseTalk Colab 設定。")


def _write_task_config(path: Path, source: str, audio: str) -> None:
    # JSON strings are valid YAML scalars and safely escape Windows paths.
    path.write_text(
        "task_0:\n"
        f"  video_path: {json.dumps(source, ensure_ascii=False)}\n"
        f"  audio_path: {json.dumps(audio, ensure_ascii=False)}\n",
        encoding="utf-8",
    )


def generate_musetalk_lip_sync(
    source_path: str,
    audio_path: str,
    output_path: str,
    *,
    musetalk_dir: str | None = None,
    python_executable: str | None = None,
    use_float16: bool = True,
    batch_size: int = 4,
    fps: int = 25,
) -> str:
    """Generate one MuseTalk 1.5 MP4 from an image or video source using CUDA."""
    source = Path(source_path).expanduser().resolve()
    audio = Path(audio_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("找不到可供 MuseTalk 使用的人物照片或影片。")
    if not audio.is_file() or audio.stat().st_size == 0:
        raise ValueError("找不到可供 MuseTalk 使用的語音檔。")
    if batch_size < 1:
        raise ValueError("MuseTalk batch size 必須至少為 1。")
    if fps < 1:
        raise ValueError("MuseTalk FPS 必須至少為 1。")

    root = Path(musetalk_dir).expanduser().resolve() if musetalk_dir else _default_root()
    python = python_executable or _default_python()
    _validate_install(root)
    get_musetalk_runtime_info(python_executable=python, musetalk_dir=str(root))

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise MuseTalkError("找不到 FFmpeg。MuseTalk 1.5 高品質模式需要 FFmpeg。")

    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch_root = root / ".ai-talking-photo-temp"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary_output: Path | None = None
    with tempfile.TemporaryDirectory(prefix=f"job-{uuid4().hex[:8]}-", dir=scratch_root) as temp_name:
        workspace = Path(temp_name)
        # Preserve the original suffix. Upstream determines image vs video from
        # the source path, so renaming an MP4 to input.png would break video mode.
        suffix = source.suffix.lower() or ".bin"
        local_source = workspace / f"input{suffix}"
        local_audio = workspace / "speech.wav"
        shutil.copy2(source, local_source)
        shutil.copy2(audio, local_audio)
        config = workspace / "task.yaml"
        results = workspace / "results"

        # Keep all paths passed to upstream relative to its cwd. The official
        # inference script builds some ffmpeg commands as strings, so this also
        # avoids breakage when the user's project path contains spaces.
        rel_source = local_source.relative_to(root).as_posix()
        rel_audio = local_audio.relative_to(root).as_posix()
        rel_config = config.relative_to(root).as_posix()
        rel_results = results.relative_to(root).as_posix()
        _write_task_config(config, rel_source, rel_audio)

        command = [
            python,
            "-m", "scripts.inference",
            "--inference_config", rel_config,
            "--result_dir", rel_results,
            "--unet_model_path", "models/musetalkV15/unet.pth",
            "--unet_config", "models/musetalkV15/musetalk.json",
            "--whisper_dir", "models/whisper",
            "--version", "v15",
            "--fps", str(fps),
            "--batch_size", str(batch_size),
            "--output_vid_name", "result.mp4",
            "--ffmpeg_path", str(Path(ffmpeg).parent),
        ]
        if use_float16:
            command.append("--use_float16")

        result = _run(command, cwd=root, check=False)
        expected = results / "v15" / "result.mp4"
        if result.returncode != 0 or not expected.is_file() or expected.stat().st_size == 0:
            text = (_decode(result.stdout) + "\n" + _decode(result.stderr)).lower()
            if "out of memory" in text or ("cuda" in text and "memory" in text):
                raise MuseTalkError("MuseTalk 1.5 顯示記憶體不足。請改用較大 Colab GPU、縮短講稿或降低人物素材解析度。")
            if "face" in text and ("detect" in text or "landmark" in text):
                raise MuseTalkError("MuseTalk 找不到可用的人臉。請改用正面、清楚、嘴部未遮擋的照片或影片。")
            raise MuseTalkError("MuseTalk 1.5 沒有產生有效影片。請檢查 Colab GPU、模型與人物素材。")

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-", suffix=".mp4", dir=destination.parent
        )
        os.close(fd)
        temporary_output = Path(tmp_name)
        try:
            shutil.copy2(expected, temporary_output)
            if temporary_output.stat().st_size == 0:
                raise MuseTalkError("MuseTalk 1.5 產生的影片檔案無效。")
            os.replace(temporary_output, destination)
            temporary_output = None
        finally:
            if temporary_output is not None:
                temporary_output.unlink(missing_ok=True)

    try:
        scratch_root.rmdir()
    except OSError:
        pass
    return str(destination)
