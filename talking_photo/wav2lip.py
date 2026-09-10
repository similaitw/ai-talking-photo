"""Subprocess wrapper around an external Wav2Lip checkout."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAV2LIP_DIR = PROJECT_ROOT / "vendor" / "Wav2Lip"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "wav2lip_gan.pth"


class Wav2LipError(RuntimeError):
    """Raised when Wav2Lip inference cannot complete."""


class Wav2LipConfigurationError(Wav2LipError):
    """Raised when the local Wav2Lip runtime is not ready."""


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _runtime_paths() -> tuple[Path, Path, Path]:
    wav2lip_dir = Path(
        os.environ.get("WAV2LIP_DIR", DEFAULT_WAV2LIP_DIR)
    ).expanduser().resolve()
    checkpoint = Path(
        os.environ.get("WAV2LIP_CHECKPOINT", DEFAULT_CHECKPOINT)
    ).expanduser().resolve()
    return wav2lip_dir, wav2lip_dir / "inference.py", checkpoint


def _cuda_index(device: str) -> int | None:
    normalized = device.strip().lower()
    if normalized == "cpu":
        return None
    if normalized == "cuda":
        index = 0
    else:
        match = re.fullmatch(r"cuda:(\d+)", normalized)
        if not match:
            raise ValueError("device 必須是 cpu、cuda 或 cuda:N。")
        index = int(match.group(1))

    try:
        torch = import_module("torch")
        if not torch.cuda.is_available():
            raise Wav2LipConfigurationError(
                "已指定 CUDA，但目前環境沒有可用的 CUDA；不會自動改用 CPU。"
            )
        count = int(torch.cuda.device_count())
    except Wav2LipConfigurationError:
        raise
    except Exception as exc:
        raise Wav2LipConfigurationError(
            "無法確認 CUDA 狀態，請先執行 python scripts/doctor.py。"
        ) from exc

    if index >= count:
        raise Wav2LipConfigurationError(
            f"找不到 CUDA 裝置 {index}；目前偵測到 {count} 個 CUDA 裝置。"
        )
    return index


def _build_command(
    inference_script: Path,
    checkpoint: Path,
    image: Path,
    audio: Path,
    output: Path,
    *,
    low_vram: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(inference_script),
        "--checkpoint_path",
        str(checkpoint),
        "--face",
        str(image),
        "--audio",
        str(audio),
        "--outfile",
        str(output),
        "--static",
        "True",
    ]
    if low_vram:
        command.extend(
            [
                "--wav2lip_batch_size",
                "1",
                "--face_det_batch_size",
                "1",
                "--resize_factor",
                "2",
            ]
        )
    return command


def _friendly_failure(output: str) -> str:
    lowered = output.lower()
    if "out of memory" in lowered or "image too big to run face detection on gpu" in lowered:
        return (
            "CUDA 顯示記憶體不足（GTX 1050 2GB 容量有限）。建議下一階段使用 Google Colab GPU、降低圖片解析度，"
            "或縮短單段音訊；系統不會自動改用 CPU。"
        )
    if "face not detected" in lowered or "face not found" in lowered:
        return "找不到可用的人臉，請改用正面、清楚且嘴部未遮擋的照片。"
    return (
        "Wav2Lip 推論失敗，請先執行 python scripts/doctor.py，"
        "並檢查模型、相依套件與輸入檔案。"
    )


def generate_lip_sync(
    image_path: str,
    audio_path: str,
    output_path: str,
    device: str,
    low_vram: bool = False,
) -> str:
    """Run Wav2Lip for a still image and normalized audio, returning an MP4 path.

    Wav2Lip itself is kept as an external checkout. Set ``WAV2LIP_DIR`` to its
    repository directory and optionally ``WAV2LIP_CHECKPOINT`` to a checkpoint.
    A temporary output is used so failed inference does not destroy an existing file.
    """
    image = Path(image_path).expanduser().resolve()
    audio = Path(audio_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()

    if not _nonempty_file(image):
        raise ValueError("找不到人物照片或檔案為空白。")
    if not _nonempty_file(audio):
        raise ValueError("找不到輸入音訊或檔案為空白。")
    if destination in {image, audio}:
        raise ValueError("輸出影片不可覆蓋輸入照片或音訊。")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("輸出影片必須使用 .mp4 副檔名。")

    cuda_index = _cuda_index(device)
    wav2lip_dir, inference_script, checkpoint = _runtime_paths()
    if not inference_script.is_file():
        raise Wav2LipConfigurationError(
            "找不到 Wav2Lip/inference.py。請將官方 Wav2Lip 專案放到 vendor/Wav2Lip，"
            "或設定 WAV2LIP_DIR。"
        )
    if not _nonempty_file(checkpoint):
        raise Wav2LipConfigurationError(
            "找不到 Wav2Lip checkpoint。請放入 models/wav2lip_gan.pth，"
            "或設定 WAV2LIP_CHECKPOINT。"
        )

    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix="wav2lip-", suffix=".mp4", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        temporary.unlink(missing_ok=True)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        if cuda_index is None:
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_index)

        command = _build_command(
            inference_script,
            checkpoint,
            image,
            audio,
            temporary,
            low_vram=low_vram,
        )
        try:
            # Upstream writes temp/result.avi relative to cwd: isolate every call.
            with tempfile.TemporaryDirectory(prefix="inference-", dir=destination.parent) as work:
                (Path(work) / "temp").mkdir()
                result = subprocess.run(
                    command,
                    cwd=work,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
        except FileNotFoundError as exc:
            raise Wav2LipConfigurationError(
                "無法啟動 Python 執行 Wav2Lip，請檢查目前 Python 環境。"
            ) from exc
        except subprocess.CalledProcessError as exc:
            combined = f"{exc.stdout or ''}\n{exc.stderr or ''}"
            raise Wav2LipError(_friendly_failure(combined)) from exc

        combined = f"{result.stdout or ''}\n{result.stderr or ''}"
        if not _nonempty_file(temporary):
            raise Wav2LipError(
                "Wav2Lip 已結束但沒有產生有效的 MP4，請檢查推論輸出與 FFmpeg。"
            )
        if cuda_index is not None and "Using cpu for inference" in combined:
            raise Wav2LipError(
                "已指定 CUDA，但 Wav2Lip 實際改用 CPU；"
                "為避免長時間無提示運算，本次已停止採用該結果。"
            )
        os.replace(temporary, destination)
        temporary = None
        return str(destination)
    except OSError as exc:
        raise Wav2LipError(
            "無法寫入 Wav2Lip 輸出影片，請檢查資料夾權限與可用空間。"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
