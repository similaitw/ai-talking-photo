"""Optional GFPGAN post-processing for talking-photo videos."""

from __future__ import annotations

from fractions import Fraction
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


class GFPGANError(RuntimeError):
    """Raised when GFPGAN enhancement cannot produce a usable video."""


def _run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(command, cwd=cwd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise GFPGANError("找不到 FFmpeg／ffprobe 或 GFPGAN 執行環境，請先完成 GFPGAN 設定。") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        if "out of memory" in stderr.lower() or "cuda" in stderr.lower() and "memory" in stderr.lower():
            raise GFPGANError(
                "GFPGAN 顯示記憶體不足。GTX 1050 2GB 建議關閉背景放大、使用 1× 修復；若仍失敗請改用 Colab。"
            ) from exc
        raise GFPGANError("GFPGAN 高清修復失敗，請檢查安裝、模型與 FFmpeg。") from exc


def _video_fps(video_path: Path) -> str:
    result = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate", "-of", "default=nw=1:nk=1",
        str(video_path),
    ])
    raw = result.stdout.decode("utf-8", errors="ignore").strip()
    try:
        value = Fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        raise GFPGANError("無法讀取來源影片幀率。") from exc
    if value <= 0:
        raise GFPGANError("來源影片幀率無效。")
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def enhance_video_with_gfpgan(
    input_path: str,
    output_path: str,
    *,
    gfpgan_dir: str | None = None,
    python_executable: str | None = None,
    version: str = "1.3",
    weight: float = 0.4,
) -> str:
    """Restore the center face in every frame while preserving video dimensions.

    GFPGAN is kept as an external checkout. The default profile intentionally
    disables Real-ESRGAN background upsampling and uses scale 1 so a 2GB GPU
    only spends memory on face restoration.
    """

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError("找不到可供 GFPGAN 修復的影片。")
    if source == destination:
        raise ValueError("GFPGAN 輸入與輸出影片不可使用同一路徑。")
    if version not in {"1.3", "1.4"}:
        raise ValueError("GFPGAN 版本目前只支援 1.3 或 1.4。")
    if not 0 <= weight <= 1:
        raise ValueError("GFPGAN 修復強度必須介於 0 到 1。")

    root = Path(
        gfpgan_dir or os.environ.get("GFPGAN_DIR", "vendor/GFPGAN")
    ).expanduser().resolve()
    inference = root / "inference_gfpgan.py"
    if not inference.is_file():
        raise GFPGANError(
            "找不到 GFPGAN。請先執行 scripts/setup_gfpgan.ps1，或設定 GFPGAN_DIR。"
        )

    python = python_executable or os.environ.get("GFPGAN_PYTHON") or sys.executable
    destination.parent.mkdir(parents=True, exist_ok=True)
    fps = _video_fps(source)

    temporary_output: Path | None = None
    with tempfile.TemporaryDirectory(prefix="gfpgan-", dir=str(destination.parent)) as temp_name:
        temp = Path(temp_name)
        frames = temp / "frames"
        restored = temp / "restored"
        frames.mkdir()

        _run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:v:0", "-vsync", "0",
            str(frames / "%08d.png"),
        ])
        if not any(frames.glob("*.png")):
            raise GFPGANError("影片拆幀失敗，沒有可修復的畫面。")

        _run([
            python, str(inference),
            "-i", str(frames),
            "-o", str(restored),
            "-v", version,
            "-s", "1",
            "--bg_upsampler", "none",
            "--only_center_face",
            "--ext", "png",
            "-w", str(weight),
        ], cwd=root)

        restored_frames = restored / "restored_imgs"
        if not restored_frames.is_dir() or not any(restored_frames.glob("*.png")):
            raise GFPGANError("GFPGAN 完成後找不到修復影格。")

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-", suffix=".mp4", dir=destination.parent
        )
        os.close(fd)
        temporary_output = Path(tmp_name)
        temporary_output.unlink(missing_ok=True)

        _run([
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", fps, "-i", str(restored_frames / "%08d.png"),
            "-i", str(source),
            "-map", "0:v:0", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-movflags", "+faststart", str(temporary_output),
        ])

        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise GFPGANError("GFPGAN 影片合成完成但輸出檔案無效。")
        os.replace(temporary_output, destination)
        temporary_output = None

    return str(destination)
