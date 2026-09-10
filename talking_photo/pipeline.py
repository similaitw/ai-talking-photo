"""Photo-to-video orchestration using the existing inference wrapper."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path
import shutil
import subprocess
from uuid import uuid4

from PIL import Image, ImageOps

from talking_photo.config import TEMP_DIR, OUTPUT_DIR
from talking_photo.device import get_device_info
from talking_photo.media import normalize_audio
from talking_photo.tts import synthesize_speech, resolve_voice, format_edge_rate
from talking_photo.validation import validate_inputs
from talking_photo.wav2lip import generate_lip_sync


class PipelineError(RuntimeError):
    """A stage failed without a usable final video."""


def finalize_video(source: Path, destination: Path) -> None:
    """Encode browser-compatible H.264/AAC MP4 before publishing the result."""
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(source),
             "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:a", "aac", "-movflags", "+faststart", str(destination)],
            check=True, capture_output=True,
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise PipelineError("影片封裝未產生有效的 MP4。")
    except (OSError, subprocess.SubprocessError) as exc:
        raise PipelineError("影片封裝失敗，請檢查 FFmpeg 與可用磁碟空間。") from exc


def generate_talking_video(
    image_path: str,
    text: str,
    voice: str,
    rate: float,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    """Return job_id, audio_path, video_path, device, gpu_name and low_vram.

    Successful jobs retain only preview MP3 and final MP4. Failed jobs are removed.
    TALKING_PHOTO_DEVICE=cpu explicitly selects CPU; CUDA errors never retry on CPU.
    """
    def report(value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    report(0, "正在驗證照片與講稿。")
    image, script = validate_inputs(image_path, text)
    resolve_voice(voice)
    format_edge_rate(rate)
    info = get_device_info()
    requested = os.environ.get("TALKING_PHOTO_DEVICE", "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise PipelineError("TALKING_PHOTO_DEVICE 必須是 auto、cpu 或 cuda。")
    if info.error:
        raise PipelineError(info.error)
    device = info.device if requested == "auto" else requested
    # CPU fallback also uses batch size 1 to bound memory usage.
    low_vram = info.low_vram or device == "cpu"
    job_id = uuid4().hex
    job = TEMP_DIR / job_id
    destination = OUTPUT_DIR / f"{job_id}.mp4"
    try:
        job.mkdir(parents=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        prepared = job / "input.png"
        with Image.open(image) as portrait:
            portrait = ImageOps.exif_transpose(portrait).convert("RGB")
            if low_vram:
                portrait.thumbnail((512, 512))
            portrait.save(prepared)
        mode = "CPU 備援，可能需要較長時間" if device == "cpu" else "CUDA 低顯示記憶體模式" if low_vram else "CUDA"
        report(0.1, f"正在產生台灣中文語音；推論將使用 {mode}。")
        audio = synthesize_speech(script, voice, rate, str(job / "speech.mp3"))
        report(0.3, "正在轉換為 16 kHz 單聲道 PCM WAV。")
        wav = normalize_audio(audio, str(job / "speech.wav"))
        report(0.45, f"正在使用 Wav2Lip 產生嘴型同步影片（{mode}）。")
        raw = generate_lip_sync(str(prepared), wav, str(job / "raw.mp4"), device, low_vram=low_vram)
        report(0.9, "正在封裝可播放的 MP4。")
        final = job / "final.mp4"
        finalize_video(Path(raw), final)
        os.replace(final, destination)
        for intermediate in (prepared, job / "speech.wav", job / "raw.mp4"):
            intermediate.unlink(missing_ok=True)
        report(1, "影片已產生，可播放語音與影片。")
        return dict(job_id=job_id, audio_path=audio, video_path=str(destination),
                    device=device, gpu_name=info.gpu_name, low_vram=low_vram)
    except BaseException:
        # Only remove paths created for this UUID job, never user inputs.
        shutil.rmtree(job, ignore_errors=True)
        destination.unlink(missing_ok=True)
        raise
