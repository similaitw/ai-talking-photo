"""Portrait-media-to-video orchestration using selectable lip-sync backends."""

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
from talking_photo.gfpgan import enhance_video_with_gfpgan
from talking_photo.media import normalize_audio
from talking_photo.musetalk import (
    MIN_MUSETALK_VRAM_GB,
    generate_musetalk_lip_sync,
)
from talking_photo.quality import detect_face_box, limit_low_vram_output
from talking_photo.tts import synthesize_speech, resolve_voice, format_edge_rate
from talking_photo.validation import media_kind, probe_video_size, validate_inputs
from talking_photo.wav2lip import generate_lip_sync


class PipelineError(RuntimeError):
    """A stage failed without a usable final video."""


BACKEND_WAV2LIP = "Wav2Lip（快速／低顯存）"
BACKEND_MUSETALK = "MuseTalk 1.5（高品質／建議 Colab）"
BACKEND_OPTIONS = (BACKEND_WAV2LIP, BACKEND_MUSETALK)

ENHANCEMENT_NONE = "關閉（較快）"
ENHANCEMENT_GFPGAN = "GFPGAN 高清修復（實驗）"
ENHANCEMENT_OPTIONS = (ENHANCEMENT_NONE, ENHANCEMENT_GFPGAN)


def finalize_video(source: Path, destination: Path) -> None:
    """Encode browser-compatible H.264/AAC MP4 with low visible recompression loss."""
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(source),
             "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264",
             "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:a", "aac", "-movflags", "+faststart", str(destination)],
            check=True, capture_output=True,
        )
        if not destination.is_file() or destination.stat().st_size == 0:
            raise PipelineError("影片封裝未產生有效的 MP4。")
    except (OSError, subprocess.SubprocessError) as exc:
        raise PipelineError("影片封裝失敗，請檢查 FFmpeg 與可用磁碟空間。") from exc


def generate_talking_video(
    media_path: str,
    text: str,
    voice: str,
    rate: float,
    progress_callback: Callable[[float, str], None] | None = None,
    *,
    enhancement: str = ENHANCEMENT_NONE,
    backend: str = BACKEND_WAV2LIP,
) -> dict:
    """Return paths and runtime metadata for one image- or video-driven job.

    Wav2Lip remains the fast/low-VRAM path. MuseTalk 1.5 is an explicit CUDA
    high-quality backend intended for Colab or GPUs with at least 4GB VRAM; it
    never silently falls back to CPU. Both backends accept a portrait image or
    a source video. The generated Edge TTS audio replaces any source-video audio.
    """
    def report(value: float, message: str) -> None:
        if progress_callback is not None:
            progress_callback(value, message)

    if enhancement not in ENHANCEMENT_OPTIONS:
        raise ValueError("不支援的畫質後處理模式。")
    if backend not in BACKEND_OPTIONS:
        raise ValueError("不支援的嘴型引擎。")

    report(0, "正在驗證人物素材與講稿。")
    media, script = validate_inputs(media_path, text)
    source_kind = media_kind(media)
    is_video = source_kind == "video"
    resolve_voice(voice)
    format_edge_rate(rate)
    info = get_device_info()
    requested = os.environ.get("TALKING_PHOTO_DEVICE", "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise PipelineError("TALKING_PHOTO_DEVICE 必須是 auto、cpu 或 cuda。")
    if info.error:
        raise PipelineError(info.error)
    device = info.device if requested == "auto" else requested

    if backend == BACKEND_MUSETALK:
        if requested == "cpu" or device == "cpu" or not str(device).startswith("cuda"):
            raise PipelineError("MuseTalk 1.5 高品質模式需要 CUDA GPU，不會自動改用 CPU。請使用 Google Colab GPU。")
        if info.vram_gb is None or info.vram_gb < MIN_MUSETALK_VRAM_GB:
            raise PipelineError(
                f"MuseTalk 1.5 高品質模式在本專案要求至少 {MIN_MUSETALK_VRAM_GB:g}GB 顯示記憶體；"
                "GTX 1050 2GB 請改用 Google Colab GPU。"
            )

    # CPU fallback and low-VRAM Wav2Lip use batch size 1 to bound memory usage.
    low_vram = info.low_vram or device == "cpu"
    job_id = uuid4().hex
    job = TEMP_DIR / job_id
    destination = OUTPUT_DIR / f"{job_id}.mp4"
    face_box: tuple[int, int, int, int] | None = None
    quality_mode = "標準模式"
    source_size: tuple[int, int] | None = None
    prepared_size: tuple[int, int] | None = None
    enhancement_mode = "未啟用"
    backend_mode = backend
    prepared: Path | None = None

    try:
        job.mkdir(parents=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        if is_video:
            source_size = probe_video_size(media)
            prepared_size = source_size
            prepared = job / f"input{media.suffix.lower()}"
            shutil.copy2(media, prepared)
            if backend == BACKEND_MUSETALK:
                quality_mode = "MuseTalk 1.5 影片高品質模式"
            elif low_vram:
                quality_mode = "影片低顯示記憶體模式"
            else:
                quality_mode = "影片模式"
        else:
            prepared = job / "input.png"
            with Image.open(media) as opened:
                portrait = ImageOps.exif_transpose(opened).convert("RGB")
                source_size = portrait.size
                if backend == BACKEND_MUSETALK:
                    # MuseTalk works on a 256x256 face crop; 1280px preserves a
                    # clear portrait/background while bounding preprocessing cost.
                    portrait.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                    quality_mode = "MuseTalk 1.5 高品質模式"
                elif low_vram:
                    portrait = limit_low_vram_output(portrait)
                    face_box = detect_face_box(portrait)
                    if face_box is not None:
                        quality_mode = "低顯示記憶體清晰模式"
                    else:
                        portrait.thumbnail((512, 512), Image.Resampling.LANCZOS)
                        quality_mode = "低顯示記憶體相容模式"
                prepared_size = portrait.size
                portrait.save(prepared, compress_level=2)

        if backend == BACKEND_MUSETALK:
            mode = f"CUDA／{quality_mode}"
        elif device == "cpu":
            mode = f"CPU 備援／{quality_mode}，可能需要較長時間"
        elif low_vram:
            mode = f"CUDA／{quality_mode}"
        else:
            mode = f"CUDA／{quality_mode}"

        report(0.1, f"正在產生台灣中文語音；推論將使用 {mode}。")
        audio = synthesize_speech(script, voice, rate, str(job / "speech.mp3"))
        report(0.3, "正在轉換為 16 kHz 單聲道 PCM WAV。")
        wav = normalize_audio(audio, str(job / "speech.wav"))

        if backend == BACKEND_MUSETALK:
            if is_video:
                report(0.45, "正在用 MuseTalk 1.5 保留原影片動作並重新產生嘴型；原影片聲音會由新語音取代。")
            else:
                report(0.45, "正在使用 MuseTalk 1.5 高品質模式產生自然嘴型；建議在 Colab GPU 執行。")
            raw = generate_musetalk_lip_sync(
                str(prepared),
                wav,
                str(job / "raw.mp4"),
                use_float16=True,
                batch_size=4,
                fps=25,
            )
        else:
            if is_video:
                report(0.45, f"正在用 Wav2Lip 保留原影片動作並產生嘴型同步（{mode}）。")
            elif face_box is not None:
                report(0.45, "已用低解析度預覽定位人臉；正在以較高解析度原圖產生並柔和融合嘴型。")
            else:
                report(0.45, f"正在使用 Wav2Lip 產生嘴型同步影片（{mode}）。")
            raw = generate_lip_sync(
                str(prepared),
                wav,
                str(job / "raw.mp4"),
                device,
                low_vram=low_vram,
                face_box=face_box,
                soft_blend=bool((not is_video) and low_vram and face_box is not None),
                static=not is_video,
            )

        report(0.82, "正在以高品質 H.264 封裝可播放的 MP4。")
        final = job / "final.mp4"
        finalize_video(Path(raw), final)

        if enhancement == ENHANCEMENT_GFPGAN:
            report(0.88, "正在用 GFPGAN V1.3 修復中心人臉；GTX 1050 會使用 1× 低顯存設定。")
            enhanced = job / "enhanced.mp4"
            enhance_video_with_gfpgan(str(final), str(enhanced), version="1.3", weight=0.4)
            os.replace(enhanced, final)
            enhancement_mode = "GFPGAN V1.3 高清修復"

        os.replace(final, destination)
        for intermediate in (prepared, job / "speech.wav", job / "raw.mp4"):
            if intermediate is not None:
                intermediate.unlink(missing_ok=True)
        report(1, f"影片已產生（{quality_mode}／{enhancement_mode}），可播放語音與影片。")
        return dict(
            job_id=job_id,
            audio_path=audio,
            video_path=str(destination),
            device=device,
            gpu_name=info.gpu_name,
            low_vram=low_vram,
            backend=backend_mode,
            quality_mode=quality_mode,
            enhancement_mode=enhancement_mode,
            source_type="影片" if is_video else "照片",
            source_size=source_size,
            prepared_size=prepared_size,
            face_box=face_box,
        )
    except BaseException:
        # Only remove paths created for this UUID job, never user inputs.
        shutil.rmtree(job, ignore_errors=True)
        destination.unlink(missing_ok=True)
        raise
