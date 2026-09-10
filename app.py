"""Gradio application for AI Talking Photo."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import gradio as gr

from talking_photo.config import APP_NAME, DEFAULT_RATE, DEFAULT_VOICE
from talking_photo.tts import TTSGenerationError, synthesize_speech
from talking_photo.validation import ValidationError, validate_script
from talking_photo.pipeline import (
    BACKEND_MUSETALK,
    BACKEND_OPTIONS,
    BACKEND_WAV2LIP,
    ENHANCEMENT_NONE,
    ENHANCEMENT_OPTIONS,
    generate_talking_video,
    PipelineError,
)
from talking_photo.gfpgan import GFPGANError
from talking_photo.media import AudioNormalizationError
from talking_photo.musetalk import MuseTalkError
from talking_photo.wav2lip import Wav2LipError

VIDEO_READY = "影片已產生，可播放或下載語音與影片。"
AUDIO_PREVIEW_READY = "語音預覽已產生，可先播放確認聲音與語速。"
AUDIO_PREVIEW_DIR = Path("temp") / "audio-preview"


def _default_backend() -> str:
    return BACKEND_MUSETALK if os.environ.get("TALKING_PHOTO_DEFAULT_BACKEND", "").lower() == "musetalk" else BACKEND_WAV2LIP


def generate_audio_preview(
    script: str,
    voice: str,
    rate: float,
) -> tuple[str | None, str]:
    """Generate a Taiwan Mandarin TTS preview without requiring portrait media."""

    try:
        normalized_script = validate_script(script)
        output_path = AUDIO_PREVIEW_DIR / f"{uuid4().hex}.mp3"
        audio_path = synthesize_speech(
            normalized_script,
            voice,
            float(rate),
            str(output_path),
        )
    except ValidationError as exc:
        return None, f"輸入有誤：{exc}"
    except ValueError as exc:
        return None, f"輸入有誤：{exc}"
    except TTSGenerationError as exc:
        return None, f"語音產生失敗：{exc}"

    return audio_path, AUDIO_PREVIEW_READY


def generate_video(
    media: str | None,
    script: str,
    voice: str,
    rate: float,
    enhancement: str = ENHANCEMENT_NONE,
    backend: str = BACKEND_WAV2LIP,
    progress=gr.Progress(),
) -> tuple[str | None, str | None, str]:
    """Run the real pipeline and update both media previews."""

    try:
        result = generate_talking_video(
            media,
            script,
            voice,
            float(rate),
            progress_callback=lambda value, message: progress(value, desc=message),
            enhancement=enhancement,
            backend=backend,
        )
    except ValueError as exc:
        return None, None, f"輸入有誤：{exc}"
    except (
        PipelineError,
        TTSGenerationError,
        AudioNormalizationError,
        Wav2LipError,
        MuseTalkError,
        GFPGANError,
        OSError,
    ) as exc:
        return None, None, f"影片產生失敗：{exc}"
    device_mode = "CPU 備援" if result["device"] == "cpu" else "CUDA"
    backend_mode = result.get("backend", BACKEND_WAV2LIP)
    quality_mode = result.get("quality_mode", "標準模式")
    enhancement_mode = result.get("enhancement_mode", "未啟用")
    source_type = result.get("source_type", "照片")
    return (
        result["audio_path"],
        result["video_path"],
        f"{VIDEO_READY}（{source_type}／{device_mode}／{backend_mode}／{quality_mode}／{enhancement_mode}）",
    )


def build_app() -> gr.Blocks:
    """Build the single-page Gradio UI."""

    with gr.Blocks(title=APP_NAME) as demo:
        gr.Markdown(
            "# AI Talking Photo\n"
            "人物照片或影片 + 中文講稿 → AI 說話影片\n\n"
            "建議使用正面、清楚、嘴部未遮擋的人像。影片若有自然眨眼、微點頭或呼吸感，成品通常比單張照片更自然。"
        )

        with gr.Row():
            with gr.Column():
                portrait_media = gr.File(
                    type="filepath",
                    file_types=["image", "video"],
                    label="人物素材（照片或影片）",
                )
                script = gr.Textbox(
                    label="講稿",
                    lines=12,
                    placeholder="請貼上要讓人物說出的文字……",
                )
                voice = gr.Dropdown(
                    choices=["台灣女聲", "台灣男聲"],
                    value=DEFAULT_VOICE,
                    label="聲音",
                )
                rate = gr.Slider(
                    minimum=0.8,
                    maximum=1.2,
                    value=DEFAULT_RATE,
                    step=0.05,
                    label="語速",
                )
                backend = gr.Dropdown(
                    choices=list(BACKEND_OPTIONS),
                    value=_default_backend(),
                    label="嘴型引擎",
                    info="影片輸入可保留原本眨眼與頭部動作。GTX 1050 2GB 請用 Wav2Lip；MuseTalk 1.5 建議 Google Colab GPU。",
                )
                enhancement = gr.Dropdown(
                    choices=list(ENHANCEMENT_OPTIONS),
                    value=ENHANCEMENT_NONE,
                    label="畫質後處理",
                    info="GFPGAN 可提高臉部清晰度，但不會重新計算嘴型同步。",
                )
                gr.Markdown(
                    "**影片輸入**：原影片聲音會由 Edge TTS 取代；短影片可由嘴型引擎循環幀以配合較長講稿。  \n"
                    "**Wav2Lip**：速度快、GTX 1050 2GB 可用。  \n"
                    "**MuseTalk 1.5**：嘴型自然度優先，至少需 4GB 顯示記憶體，本專案建議 Colab。  \n"
                    "**GFPGAN 高清修復（實驗）**：只做臉部清晰度後處理，不取代嘴型引擎。"
                )
                preview_audio = gr.Button("產生語音預覽")
                generate = gr.Button("產生影片", variant="primary")

            with gr.Column():
                status = gr.Markdown("尚未開始產生。")
                audio = gr.Audio(label="語音預覽", type="filepath")
                video = gr.Video(label="影片預覽")

        gr.Markdown(
            "請只使用你有權使用的人像與聲音。"
            "產生的內容應避免冒充、誤導或未經本人同意的公開發布。"
        )

        preview_audio.click(
            fn=generate_audio_preview,
            inputs=[script, voice, rate],
            outputs=[audio, status],
        )

        generate.click(
            fn=generate_video,
            # Preserve the existing enhancement positional argument and add the
            # backend after it; visual component order remains independent.
            inputs=[portrait_media, script, voice, rate, enhancement, backend],
            outputs=[audio, video, status],
            concurrency_limit=1,
        )

    return demo.queue()


def main() -> None:
    """Launch the local Gradio app."""

    build_app().launch()


if __name__ == "__main__":
    main()
