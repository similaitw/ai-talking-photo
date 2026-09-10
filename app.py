"""Gradio application for AI Talking Photo."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import gradio as gr

from talking_photo.config import APP_NAME, DEFAULT_RATE, DEFAULT_VOICE
from talking_photo.tts import TTSGenerationError, synthesize_speech
from talking_photo.validation import ValidationError, validate_script
from talking_photo.pipeline import generate_talking_video, PipelineError
from talking_photo.media import AudioNormalizationError
from talking_photo.wav2lip import Wav2LipError

VIDEO_READY = "影片已產生，可播放或下載語音與影片。"
AUDIO_PREVIEW_READY = "語音預覽已產生，可先播放確認聲音與語速。"
AUDIO_PREVIEW_DIR = Path("temp") / "audio-preview"


def generate_audio_preview(
    script: str,
    voice: str,
    rate: float,
) -> tuple[str | None, str]:
    """Generate a Taiwan Mandarin TTS preview without requiring a portrait."""

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
    image: str | None,
    script: str,
    voice: str,
    rate: float,
    progress=gr.Progress(),
) -> tuple[str | None, str | None, str]:
    """Run the real pipeline and update both media previews."""

    try:
        result = generate_talking_video(
            image, script, voice, float(rate),
            progress_callback=lambda value, message: progress(value, desc=message),
        )
    except ValueError as exc:
        return None, None, f"輸入有誤：{exc}"
    except (PipelineError, TTSGenerationError, AudioNormalizationError, Wav2LipError, OSError) as exc:
        return None, None, f"影片產生失敗：{exc}"
    mode = "CPU 備援" if result["device"] == "cpu" else "CUDA"
    return result["audio_path"], result["video_path"], f"{VIDEO_READY}（{mode}）"


def build_app() -> gr.Blocks:
    """Build the single-page Gradio UI."""

    with gr.Blocks(title=APP_NAME) as demo:
        gr.Markdown(
            "# AI Talking Photo\n"
            "單張照片 + 中文講稿 → AI 說話影片\n\n"
            "建議使用正面、清楚、嘴部未遮擋的人像照片。"
        )

        with gr.Row():
            with gr.Column():
                portrait = gr.Image(type="filepath", label="人物照片")
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
            inputs=[portrait, script, voice, rate],
            outputs=[audio, video, status],
            concurrency_limit=1,
        )

    return demo.queue()


def main() -> None:
    """Launch the local Gradio app."""

    build_app().launch()


if __name__ == "__main__":
    main()
