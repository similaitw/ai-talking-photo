"""Gradio application shell for AI Talking Photo."""

from __future__ import annotations

import gradio as gr

APP_TITLE = "AI Talking Photo"
PIPELINE_NOT_READY = "Pipeline 尚未啟用"


def pipeline_not_ready(
    _image: str | None,
    _script: str,
    _voice: str,
    _rate: float,
) -> tuple[None, None, str]:
    """Return an explicit placeholder state until the real pipeline exists."""

    return None, None, PIPELINE_NOT_READY


def build_app() -> gr.Blocks:
    """Build the single-page Gradio UI defined by milestone M1.1."""

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(
            "# AI Talking Photo\n"
            "單張照片 + 中文講稿 → AI 說話影片\n\n"
            "建議使用正面、清楚、嘴部未遮擋的人像照片。"
        )

        with gr.Row():
            with gr.Column():
                portrait = gr.Image(
                    type="filepath",
                    label="人物照片",
                )
                script = gr.Textbox(
                    label="講稿",
                    lines=12,
                    placeholder="請貼上要讓人物說出的文字……",
                )
                voice = gr.Dropdown(
                    choices=["台灣女聲", "台灣男聲"],
                    value="台灣女聲",
                    label="聲音",
                )
                rate = gr.Slider(
                    minimum=0.8,
                    maximum=1.2,
                    value=0.95,
                    step=0.05,
                    label="語速",
                )
                generate = gr.Button("產生影片", variant="primary")

            with gr.Column():
                status = gr.Markdown("尚未開始產生。")
                audio = gr.Audio(label="語音預覽", type="filepath")
                video = gr.Video(label="影片預覽")

        gr.Markdown(
            "請只使用你有權使用的人像與聲音。"
            "產生的內容應避免冒充、誤導或未經本人同意的公開發布。"
        )

        generate.click(
            fn=pipeline_not_ready,
            inputs=[portrait, script, voice, rate],
            outputs=[audio, video, status],
        )

    return demo


def main() -> None:
    """Launch the local Gradio app."""

    build_app().launch()


if __name__ == "__main__":
    main()
