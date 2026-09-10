from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app import (
    AUDIO_PREVIEW_READY,
    VIDEO_READY,
    build_app,
    generate_audio_preview,
    generate_video,
)
from talking_photo.tts import TTSGenerationError


def test_video_rejects_missing_image() -> None:
    audio, video, status = generate_video(None, "測試", "台灣女聲", 0.95)
    assert audio is None
    assert video is None
    assert "請上傳人物照片" in status


def test_video_returns_both_previews_after_success(tmp_path: Path) -> None:
    portrait = tmp_path / "portrait.png"
    Image.new("RGB", (512, 512), "white").save(portrait)
    result = dict(
        audio_path="speech.mp3",
        video_path="real.mp4",
        device="cuda:0",
        quality_mode="低顯示記憶體清晰模式",
    )
    with patch("app.generate_talking_video", return_value=result) as pipeline:
        audio, video, status = generate_video(str(portrait), "測試", "台灣女聲", 0.95)
    assert audio == "speech.mp3"
    assert video == "real.mp4"
    assert VIDEO_READY in status
    assert "CUDA" in status
    assert "低顯示記憶體清晰模式" in status
    assert pipeline.call_args.args == (str(portrait), "測試", "台灣女聲", 0.95)


def test_video_reports_oom_and_clears_stale_previews() -> None:
    from talking_photo.wav2lip import Wav2LipError
    with patch("app.generate_talking_video", side_effect=Wav2LipError("GTX 1050 2GB 顯示記憶體不足，請使用 Google Colab GPU 或降低圖片解析度。")):
        audio, video, status = generate_video("portrait.png", "測試", "台灣女聲", 1)
    assert audio is None and video is None
    assert "GTX 1050 2GB" in status and "Colab" in status


def test_audio_preview_does_not_require_portrait(tmp_path: Path) -> None:
    expected = tmp_path / "speech.mp3"
    expected.write_bytes(b"fake-mp3")

    with patch("app.synthesize_speech", return_value=str(expected)) as synthesize:
        audio, status = generate_audio_preview("  大家好。  ", "台灣女聲", 0.95)

    assert audio == str(expected)
    assert status == AUDIO_PREVIEW_READY
    args = synthesize.call_args.args
    assert args[0] == "大家好。"
    assert args[1] == "台灣女聲"
    assert args[2] == 0.95
    assert args[3].endswith(".mp3")
    assert "audio-preview" in args[3]


def test_audio_preview_rejects_blank_script() -> None:
    audio, status = generate_audio_preview("   ", "台灣女聲", 0.95)
    assert audio is None
    assert "請輸入講稿" in status


def test_audio_preview_reports_tts_failure() -> None:
    with patch(
        "app.synthesize_speech",
        side_effect=TTSGenerationError("網路連線異常"),
    ):
        audio, status = generate_audio_preview("測試", "台灣女聲", 0.95)

    assert audio is None
    assert "語音產生失敗" in status
    assert "網路連線異常" in status


def test_gradio_shell_contains_required_components() -> None:
    demo = build_app()
    config = demo.get_config_file()
    components = config["components"]

    component_types = {component["type"] for component in components}
    labels = {
        component.get("props", {}).get("label")
        for component in components
        if component.get("props", {}).get("label")
    }
    button_values = {
        component.get("props", {}).get("value")
        for component in components
        if component["type"] == "button"
    }

    assert {"image", "textbox", "dropdown", "slider", "audio", "video"} <= component_types
    assert {"人物照片", "講稿", "聲音", "語速", "語音預覽", "影片預覽"} <= labels
    assert {"產生語音預覽", "產生影片"} <= button_values
    video_event = next(fn for fn in demo.fns.values() if fn.fn == generate_video)
    assert [component.get_block_name() for component in video_event.outputs] == ["audio", "video", "markdown"]
