from pathlib import Path
from types import SimpleNamespace

import pytest

from talking_photo import musetalk, pipeline, wav2lip
from talking_photo.device import DeviceInfo


def _common_pipeline_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, low_vram: bool):
    monkeypatch.delenv("TALKING_PHOTO_DEVICE", raising=False)
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(
        pipeline,
        "get_device_info",
        lambda: DeviceInfo(
            device="cuda:0",
            gpu_name="GTX 1050" if low_vram else "Tesla T4",
            vram_gb=2 if low_vram else 15,
            low_vram=low_vram,
            torch_version="test",
        ),
    )

    def tts(text, voice, rate, output):
        Path(output).write_bytes(b"mp3")
        return output

    def normalize(source, output):
        Path(output).write_bytes(b"wav")
        return output

    monkeypatch.setattr(pipeline, "synthesize_speech", tts)
    monkeypatch.setattr(pipeline, "normalize_audio", normalize)
    monkeypatch.setattr(
        pipeline,
        "finalize_video",
        lambda source, output: output.write_bytes(b"final"),
    )


def test_wav2lip_video_command_does_not_force_static(tmp_path: Path) -> None:
    command = wav2lip._build_command(
        tmp_path / "inference.py",
        tmp_path / "model.pth",
        tmp_path / "portrait.mp4",
        tmp_path / "speech.wav",
        tmp_path / "out.mp4",
        low_vram=True,
        static=False,
    )
    assert "--static" not in command
    assert command[command.index("--face") + 1].endswith("portrait.mp4")
    assert "--wav2lip_batch_size" in command


def test_wav2lip_video_pipeline_preserves_motion_source_and_disables_static(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _common_pipeline_stages(tmp_path, monkeypatch, low_vram=True)
    source = tmp_path / "人物影片.mp4"
    source.write_bytes(b"video-source")
    monkeypatch.setattr(pipeline, "validate_inputs", lambda media, text: (source.resolve(), "大家好。"))
    monkeypatch.setattr(pipeline, "media_kind", lambda media: "video")
    monkeypatch.setattr(pipeline, "probe_video_size", lambda media: (1280, 720))
    seen = {}

    def infer(media, audio, output, device, **kwargs):
        seen["source"] = Path(media)
        seen["kwargs"] = kwargs
        assert Path(media).suffix == ".mp4"
        assert Path(media).read_bytes() == b"video-source"
        assert Path(audio).read_bytes() == b"wav"
        Path(output).write_bytes(b"raw")
        return output

    monkeypatch.setattr(pipeline, "generate_lip_sync", infer)
    monkeypatch.setattr(
        pipeline,
        "generate_musetalk_lip_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不可呼叫 MuseTalk")),
    )

    result = pipeline.generate_talking_video(str(source), "大家好。", "台灣女聲", 1)

    assert seen["kwargs"]["static"] is False
    assert seen["kwargs"]["low_vram"] is True
    assert "face_box" not in seen["kwargs"]
    assert "soft_blend" not in seen["kwargs"]
    assert result["source_type"] == "影片"
    assert result["source_size"] == (1280, 720)
    assert result["prepared_size"] == (1280, 720)
    assert result["face_box"] is None
    assert result["quality_mode"] == "影片低顯示記憶體模式"
    assert source.read_bytes() == b"video-source"
    assert Path(result["video_path"]).read_bytes() == b"final"


def test_musetalk_video_pipeline_routes_video_without_wav2lip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _common_pipeline_stages(tmp_path, monkeypatch, low_vram=False)
    source = tmp_path / "portrait.mov"
    source.write_bytes(b"mov-source")
    monkeypatch.setattr(pipeline, "validate_inputs", lambda media, text: (source.resolve(), "大家好。"))
    monkeypatch.setattr(pipeline, "media_kind", lambda media: "video")
    monkeypatch.setattr(pipeline, "probe_video_size", lambda media: (1080, 1920))
    seen = {}

    def muse(media, audio, output, **kwargs):
        seen["media"] = Path(media)
        seen["kwargs"] = kwargs
        assert Path(media).suffix == ".mov"
        assert Path(media).read_bytes() == b"mov-source"
        Path(output).write_bytes(b"muse-raw")
        return output

    monkeypatch.setattr(pipeline, "generate_musetalk_lip_sync", muse)
    monkeypatch.setattr(
        pipeline,
        "generate_lip_sync",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("MuseTalk 模式不可呼叫 Wav2Lip")),
    )

    result = pipeline.generate_talking_video(
        str(source),
        "大家好。",
        "台灣女聲",
        1,
        backend=pipeline.BACKEND_MUSETALK,
    )

    assert seen["kwargs"] == {"use_float16": True, "batch_size": 4, "fps": 25}
    assert result["source_type"] == "影片"
    assert result["quality_mode"] == "MuseTalk 1.5 影片高品質模式"
    assert result["source_size"] == (1080, 1920)
    assert result["prepared_size"] == (1080, 1920)
    assert Path(result["video_path"]).read_bytes() == b"final"


def test_musetalk_wrapper_preserves_video_suffix_in_upstream_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "MuseTalk"
    root.mkdir()
    source = tmp_path / "portrait.webm"
    source.write_bytes(b"video")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio")
    output = tmp_path / "result.mp4"

    monkeypatch.setattr(musetalk, "_validate_install", lambda root: None)
    monkeypatch.setattr(
        musetalk,
        "get_musetalk_runtime_info",
        lambda **kwargs: {"cuda": True, "name": "T4", "vram_gb": 15.0},
    )
    monkeypatch.setattr(musetalk.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    seen = {}

    def fake_run(command, *, cwd, check=True):
        config_arg = command[command.index("--inference_config") + 1]
        config = Path(cwd) / config_arg
        text = config.read_text(encoding="utf-8")
        seen["config"] = text
        result_dir = Path(cwd) / command[command.index("--result_dir") + 1]
        expected = result_dir / "v15" / "result.mp4"
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"result")
        return SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(musetalk, "_run", fake_run)
    musetalk.generate_musetalk_lip_sync(
        str(source),
        str(audio),
        str(output),
        musetalk_dir=str(root),
        python_executable="python",
    )

    assert "input.webm" in seen["config"]
    assert "input.png" not in seen["config"]
    assert output.read_bytes() == b"result"
