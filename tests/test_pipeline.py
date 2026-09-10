from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from talking_photo import pipeline
from talking_photo.device import DeviceInfo
from talking_photo.wav2lip import Wav2LipError


@pytest.fixture
def stages(tmp_path, monkeypatch):
    monkeypatch.delenv("TALKING_PHOTO_DEVICE", raising=False)
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "get_device_info", lambda: DeviceInfo(
        device="cuda:0", gpu_name="GTX 1050", vram_gb=2, low_vram=True, torch_version="test"))
    calls = []
    def tts(text, voice, rate, output):
        calls.append("tts")
        assert text == "大家好。"
        Path(output).write_bytes(b"mp3")
        return output
    def normalize(source, output):
        calls.append("normalize")
        assert Path(source).read_bytes() == b"mp3"
        Path(output).write_bytes(b"wav")
        return output
    def infer(image, audio, output, device, low_vram):
        calls.append("infer")
        assert low_vram
        with Image.open(image) as portrait:
            assert max(portrait.size) <= 512
            assert portrait.mode == "RGB"
        assert Path(audio).read_bytes() == b"wav"
        Path(output).write_bytes(b"raw")
        return output
    def finalize(source, output):
        calls.append("finalize")
        assert source.read_bytes() == b"raw"
        output.write_bytes(b"final")
    for name, implementation in [("synthesize_speech", tts), ("normalize_audio", normalize),
                                  ("generate_lip_sync", infer), ("finalize_video", finalize)]:
        monkeypatch.setattr(pipeline, name, implementation)
    image = tmp_path / "人物.test.webp"
    Image.new("RGB", (1024, 768)).save(image)
    return image, calls


def test_pipeline_order_unique_jobs_and_cleanup(stages):
    image, calls = stages
    progress = []
    first = pipeline.generate_talking_video(str(image), " **大家好。** ", "台灣女聲", 1,
                                           lambda value, text: progress.append((value, text)))
    second = pipeline.generate_talking_video(str(image), "大家好。", "台灣女聲", 1)
    assert calls == ["tts", "normalize", "infer", "finalize"] * 2
    assert first["job_id"] != second["job_id"]
    assert first["device"] == "cuda:0" and first["low_vram"]
    assert Path(first["video_path"]).read_bytes() == b"final"
    assert list(Path(first["audio_path"]).parent.iterdir()) == [Path(first["audio_path"])]
    assert image.is_file()
    assert [p[0] for p in progress] == sorted(p[0] for p in progress)
    assert progress[-1][0] == 1


@pytest.mark.parametrize("stage", ["synthesize_speech", "normalize_audio", "generate_lip_sync", "finalize_video"])
def test_failure_cleans_job_and_never_retries(stages, monkeypatch, stage):
    image, calls = stages
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(1)
        raise Wav2LipError("CUDA 顯示記憶體不足")
    monkeypatch.setattr(pipeline, stage, fail)
    with pytest.raises(Wav2LipError, match="顯示記憶體不足"):
        pipeline.generate_talking_video(str(image), "大家好。", "台灣女聲", 1)
    assert attempts == [1]
    assert list(pipeline.TEMP_DIR.iterdir()) == []
    assert list(pipeline.OUTPUT_DIR.iterdir()) == []
    assert image.is_file()


def test_explicit_cpu(stages, monkeypatch):
    monkeypatch.setenv("TALKING_PHOTO_DEVICE", "cpu")
    result = pipeline.generate_talking_video(str(stages[0]), "大家好。", "台灣女聲", 1)
    assert result["device"] == "cpu"
    assert result["low_vram"]


def test_validation_precedes_work(stages):
    with pytest.raises(ValueError):
        pipeline.generate_talking_video(str(stages[0]), "", "台灣女聲", 1)
    assert not pipeline.TEMP_DIR.exists()
    assert stages[1] == []


def test_device_errors_do_not_fallback(stages, monkeypatch):
    monkeypatch.setattr(pipeline, "get_device_info", lambda: DeviceInfo(error="CUDA 偵測失敗"))
    with pytest.raises(pipeline.PipelineError, match="CUDA 偵測失敗"):
        pipeline.generate_talking_video(str(stages[0]), "大家好。", "台灣女聲", 1)
    assert stages[1] == []


def test_finalize_requires_audio_and_browser_codecs(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert "0:a:0" in command and "0:v:0" in command
        assert "libx264" in command and "aac" in command and "yuv420p" in command
        assert kwargs["check"]
        Path(command[-1]).write_bytes(b"mp4")
        return SimpleNamespace()
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.finalize_video(tmp_path / "raw.mp4", tmp_path / "final.mp4")
