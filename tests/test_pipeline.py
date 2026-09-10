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
    monkeypatch.setattr(pipeline, "detect_face_box", lambda portrait: (20, 700, 120, 900))
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

    def infer(image, audio, output, device, low_vram, face_box=None, soft_blend=False):
        calls.append("infer")
        assert low_vram
        assert face_box == (20, 700, 120, 900)
        assert soft_blend
        with Image.open(image) as portrait:
            assert portrait.size == (1024, 768)
            assert portrait.mode == "RGB"
        assert Path(audio).read_bytes() == b"wav"
        Path(output).write_bytes(b"raw")
        return output

    def finalize(source, output):
        calls.append("finalize")
        assert source.read_bytes() == b"raw"
        output.write_bytes(b"final")

    for name, implementation in [
        ("synthesize_speech", tts),
        ("normalize_audio", normalize),
        ("generate_lip_sync", infer),
        ("finalize_video", finalize),
    ]:
        monkeypatch.setattr(pipeline, name, implementation)
    image = tmp_path / "人物.test.webp"
    Image.new("RGB", (1024, 768)).save(image)
    return image, calls


def test_pipeline_order_unique_jobs_and_cleanup(stages):
    image, calls = stages
    progress = []
    first = pipeline.generate_talking_video(
        str(image), " **大家好。** ", "台灣女聲", 1,
        lambda value, text: progress.append((value, text)),
    )
    second = pipeline.generate_talking_video(str(image), "大家好。", "台灣女聲", 1)
    assert calls == ["tts", "normalize", "infer", "finalize"] * 2
    assert first["job_id"] != second["job_id"]
    assert first["device"] == "cuda:0" and first["low_vram"]
    assert first["backend"] == pipeline.BACKEND_WAV2LIP
    assert first["quality_mode"] == "低顯示記憶體清晰模式"
    assert first["enhancement_mode"] == "未啟用"
    assert first["source_size"] == (1024, 768)
    assert first["prepared_size"] == (1024, 768)
    assert first["face_box"] == (20, 700, 120, 900)
    assert Path(first["video_path"]).read_bytes() == b"final"
    assert list(Path(first["audio_path"]).parent.iterdir()) == [Path(first["audio_path"])]
    assert image.is_file()
    assert [p[0] for p in progress] == sorted(p[0] for p in progress)
    assert progress[-1][0] == 1


def test_gfpgan_runs_only_when_selected(stages, monkeypatch):
    image, calls = stages

    def enhance(source, output, **kwargs):
        calls.append("gfpgan")
        assert Path(source).read_bytes() == b"final"
        assert kwargs == {"version": "1.3", "weight": 0.4}
        Path(output).write_bytes(b"enhanced")
        return output

    monkeypatch.setattr(pipeline, "enhance_video_with_gfpgan", enhance)
    result = pipeline.generate_talking_video(
        str(image), "大家好。", "台灣女聲", 1,
        enhancement=pipeline.ENHANCEMENT_GFPGAN,
    )
    assert calls == ["tts", "normalize", "infer", "finalize", "gfpgan"]
    assert Path(result["video_path"]).read_bytes() == b"enhanced"
    assert result["enhancement_mode"] == "GFPGAN V1.3 高清修復"


def test_invalid_enhancement_fails_before_work(stages):
    image, calls = stages
    with pytest.raises(ValueError, match="畫質後處理"):
        pipeline.generate_talking_video(
            str(image), "大家好。", "台灣女聲", 1, enhancement="unknown"
        )
    assert calls == []


def test_invalid_backend_fails_before_work(stages):
    image, calls = stages
    with pytest.raises(ValueError, match="嘴型引擎"):
        pipeline.generate_talking_video(
            str(image), "大家好。", "台灣女聲", 1, backend="unknown"
        )
    assert calls == []


def test_low_vram_caps_large_portrait_at_1280(tmp_path, monkeypatch):
    monkeypatch.delenv("TALKING_PHOTO_DEVICE", raising=False)
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "get_device_info", lambda: DeviceInfo(
        device="cuda:0", gpu_name="GTX 1050", vram_gb=2, low_vram=True, torch_version="test"))
    seen = {}

    def detect(portrait):
        seen["detect_size"] = portrait.size
        return (10, 600, 100, 900)

    def tts(text, voice, rate, output):
        Path(output).write_bytes(b"mp3")
        return output

    def normalize(source, output):
        Path(output).write_bytes(b"wav")
        return output

    def infer(image, audio, output, device, low_vram, face_box=None, soft_blend=False):
        with Image.open(image) as portrait:
            seen["prepared_size"] = portrait.size
        Path(output).write_bytes(b"raw")
        return output

    monkeypatch.setattr(pipeline, "detect_face_box", detect)
    monkeypatch.setattr(pipeline, "synthesize_speech", tts)
    monkeypatch.setattr(pipeline, "normalize_audio", normalize)
    monkeypatch.setattr(pipeline, "generate_lip_sync", infer)
    monkeypatch.setattr(pipeline, "finalize_video", lambda source, output: output.write_bytes(b"final"))
    image = tmp_path / "large.png"
    Image.new("RGB", (4000, 3000)).save(image)
    result = pipeline.generate_talking_video(str(image), "大家好。", "台灣女聲", 1)
    assert seen["detect_size"] == (1280, 960)
    assert seen["prepared_size"] == (1280, 960)
    assert result["prepared_size"] == (1280, 960)


def test_low_vram_falls_back_to_512_when_preview_face_detection_misses(stages, monkeypatch):
    image, _ = stages
    monkeypatch.setattr(pipeline, "detect_face_box", lambda portrait: None)
    observed = {}

    def infer(image, audio, output, device, low_vram, face_box=None, soft_blend=False):
        with Image.open(image) as portrait:
            observed["size"] = portrait.size
        observed["face_box"] = face_box
        observed["soft_blend"] = soft_blend
        Path(output).write_bytes(b"raw")
        return output

    monkeypatch.setattr(pipeline, "generate_lip_sync", infer)
    result = pipeline.generate_talking_video(str(image), "大家好。", "台灣女聲", 1)
    assert max(observed["size"]) <= 512
    assert observed["face_box"] is None
    assert not observed["soft_blend"]
    assert result["quality_mode"] == "低顯示記憶體相容模式"


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


def test_musetalk_rejects_gtx1050_before_tts_or_inference(stages, monkeypatch):
    image, calls = stages
    musetalk_calls = []
    monkeypatch.setattr(
        pipeline,
        "generate_musetalk_lip_sync",
        lambda *args, **kwargs: musetalk_calls.append(1),
    )
    with pytest.raises(pipeline.PipelineError, match="至少 4GB"):
        pipeline.generate_talking_video(
            str(image), "大家好。", "台灣女聲", 1,
            backend=pipeline.BACKEND_MUSETALK,
        )
    assert calls == []
    assert musetalk_calls == []


def test_musetalk_uses_only_high_quality_backend_on_large_cuda_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("TALKING_PHOTO_DEVICE", raising=False)
    monkeypatch.setattr(pipeline, "TEMP_DIR", tmp_path / "temp")
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(pipeline, "get_device_info", lambda: DeviceInfo(
        device="cuda:0", gpu_name="Tesla T4", vram_gb=15, low_vram=False, torch_version="test"))
    calls = []

    def tts(text, voice, rate, output):
        calls.append("tts")
        Path(output).write_bytes(b"mp3")
        return output

    def normalize(source, output):
        calls.append("normalize")
        Path(output).write_bytes(b"wav")
        return output

    def musetalk_infer(image, audio, output, **kwargs):
        calls.append("musetalk")
        assert kwargs == {"use_float16": True, "batch_size": 4, "fps": 25}
        with Image.open(image) as portrait:
            assert portrait.size == (1280, 960)
        Path(output).write_bytes(b"raw-musetalk")
        return output

    def wav2lip_fail(*args, **kwargs):
        raise AssertionError("MuseTalk 模式不可呼叫 Wav2Lip")

    monkeypatch.setattr(pipeline, "synthesize_speech", tts)
    monkeypatch.setattr(pipeline, "normalize_audio", normalize)
    monkeypatch.setattr(pipeline, "generate_musetalk_lip_sync", musetalk_infer)
    monkeypatch.setattr(pipeline, "generate_lip_sync", wav2lip_fail)
    monkeypatch.setattr(
        pipeline, "finalize_video",
        lambda source, output: (calls.append("finalize"), output.write_bytes(b"final")),
    )
    image = tmp_path / "large.png"
    Image.new("RGB", (4000, 3000)).save(image)

    result = pipeline.generate_talking_video(
        str(image), "大家好。", "台灣女聲", 1,
        backend=pipeline.BACKEND_MUSETALK,
    )
    assert calls == ["tts", "normalize", "musetalk", "finalize"]
    assert result["backend"] == pipeline.BACKEND_MUSETALK
    assert result["quality_mode"] == "MuseTalk 1.5 高品質模式"
    assert result["prepared_size"] == (1280, 960)
    assert result["face_box"] is None
    assert Path(result["video_path"]).read_bytes() == b"final"


def test_finalize_requires_audio_browser_codecs_and_high_quality_h264(tmp_path, monkeypatch):
    def run(command, **kwargs):
        assert "0:a:0" in command and "0:v:0" in command
        assert "libx264" in command and "aac" in command and "yuv420p" in command
        assert command[command.index("-crf") + 1] == "18"
        assert command[command.index("-preset") + 1] == "medium"
        assert kwargs["check"]
        Path(command[-1]).write_bytes(b"mp4")
        return SimpleNamespace()

    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.finalize_video(tmp_path / "raw.mp4", tmp_path / "final.mp4")
