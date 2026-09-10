from pathlib import Path
from types import SimpleNamespace

import pytest

from talking_photo import musetalk


def _fake_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "MuseTalk root"
    root.mkdir()
    image = tmp_path / "人物 照片.png"
    audio = tmp_path / "語音 檔.wav"
    output = tmp_path / "輸出 影片.mp4"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    return root, image, audio, output


def test_runtime_requires_cuda_and_never_falls_back_to_cpu(tmp_path, monkeypatch):
    root = tmp_path / "MuseTalk"
    root.mkdir()
    monkeypatch.setattr(
        musetalk.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b'{"cuda": false, "name": null, "vram_gb": 0.0}\n',
            stderr=b"",
        ),
    )
    with pytest.raises(musetalk.MuseTalkError, match="不會自動改用 CPU"):
        musetalk.get_musetalk_runtime_info(
            python_executable="python", musetalk_dir=str(root)
        )


def test_runtime_rejects_gtx1050_2gb(tmp_path, monkeypatch):
    root = tmp_path / "MuseTalk"
    root.mkdir()
    monkeypatch.setattr(
        musetalk.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b'{"cuda": true, "name": "GTX 1050", "vram_gb": 2.0}\n',
            stderr=b"",
        ),
    )
    with pytest.raises(musetalk.MuseTalkError, match="至少 4GB"):
        musetalk.get_musetalk_runtime_info(
            python_executable="python", musetalk_dir=str(root)
        )


def test_runtime_accepts_colab_class_gpu(tmp_path, monkeypatch):
    root = tmp_path / "MuseTalk"
    root.mkdir()
    monkeypatch.setattr(
        musetalk.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b'warning first\n{"cuda": true, "name": "Tesla T4", "vram_gb": 15.0}\n',
            stderr=b"",
        ),
    )
    info = musetalk.get_musetalk_runtime_info(
        python_executable="python", musetalk_dir=str(root)
    )
    assert info["cuda"] is True
    assert info["name"] == "Tesla T4"
    assert info["vram_gb"] == 15.0


def test_task_config_uses_yaml_safe_json_strings(tmp_path):
    config = tmp_path / "task.yaml"
    musetalk._write_task_config(
        config,
        "folder/人物 照片.png",
        "folder/語音 檔.wav",
    )
    text = config.read_text(encoding="utf-8")
    assert 'video_path: "folder/人物 照片.png"' in text
    assert 'audio_path: "folder/語音 檔.wav"' in text


def test_inference_uses_v15_fp16_relative_workspace_and_atomic_output(
    tmp_path, monkeypatch
):
    root, image, audio, output = _fake_inputs(tmp_path)
    monkeypatch.setattr(musetalk, "_validate_install", lambda root: None)
    monkeypatch.setattr(
        musetalk,
        "get_musetalk_runtime_info",
        lambda **kwargs: {"cuda": True, "name": "T4", "vram_gb": 15.0},
    )
    monkeypatch.setattr(musetalk.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    seen = {}

    def fake_run(command, *, cwd, check=True):
        seen["command"] = command
        seen["cwd"] = cwd
        assert check is False
        result_dir = Path(cwd) / command[command.index("--result_dir") + 1]
        expected = result_dir / "v15" / "result.mp4"
        expected.parent.mkdir(parents=True)
        expected.write_bytes(b"musetalk-mp4")
        return SimpleNamespace(returncode=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(musetalk, "_run", fake_run)
    result = musetalk.generate_musetalk_lip_sync(
        str(image),
        str(audio),
        str(output),
        musetalk_dir=str(root),
        python_executable="python",
    )

    command = seen["command"]
    assert seen["cwd"] == root
    assert command[:3] == ["python", "-m", "scripts.inference"]
    assert command[command.index("--version") + 1] == "v15"
    assert command[command.index("--batch_size") + 1] == "4"
    assert command[command.index("--fps") + 1] == "25"
    assert "--use_float16" in command
    assert command[command.index("--unet_model_path") + 1] == "models/musetalkV15/unet.pth"
    config_arg = command[command.index("--inference_config") + 1]
    result_arg = command[command.index("--result_dir") + 1]
    assert not Path(config_arg).is_absolute()
    assert not Path(result_arg).is_absolute()
    assert result == str(output.resolve())
    assert output.read_bytes() == b"musetalk-mp4"


def test_upstream_zero_exit_without_mp4_is_still_failure(tmp_path, monkeypatch):
    root, image, audio, output = _fake_inputs(tmp_path)
    monkeypatch.setattr(musetalk, "_validate_install", lambda root: None)
    monkeypatch.setattr(
        musetalk,
        "get_musetalk_runtime_info",
        lambda **kwargs: {"cuda": True, "name": "T4", "vram_gb": 15.0},
    )
    monkeypatch.setattr(musetalk.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        musetalk,
        "_run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"Error occurred during processing: test failure",
            stderr=b"",
        ),
    )

    with pytest.raises(musetalk.MuseTalkError, match="沒有產生有效影片"):
        musetalk.generate_musetalk_lip_sync(
            str(image),
            str(audio),
            str(output),
            musetalk_dir=str(root),
            python_executable="python",
        )
    assert not output.exists()


def test_oom_detected_even_when_upstream_catches_exception(tmp_path, monkeypatch):
    root, image, audio, output = _fake_inputs(tmp_path)
    monkeypatch.setattr(musetalk, "_validate_install", lambda root: None)
    monkeypatch.setattr(
        musetalk,
        "get_musetalk_runtime_info",
        lambda **kwargs: {"cuda": True, "name": "T4", "vram_gb": 15.0},
    )
    monkeypatch.setattr(musetalk.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        musetalk,
        "_run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=b"CUDA out of memory",
            stderr=b"",
        ),
    )
    with pytest.raises(musetalk.MuseTalkError, match="顯示記憶體不足"):
        musetalk.generate_musetalk_lip_sync(
            str(image), str(audio), str(output),
            musetalk_dir=str(root), python_executable="python",
        )
