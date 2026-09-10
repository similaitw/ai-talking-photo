from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest

from talking_photo import gfpgan


def test_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="找不到"):
        gfpgan.enhance_video_with_gfpgan(
            str(tmp_path / "missing.mp4"), str(tmp_path / "out.mp4")
        )


def test_rejects_missing_gfpgan_checkout(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    with pytest.raises(gfpgan.GFPGANError, match="setup_gfpgan"):
        gfpgan.enhance_video_with_gfpgan(
            str(source), str(tmp_path / "out.mp4"), gfpgan_dir=str(tmp_path / "none")
        )


def test_rejects_invalid_options(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    with pytest.raises(ValueError, match="版本"):
        gfpgan.enhance_video_with_gfpgan(str(source), str(tmp_path / "out.mp4"), version="2")
    with pytest.raises(ValueError, match="強度"):
        gfpgan.enhance_video_with_gfpgan(str(source), str(tmp_path / "out.mp4"), weight=1.2)


def test_gfpgan_pipeline_uses_center_face_scale_one_and_preserves_audio(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"video")
    root = tmp_path / "GFPGAN"
    root.mkdir()
    (root / "inference_gfpgan.py").write_text("# test", encoding="utf-8")
    fake_python = tmp_path / "python.exe"
    fake_python.write_bytes(b"python")
    commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd=None):
        commands.append(command)
        if command[0] == "ffprobe":
            return SimpleNamespace(stdout=b"25/1")
        if command[0] == "ffmpeg" and "%08d.png" in command[-1]:
            frame = Path(command[-1].replace("%08d", "00000001"))
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(b"frame")
            return SimpleNamespace(stdout=b"")
        if command[0] == str(fake_python):
            restored = Path(command[command.index("-o") + 1]) / "restored_imgs"
            restored.mkdir(parents=True)
            (restored / "00000001.png").write_bytes(b"restored")
            return SimpleNamespace(stdout=b"")
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(b"enhanced-video")
            return SimpleNamespace(stdout=b"")
        raise AssertionError(command)

    monkeypatch.setattr(gfpgan, "_run", fake_run)
    result = gfpgan.enhance_video_with_gfpgan(
        str(source), str(output), gfpgan_dir=str(root), python_executable=str(fake_python)
    )

    assert result == str(output.resolve())
    assert output.read_bytes() == b"enhanced-video"
    inference = next(command for command in commands if command[0] == str(fake_python))
    assert inference[inference.index("-v") + 1] == "1.3"
    assert inference[inference.index("-s") + 1] == "1"
    assert inference[inference.index("--bg_upsampler") + 1] == "none"
    assert "--only_center_face" in inference
    assert inference[inference.index("-w") + 1] == "0.4"
    mux = commands[-1]
    assert "1:a?" in mux
    assert "libx264" in mux and "aac" in mux


def test_cuda_oom_has_traditional_chinese_guidance(monkeypatch) -> None:
    error = subprocess.CalledProcessError(
        1, ["python"], stderr=b"CUDA out of memory"
    )
    monkeypatch.setattr(gfpgan.subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(gfpgan.GFPGANError, match="GTX 1050 2GB"):
        gfpgan._run(["python"])


def test_fractional_fps_is_parsed(monkeypatch) -> None:
    monkeypatch.setattr(gfpgan, "_run", lambda command, cwd=None: SimpleNamespace(stdout=b"30000/1001\n"))
    assert gfpgan._video_fps(Path("video.mp4")).startswith("29.970")
