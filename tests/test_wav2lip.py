from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import talking_photo.wav2lip as wav2lip
from talking_photo.wav2lip import (
    Wav2LipConfigurationError,
    Wav2LipError,
    generate_lip_sync,
)


def prepare_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    image = tmp_path / "portrait.png"
    audio = tmp_path / "speech.wav"
    output = tmp_path / "result.mp4"
    image.write_bytes(b"image")
    audio.write_bytes(b"audio")
    repo = tmp_path / "Wav2Lip"
    repo.mkdir()
    (repo / "inference.py").write_text("# test", encoding="utf-8")
    checkpoint = tmp_path / "wav2lip_gan.pth"
    checkpoint.write_bytes(b"weights")
    monkeypatch.setenv("WAV2LIP_DIR", str(repo))
    monkeypatch.setenv("WAV2LIP_CHECKPOINT", str(checkpoint))
    return image, audio, output, repo


def test_low_vram_command_uses_safe_batch_sizes(tmp_path: Path) -> None:
    command = wav2lip._build_command(
        tmp_path / "inference.py",
        tmp_path / "model.pth",
        tmp_path / "face.png",
        tmp_path / "audio.wav",
        tmp_path / "out.mp4",
        low_vram=True,
    )
    assert command[-6:] == [
        "--wav2lip_batch_size",
        "1",
        "--face_det_batch_size",
        "1",
        "--resize_factor",
        "2",
    ]
    start = command.index("--static")
    assert command[start : start + 2] == ["--static", "True"]


def test_quality_command_uses_fixed_box_and_soft_blend(tmp_path: Path) -> None:
    command = wav2lip._build_command(
        tmp_path / "inference.py",
        tmp_path / "model.pth",
        tmp_path / "face.png",
        tmp_path / "audio.wav",
        tmp_path / "out.mp4",
        low_vram=True,
        face_box=(20, 220, 30, 210),
        soft_blend=True,
    )
    index = command.index("--box")
    assert command[index : index + 5] == ["--box", "20", "220", "30", "210"]
    assert "--soft_blend" in command


def test_invalid_face_box_is_rejected() -> None:
    with pytest.raises(ValueError, match="face_box"):
        wav2lip._validate_face_box((10, 5, 0, 20))
    with pytest.raises(ValueError, match="face_box"):
        wav2lip._validate_face_box((-1, 20, 0, 20))


def test_normal_command_does_not_force_low_vram_flags(tmp_path: Path) -> None:
    command = wav2lip._build_command(
        tmp_path / "inference.py",
        tmp_path / "model.pth",
        tmp_path / "face.png",
        tmp_path / "audio.wav",
        tmp_path / "out.mp4",
        low_vram=False,
    )
    assert "--wav2lip_batch_size" not in command
    assert "--resize_factor" not in command
    assert "--box" not in command
    assert "--soft_blend" not in command


@pytest.mark.parametrize("device", ["gpu", "cuda:x", "cuda:-1"])
def test_invalid_device_name_is_rejected(device: str) -> None:
    with pytest.raises(ValueError, match="device"):
        wav2lip._cuda_index(device)


def test_cpu_hides_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image, audio, output, _ = prepare_runtime(tmp_path, monkeypatch)
    workdirs = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        Path(command[command.index("--outfile") + 1]).write_bytes(b"mp4")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        work = Path(kwargs["cwd"])
        assert (work / "temp").is_dir()
        (work / "temp" / "result.avi").write_bytes(b"intermediate")
        workdirs.append(work)
        return SimpleNamespace(stdout="Using cpu for inference.", stderr="")

    monkeypatch.setattr(wav2lip.subprocess, "run", fake_run)
    assert generate_lip_sync(str(image), str(audio), str(output), "cpu") == str(
        output.resolve()
    )
    assert output.read_bytes() == b"mp4"
    generate_lip_sync(str(image), str(audio), str(output), "cpu")
    assert workdirs[0] != workdirs[1]
    assert all(not work.exists() for work in workdirs)


def test_cuda_selects_requested_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, audio, output, _ = prepare_runtime(tmp_path, monkeypatch)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
    )
    monkeypatch.setattr(wav2lip, "import_module", lambda name: fake_torch)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        Path(command[command.index("--outfile") + 1]).write_bytes(b"mp4")
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        return SimpleNamespace(stdout="Using cuda for inference.", stderr="")

    monkeypatch.setattr(wav2lip.subprocess, "run", fake_run)
    generate_lip_sync(str(image), str(audio), str(output), "cuda:1")


def test_cuda_does_not_silently_fallback_to_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, audio, output, _ = prepare_runtime(tmp_path, monkeypatch)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    )
    monkeypatch.setattr(wav2lip, "import_module", lambda name: fake_torch)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        Path(command[command.index("--outfile") + 1]).write_bytes(b"mp4")
        return SimpleNamespace(stdout="Using cpu for inference.", stderr="")

    monkeypatch.setattr(wav2lip.subprocess, "run", fake_run)
    with pytest.raises(Wav2LipError, match="實際改用 CPU"):
        generate_lip_sync(str(image), str(audio), str(output), "cuda")
    assert not output.exists()


def test_cuda_unavailable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    )
    monkeypatch.setattr(wav2lip, "import_module", lambda name: fake_torch)
    with pytest.raises(Wav2LipConfigurationError, match="不會自動改用 CPU"):
        wav2lip._cuda_index("cuda")


@pytest.mark.parametrize("missing", ["image", "audio"])
def test_missing_inputs_are_rejected(tmp_path: Path, missing: str) -> None:
    image = tmp_path / "portrait.png"
    audio = tmp_path / "speech.wav"
    if missing != "image":
        image.write_bytes(b"image")
    if missing != "audio":
        audio.write_bytes(b"audio")
    with pytest.raises(ValueError):
        generate_lip_sync(str(image), str(audio), str(tmp_path / "out.mp4"), "cpu")


def test_missing_runtime_and_checkpoint_have_clear_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "portrait.png"
    image.write_bytes(b"image")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio")
    monkeypatch.setenv("WAV2LIP_DIR", str(tmp_path / "missing"))
    with pytest.raises(Wav2LipConfigurationError, match="inference.py"):
        generate_lip_sync(str(image), str(audio), str(tmp_path / "out.mp4"), "cpu")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "inference.py").write_text("# test", encoding="utf-8")
    monkeypatch.setenv("WAV2LIP_DIR", str(repo))
    monkeypatch.setenv("WAV2LIP_CHECKPOINT", str(tmp_path / "missing.pth"))
    with pytest.raises(Wav2LipConfigurationError, match="checkpoint"):
        generate_lip_sync(str(image), str(audio), str(tmp_path / "out.mp4"), "cpu")


def test_oom_gets_specific_message_and_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, audio, output, _ = prepare_runtime(tmp_path, monkeypatch)
    output.write_bytes(b"old")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    )
    monkeypatch.setattr(wav2lip, "import_module", lambda name: fake_torch)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(1, command, stderr="CUDA out of memory")

    monkeypatch.setattr(wav2lip.subprocess, "run", fake_run)
    with pytest.raises(Wav2LipError, match="顯示記憶體不足"):
        generate_lip_sync(str(image), str(audio), str(output), "cuda", low_vram=True)
    assert output.read_bytes() == b"old"
    assert not list(output.parent.glob("wav2lip-*.mp4"))


def test_face_detection_failure_gets_specific_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, audio, output, _ = prepare_runtime(tmp_path, monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(1, command, stderr="Face not detected!")

    monkeypatch.setattr(wav2lip.subprocess, "run", fake_run)
    with pytest.raises(Wav2LipError, match="找不到可用的人臉"):
        generate_lip_sync(str(image), str(audio), str(output), "cpu")


def test_old_unpatched_runtime_gets_prepare_hint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, audio, output, _ = prepare_runtime(tmp_path, monkeypatch)

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        raise subprocess.CalledProcessError(
            2, command, stderr="error: unrecognized arguments: --soft_blend"
        )

    monkeypatch.setattr(wav2lip.subprocess, "run", fake_run)
    with pytest.raises(Wav2LipError, match="prepare_wav2lip.py"):
        generate_lip_sync(
            str(image), str(audio), str(output), "cpu", soft_blend=True
        )


def test_success_requires_nonempty_mp4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, audio, output, _ = prepare_runtime(tmp_path, monkeypatch)
    monkeypatch.setattr(
        wav2lip.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", stderr=""),
    )
    with pytest.raises(Wav2LipError, match="沒有產生有效的 MP4"):
        generate_lip_sync(str(image), str(audio), str(output), "cpu")
    assert not output.exists()


def test_output_extension_and_input_overwrite_are_rejected(tmp_path: Path) -> None:
    image = tmp_path / "portrait.png"
    image.write_bytes(b"image")
    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio")
    with pytest.raises(ValueError, match=".mp4"):
        generate_lip_sync(str(image), str(audio), str(tmp_path / "out.avi"), "cpu")
    with pytest.raises(ValueError, match="不可覆蓋"):
        generate_lip_sync(str(image), str(audio), str(image), "cpu")
