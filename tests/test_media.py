from pathlib import Path
import shutil
import subprocess
import wave

import pytest

from talking_photo.media import AudioNormalizationError, normalize_audio


def write_wav(path: Path, rate: int = 16000, channels: int = 1, frames: int = 1600) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"\x01\x00" * channels * frames)


@pytest.mark.parametrize("invalid", ["missing", "directory", "same", "extension"])
def test_invalid_paths(tmp_path: Path, invalid: str) -> None:
    source = tmp_path / "input.wav"
    write_wav(source)
    output = tmp_path / "output.wav"
    if invalid == "missing":
        source = tmp_path / "missing.wav"
    elif invalid == "directory":
        source = tmp_path
    elif invalid == "same":
        output = source
    else:
        output = tmp_path / "output.mp3"
    with pytest.raises(ValueError):
        normalize_audio(str(source), str(output))


@pytest.mark.parametrize("failure", ["missing_ffmpeg", "conversion", "os", "empty", "invalid", "format", "zero_frames"])
def test_failure_preserves_output_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    source = tmp_path / "input.mp3"
    source.write_bytes(b"input")
    output = tmp_path / "output.wav"
    output.write_bytes(b"existing output")

    def run(command: list[str], **kwargs: object) -> None:
        if failure == "missing_ffmpeg":
            raise FileNotFoundError()
        if failure == "conversion":
            raise subprocess.CalledProcessError(1, command, stderr=b"invalid input")
        if failure == "os":
            raise PermissionError()
        if failure == "invalid":
            Path(command[-1]).write_bytes(b"not a wav")
        if failure == "format":
            write_wav(Path(command[-1]), rate=44100, channels=2)
        if failure == "zero_frames":
            write_wav(Path(command[-1]), frames=0)

    monkeypatch.setattr("talking_photo.media.subprocess.run", run)
    with pytest.raises(AudioNormalizationError):
        normalize_audio(str(source), str(output))
    assert output.read_bytes() == b"existing output"
    assert source.read_bytes() == b"input"
    assert set(tmp_path.iterdir()) == {source, output}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="未安裝 FFmpeg，略過實際音訊轉換測試")
@pytest.mark.parametrize("input_format", ["wav", "mp3"])
def test_real_ffmpeg_normalizes_audio(tmp_path: Path, input_format: str) -> None:
    source = tmp_path / "原始 音訊.wav"
    write_wav(source, rate=44100, channels=2, frames=44100)
    if input_format == "mp3":
        mp3 = tmp_path / "語音 預覽.mp3"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(source), str(mp3)],
            check=True, capture_output=True,
        )
        source = mp3
    original = source.read_bytes()
    output = tmp_path / "新 資料夾" / "語音.wav"
    result = normalize_audio(str(source), str(output))
    assert result == str(output.resolve())
    with wave.open(result, "rb") as audio:
        assert audio.getframerate() == 16000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert audio.getcomptype() == "NONE"
        assert audio.getnframes() / audio.getframerate() == pytest.approx(1, abs=0.1)
    assert source.read_bytes() == original
    assert list(output.parent.iterdir()) == [output]
    # A second conversion must safely replace an existing output too.
    assert normalize_audio(str(source), str(output)) == result


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="未安裝 FFmpeg，略過實際音訊轉換測試")
def test_real_ffmpeg_rejects_corrupt_input(tmp_path: Path) -> None:
    source = tmp_path / "broken.mp3"
    source.write_bytes(b"invalid audio")
    output = tmp_path / "output.wav"
    with pytest.raises(AudioNormalizationError, match="音訊轉換失敗"):
        normalize_audio(str(source), str(output))
    assert list(tmp_path.iterdir()) == [source]
