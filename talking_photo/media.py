"""FFmpeg audio preparation for Wav2Lip."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import wave


class AudioNormalizationError(RuntimeError):
    """Raised when audio cannot be converted to the required WAV format."""


def normalize_audio(input_path: str, output_path: str) -> str:
    """Return an absolute path to 16 kHz mono signed 16-bit PCM WAV audio.

    Convert into a temporary sibling file so failures preserve existing output.
    FFmpeg must be installed and available on PATH.
    """
    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("找不到輸入音訊檔案，請先產生或選擇音訊。")
    if source == destination or (
        destination.exists() and source.samefile(destination)
    ):
        raise ValueError("輸入與輸出音訊不可使用同一個檔案。")
    if destination.suffix.lower() != ".wav":
        raise ValueError("輸出音訊必須使用 .wav 副檔名。")

    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-y", "-i", str(source), "-map", "0:a:0", "-vn",
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                    "-f", "wav", str(temporary),
                ],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError as exc:
            raise AudioNormalizationError(
                "找不到 FFmpeg，請先安裝 FFmpeg 並加入 PATH 後再試。"
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise AudioNormalizationError(
                "音訊轉換失敗，請確認輸入檔案包含可讀取的音訊。"
            ) from exc

        try:
            with wave.open(str(temporary), "rb") as audio:
                valid = (
                    audio.getframerate() == 16000
                    and audio.getnchannels() == 1
                    and audio.getsampwidth() == 2
                    and audio.getcomptype() == "NONE"
                    and audio.getnframes() > 0
                    and bool(audio.readframes(1))
                )
        except (wave.Error, EOFError) as exc:
            raise AudioNormalizationError("音訊轉換未產生有效的 WAV 檔案。") from exc
        if not valid:
            raise AudioNormalizationError("音訊轉換結果不是有效的 16 kHz 單聲道 PCM WAV。")
        os.replace(temporary, destination)
    except OSError as exc:
        raise AudioNormalizationError(
            "無法執行音訊轉換或寫入檔案，請確認 FFmpeg、資料夾權限與可用空間。"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(destination)
