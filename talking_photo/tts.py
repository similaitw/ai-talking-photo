"""Taiwan Mandarin text-to-speech helpers backed by edge-tts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from talking_photo.validation import normalize_script

VOICE_MAP = {
    "台灣女聲": "zh-TW-HsiaoChenNeural",
    "台灣男聲": "zh-TW-YunJheNeural",
}

MIN_RATE = 0.8
MAX_RATE = 1.2


class TTSGenerationError(RuntimeError):
    """Raised when speech generation cannot be completed."""


def resolve_voice(voice: str) -> str:
    """Map a user-facing voice label to an Edge TTS voice id."""

    try:
        return VOICE_MAP[voice]
    except KeyError as exc:
        raise ValueError(f"不支援的語音：{voice}") from exc


def format_edge_rate(rate: float) -> str:
    """Convert a speed multiplier such as 0.95 to Edge TTS rate syntax."""

    if not MIN_RATE <= rate <= MAX_RATE:
        raise ValueError(f"語速必須介於 {MIN_RATE:.1f} 到 {MAX_RATE:.1f} 之間。")

    percent = round((rate - 1.0) * 100)
    return f"{percent:+d}%"


async def _synthesize_speech_async(
    text: str,
    voice_id: str,
    rate: str,
    output_path: Path,
) -> None:
    """Generate speech asynchronously with edge-tts."""

    try:
        import edge_tts
    except ImportError as exc:
        raise TTSGenerationError(
            "尚未安裝 edge-tts，請先安裝專案相依套件後再試。"
        ) from exc

    try:
        communicate = edge_tts.Communicate(text, voice_id, rate=rate)
        await communicate.save(str(output_path))
    except Exception as exc:
        raise TTSGenerationError(
            "語音產生失敗，請確認網路連線後再試；若持續失敗，可稍後重新產生。"
        ) from exc


def synthesize_speech(
    text: str,
    voice: str,
    rate: float,
    output_path: str,
) -> str:
    """Create an MP3 speech file and return its absolute output path.

    The public API intentionally stays synchronous because the generation pipeline and
    Gradio worker callbacks are synchronous. Network access is performed only when this
    function is invoked; importing this module does not require ``edge-tts`` to be
    installed, which keeps unit tests and diagnostic tooling lightweight.
    """

    normalized = normalize_script(text)
    if not normalized:
        raise ValueError("請輸入講稿。")

    voice_id = resolve_voice(voice)
    edge_rate = format_edge_rate(rate)

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(
            _synthesize_speech_async(
                text=normalized,
                voice_id=voice_id,
                rate=edge_rate,
                output_path=destination,
            )
        )
    except TTSGenerationError:
        raise
    except RuntimeError as exc:
        raise TTSGenerationError("語音產生程序無法啟動，請重新執行後再試。") from exc

    if not destination.is_file() or destination.stat().st_size == 0:
        raise TTSGenerationError("語音產生完成但找不到輸出檔案，請重新產生。")

    return str(destination)
