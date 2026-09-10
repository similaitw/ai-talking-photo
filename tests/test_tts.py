from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from talking_photo.tts import (
    TTSGenerationError,
    format_edge_rate,
    resolve_voice,
    synthesize_speech,
)


def test_resolve_voice_maps_taiwan_voices() -> None:
    assert resolve_voice("台灣女聲") == "zh-TW-HsiaoChenNeural"
    assert resolve_voice("台灣男聲") == "zh-TW-YunJheNeural"


def test_resolve_voice_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="不支援的語音"):
        resolve_voice("不存在的聲音")


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        (0.8, "-20%"),
        (0.95, "-5%"),
        (1.0, "+0%"),
        (1.2, "+20%"),
    ],
)
def test_format_edge_rate(rate: float, expected: str) -> None:
    assert format_edge_rate(rate) == expected


def test_format_edge_rate_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="語速必須介於"):
        format_edge_rate(1.25)


def test_synthesize_speech_normalizes_text_and_writes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakeCommunicate:
        def __init__(self, text: str, voice: str, rate: str) -> None:
            calls.update(text=text, voice=voice, rate=rate)

        async def save(self, output_path: str) -> None:
            Path(output_path).write_bytes(b"fake-mp3")
            calls["output_path"] = output_path

    monkeypatch.setitem(sys.modules, "edge_tts", SimpleNamespace(Communicate=FakeCommunicate))

    output = tmp_path / "nested" / "speech.mp3"
    result = synthesize_speech(
        "  **大家好。**\r\n  我是   林春秀。  ",
        "台灣女聲",
        0.95,
        str(output),
    )

    assert Path(result) == output.resolve()
    assert output.read_bytes() == b"fake-mp3"
    assert calls == {
        "text": "大家好。\n我是 林春秀。",
        "voice": "zh-TW-HsiaoChenNeural",
        "rate": "-5%",
        "output_path": str(output.resolve()),
    }


def test_synthesize_speech_rejects_blank_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="請輸入講稿"):
        synthesize_speech("  \n ", "台灣女聲", 0.95, str(tmp_path / "speech.mp3"))


def test_synthesize_speech_wraps_edge_tts_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenCommunicate:
        def __init__(self, text: str, voice: str, rate: str) -> None:
            pass

        async def save(self, output_path: str) -> None:
            raise OSError("network unavailable")

    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=BrokenCommunicate)
    )

    with pytest.raises(TTSGenerationError, match="語音產生失敗"):
        synthesize_speech(
            "大家好。", "台灣男聲", 1.0, str(tmp_path / "speech.mp3")
        )


def test_synthesize_speech_rejects_missing_output_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoOutputCommunicate:
        def __init__(self, text: str, voice: str, rate: str) -> None:
            pass

        async def save(self, output_path: str) -> None:
            return None

    monkeypatch.setitem(
        sys.modules, "edge_tts", SimpleNamespace(Communicate=NoOutputCommunicate)
    )

    with pytest.raises(TTSGenerationError, match="找不到輸出檔案"):
        synthesize_speech(
            "大家好。", "台灣女聲", 0.95, str(tmp_path / "speech.mp3")
        )
