from PIL import Image
import pytest

from talking_photo import quality


def test_expand_face_box_maps_padding_and_clamps() -> None:
    assert quality._expand_face_box(100, 80, 200, 240, 1024, 768) == (72, 349, 88, 312)
    assert quality._expand_face_box(0, 0, 100, 100, 120, 120) == (0, 113, 0, 106)


def test_limit_low_vram_output_keeps_hd_detail() -> None:
    large = Image.new("RGB", (4000, 3000))
    resized = quality.limit_low_vram_output(large)
    assert resized.size == (1280, 960)

    modest = Image.new("RGB", (1024, 768))
    kept = quality.limit_low_vram_output(modest)
    assert kept.size == (1024, 768)


def test_detect_face_box_uses_small_preview_and_maps_back(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def detect(preview):
        calls["size"] = preview.size
        return (50, 40, 150, 160)

    monkeypatch.setattr(quality, "_detect_preview_rect", detect)
    portrait = Image.new("RGB", (1024, 768))
    assert quality.detect_face_box(portrait) == (72, 349, 88, 312)
    assert calls["size"] == (512, 384)
    assert max(calls["size"]) <= quality.DETECTION_MAX_EDGE


def test_detect_face_box_returns_none_when_preview_detector_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quality, "_detect_preview_rect", lambda preview: None)
    assert quality.detect_face_box(Image.new("RGB", (800, 600))) is None
