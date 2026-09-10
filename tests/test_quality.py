from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from talking_photo import quality


def test_expand_face_box_maps_padding_and_clamps() -> None:
    assert quality._expand_face_box(100, 80, 200, 240, 1024, 768) == (72, 349, 88, 312)
    assert quality._expand_face_box(0, 0, 100, 100, 120, 120) == (0, 112, 0, 106)


def test_limit_low_vram_output_keeps_hd_detail() -> None:
    large = Image.new("RGB", (4000, 3000))
    resized = quality.limit_low_vram_output(large)
    assert resized.size == (1280, 960)

    modest = Image.new("RGB", (1024, 768))
    kept = quality.limit_low_vram_output(modest)
    assert kept.size == (1024, 768)


def test_detect_face_box_uses_small_preview_and_largest_face(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    class FakeCascade:
        def empty(self) -> bool:
            return False

        def detectMultiScale(self, gray, **kwargs):
            calls["shape"] = gray.shape
            calls["kwargs"] = kwargs
            return np.array([[50, 40, 100, 120], [10, 10, 30, 30]])

    monkeypatch.setattr(quality.cv2, "CascadeClassifier", lambda path: FakeCascade())
    portrait = Image.new("RGB", (1024, 768))
    assert quality.detect_face_box(portrait) == (72, 349, 88, 312)
    assert max(calls["shape"]) <= quality.DETECTION_MAX_EDGE
    assert calls["kwargs"]["minNeighbors"] == 5


def test_detect_face_box_returns_none_when_preview_detector_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCascade:
        def empty(self) -> bool:
            return False

        def detectMultiScale(self, gray, **kwargs):
            return np.empty((0, 4), dtype=int)

    monkeypatch.setattr(quality.cv2, "CascadeClassifier", lambda path: FakeCascade())
    assert quality.detect_face_box(Image.new("RGB", (800, 600))) is None
