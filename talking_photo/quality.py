"""Quality-preserving helpers for low-VRAM still-image inference."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DETECTION_MAX_EDGE = 512
LOW_VRAM_OUTPUT_MAX_EDGE = 1280


def _expand_face_box(
    x: float,
    y: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Return Wav2Lip box order: top, bottom, left, right."""

    left = max(0, math.floor(x - width * 0.06))
    right = min(image_width, math.ceil(x + width * 1.06))
    top = max(0, math.floor(y - height * 0.03))
    bottom = min(image_height, math.ceil(y + height * 1.12))
    return top, bottom, left, right


def detect_face_box(portrait: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect the largest frontal face on a small CPU preview and map it back."""

    preview = portrait.convert("RGB").copy()
    preview.thumbnail((DETECTION_MAX_EDGE, DETECTION_MAX_EDGE), Image.Resampling.LANCZOS)
    if preview.width == 0 or preview.height == 0:
        return None

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return None

    gray = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2GRAY)
    minimum = max(40, min(preview.size) // 8)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(minimum, minimum),
    )
    if len(faces) == 0:
        return None

    x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    scale_x = portrait.width / preview.width
    scale_y = portrait.height / preview.height
    return _expand_face_box(
        float(x) * scale_x,
        float(y) * scale_y,
        float(width) * scale_x,
        float(height) * scale_y,
        portrait.width,
        portrait.height,
    )


def limit_low_vram_output(portrait: Image.Image) -> Image.Image:
    """Keep substantially more detail than the old 512 px path without huge frames."""

    if max(portrait.size) <= LOW_VRAM_OUTPUT_MAX_EDGE:
        return portrait
    resized = portrait.copy()
    resized.thumbnail(
        (LOW_VRAM_OUTPUT_MAX_EDGE, LOW_VRAM_OUTPUT_MAX_EDGE),
        Image.Resampling.LANCZOS,
    )
    return resized
