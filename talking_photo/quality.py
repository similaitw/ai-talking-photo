"""Quality-preserving helpers for low-VRAM still-image inference."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def _wav2lip_dir() -> Path:
    return Path(os.environ.get("WAV2LIP_DIR", PROJECT_ROOT / "vendor" / "Wav2Lip")).expanduser().resolve()


@lru_cache(maxsize=1)
def _load_cpu_face_detector():
    """Load the pinned Wav2Lip S3FD detector on CPU only; never consume GPU VRAM."""

    repo = _wav2lip_dir()
    detector_weights = repo / "face_detection" / "detection" / "sfd" / "s3fd.pth"
    if not detector_weights.is_file():
        return None

    repo_text = str(repo)
    inserted = repo_text not in sys.path
    if inserted:
        sys.path.insert(0, repo_text)
    try:
        face_detection = import_module("face_detection")
        return face_detection.FaceAlignment(
            face_detection.LandmarksType._2D,
            flip_input=False,
            device="cpu",
        )
    except Exception:
        return None
    finally:
        if inserted:
            try:
                sys.path.remove(repo_text)
            except ValueError:
                pass


def _detect_preview_rect(preview: Image.Image) -> tuple[int, int, int, int] | None:
    detector = _load_cpu_face_detector()
    if detector is None:
        return None
    try:
        # Upstream FaceAlignment reverses BGR to RGB internally, matching cv2 input.
        bgr = np.asarray(preview.convert("RGB"), dtype=np.uint8)[..., ::-1].copy()
        rect = detector.get_detections_for_batch(np.asarray([bgr]))[0]
    except Exception:
        return None
    if rect is None:
        return None
    x1, y1, x2, y2 = (int(value) for value in rect)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def detect_face_box(portrait: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect one face on a 512 px CPU S3FD preview and map it back."""

    preview = portrait.convert("RGB").copy()
    preview.thumbnail((DETECTION_MAX_EDGE, DETECTION_MAX_EDGE), Image.Resampling.LANCZOS)
    if preview.width == 0 or preview.height == 0:
        return None

    rect = _detect_preview_rect(preview)
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    scale_x = portrait.width / preview.width
    scale_y = portrait.height / preview.height
    return _expand_face_box(
        float(x1) * scale_x,
        float(y1) * scale_y,
        float(x2 - x1) * scale_x,
        float(y2 - y1) * scale_y,
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
