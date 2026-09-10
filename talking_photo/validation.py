"""Input validation and script normalization."""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from talking_photo.config import MAX_TEXT_LENGTH, MIN_IMAGE_SIZE

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


class ValidationError(ValueError):
    """Raised when an end-user input cannot be accepted."""


def normalize_script(text: str) -> str:
    """Normalize pasted script text while preserving useful paragraph breaks."""

    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("**", "").replace("__", "")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)
    normalized = re.sub(r" *\n *", "\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def validate_script(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Validate and return a normalized script."""

    normalized = normalize_script(text)
    if not normalized:
        raise ValidationError("請輸入講稿。")
    if len(normalized) > max_length:
        raise ValidationError(
            "講稿過長，請分段產生。建議每段 10～30 秒，效果通常較自然。"
        )
    return normalized


def validate_image(
    image_path: str | Path | None,
    min_size: int = MIN_IMAGE_SIZE,
) -> Path:
    """Validate an uploaded portrait and return its resolved path."""

    if not image_path:
        raise ValidationError("請上傳人物照片。")

    path = Path(image_path)
    if not path.is_file():
        raise ValidationError("找不到上傳的圖片，請重新上傳。")

    try:
        with Image.open(path) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValidationError("圖片無法讀取，請改用 JPG、PNG 或 WebP。") from exc

    if image_format not in SUPPORTED_IMAGE_FORMATS:
        raise ValidationError("不支援此圖片格式，請改用 JPG、PNG 或 WebP。")

    if width < min_size or height < min_size:
        raise ValidationError(
            f"圖片尺寸太小，寬與高都必須至少 {min_size} px。建議使用 512×512 以上照片。"
        )

    return path.resolve()


def validate_inputs(
    image_path: str | Path | None,
    text: str,
) -> tuple[Path, str]:
    """Validate both primary user inputs in UI order."""

    image = validate_image(image_path)
    script = validate_script(text)
    return image, script
