"""Input validation and script normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
import subprocess

from PIL import Image, UnidentifiedImageError

from talking_photo.config import MAX_TEXT_LENGTH, MIN_IMAGE_SIZE

SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv"}


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
    """Validate an uploaded portrait image and return its resolved path."""

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


def probe_video_size(video_path: str | Path) -> tuple[int, int]:
    """Return the first video stream size using FFprobe."""

    path = Path(video_path).expanduser().resolve()
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ValidationError("找不到 FFprobe，無法檢查人物影片。請先安裝 FFmpeg。") from exc
    except subprocess.SubprocessError as exc:
        raise ValidationError("影片無法讀取，請改用 MP4、MOV、WebM 或 MKV。") from exc

    try:
        streams = json.loads(result.stdout).get("streams", [])
        width = int(streams[0]["width"])
        height = int(streams[0]["height"])
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationError("影片沒有可用的影像軌，請重新選擇人物影片。") from exc
    if width <= 0 or height <= 0:
        raise ValidationError("影片尺寸無效，請重新選擇人物影片。")
    return width, height


def validate_video(
    video_path: str | Path | None,
    min_size: int = MIN_IMAGE_SIZE,
) -> Path:
    """Validate an uploaded portrait video and return its resolved path."""

    if not video_path:
        raise ValidationError("請上傳人物影片。")
    path = Path(video_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValidationError("找不到上傳的影片，請重新上傳。")
    if path.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValidationError("不支援此影片格式，請改用 MP4、MOV、WebM 或 MKV。")
    width, height = probe_video_size(path)
    if width < min_size or height < min_size:
        raise ValidationError(
            f"影片尺寸太小，寬與高都必須至少 {min_size} px。建議使用 720p 以上影片。"
        )
    return path.resolve()


def media_kind(media_path: str | Path) -> str:
    """Return ``image`` or ``video`` for a validated supported input path."""

    suffix = Path(media_path).suffix.lower()
    if suffix in SUPPORTED_VIDEO_SUFFIXES:
        return "video"
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return "image"
    raise ValidationError("不支援此人物素材格式。請使用 JPG、PNG、WebP、MP4、MOV、WebM 或 MKV。")


def validate_media(media_path: str | Path | None) -> Path:
    """Validate a portrait image or video selected by the user."""

    if not media_path:
        raise ValidationError("請上傳人物照片或影片。")
    path = Path(media_path)
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_VIDEO_SUFFIXES:
        return validate_video(path)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return validate_image(path)
    raise ValidationError("不支援此人物素材格式。請使用 JPG、PNG、WebP、MP4、MOV、WebM 或 MKV。")


def validate_inputs(
    media_path: str | Path | None,
    text: str,
) -> tuple[Path, str]:
    """Validate the portrait media and script in UI order."""

    media = validate_media(media_path)
    script = validate_script(text)
    return media, script
