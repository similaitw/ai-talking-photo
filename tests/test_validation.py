from pathlib import Path

import pytest
from PIL import Image

from talking_photo.validation import (
    ValidationError,
    normalize_script,
    validate_image,
    validate_inputs,
    validate_script,
)


def make_image(path: Path, size: tuple[int, int] = (512, 512), fmt: str = "PNG") -> Path:
    Image.new("RGB", size, "white").save(path, format=fmt)
    return path


def test_normalize_script_removes_markdown_and_extra_whitespace() -> None:
    source = "  **大家好。**\r\n\r\n\r\n  我是   林春秀。  "
    assert normalize_script(source) == "大家好。\n\n我是 林春秀。"


def test_validate_script_rejects_blank_text() -> None:
    with pytest.raises(ValidationError, match="請輸入講稿"):
        validate_script(" \n\t ")


def test_validate_script_rejects_too_long_text() -> None:
    with pytest.raises(ValidationError, match="講稿過長"):
        validate_script("字" * 3001)


def test_validate_script_accepts_exact_limit() -> None:
    text = "字" * 3000
    assert validate_script(text) == text


def test_validate_image_requires_file() -> None:
    with pytest.raises(ValidationError, match="請上傳人物照片"):
        validate_image(None)


def test_validate_image_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="找不到上傳的圖片"):
        validate_image(tmp_path / "missing.png")


def test_validate_image_rejects_invalid_file(tmp_path: Path) -> None:
    invalid = tmp_path / "not-an-image.png"
    invalid.write_text("hello", encoding="utf-8")
    with pytest.raises(ValidationError, match="圖片無法讀取"):
        validate_image(invalid)


def test_validate_image_rejects_small_image(tmp_path: Path) -> None:
    image = make_image(tmp_path / "small.png", (255, 512))
    with pytest.raises(ValidationError, match="尺寸太小"):
        validate_image(image)


def test_validate_image_accepts_png(tmp_path: Path) -> None:
    image = make_image(tmp_path / "portrait.png")
    assert validate_image(image) == image.resolve()


def test_validate_inputs_returns_normalized_values(tmp_path: Path) -> None:
    image = make_image(tmp_path / "portrait.png")
    result_image, result_script = validate_inputs(image, " **大家好。** ")
    assert result_image == image.resolve()
    assert result_script == "大家好。"
