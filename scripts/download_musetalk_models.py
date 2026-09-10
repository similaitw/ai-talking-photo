"""Download only the official model files required for MuseTalk 1.5 inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _snapshot(repo_id: str, target: Path, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        allow_patterns=patterns,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        default="vendor/MuseTalk/models",
        help="MuseTalk models directory",
    )
    args = parser.parse_args()
    target = Path(args.target).expanduser().resolve()

    print("下載 MuseTalk 1.5 官方模型……")
    _snapshot(
        "TMElyralab/MuseTalk",
        target,
        ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"],
    )
    print("下載 SD VAE……")
    _snapshot(
        "stabilityai/sd-vae-ft-mse",
        target / "sd-vae",
        ["config.json", "diffusion_pytorch_model.bin"],
    )
    print("下載 Whisper tiny……")
    _snapshot(
        "openai/whisper-tiny",
        target / "whisper",
        ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    )
    print("下載 DWPose……")
    _snapshot(
        "yzd-v/DWPose",
        target / "dwpose",
        ["dw-ll_ucoco_384.pth"],
    )
    print("下載 Face Parse BiSeNet……")
    _snapshot(
        "ManyOtherFunctions/face-parse-bisent",
        target / "face-parse-bisent",
        ["79999_iter.pth", "resnet18-5c106cde.pth"],
    )

    required = [
        target / "musetalkV15" / "musetalk.json",
        target / "musetalkV15" / "unet.pth",
        target / "sd-vae" / "config.json",
        target / "sd-vae" / "diffusion_pytorch_model.bin",
        target / "whisper" / "config.json",
        target / "whisper" / "pytorch_model.bin",
        target / "whisper" / "preprocessor_config.json",
        target / "dwpose" / "dw-ll_ucoco_384.pth",
        target / "face-parse-bisent" / "79999_iter.pth",
        target / "face-parse-bisent" / "resnet18-5c106cde.pth",
    ]
    missing = [str(path.relative_to(target)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("MuseTalk 模型下載不完整：" + "、".join(missing))
    print("MuseTalk 1.5 推論所需模型已準備完成。")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    main()
