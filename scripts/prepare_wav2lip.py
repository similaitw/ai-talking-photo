"""Reproducible compatibility and quality fixes for pinned official Wav2Lip."""

import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "bac9a81e63ecc153202353372e5724b83d9e6322"
OFFICIAL_GAN_SHA256 = "180cfd49d31d47f195d5bfc62830ffe2c40724b7bbbce6faadc73ac6ab1a3b8e"
S3FD_SHA256 = "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"
SOFT_BLEND_MARKER = "AI_TALKING_PHOTO_SOFT_BLEND"


def patch_inference(text: str) -> str:
    """Apply idempotent compatibility and optional soft-blend changes."""

    text = text.replace("args.face.split('.')[1]", "os.path.splitext(args.face)[1][1:].lower()")

    old_ffmpeg = "command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)\n\tsubprocess.call(command, shell=platform.system() != 'Windows')"
    new_ffmpeg = "command = ['ffmpeg', '-nostdin', '-y', '-i', args.audio, '-i', 'temp/result.avi', '-strict', '-2', '-q:v', '1', args.outfile]\n\tsubprocess.run(command, check=True)"
    if old_ffmpeg in text:
        text = text.replace(old_ffmpeg, new_ffmpeg)
    elif new_ffmpeg not in text:
        raise RuntimeError("官方 FFmpeg 呼叫內容不符，停止修改。")

    if "--soft_blend" not in text:
        marker = "args = parser.parse_args()"
        if marker not in text:
            raise RuntimeError("找不到 Wav2Lip argparse 入口，停止加入品質修正。")
        argument = (
            "parser.add_argument('--soft_blend', action='store_true', "
            "help='Blend generated lower face into the original frame with a soft mask')\n\n"
        )
        text = text.replace(marker, argument + marker, 1)

    if SOFT_BLEND_MARKER not in text:
        old_paste = "\t\t\tf[y1:y2, x1:x2] = p"
        if old_paste not in text:
            raise RuntimeError("找不到 Wav2Lip 臉部貼回位置，停止加入品質修正。")
        new_paste = """\t\t\t# AI_TALKING_PHOTO_SOFT_BLEND: preserve original upper-face detail.\n\t\t\tif args.soft_blend:\n\t\t\t\toriginal = f[y1:y2, x1:x2].copy()\n\t\t\t\tmask = np.zeros((p.shape[0], p.shape[1]), dtype=np.float32)\n\t\t\t\tcenter = (p.shape[1] // 2, int(p.shape[0] * 0.72))\n\t\t\t\taxes = (max(1, int(p.shape[1] * 0.44)), max(1, int(p.shape[0] * 0.32)))\n\t\t\t\tcv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)\n\t\t\t\tblur = max(3, int(min(p.shape[:2]) * 0.11))\n\t\t\t\tif blur % 2 == 0: blur += 1\n\t\t\t\tmask = cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]\n\t\t\t\tp = (original.astype(np.float32) * (1.0 - mask) + p.astype(np.float32) * mask).astype(np.uint8)\n\n\t\t\tf[y1:y2, x1:x2] = p"""
        text = text.replace(old_paste, new_paste, 1)

    return text


def main() -> None:
    repo = ROOT / "vendor" / "Wav2Lip"
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise RuntimeError("官方 Wav2Lip 版本不符，請依 README 切換到指定 commit。")
    detector = repo / "face_detection" / "detection" / "sfd" / "s3fd.pth"
    if hashlib.sha256(detector.read_bytes()).hexdigest() != S3FD_SHA256:
        raise RuntimeError("官方指定的 S3FD 權重 SHA-256 不符。")
    checkpoint = ROOT / "models" / "wav2lip_gan.pth"
    original = ROOT / "models" / "wav2lip_gan.official.torchscript"
    if not original.exists():
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != OFFICIAL_GAN_SHA256:
            raise RuntimeError("官方下載檔 SHA-256 不符，停止轉換；請確認來源與版本。")
        checkpoint.rename(original)
    if hashlib.sha256(original.read_bytes()).hexdigest() != OFFICIAL_GAN_SHA256:
        raise RuntimeError("保留的官方下載檔 SHA-256 不符。")
    import torch
    sys.path.insert(0, str(repo))
    from models import Wav2Lip
    state = torch.jit.load(str(original), map_location="cpu").state_dict()
    model = Wav2Lip()
    model.load_state_dict(state, strict=True)
    torch.save({"state_dict": state}, checkpoint)

    script = repo / "inference.py"
    patched = patch_inference(script.read_text(encoding="utf-8"))
    script.write_text(patched, encoding="utf-8")
    print("官方權重已嚴格驗證；已套用路徑、FFmpeg 與可選嘴周柔和融合修正。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
