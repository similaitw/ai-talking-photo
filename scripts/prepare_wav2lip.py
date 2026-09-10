"""Reproducible compatibility fixes for the pinned official Wav2Lip checkout."""

import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = "bac9a81e63ecc153202353372e5724b83d9e6322"
OFFICIAL_GAN_SHA256 = "180cfd49d31d47f195d5bfc62830ffe2c40724b7bbbce6faadc73ac6ab1a3b8e"
S3FD_SHA256 = "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"


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

    # Small compatibility edits only; model architecture/inference remain upstream.
    script = repo / "inference.py"
    text = script.read_text(encoding="utf-8")
    text = text.replace("args.face.split('.')[1]", "os.path.splitext(args.face)[1][1:].lower()")
    old = "command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)\n\tsubprocess.call(command, shell=platform.system() != 'Windows')"
    new = "command = ['ffmpeg', '-nostdin', '-y', '-i', args.audio, '-i', 'temp/result.avi', '-strict', '-2', '-q:v', '1', args.outfile]\n\tsubprocess.run(command, check=True)"
    if old not in text and new not in text:
        raise RuntimeError("官方 FFmpeg 呼叫內容不符，停止修改。")
    script.write_text(text.replace(old, new), encoding="utf-8")
    print("官方權重已轉換並嚴格驗證；已修正路徑辨識與 FFmpeg 路徑參數。")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
