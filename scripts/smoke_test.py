"""Opt-in real E2E test. Never run as part of pytest or CI."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from talking_photo.pipeline import generate_talking_video


def main() -> int:
    parser = argparse.ArgumentParser(description="使用指定的非私人人像執行真實短片測試")
    parser.add_argument("image", help="測試人像路徑")
    args = parser.parse_args()
    try:
        result = generate_talking_video(
            args.image, "大家好，這是一段說話照片測試。祝大家今天心情愉快，事事順利。",
            "台灣女聲", 1.0, lambda value, message: print(f"{value:.0%} {message}", flush=True),
        )
    except Exception as exc:
        print(f"真實測試失敗：{exc}", file=sys.stderr)
        cause = exc.__cause__
        if isinstance(cause, subprocess.CalledProcessError):
            print(cause.stdout or "", file=sys.stderr)
            print(cause.stderr or "", file=sys.stderr)
        return 1
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", result["video_path"]],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    details = json.loads(probe.stdout)
    streams = details["streams"]
    assert any(s["codec_name"] == "h264" for s in streams)
    assert any(s["codec_name"] == "aac" for s in streams)
    duration = float(details["format"]["duration"])
    assert 5 <= duration <= 10, duration
    subprocess.run(["ffmpeg", "-v", "error", "-i", result["video_path"], "-f", "null", "-"], check=True)
    result["duration_seconds"] = duration
    report = ROOT / "temp" / "smoke-result.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
