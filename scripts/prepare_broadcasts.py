"""Extract a sparse, timestamped labeling set from complete broadcast videos."""
from argparse import ArgumentParser
from fractions import Fraction
from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def probe(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    data = json.loads(subprocess.check_output(command, text=True))
    return float(data["format"]["duration"]), Fraction(data["streams"][0]["avg_frame_rate"])


def extract(video, source_id, interval):
    destination = ROOT / "broadcasts" / source_id
    frames = destination / "frames"
    timing_path = destination / "timing.json"
    frames.mkdir(parents=True, exist_ok=True)
    if any(frames.glob("frame_*.jpg")):
        raise SystemExit(f"Refusing to overwrite existing extraction: {frames}")

    duration, fps = probe(video)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps=1/{interval}:start_time=0", "-q:v", "3",
        str(frames / "frame_%06d.jpg"),
    ], check=True)

    images = sorted(frames.glob("frame_*.jpg"))
    timing = {}
    for index, image in enumerate(images):
        timestamp = index * interval
        timing[image.name] = {
            "source_video": str(video.relative_to(ROOT)),
            "source_frame_index": round(timestamp * float(fps)),
            "timestamp_seconds": float(timestamp),
        }
    timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps({
        "source_id": source_id,
        "video": str(video),
        "duration_seconds": duration,
        "fps": str(fps),
        "interval_seconds": interval,
        "frames": len(images),
        "timing": str(timing_path),
    }))


def main():
    parser = ArgumentParser()
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("video", type=Path)
    parser.add_argument("source_id")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    extract(args.video.resolve(), args.source_id, args.interval)


if __name__ == "__main__":
    main()
