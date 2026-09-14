"""Analyze footage with hardware FFmpeg; save anonymous bounds and review frames."""
import argparse
from contextlib import closing
import dataclasses
import json
import math
import time
from pathlib import Path

import cv2

from .core import DecodeConfig, frames, hardware_command
from .detectors import Performers, stage_candidates


def analyze(config, models, output, *, gpu=False, review_every=30):
    if not math.isfinite(review_every) or review_every <= 0:
        raise ValueError("Review interval must be finite and positive")
    command = hardware_command(config)
    output = Path(output)
    # Refuse to overwrite a previous analysis, including its completion manifest.
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"decode": dataclasses.asdict(config), "command": command,
                "inference": "CUDA person + CPU face" if gpu else "CPU",
                "status": "initializing", "frames": 0}
    manifest_path = output / "manifest.json"

    def save_manifest():
        temporary = output / ".manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(manifest_path)

    save_manifest()
    try:
        detector = Performers(Path(models), gpu=gpu, threads=2)
        manifest["status"] = "running"
        save_manifest()
        next_review = config.start
        with (output / "detections.jsonl").open("x") as records, closing(frames(config)) as reader:
            for index, (t, frame) in enumerate(reader):
                detections = detector.detect(frame)
                accepted, rejected = stage_candidates(detections)
                row = {"time": t, **detections, "stage_candidates": accepted,
                       "rejected_candidates": rejected}
                records.write(json.dumps(row, separators=(",", ":")) + "\n")
                if t >= next_review:
                    # Unannotated frames permit independent framing review.
                    if not cv2.imwrite(str(output / f"frame_{t:010.3f}.jpg"), frame):
                        raise OSError("Could not write the analysis review frame")
                    next_review += review_every
                manifest["frames"] = index + 1
                if index % 50 == 0:
                    records.flush()
                    manifest["elapsed"] = time.monotonic() - started
                    save_manifest()
                    print(json.dumps({"time": t, "frames": index + 1,
                                      "elapsed": round(manifest["elapsed"], 2)}), flush=True)
        if not manifest["frames"]:
            raise RuntimeError("Hardware decoder produced no analysis frames")
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=f"{type(error).__name__}: {error}", elapsed=time.monotonic() - started)
        save_manifest()
        raise
    manifest.update(status="complete", elapsed=time.monotonic() - started)
    save_manifest()
    print(json.dumps(manifest), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--review-every", type=float, default=30)
    args = parser.parse_args()
    config = DecodeConfig(**json.loads(args.config.read_text()))
    analyze(config, args.models, args.output, gpu=args.gpu, review_every=args.review_every)


if __name__ == "__main__":
    main()
