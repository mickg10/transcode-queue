"""Run hardware-decoded samples and write bounds plus annotated review frames."""
import argparse
import dataclasses
import json
from pathlib import Path

import cv2

from .core import DecodeConfig, frames
from .detectors import Performers, stage_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON DecodeConfig, including an optional SSH/Docker prefix")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="Require CUDA person inference")
    args = parser.parse_args()
    config = DecodeConfig(**json.loads(args.config.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    detector = Performers(args.models, gpu=args.gpu)
    records = []
    for t, frame in frames(config):
        detections = detector.detect(frame)
        accepted, rejected = stage_candidates(detections)
        row = {"time": t, **detections, "stage_candidates": accepted, "rejected_candidates": rejected}
        records.append(row)
        for person in detections["people"]:
            color = (100, 230, 120) if person in accepted else (100, 120, 230)
            x1, y1, x2, y2 = person["box"]
            a = (round(x1 * config.width), round(y1 * config.height))
            b = (round(x2 * config.width), round(y2 * config.height))
            cv2.rectangle(frame, a, b, color, 2)
            cv2.putText(frame, f'{person["confidence"]:.2f}', a, cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        for face in detections["faces"]:
            x1, y1, x2, y2 = face["box"]
            cv2.rectangle(frame, (round(x1 * config.width), round(y1 * config.height)),
                          (round(x2 * config.width), round(y2 * config.height)), (255, 190, 80), 1)
        cv2.imwrite(str(args.output / f"frame_{t:010.3f}.jpg"), frame)
        print(json.dumps({"time": t, "people": len(accepted), "faces": len(detections["faces"])}), flush=True)
    (args.output / "detections.json").write_text(json.dumps(
        {"decode": dataclasses.asdict(config), "records": records}, indent=2))


if __name__ == "__main__":
    main()
