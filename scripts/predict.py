import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from qes6d import DEFAULT_WEIGHTS, QES6D


def parse_args():
    parser = argparse.ArgumentParser(description="Run QES6D prediction on one image")
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--weights", default=None, type=str)
    parser.add_argument("--output", default="", type=str)
    parser.add_argument("--device", default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    model = QES6D(weights=args.weights or DEFAULT_WEIGHTS, device=args.device)
    pitch, yaw, roll = model.predict(image)
    print(f"pitch={pitch:.4f}, yaw={yaw:.4f}, roll={roll:.4f}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        model.draw_axis(image, yaw, pitch, roll)
        cv2.imwrite(str(output), image)


if __name__ == "__main__":
    main()
