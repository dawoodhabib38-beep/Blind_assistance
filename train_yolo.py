"""
Fine-tune YOLOv8 on the UNIVERSIA door dataset (closed_door / open_door).

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 50 --batch 8 --device cpu
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "yolov8n.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Train door detection model")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Base YOLO weights")
    parser.add_argument("--data", default=str(DATA_YAML), help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cpu", help="cpu, 0, cuda:0, etc.")
    parser.add_argument("--name", default="universia_doors")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset config not found: {args.data}")

    weights = args.weights
    if not Path(weights).exists():
        print(f"[train] Local weights not found at {weights}, using yolov8n.pt download")
        weights = "yolov8n.pt"

    print(f"[train] Data: {args.data}")
    print(f"[train] Weights: {weights}")
    print(f"[train] Device: {args.device}")

    model = YOLO(weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        pretrained=True,
        augment=True,
    )

    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print(f"\n[train] Done. Best weights: {best}")
    print(f"[train] Run app with: python main.py 0 {best}")
    return results


if __name__ == "__main__":
    main()
