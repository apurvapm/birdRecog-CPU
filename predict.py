"""Run a trained checkpoint on a single image and print the top-5 species.

Usage:
    python predict.py path/to/bird.jpg
    python predict.py path/to/bird.jpg --topk 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

import config
from dataset import eval_transforms


def load_checkpoint(ckpt_path: Path):
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run `python train.py` first."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = ckpt["model_name"]
    num_classes = ckpt["num_classes"]
    class_names = ckpt["class_names"]

    factory = getattr(models, model_name)
    model = factory(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, class_names, ckpt.get("image_size", config.IMAGE_SIZE)


def predict(image_path: Path, model, class_names, image_size: int, topk: int = 5):
    tf = eval_transforms(image_size)
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)  # add batch dim
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(topk)
    return [
        (class_names[i.item()], float(p))
        for p, i in zip(top_probs, top_idx)
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("image", type=Path, help="Path to a JPG/PNG bird photo")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=config.CHECKPOINT_DIR / "best.pt",
        help="Path to the checkpoint file",
    )
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}", file=sys.stderr)
        return 1

    model, names, image_size = load_checkpoint(args.ckpt)
    results = predict(args.image, model, names, image_size, args.topk)

    print(f"\n{args.image.name}")
    print("-" * (len(args.image.name) + 2))
    for name, prob in results:
        bar = "#" * int(round(prob * 30))
        print(f"  {name:<35s} {prob:6.2%}  {bar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
