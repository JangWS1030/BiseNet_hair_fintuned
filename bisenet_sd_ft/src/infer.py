from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image inference for BiSeNet.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--output-size", type=int, default=1024)
    return parser.parse_args()


def preprocess_image(image_path: str, input_size: int) -> tuple[torch.Tensor, np.ndarray]:
    from src.utils.common import IMAGENET_MEAN, IMAGENET_STD

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0), image_rgb


def save_outputs(image_rgb: np.ndarray, pred: np.ndarray, output_path: str) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hair_binary = np.where(pred == 10, 255, 0).astype(np.uint8)
    overlay = image_rgb.copy()
    overlay[pred == 10] = np.array([255, 80, 0], dtype=np.uint8)
    overlay[pred == 1] = np.array([0, 180, 255], dtype=np.uint8)
    blended = cv2.addWeighted(image_rgb, 0.45, overlay, 0.55, 0)

    cv2.imwrite(str(out_path), pred.astype(np.uint8))
    cv2.imwrite(str(out_path.with_name(out_path.stem + "_hair.png")), hair_binary)
    cv2.imwrite(str(out_path.with_name(out_path.stem + "_overlay.png")), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


def main() -> None:
    args = parse_args()
    from src.models import BiSeNet
    from src.utils.checkpoint import load_model_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiSeNet(n_classes=16, pretrained_backbone=False).to(device)
    load_model_checkpoint(model, args.checkpoint, strict=False, allow_partial=True)
    model.eval()

    tensor, image_rgb = preprocess_image(args.image, input_size=args.input_size)
    with torch.no_grad():
        logits, _, _ = model(tensor.to(device))
        logits = torch.nn.functional.interpolate(
            logits,
            size=(args.output_size, args.output_size),
            mode="bilinear",
            align_corners=True,
        )
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    resized_rgb = cv2.resize(image_rgb, (args.output_size, args.output_size), interpolation=cv2.INTER_LINEAR)
    save_outputs(image_rgb=resized_rgb, pred=pred, output_path=args.out)


if __name__ == "__main__":
    main()
