from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lawn_segmentation.constants import CLASS_NAMES, CLASS_PALETTE
from lawn_segmentation.data import build_dataset, read_manifest
from lawn_segmentation.metrics import confusion_matrix, summarize_confusion
from infer import load_model, run_one


def colorize(mask: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_PALETTE.items(): result[mask == class_id] = color
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--data", type=Path, default=Path("data/processed")); parser.add_argument("--output", type=Path, default=Path("artifacts/eval")); parser.add_argument("--visualize", type=int, default=30); parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import cv2
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    records = read_manifest(args.data / "val.jsonl")
    loader = DataLoader(build_dataset(records, training=False), batch_size=1, shuffle=False)
    model = load_model(args.checkpoint, args.device); matrix = np.zeros((3, 3), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(args.device)).logits
            prediction = F.interpolate(logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False).argmax(1).cpu().numpy()
            matrix += confusion_matrix(prediction, batch["labels"].numpy(), 3)
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = summarize_confusion(matrix, CLASS_NAMES)
    (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    visual_dir = args.output / "visual_review"
    for record in records[:args.visualize]:
        run_one(model, Path(record["image"]), visual_dir, args.device)
        truth = cv2.imread(record["mask"], cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(visual_dir / f"{Path(record['image']).stem}_truth_color.png"), cv2.cvtColor(colorize(truth), cv2.COLOR_RGB2BGR))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
