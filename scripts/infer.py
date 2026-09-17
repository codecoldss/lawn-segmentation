from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lawn_segmentation.constants import CLASS_NAMES, CLASS_PALETTE


def load_model(checkpoint: Path, device: str):
    import torch
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    state = torch.load(checkpoint, map_location=device)
    config = SegformerConfig.from_pretrained(state["config"]["model_name"], num_labels=3, id2label={i: name for i, name in CLASS_NAMES.items()}, label2id={name: i for i, name in CLASS_NAMES.items()})
    model = SegformerForSemanticSegmentation(config); model.load_state_dict(state["state_dict"]); return model.to(device).eval()


def run_one(model, image_path: Path, output: Path, device: str) -> None:
    import cv2
    import torch
    import torch.nn.functional as F
    bgr = cv2.imread(str(image_path));
    if bgr is None: raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    with torch.no_grad():
        logits = model(pixel_values=torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)).logits
        mask = F.interpolate(logits, size=rgb.shape[:2], mode="bilinear", align_corners=False).argmax(1)[0].cpu().numpy().astype(np.uint8)
    color = np.zeros_like(rgb, dtype=np.uint8)
    for class_id, color_value in CLASS_PALETTE.items(): color[mask == class_id] = color_value
    output.mkdir(parents=True, exist_ok=True); stem = image_path.stem
    cv2.imwrite(str(output / f"{stem}_mask.png"), mask)
    cv2.imwrite(str(output / f"{stem}_color.png"), cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output / f"{stem}_overlay.png"), cv2.addWeighted(bgr, 0.55, cv2.cvtColor(color, cv2.COLOR_RGB2BGR), 0.45, 0))
    cv2.imwrite(str(output / f"{stem}_traversable.png"), ((mask == 1).astype(np.uint8) * 255))
    coords = {name: np.argwhere(mask == class_id).tolist() for class_id, name in CLASS_NAMES.items()}
    (output / f"{stem}_pixels.json").write_text(json.dumps({"shape_hw": list(mask.shape), "coordinates_yx": coords}, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--input", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("artifacts/infer")); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); model = load_model(args.checkpoint, args.device)
    paths = [args.input] if args.input.is_file() else sorted(path for path in args.input.rglob("*.png"))
    if not paths: raise SystemExit("No PNG inputs found")
    for path in paths: run_one(model, path, args.output, args.device)


if __name__ == "__main__": main()
