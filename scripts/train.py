from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lawn_segmentation.constants import CLASS_NAMES
from lawn_segmentation.data import build_dataset, read_manifest, require_training_dependencies
from lawn_segmentation.metrics import confusion_matrix, summarize_confusion


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    import torch
    import torch.nn.functional as F
    matrix = np.zeros((3, 3), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(pixel_values=batch["pixel_values"].to(device)).logits
            logits = F.interpolate(logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False)
            matrix += confusion_matrix(logits.argmax(1).cpu().numpy(), batch["labels"].numpy(), 3)
    return summarize_confusion(matrix, CLASS_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the three-class SegFormer-B0 baseline.")
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--config", type=Path, default=Path("configs/segformer_b0_3class.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/train"))
    args = parser.parse_args()
    require_training_dependencies()
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import SegformerForSemanticSegmentation
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this baseline. Prepare/audit data on CPU, then train on a CUDA host.")
    seed_everything(config["seed"])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train_loader = DataLoader(build_dataset(read_manifest(args.data / "train.jsonl"), training=True), batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], pin_memory=True)
    val_loader = DataLoader(build_dataset(read_manifest(args.data / "val.jsonl"), training=False), batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"], pin_memory=True)
    model = SegformerForSemanticSegmentation.from_pretrained(config["model_name"], num_labels=3, id2label={i: name for i, name in CLASS_NAMES.items()}, label2id={name: i for i, name in CLASS_NAMES.items()}, ignore_mismatched_sizes=True).cuda()
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_miou = -1.0
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train(); losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            result = model(pixel_values=batch["pixel_values"].cuda(non_blocking=True), labels=batch["labels"].cuda(non_blocking=True))
            result.loss.backward(); optimizer.step(); losses.append(float(result.loss.detach().cpu()))
        metrics = evaluate(model, val_loader, "cuda")
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **metrics}; history.append(row)
        print(json.dumps(row, ensure_ascii=False)); (args.output / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        if metrics["miou"] is not None and metrics["miou"] > best_miou:
            best_miou = metrics["miou"]
            torch.save({"state_dict": model.state_dict(), "config": config, "metrics": metrics}, args.output / "best.pt")


if __name__ == "__main__":
    main()
