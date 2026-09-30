"""Audit the effective dataset, re-evaluate the best checkpoint and export review cases."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lawn_segmentation.constants import CLASS_NAMES, CLASS_PALETTE
from lawn_segmentation.data import build_dataset, read_manifest
from lawn_segmentation.metrics import confusion_matrix, summarize_confusion
from infer import load_model


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def colorize(mask):
    colors = np.array([CLASS_PALETTE[i] for i in range(3)], np.uint8)
    return colors[mask][..., ::-1].copy()


def audit(train, val):
    import cv2
    counts = {"train": len(train), "validation": len(val)}
    names, hashes, invalid, duplicate_names, overlap_hashes = {}, {}, [], [], []
    for split, records in (("train", train), ("validation", val)):
        for record in records:
            name = Path(record["image"]).name
            if name in names:
                duplicate_names.append(name)
            names[name] = split
            image = cv2.imread(record["image"])
            mask = cv2.imread(record["mask"], cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None or image.shape[:2] != (480, 640) or mask.shape != (480, 640):
                invalid.append({"file": name, "reason": "unreadable or wrong dimensions"})
                continue
            values = np.unique(mask).tolist()
            if not set(values).issubset({0, 1, 2}):
                invalid.append({"file": name, "reason": "invalid labels", "labels": values})
            if Path(record["mask"]).name != name:
                invalid.append({"file": name, "reason": "image/mask filename mismatch"})
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            if digest in hashes and hashes[digest][0] != split:
                overlap_hashes.append([hashes[digest][1], name])
            hashes[digest] = (split, name)
    groups_train = {r["group"] for r in train}
    groups_val = {r["group"] for r in val}
    valid = not (invalid or duplicate_names or overlap_hashes or groups_train & groups_val)
    valid = valid and counts == {"train": 2652, "validation": 1244} and groups_val == {"2", "3", "4"}
    return {"effective_dataset_pass": valid, "counts": counts,
            "train_groups": sorted(groups_train), "validation_groups": sorted(groups_val),
            "invalid": invalid, "duplicate_names": duplicate_names, "cross_split_identical_images": overlap_hashes,
            "scope": "3896 effective pairs; the original 4145-pair requirement still has 249 empty masks"}


def select_cases(rows, count):
    selected, reasons = [], {}
    def add(index, reason):
        if index not in selected and len(selected) < count:
            selected.append(index)
            reasons[index] = reason
    for group in ("2", "3", "4"):
        indexes = [i for i, row in enumerate(rows) if row["group"] == group]
        for k in np.linspace(0, len(indexes) - 1, 4, dtype=int):
            add(indexes[k], "sequence_coverage")
    for key, number, descending, reason in (("grass_soil_confusion_rate", 6, True, "grass_soil_confusion"),
                                           ("grass_mean_brightness", 4, False, "dark_grass_candidate"),
                                           ("red_grass_fraction", 4, True, "red_grass_candidate"),
                                           ("bright_fraction", 4, True, "bright_scene_candidate")):
        eligible = [i for i, r in enumerate(rows) if r["grass_pixels"] > 1000]
        if key == "grass_soil_confusion_rate":
            eligible = [i for i in eligible if rows[i]["soil_pixels"] > 100]
        added = 0
        for i in sorted(eligible, key=lambda i: rows[i][key], reverse=descending):
            before = len(selected)
            add(i, reason)
            added += len(selected) - before
            if added >= number:
                break
    for i in sorted(range(len(rows)), key=lambda i: rows[i]["miou"]):
        add(i, "low_iou_candidate")
    return selected, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    import cv2
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    train, val = read_manifest(args.data / "train.jsonl"), read_manifest(args.data / "val.jsonl")
    audit_result = audit(train, val)
    write_json(args.output / "data_audit.json", audit_result)
    if not audit_result["effective_dataset_pass"]:
        raise SystemExit("Effective dataset audit failed; see data_audit.json")
    print(f"Effective dataset audited: {len(train)} train, {len(val)} validation", flush=True)
    model = load_model(args.checkpoint, args.device)
    loader = DataLoader(build_dataset(val, training=False), batch_size=4, shuffle=False, num_workers=2)
    matrix, rows, encoded_masks, cursor = np.zeros((3, 3), np.int64), [], [], 0
    with torch.inference_mode():
        for batch in loader:
            pred = F.interpolate(model(pixel_values=batch["pixel_values"].to(args.device)).logits,
                                 size=(480, 640), mode="bilinear", align_corners=False).argmax(1).cpu().numpy().astype(np.uint8)
            truths = batch["labels"].numpy()
            for mask, truth in zip(pred, truths):
                record = val[cursor]
                cm = confusion_matrix(mask, truth, 3)
                matrix += cm
                image = cv2.imread(record["image"])
                grass = truth == 1
                rgb = image[..., ::-1].astype(np.float32)
                brightness = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                metrics = summarize_confusion(cm, CLASS_NAMES)
                rows.append({"index": cursor, "name": Path(record["image"]).name, "group": record["group"],
                             "miou": metrics["miou"], "iou": metrics["iou"],
                             "grass_pixels": int(grass.sum()), "soil_pixels": int((truth == 2).sum()),
                             "grass_soil_confusion_pixels": int(cm[1, 2] + cm[2, 1]),
                             "grass_soil_confusion_rate": float((cm[1, 2] + cm[2, 1]) / max(1, cm[1:].sum())),
                             "grass_mean_brightness": float(brightness[grass].mean()) if grass.any() else 255.0,
                             "red_grass_fraction": float((rgb[..., 0][grass] > rgb[..., 1][grass] * 1.08).mean()) if grass.any() else 0.0,
                             "bright_fraction": float((brightness > 235).mean())})
                encoded_masks.append(cv2.imencode(".png", mask)[1])
                cursor += 1
            if cursor % 200 == 0 or cursor == len(val):
                print(f"Evaluated {cursor}/{len(val)}", flush=True)
    metrics = summarize_confusion(matrix, CLASS_NAMES)
    history = json.loads(args.history.read_text())
    best = max(history, key=lambda row: row["miou"])
    metrics.update({"samples": len(val), "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    "reference_epoch": best["epoch"], "reference_miou": best["miou"],
                    "matches_training_confusion_matrix": metrics["confusion_matrix"] == best["confusion_matrix"]})
    write_json(args.output / "metrics.json", metrics)
    write_json(args.output / "per_sample_metrics.json", rows)
    selected, reasons = select_cases(rows, args.samples)
    visual = args.output / "visual_review"
    visual.mkdir(exist_ok=True)
    review, tiles = [], []
    for ordinal, index in enumerate(selected, 1):
        record, row = val[index], rows[index]
        image, truth = cv2.imread(record["image"]), cv2.imread(record["mask"], 0)
        mask = cv2.imdecode(encoded_masks[index], cv2.IMREAD_GRAYSCALE)
        prediction = colorize(mask)
        overlay = cv2.addWeighted(image, 0.55, prediction, 0.45, 0)
        stem = f"{ordinal:02d}"
        for suffix, value in (("original", image), ("truth", colorize(truth)), ("prediction", prediction),
                              ("overlay", overlay), ("mask", mask), ("traversable", ((mask == 1) * 255).astype(np.uint8))):
            cv2.imwrite(str(visual / f"{stem}_{suffix}.png"), value)
        # Keep columns aligned at half resolution for readable six-case contact sheets.
        tile = np.full((280, 1280, 3), 245, np.uint8)
        text = f"{ordinal:02d} group={row['group']} mIoU={row['miou']:.3f} grass<->soil={row['grass_soil_confusion_pixels']} {reasons[index]}"
        cv2.putText(tile, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        for column, value in enumerate((image, colorize(truth), prediction, overlay)):
            tile[40:280, column * 320:(column + 1) * 320] = cv2.resize(value, (320, 240), interpolation=cv2.INTER_NEAREST if column in (1, 2) else cv2.INTER_AREA)
        tiles.append(tile)
        review.append({"review_id": stem, **row, "reason": reasons[index], "image": record["image"], "mask": record["mask"]})
    write_json(args.output / "selected_samples.json", review)
    confusion_cases = sorted(rows, key=lambda row: row["grass_soil_confusion_rate"], reverse=True)[:30]
    write_json(args.output / "grass_soil_confusion_cases.json", confusion_cases)
    for offset in range(0, len(tiles), 6):
        header = np.full((45, 1280, 3), 255, np.uint8)
        for col, title in enumerate(("ORIGINAL", "TRUTH", "PREDICTION", "OVERLAY")):
            cv2.putText(header, title, (col * 320 + 15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1)
        cv2.imwrite(str(args.output / f"review_sheet_{offset // 6 + 1:02d}.jpg"), np.vstack([header, *tiles[offset:offset+6]]), [cv2.IMWRITE_JPEG_QUALITY, 95])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history]); axes[0].set(xlabel="Epoch", ylabel="Training loss")
    axes[1].plot(epochs, [row["miou"] for row in history]); axes[1].scatter([best["epoch"]], [best["miou"]], color="red", label=f"Best: epoch {best['epoch']}")
    axes[1].set(xlabel="Epoch", ylabel="Validation mIoU"); axes[1].legend()
    fig.tight_layout(); fig.savefig(args.output / "training_curve.png", dpi=160); plt.close(fig)
    assert metrics["matches_training_confusion_matrix"], "Best checkpoint re-evaluation differs from archived training confusion matrix"
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
