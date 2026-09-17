from __future__ import annotations

import json
from pathlib import Path


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require_training_dependencies() -> None:
    try:
        import albumentations  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install a CUDA-compatible PyTorch build first, "
            "then run: pip install -r requirements.txt"
        ) from exc


def build_dataset(records: list[dict], training: bool):
    require_training_dependencies()
    import albumentations as A
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomScale(scale_limit=0.12, p=0.35),
        A.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.06, p=0.55),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=18, val_shift_limit=18, p=0.35),
        A.RandomGamma(gamma_limit=(80, 125), p=0.35),
    ] if training else []
    transform = A.Compose(transforms)

    class LawnDataset(Dataset):
        def __len__(self):
            return len(records)

        def __getitem__(self, index):
            record = records[index]
            image = cv2.cvtColor(cv2.imread(record["image"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(record["mask"], cv2.IMREAD_GRAYSCALE)
            if image is None or mask is None:
                raise FileNotFoundError(f"Unreadable sample: {record}")
            result = transform(image=image, mask=mask)
            image = result["image"].astype(np.float32) / 255.0
            image = (image - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
            return {
                "pixel_values": torch.from_numpy(image.transpose(2, 0, 1)),
                "labels": torch.from_numpy(result["mask"].astype(np.int64)),
                "record": record,
            }

    return LawnDataset()
