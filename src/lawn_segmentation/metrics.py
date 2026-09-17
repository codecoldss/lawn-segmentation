from __future__ import annotations

import numpy as np


def confusion_matrix(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    """Return rows=true class and columns=predicted class."""
    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    valid = (target >= 0) & (target < num_classes)
    encoded = num_classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def summarize_confusion(matrix: np.ndarray, class_names: dict[int, str]) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    total = int(matrix.sum())
    ious: dict[str, float | None] = {}
    for class_id, name in class_names.items():
        tp = int(matrix[class_id, class_id])
        denom = int(matrix[class_id, :].sum() + matrix[:, class_id].sum() - tp)
        ious[name] = None if denom == 0 else tp / denom
    present = [value for value in ious.values() if value is not None]
    return {
        "pixel_accuracy": None if total == 0 else int(np.trace(matrix)) / total,
        "iou": ious,
        "miou": None if not present else float(np.mean(present)),
        "confusion_matrix": matrix.tolist(),
    }
