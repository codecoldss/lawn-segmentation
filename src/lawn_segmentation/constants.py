from __future__ import annotations

from pathlib import Path

CLASS_NAMES = {0: "background", 1: "grass", 2: "soil_grass"}
CLASS_PALETTE = {0: (128, 0, 0), 1: (0, 128, 0), 2: (128, 128, 0)}
IMAGE_SIZE = (640, 480)  # width, height
VALID_MASK_VALUES = set(CLASS_NAMES)

IMAGE_ARCHIVES = {
    "1": "1.zip", "2": "2.zip", "3": "3.zip", "4": "4.zip",
    "5": "5.zip", "6": "6.zip", "7": "7.zip", "8": "8.zip",
    "supplement": "new_images_1002data (3).zip",
}
GRAY_LABEL_ARCHIVE = "labels_gray_1002data.zip"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
