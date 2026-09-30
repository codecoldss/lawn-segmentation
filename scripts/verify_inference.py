"""Exercise single-image and directory inference using real validation images."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import cv2
    import numpy as np
    records = json.loads(args.samples_json.read_text())[:3]
    assert len(records) == 3
    inputs = args.output / "folder_inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(records[1:], 1):
        shutil.copyfile(record["image"], inputs / f"sample_{i}.png")
    cases = [("single", Path(records[0]["image"]), [Path(records[0]["image"]).stem]),
             ("folder", inputs, ["sample_1", "sample_2"])]
    report = {"passed": True, "cases": []}
    for mode, path, stems in cases:
        output = args.output / mode
        command = [sys.executable, str(Path(__file__).with_name("infer.py")), "--checkpoint", str(args.checkpoint),
                   "--input", str(path), "--output", str(output), "--device", args.device]
        subprocess.run(command, check=True)
        for stem in stems:
            mask = cv2.imread(str(output / f"{stem}_mask.png"), 0)
            traversable = cv2.imread(str(output / f"{stem}_traversable.png"), 0)
            assert mask.shape == (480, 640) and set(np.unique(mask)).issubset({0, 1, 2})
            assert np.array_equal(traversable, (mask == 1).astype(np.uint8) * 255)
            pixels = json.loads((output / f"{stem}_pixels.json").read_text())
            assert pixels["shape_hw"] == [480, 640]
            for value, name in enumerate(("background", "grass", "soil_grass")):
                coords = np.asarray(pixels["coordinates_yx"][name], dtype=np.int64).reshape(-1, 2)
                assert len(coords) == int((mask == value).sum())
                assert (mask[coords[:, 0], coords[:, 1]] == value).all()
                assert len(np.unique(coords, axis=0)) == len(coords)
            for suffix in ("color", "overlay"):
                assert cv2.imread(str(output / f"{stem}_{suffix}.png")).shape == (480, 640, 3)
        report["cases"].append({"mode": mode, "command": command, "image_count": len(stems),
                                "mask_shape": [480, 640], "labels": [0, 1, 2],
                                "all_pixel_coordinates_verified": True, "traversable_matches_grass": True})
    (args.output / "inference_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
