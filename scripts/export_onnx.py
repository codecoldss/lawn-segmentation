from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lawn_segmentation.constants import IMAGE_SIZE
from infer import load_model


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", type=Path, required=True); parser.add_argument("--output", type=Path, default=Path("artifacts/segformer_b0_3class_640x480.onnx")); args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    model = load_model(args.checkpoint, "cpu")
    class Exportable(torch.nn.Module):
        def __init__(self, wrapped): super().__init__(); self.wrapped = wrapped
        def forward(self, images): return F.interpolate(self.wrapped(pixel_values=images).logits, size=(IMAGE_SIZE[1], IMAGE_SIZE[0]), mode="bilinear", align_corners=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(Exportable(model), torch.zeros(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]), args.output, input_names=["images"], output_names=["logits"], opset_version=17)


if __name__ == "__main__": main()
