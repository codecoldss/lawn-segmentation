"""Check the ONNX graph and compare real-image logits with the saved PyTorch model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from infer import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--samples-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    import cv2
    import onnx
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    onnx.checker.check_model(onnx.load(args.onnx))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), options, providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    assert inp.name == "images" and inp.type == "tensor(float)" and inp.shape == [1, 3, 480, 640]
    assert out.name == "logits" and out.type == "tensor(float)" and out.shape == [1, 3, 480, 640]
    model = load_model(args.checkpoint, args.device)
    rows = []
    for record in json.loads(args.samples_json.read_text())[:args.samples]:
        rgb = cv2.cvtColor(cv2.imread(record["image"]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        images = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
        actual = session.run(["logits"], {"images": images})[0]
        with torch.inference_mode():
            reference = F.interpolate(model(pixel_values=torch.from_numpy(images).to(args.device)).logits,
                                      size=(480, 640), mode="bilinear", align_corners=False).cpu().numpy()
        delta = np.abs(actual - reference)
        agreement = float((actual.argmax(1) == reference.argmax(1)).mean())
        finite = bool(np.isfinite(actual).all() and np.isfinite(reference).all())
        passed = finite and bool(np.allclose(actual, reference, atol=5e-4, rtol=1e-3)) and agreement >= 0.9999
        rows.append({"name": record["name"], "max_absolute_logit_error": float(delta.max()),
                     "mean_absolute_logit_error": float(delta.mean()), "mask_pixel_agreement": agreement, "passed": passed})
        print(f"{record['review_id']}: max_error={delta.max():.6g}, agreement={agreement:.8f}", flush=True)
    report = {"passed": bool(rows) and all(r["passed"] for r in rows), "graph_checker_passed": True,
              "input": {"name": inp.name, "shape": inp.shape, "type": inp.type},
              "output": {"name": out.name, "shape": out.shape, "type": out.type},
              "comparison": "PyTorch CUDA vs ONNX Runtime CPU; real validation images",
              "thresholds": {"logits_atol": 5e-4, "logits_rtol": 1e-3, "minimum_mask_agreement": 0.9999},
              "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(), "samples": rows,
              "onnxruntime_version": ort.__version__, "rknn_board_validation": "not_performed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert report["passed"], "ONNX numerical comparison failed; inspect report"


if __name__ == "__main__":
    main()
