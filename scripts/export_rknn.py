"""RK3588 non-quantized conversion entry; requires separately installed RKNN Toolkit2."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.onnx.is_file():
        raise SystemExit(f"ONNX not found: {args.onnx}")
    try:
        from rknn.api import RKNN
    except ImportError:
        raise SystemExit("Install the matching RKNN Toolkit2 in a separate conversion environment")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=True)
    try:
        # Input already uses ImageNet normalization; preserve the ONNX input contract.
        operations = [
            ("config", lambda: rknn.config(target_platform="rk3588", mean_values=[[0, 0, 0]], std_values=[[1, 1, 1]])),
            ("load_onnx", lambda: rknn.load_onnx(model=str(args.onnx))),
            ("build", lambda: rknn.build(do_quantization=False)),
            ("export_rknn", lambda: rknn.export_rknn(str(args.output))),
        ]
        for name, operation in operations:
            result = operation()
            if result != 0:
                raise SystemExit(f"RKNN {name} failed: {result}")
        print(f"RKNN exported: {args.output}; board accuracy and runtime remain unverified")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
