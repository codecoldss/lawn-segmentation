"""Produce a dataset report and plots from the read-only ZIP scan, without changing labels."""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--plot-deps", type=Path)
    args = parser.parse_args()
    if args.plot_deps:
        sys.path.insert(0, str(args.plot_deps.resolve()))
    import numpy as np
    from PIL import Image, ImageDraw
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    root, out = args.root.resolve(), args.analysis.resolve()
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    duplicates = json.loads((out / "duplicate_groups.json").read_text(encoding="utf-8"))
    with (out / "file_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for r in rows:
        r["usable"] = r["usable"] == "True"
        r["pixels"] = ast.literal_eval(r["pixels"])
    valid = [r for r in rows if r["usable"]]
    train = [r for r in valid if r["split"] == "train"]
    val = [r for r in valid if r["split"] == "validation"]
    # A second, explicitly bounded perceptual check across every effective split pair.
    train_hashes = [(r, int(r["dhash64"], 16)) for r in train]
    cross_pairs = []
    for b in val:
        bh = int(b["dhash64"], 16)
        for a, ah in train_hashes:
            distance = (ah ^ bh).bit_count()
            if distance <= 3:
                cross_pairs.append({"train_group": a["group"], "train_name": a["name"],
                                    "validation_group": b["group"], "validation_name": b["name"],
                                    "hamming_distance": distance})
    cross_pairs.sort(key=lambda r: r["hamming_distance"])
    (out / "cross_split_dhash_candidates.json").write_text(json.dumps(cross_pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    near = json.loads((out / "near_duplicate_candidates.json").read_text(encoding="utf-8"))
    summary["cross_split_dhash"] = {"compared_pairs": len(train) * len(val), "threshold": 3,
                                      "candidate_pairs": len(cross_pairs), "interpretation": "candidates only; not confirmed leakage or exhaustive perceptual matching"}
    summary["adjacent_dhash_unique_images"] = len({(p["group"], p[k]) for p in near for k in ("first", "second")})
    capture_dates = Counter()
    for r in rows:
        match = re.search(r"_(20\d{6})_\d{6}_", r["name"])
        capture_dates[match.group(1) if match else "unknown"] += 1
    summary["filename_capture_date_counts"] = dict(capture_dates)
    summary["image_dimensions_wh"] = dict(Counter("x".join(map(str, ast.literal_eval(r["size_wh"]))) for r in rows))
    summary["image_modes"] = dict(Counter(r["image_mode"] for r in rows))
    summary["valid_mask_values"] = sorted({v for r in valid for v in ast.literal_eval(r["mask_values"])})
    quality = []
    label_zip = zipfile.ZipFile(root / "labels_gray_1002data.zip")
    archive_cache = {}
    def read_image(row):
        if row["archive"] not in archive_cache:
            archive_cache[row["archive"]] = zipfile.ZipFile(root / row["archive"])
        return Image.open(io.BytesIO(archive_cache[row["archive"]].read(row["member"]))).convert("RGB")
    for r in valid:
        dark_frame = float(r["mean_brightness"]) < 15 and float(r["dark_pixel_fraction"]) > .95
        single_background = r["pixels"][0] == 480 * 640
        if dark_frame or single_background:
            rgb = np.asarray(read_image(r))
            quality.append({"group": r["group"], "name": r["name"], "split": r["split"], "pixels": r["pixels"],
                            "mean_brightness": float(r["mean_brightness"]), "rgb_min": int(rgb.min()),
                            "rgb_max": int(rgb.max()), "channel_standard_deviation": rgb.std(axis=(0, 1)).tolist(),
                            "status": "suspected_uninformative_dark_frame" if dark_frame else "all_background_mask_requires_visual_review"})
    previous = json.loads((root / "results/segformer_b0_3class_20260928/acceptance_20260930/per_sample_metrics.json").read_text(encoding="utf-8"))
    previous_by_name = {r["name"]: r for r in previous}
    for q in quality:
        if q["name"] in previous_by_name:
            q["archived_baseline_metrics"] = previous_by_name[q["name"]]
    summary["quality_candidates"] = quality
    summary["single_class_masks"] = {str(i): sum(r["pixels"][i] == 480 * 640 for r in valid) for i in range(3)}
    (out / "quality_candidates.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.exists() else None
    class_names = ["背景", "草地", "泥土草"] if font else ["background", "grass", "soil_grass"]
    def label(ax, text, axis="title"):
        getattr(ax, "set_" + axis)(text, fontproperties=font)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    keys = ["background", "grass", "soil_grass"]
    colors = ["#a45353", "#369960", "#b9a63d"]
    values = [summary["overall"]["pixel_fraction"][k] * 100 for k in keys]
    bars = axes[0].bar(range(3), values, color=colors)
    axes[0].set_xticks(range(3), class_names, fontproperties=font)
    label(axes[0], "有效3896张：类别像素占比" if font else "Class pixel distribution: 3896 usable images")
    label(axes[0], "占全部有效像素 (%)" if font else "Share of usable pixels (%)", "ylabel")
    axes[0].set_ylim(0, 62)
    for b, v in zip(bars, values):
        axes[0].text(b.get_x() + b.get_width()/2, v + 1, f"{v:.2f}%", ha="center")
    soil = [summary["splits"][s]["pixel_fraction"]["soil_grass"] * 100 for s in ["train", "validation"]]
    axes[1].bar(range(2), soil, color=["#477bbc", "#dc9146"])
    axes[1].set_xticks(range(2), ["训练集2652张", "验证集1244张"] if font else ["Train: 2652", "Validation: 1244"], fontproperties=font)
    label(axes[1], "泥土草占比在训练/验证之间不同" if font else "Soil-class fraction differs across splits")
    label(axes[1], "占各自集合像素 (%)" if font else "Share of split pixels (%)", "ylabel")
    axes[1].set_ylim(0, 4)
    for i, v in enumerate(soil):
        axes[1].text(i, v + .08, f"{v:.2f}%", ha="center")
    fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=170); plt.close(fig)
    groups = list(summary["groups"])
    fig, ax = plt.subplots(figsize=(11, 4))
    usable = [summary["groups"][g]["usable"] for g in groups]
    empty = [summary["groups"][g]["empty_masks"] for g in groups]
    ax.bar(groups, usable, label="有效图像标签对" if font else "Usable pairs", color="#477bbc")
    ax.bar(groups, empty, bottom=usable, label="空标签" if font else "Empty masks", color="#cd6659")
    ax.legend(prop=font); label(ax, "采集组数量：249个空标签全部位于补充包" if font else "Images by group; all 249 empty masks in supplement")
    label(ax, "图片数量" if font else "Image count", "ylabel")
    for i, (a, b) in enumerate(zip(usable, empty)):
        ax.text(i, a + b + 8, str(a+b), ha="center")
    fig.tight_layout(); fig.savefig(out / "group_distribution.png", dpi=170); plt.close(fig)
    lookup = {(r["group"], r["name"]): r for r in rows}
    quality_rows = [lookup[(r["group"], r["name"])] for r in quality]
    for g in duplicates["nonempty_mask_file_md5"]:
        for r in g["members"]:
            candidate = lookup[(r["group"], r["name"])]
            if not any(x["name"] == candidate["name"] for x in quality_rows):
                quality_rows.append(candidate)
    if quality_rows:
        sheet = Image.new("RGB", (960, 270 * len(quality_rows)), "white")
        draw = ImageDraw.Draw(sheet)
        palette = np.array([[128, 0, 0], [0, 128, 0], [128, 128, 0]], dtype=np.uint8)
        for i, r in enumerate(quality_rows):
            image = read_image(r)
            mask = np.asarray(Image.open(io.BytesIO(label_zip.read(r["mask_member"]))))
            truth = Image.fromarray(palette[mask])
            draw.text((5, i * 270 + 5), f"group={r['group']} labels={r['pixels']} | ORIGINAL / TRUTH / OVERLAY", fill="black")
            for col, im in enumerate((image, truth, Image.blend(image, truth, .45))):
                sheet.paste(im.resize((320, 240)), (col * 320, i * 270 + 30))
        sheet.save(out / "quality_review.jpg", quality=95)
    if cross_pairs:
        sheet = Image.new("RGB", (640, 270 * min(8, len(cross_pairs))), "white")
        draw = ImageDraw.Draw(sheet)
        for i, p in enumerate(cross_pairs[:8]):
            draw.text((5, i*270+5), f"train group={p['train_group']} / val group={p['validation_group']}; distance={p['hamming_distance']}", fill="black")
            for col, row in enumerate((lookup[(p["train_group"], p["train_name"])], lookup[(p["validation_group"], p["validation_name"])])):
                sheet.paste(read_image(row).resize((320, 240)), (col*320, i*270+30))
        sheet.save(out / "cross_split_dhash_examples.jpg", quality=92)
    for z in archive_cache.values():
        z.close()
    label_zip.close()
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"cross_split_dhash": summary["cross_split_dhash"], "quality_candidates": quality,
                      "single_class_masks": summary["single_class_masks"], "capture_dates": capture_dates}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
