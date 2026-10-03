"""Read-only source ZIP audit: MD5 duplicates, split leakage and class/scene statistics."""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ARCHIVES = {str(i): f"{i}.zip" for i in range(1, 9)} | {"supplement": "new_images_1002data (3).zip"}
VAL_GROUPS = {"2", "3", "4"}
CLASS_NAMES = ["background", "grass", "soil_grass"]


def md5(data):
    return hashlib.md5(data).hexdigest()


def pixel_hash(array):
    return md5(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def duplicate_groups(records, key, valid_only=False):
    groups = defaultdict(list)
    for record in records:
        if record.get(key) and (not valid_only or record["usable"]):
            groups[record[key]].append(record)
    result = []
    for digest, items in groups.items():
        if len(items) > 1:
            splits = {r["split"] for r in items if r["usable"]}
            result.append({"md5": digest, "count": len(items), "redundant_copies": len(items) - 1,
                           "cross_effective_split": {"train", "validation"}.issubset(splits),
                           "members": [{k: r[k] for k in ("group", "name", "archive", "member", "split", "usable")} for r in items]})
    return sorted(result, key=lambda r: (-r["count"], r["md5"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--date", default=datetime.date.today().isoformat(), help="Report date (YYYY-MM-DD)")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    label_records, masks, issues = {}, {}, []
    with __import__("zipfile").ZipFile(root / "labels_gray_1002data.zip") as z:
        for info in z.infolist():
            if not info.filename.lower().endswith(".png"):
                continue
            name, raw = Path(info.filename).name, z.read(info)
            if name in label_records:
                issues.append({"name": name, "issue": "duplicate_mask_filename"})
            record = {"mask_member": info.filename, "mask_bytes": len(raw), "mask_file_md5": md5(raw),
                      "mask_pixel_md5": None, "mask_valid": False, "pixels": [0, 0, 0]}
            if raw:
                try:
                    with Image.open(io.BytesIO(raw)) as im:
                        a = np.array(im)
                    values, counts = np.unique(a, return_counts=True)
                    record.update(mask_shape=list(a.shape), mask_values=values.tolist(), mask_pixel_md5=pixel_hash(a))
                    record["mask_valid"] = a.shape == (480, 640) and set(values.tolist()).issubset({0, 1, 2})
                    record["pixels"] = [int(counts[values == i].sum()) for i in range(3)]
                    if record["mask_valid"]:
                        masks[name] = a
                    else:
                        issues.append({"name": name, "issue": "invalid_mask", "shape": list(a.shape), "values": values.tolist()})
                except Exception as exc:
                    issues.append({"name": name, "issue": "mask_decode_error", "error": str(exc)})
            label_records[name] = record
    print(f"Masks read: {len(label_records)}, valid={len(masks)}", flush=True)

    def scan(group, archive):
        records, errors = [], []
        with __import__("zipfile").ZipFile(root / archive) as z:
            for info in z.infolist():
                if not info.filename.lower().endswith(".png"):
                    continue
                name, raw = Path(info.filename).name, z.read(info)
                row = {"group": group, "name": name, "archive": archive, "member": info.filename,
                       "split": "validation" if group in VAL_GROUPS else "train", "image_bytes": len(raw),
                       "image_file_md5": md5(raw), "image_pixel_md5": None, "usable": False,
                       **label_records.get(name, {"mask_valid": False, "mask_bytes": None, "pixels": [0, 0, 0]})}
                try:
                    with Image.open(io.BytesIO(raw)) as im:
                        im.load()
                        row["image_mode"], row["size_wh"] = im.mode, list(im.size)
                        rgb = im.convert("RGB")
                        a = np.asarray(rgb)
                        row["image_pixel_md5"] = pixel_hash(a)
                        row["usable"] = im.size == (640, 480) and row["mask_valid"]
                        gray = np.asarray(rgb.convert("L"))
                        small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
                        bits = (small[:, 1:] > small[:, :-1]).ravel()
                        row["dhash64"] = f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"
                        row["mean_brightness"] = float(gray.mean())
                        row["dark_pixel_fraction"] = float((gray < 40).mean())
                        row["bright_pixel_fraction"] = float((gray > 235).mean())
                        if row["usable"]:
                            grass = masks[name] == 1
                            row["grass_mean_brightness"] = float(gray[grass].mean()) if grass.any() else None
                            r, g = a[..., 0].astype(np.float32), a[..., 1].astype(np.float32)
                            row["red_grass_fraction"] = float((r[grass] > 1.08 * g[grass]).mean()) if grass.any() else None
                    if row["size_wh"] != [640, 480]:
                        errors.append({"name": name, "issue": "image_wrong_dimensions"})
                except Exception as exc:
                    errors.append({"name": name, "issue": "image_decode_error", "error": str(exc)})
                if name not in label_records:
                    errors.append({"name": name, "issue": "missing_mask"})
                records.append(row)
        return records, errors

    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(scan, g, a): g for g, a in ARCHIVES.items()}
        for future in as_completed(futures):
            rows, errors = future.result()
            records.extend(rows); issues.extend(errors)
            print(f"Group {futures[future]}: {len(rows)} images scanned", flush=True)
    records.sort(key=lambda r: (r["group"], r["name"], r["member"]))
    names = Counter(r["name"] for r in records)
    orphan_masks = sorted(set(label_records) - set(names))
    for name in orphan_masks:
        issues.append({"name": name, "issue": "mask_without_image"})
    duplicate_names = {n: c for n, c in names.items() if c > 1}
    dup = {"image_file_md5": duplicate_groups(records, "image_file_md5"),
           "image_decoded_pixel_md5": duplicate_groups(records, "image_pixel_md5"),
           "nonempty_mask_file_md5": duplicate_groups([r for r in records if r.get("mask_bytes")], "mask_file_md5"),
           "mask_decoded_pixel_md5": duplicate_groups(records, "mask_pixel_md5")}
    for r in records:
        r["pair_md5"] = md5((r["image_pixel_md5"] + r["mask_pixel_md5"]).encode()) if r["usable"] else None
    dup["decoded_image_mask_pair_md5"] = duplicate_groups(records, "pair_md5", valid_only=True)
    dup_summary = {key: {"groups": len(groups), "members": sum(g["count"] for g in groups),
                         "redundant_copies": sum(g["redundant_copies"] for g in groups),
                         "cross_effective_split_groups": sum(g["cross_effective_split"] for g in groups)} for key, groups in dup.items()}
    conflicts = []
    for group in dup["image_decoded_pixel_md5"]:
        matched = [r for r in records if r["image_pixel_md5"] == group["md5"] and r["usable"]]
        if len({r["mask_pixel_md5"] for r in matched}) > 1:
            conflicts.append(group)

    def aggregate(rows):
        valid = [r for r in rows if r["usable"]]
        pixels = np.array([r["pixels"] for r in valid], np.int64).reshape(-1, 3)
        sums = pixels.sum(axis=0)
        total = int(sums.sum())
        presence = (pixels > 0).sum(axis=0)
        return {"images": len(rows), "usable": len(valid), "empty_masks": sum(r.get("mask_bytes") == 0 for r in rows),
                "invalid_or_missing": len(rows) - len(valid),
                "pixels": {name: int(sums[i]) for i, name in enumerate(CLASS_NAMES)},
                "pixel_fraction": {name: float(sums[i] / total) if total else 0 for i, name in enumerate(CLASS_NAMES)},
                "images_with_class": {name: int(presence[i]) for i, name in enumerate(CLASS_NAMES)},
                "soil_fraction_quantiles": dict(zip(["min", "p25", "median", "p75", "p90", "max"],
                     np.quantile(pixels[:, 2] / (480 * 640), [0, .25, .5, .75, .9, 1]).tolist())) if len(valid) else {}}
    adjacent = []
    for group in ARCHIVES:
        ordered = sorted([r for r in records if r["group"] == group and r.get("dhash64")], key=lambda r: r["name"])
        for a, b in zip(ordered, ordered[1:]):
            distance = (int(a["dhash64"], 16) ^ int(b["dhash64"], 16)).bit_count()
            if distance <= 3:
                adjacent.append({"group": group, "first": a["name"], "second": b["name"], "hamming_distance": distance,
                                 "first_usable": a["usable"], "second_usable": b["usable"],
                                 "same_image_md5": a["image_file_md5"] == b["image_file_md5"]})
    usable = [r for r in records if r["usable"]]
    candidates = {"dark_grass": sorted([r for r in usable if r.get("grass_mean_brightness") is not None], key=lambda r: r["grass_mean_brightness"])[:12],
                  "bright_scene": sorted(usable, key=lambda r: r["bright_pixel_fraction"], reverse=True)[:12],
                  "red_grass": sorted([r for r in usable if r.get("red_grass_fraction") is not None], key=lambda r: r["red_grass_fraction"], reverse=True)[:12],
                  "soil_rich": sorted(usable, key=lambda r: r["pixels"][2], reverse=True)[:12]}
    summary = {"date": args.date, "source": "immutable original image and grayscale-mask ZIP entries; no data modified",
               "overall": aggregate(records), "groups": {g: aggregate([r for r in records if r["group"] == g]) for g in ARCHIVES},
               "splits": {s: aggregate([r for r in records if r["split"] == s]) for s in ("train", "validation")},
               "duplicates": dup_summary, "conflicting_masks_for_identical_images": len(conflicts),
               "duplicate_filenames": duplicate_names, "issues": issues,
               "adjacent_dhash_candidates": len(adjacent), "adjacent_dhash_threshold": 3,
               "adjacent_dhash_by_group": dict(Counter(r["group"] for r in adjacent)),
               "definitions": {"MD5": "exact file or decoded-pixel equality; not a near-duplicate test",
                   "dHash": "64-bit 9x8 grayscale horizontal difference hash, consecutive filename-ordered images only; candidates require human review",
                   "mask_repeats": "identical masks alone do not establish duplicate images; 0-byte masks excluded from mask duplicate groups"}}
    write_json(output / "summary.json", summary)
    write_json(output / "duplicate_groups.json", dup)
    write_json(output / "conflicting_labels.json", conflicts)
    write_json(output / "near_duplicate_candidates.json", adjacent)
    write_json(output / "scene_candidates.json", candidates)
    write_json(output / "empty_masks.json", [{k: r[k] for k in ("group", "name", "archive", "member", "split")} for r in records if r.get("mask_bytes") == 0])
    fields = sorted({k for r in records for k in r})
    with (output / "file_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(records)
    summary["manifest_sha256"] = hashlib.sha256((output / "file_manifest.csv").read_bytes()).hexdigest()
    write_json(output / "summary.json", summary)
    # Actual image contact sheets, without automatic deletion or relabeling.
    cache = {}
    def thumbnail(row):
        if row["archive"] not in cache:
            cache[row["archive"]] = __import__("zipfile").ZipFile(root / row["archive"])
        return Image.open(io.BytesIO(cache[row["archive"]].read(row["member"]))).convert("RGB").resize((320, 240))
    for category, items in candidates.items():
        sheet = Image.new("RGB", (1280, 3 * 270), "white")
        draw = ImageDraw.Draw(sheet)
        for index, r in enumerate(items):
            x, y = (index % 4) * 320, (index // 4) * 270
            sheet.paste(thumbnail(r), (x, y + 30))
            draw.text((x + 5, y + 5), f"{index+1:02} group={r['group']} {r['split']} {category}", fill="black")
        sheet.save(output / f"scene_{category}.jpg", quality=92)
    lookup = {(r["group"], r["name"]): r for r in records}
    chosen = []
    for g in ARCHIVES:
        subset = [r for r in adjacent if r["group"] == g]
        chosen.extend(sorted(subset, key=lambda r: r["hamming_distance"])[:2])
    chosen = chosen[:12]
    if chosen:
        sheet = Image.new("RGB", (640, len(chosen) * 270), "white")
        draw = ImageDraw.Draw(sheet)
        for i, pair in enumerate(chosen):
            y = i * 270
            draw.text((5, y + 5), f"{i+1:02} group={pair['group']} distance={pair['hamming_distance']} exact={pair['same_image_md5']}", fill="black")
            for col, key in enumerate(("first", "second")):
                sheet.paste(thumbnail(lookup[(pair["group"], pair[key])]), (col * 320, y + 30))
        sheet.save(output / "near_duplicate_examples.jpg", quality=92)
    write_json(output / "near_duplicate_examples.json", chosen)
    for z in cache.values():
        z.close()
    print(json.dumps({"overall": summary["overall"], "duplicates": dup_summary,
                      "near_duplicate_candidates": len(adjacent), "issues": len(issues)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
