"""Audit immutable source archives and materialize a sequence-disjoint training set."""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lawn_segmentation.constants import GRAY_LABEL_ARCHIVE, IMAGE_ARCHIVES, IMAGE_SIZE, VALID_MASK_VALUES


def png_members(archive: zipfile.ZipFile) -> list[str]:
    return [name for name in archive.namelist() if name.lower().endswith(".png") and not name.endswith("/")]


def audit(root: Path, skip_empty_labels: bool = False) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]], dict]:
    label_archive = root / GRAY_LABEL_ARCHIVE
    if not label_archive.exists():
        raise FileNotFoundError(label_archive)
    with zipfile.ZipFile(label_archive) as labels_zip:
        labels = {Path(member).name: member for member in png_members(labels_zip)}
        if len(labels) != len(png_members(labels_zip)):
            raise ValueError("Duplicate label filenames found in the gray-label archive")
        empty_gray = {Path(info.filename).name for info in labels_zip.infolist() if info.filename.lower().endswith(".png") and info.file_size == 0}
    if empty_gray and not skip_empty_labels:
        raise ValueError(f"Found {len(empty_gray)} zero-byte gray labels. Repair the source labels or rerun with --skip-empty-labels to exclude them.")
    with zipfile.ZipFile(label_archive) as labels_zip:
        image_members: dict[str, tuple[str, str]] = {}
        image_size_by_name: dict[str, tuple[int, int]] = {}
        for group, archive_name in IMAGE_ARCHIVES.items():
            archive_path = root / archive_name
            if not archive_path.exists():
                raise FileNotFoundError(archive_path)
            with zipfile.ZipFile(archive_path) as image_zip:
                for member in png_members(image_zip):
                    filename = Path(member).name
                    if filename in image_members:
                        raise ValueError(f"Duplicate image filename: {filename}")
                    if filename not in labels:
                        raise ValueError(f"Missing gray label for: {filename}")
                    # Reading PNG headers through ZipExtFile verifies dimensions without
                    # decompressing every multi-megabyte camera frame into memory.
                    with Image.open(image_zip.open(member)) as image:
                        image_size_by_name[filename] = image.size
                    image_members[filename] = (group, member)
    usable_labels = {filename: member for filename, member in labels.items() if filename not in empty_gray}
    expected_names = set(usable_labels)
    if skip_empty_labels:
        image_members = {filename: item for filename, item in image_members.items() if filename in usable_labels}
    image_sizes = Counter(image_size_by_name[filename] for filename in image_members)
    source_image_sizes = Counter(image_size_by_name.values())
    if set(image_members) != expected_names:
        missing_images = sorted(expected_names - set(image_members))
        raise ValueError(f"Usable labels without source images: {missing_images[:5]} (total={len(missing_images)})")
    mask_sources = {filename: ("gray", member) for filename, member in usable_labels.items()}
    values = Counter()
    mask_sizes = Counter()
    with zipfile.ZipFile(label_archive) as labels_zip:
        for filename, (kind, member) in mask_sources.items():
            with Image.open(labels_zip.open(member)) as mask:
                mask_sizes[mask.size] += 1
                values.update({value: count for value, count in enumerate(mask.histogram()) if count})
    bad_values = set(values) - VALID_MASK_VALUES
    if bad_values:
        raise ValueError(f"Unexpected mask values: {sorted(bad_values)}")
    if set(image_sizes) != {IMAGE_SIZE} or set(mask_sizes) != {IMAGE_SIZE}:
        raise ValueError(f"Expected {IMAGE_SIZE}; image sizes={dict(image_sizes)}, mask sizes={dict(mask_sizes)}")
    return image_members, mask_sources, {
        "sample_count": len(image_members), "image_sizes": {str(k): v for k, v in image_sizes.items()},
        "source_image_sizes": {str(k): v for k, v in source_image_sizes.items()},
        "mask_sizes": {str(k): v for k, v in mask_sizes.items()}, "mask_value_pixels": dict(sorted(values.items())),
        "groups": dict(Counter(group for group, _ in image_members.values())), "source_sample_count": len(labels),
        "empty_gray_labels": len(empty_gray), "excluded_empty_gray_labels": len(empty_gray) if skip_empty_labels else 0,
    }


def extract_and_manifest(root: Path, output: Path, members: dict[str, tuple[str, str]], mask_sources: dict[str, tuple[str, str]], validation_groups: set[str]) -> None:
    images_dir, masks_dir = output / "images", output / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / GRAY_LABEL_ARCHIVE) as labels_zip:
        for filename, (_, member) in mask_sources.items():
            (masks_dir / filename).write_bytes(labels_zip.read(member))
    grouped: dict[str, list[tuple[str, str]]] = {}
    for filename, (group, member) in members.items():
        grouped.setdefault(group, []).append((filename, member))
    for group, files in grouped.items():
        with zipfile.ZipFile(root / IMAGE_ARCHIVES[group]) as image_zip:
            for filename, member in files:
                (images_dir / filename).write_bytes(image_zip.read(member))
    manifests = {"train": [], "val": []}
    for filename, (group, _) in sorted(members.items()):
        split = "val" if group in validation_groups else "train"
        manifests[split].append({"image": str((images_dir / filename).resolve()), "mask": str((masks_dir / filename).resolve()), "group": group})
    for split, records in manifests.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output / "audit_report.json").write_text(json.dumps({"train_count": len(manifests["train"]), "val_count": len(manifests["val"]), "validation_groups": sorted(validation_groups)}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--validation-groups", nargs="+", default=["2", "3", "4"])
    parser.add_argument("--skip-empty-labels", action="store_true", help="Exclude zero-byte gray labels; default is strict failure.")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the generated output directory.")
    args = parser.parse_args()
    members, mask_sources, report = audit(args.root.resolve(), skip_empty_labels=args.skip_empty_labels)
    report["validation_groups"] = args.validation_groups
    report["train_count"] = sum(group not in args.validation_groups for group, _ in members.values())
    report["val_count"] = sum(group in args.validation_groups for group, _ in members.values())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.audit_only:
        output = args.output if args.output.is_absolute() else args.root / args.output
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to replace {output}; pass --overwrite for generated data only")
        if output.exists():
            import shutil
            shutil.rmtree(output)
        extract_and_manifest(args.root.resolve(), output.resolve(), members, mask_sources, set(args.validation_groups))


if __name__ == "__main__":
    main()
