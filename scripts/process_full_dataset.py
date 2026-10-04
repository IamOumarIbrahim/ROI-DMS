"""
ROI-DMS Full Dataset Partition Generator
=========================================
Batch-processes the 15,723 DMD frames into the desired partitions:
- Methods: baseline, cropped, collage
- Resolutions: 1280x1280, 640x640, 320x320, 160x160, 80x80
- Splits: train, val, test

Usage:
    python scripts/process_full_dataset.py --method cropped --resolution 640x640
    python scripts/process_full_dataset.py --all
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from PIL import Image
from partition_pipeline import (
    SPLITS, RESOLUTIONS, METHODS,
    process_image, parse_yolo_labels, transform_labels
)

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES_DIR = WORKSPACE_ROOT / "data/processed/RGB/images"
RAW_LABELS_DIR = WORKSPACE_ROOT / "data/processed/RGB/labels"
PARTITIONS_DIR = WORKSPACE_ROOT / "partitions"


def get_subject_split_map():
    mapping = {}
    for split_name, subjects in SPLITS.items():
        for sub in subjects:
            mapping[sub] = split_name
    return mapping


def discover_all_samples():
    sub_map = get_subject_split_map()
    samples = []
    for sub, split_name in sub_map.items():
        sub_dir = RAW_IMAGES_DIR / sub
        if not sub_dir.exists():
            continue
        for img_path in sub_dir.glob("*/*.jpg"):
            rel_parts = img_path.relative_to(RAW_IMAGES_DIR).parts
            # (subject, video, filename)
            lbl_path = RAW_LABELS_DIR / rel_parts[0] / rel_parts[1] / (img_path.stem + ".txt")
            samples.append({
                "split": split_name,
                "subject": rel_parts[0],
                "video": rel_parts[1],
                "stem": img_path.stem,
                "img_path": img_path,
                "lbl_path": lbl_path
            })
    return samples


def process_single_item(item, method, res_w, res_h):
    res_str = f"{res_w}x{res_h}"
    split = item["split"]
    stem = item["stem"]

    target_img_dir = PARTITIONS_DIR / method / res_str / split / "images"
    target_lbl_dir = PARTITIONS_DIR / method / res_str / split / "labels"
    target_img_path = target_img_dir / f"{stem}.jpg"
    target_lbl_path = target_lbl_dir / f"{stem}.txt"

    if target_img_path.exists() and target_lbl_path.exists():
        return True

    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_img = Image.open(item["img_path"]).convert("RGB")
        processed_img = process_image(raw_img, method, (res_w, res_h))
        processed_img.save(target_img_path, quality=95)

        raw_lbl_text = item["lbl_path"].read_text(encoding="utf-8") if item["lbl_path"].exists() else ""
        boxes = parse_yolo_labels(raw_lbl_text, raw_img.width, raw_img.height)
        yolo_lines = transform_labels(boxes, method)
        target_lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
        return True
    except Exception as e:
        print(f"Error processing {stem}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Process DMD dataset partitions.")
    parser.add_argument("--method", choices=METHODS + ["all"], default="all")
    parser.add_argument("--resolution", choices=[f"{w}x{h}" for w, h in RESOLUTIONS] + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    methods = METHODS if args.method == "all" else [args.method]
    if args.resolution == "all":
        resolutions = RESOLUTIONS
    else:
        w, h = map(int, args.resolution.split("x"))
        resolutions = [(w, h)]

    all_samples = discover_all_samples()
    if args.split != "all":
        all_samples = [s for s in all_samples if s["split"] == args.split]
    if args.limit:
        all_samples = all_samples[:args.limit]

    print(f"Discovered {len(all_samples)} total frames to process.")
    print(f"Methods: {methods}")
    print(f"Resolutions: {[f'{w}x{h}' for w, h in resolutions]}")

    for method in methods:
        for res_w, res_h in resolutions:
            res_str = f"{res_w}x{res_h}"
            print(f"\nProcessing {method} @ {res_str} ({len(all_samples)} items)...")
            success_count = 0
            # Execute batch
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_single_item, item, method, res_w, res_h) for item in all_samples]
                for f in futures:
                    if f.result():
                        success_count += 1
            print(f"Completed {method} @ {res_str}: {success_count}/{len(all_samples)} successful.")


if __name__ == "__main__":
    main()
