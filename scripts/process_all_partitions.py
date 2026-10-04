"""
ROI-DMS High-Performance Full Dataset Generator
===============================================
Generates all 15 partitions (3 methods x 5 resolutions x 3 splits) for all 15,723 frames:
- Methods: baseline, cropped, collage
- Resolutions: 1280x1280, 640x640, 320x320, 160x160, 80x80 (Bilinear)
- Splits: train, val, test
- Bounding box labels strictly projected and tile-bounded for YOLO
"""

import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image

from partition_pipeline import (
    SPLITS, RESOLUTIONS, METHODS, CROP_BOX, COLLAGE_TILES,
    parse_yolo_labels, transform_labels, to_yolo_lines
)

WORKSPACE_ROOT = Path("c:/Dev/repos/Public repos/research/ROI-DMS")
RAW_IMAGES_DIR = WORKSPACE_ROOT / "data/processed/RGB/images"
RAW_LABELS_DIR = WORKSPACE_ROOT / "data/processed/RGB/labels"
PARTITIONS_DIR = WORKSPACE_ROOT / "partitions"


def get_subject_split_map():
    sub_map = {}
    for split_name, subjects in SPLITS.items():
        for sub in subjects:
            sub_map[sub] = split_name
    return sub_map


def build_work_items():
    sub_map = get_subject_split_map()
    items = []
    for sub, split_name in sub_map.items():
        sub_dir = RAW_IMAGES_DIR / sub
        if not sub_dir.exists():
            continue
        for img_path in sub_dir.glob("*/*.jpg"):
            rel = img_path.relative_to(RAW_IMAGES_DIR)
            lbl_path = RAW_LABELS_DIR / rel.parts[0] / rel.parts[1] / (img_path.stem + ".txt")
            items.append((
                str(img_path),
                str(lbl_path),
                img_path.stem,
                split_name
            ))
    return items


def pre_create_directories():
    """Pre-creates all directory paths across all partitions to avoid worker lock contention."""
    for method in METHODS:
        for res_w, res_h in RESOLUTIONS:
            res_str = f"{res_w}x{res_h}"
            for split_name in ["train", "val", "test"]:
                (PARTITIONS_DIR / method / res_str / split_name / "images").mkdir(parents=True, exist_ok=True)
                (PARTITIONS_DIR / method / res_str / split_name / "labels").mkdir(parents=True, exist_ok=True)


def process_single_frame(args):
    img_path_str, lbl_path_str, stem, split_name = args

    try:
        raw_img = Image.open(img_path_str).convert("RGB")
        lbl_p = Path(lbl_path_str)
        lbl_text = lbl_p.read_text(encoding="utf-8") if lbl_p.exists() else ""
        parsed_boxes = parse_yolo_labels(lbl_text, raw_img.width, raw_img.height)

        # 1. BASELINE
        baseline_lines = to_yolo_lines(transform_labels(parsed_boxes, "baseline"))
        baseline_label_content = "\n".join(baseline_lines) + ("\n" if baseline_lines else "")
        for res_w, res_h in RESOLUTIONS:
            res_str = f"{res_w}x{res_h}"
            out_img = PARTITIONS_DIR / "baseline" / res_str / split_name / "images" / f"{stem}.jpg"
            out_lbl = PARTITIONS_DIR / "baseline" / res_str / split_name / "labels" / f"{stem}.txt"

            if (res_w, res_h) == (640, 640):
                im = raw_img
            else:
                im = raw_img.resize((res_w, res_h), Image.Resampling.BILINEAR)
            im.save(out_img, quality=95)
            out_lbl.write_text(baseline_label_content, encoding="utf-8")

        # 2. CROPPED
        crop_img = raw_img.crop(CROP_BOX)
        cropped_lines = to_yolo_lines(transform_labels(parsed_boxes, "cropped"))
        cropped_label_content = "\n".join(cropped_lines) + ("\n" if cropped_lines else "")
        for res_w, res_h in RESOLUTIONS:
            res_str = f"{res_w}x{res_h}"
            out_img = PARTITIONS_DIR / "cropped" / res_str / split_name / "images" / f"{stem}.jpg"
            out_lbl = PARTITIONS_DIR / "cropped" / res_str / split_name / "labels" / f"{stem}.txt"

            im = crop_img.resize((res_w, res_h), Image.Resampling.BILINEAR)
            im.save(out_img, quality=95)
            out_lbl.write_text(cropped_label_content, encoding="utf-8")

        # 3. COLLAGE
        collage_img = Image.new("RGB", (640, 640))
        for (sx1, sy1, sx2, sy2), (dx1, dy1, dx2, dy2), _ in COLLAGE_TILES:
            patch = raw_img.crop((sx1, sy1, sx2, sy2))
            collage_img.paste(patch, (dx1, dy1))

        collage_lines = to_yolo_lines(transform_labels(parsed_boxes, "collage"))
        collage_label_content = "\n".join(collage_lines) + ("\n" if collage_lines else "")
        for res_w, res_h in RESOLUTIONS:
            res_str = f"{res_w}x{res_h}"
            out_img = PARTITIONS_DIR / "collage" / res_str / split_name / "images" / f"{stem}.jpg"
            out_lbl = PARTITIONS_DIR / "collage" / res_str / split_name / "labels" / f"{stem}.txt"

            if (res_w, res_h) == (640, 640):
                im = collage_img
            else:
                im = collage_img.resize((res_w, res_h), Image.Resampling.BILINEAR)
            im.save(out_img, quality=95)
            out_lbl.write_text(collage_label_content, encoding="utf-8")

        return True
    except Exception as e:
        print(f"Error processing {stem}: {e}", file=sys.stderr)
        return False


def main():
    print("Pre-creating partition directories...")
    pre_create_directories()

    items = build_work_items()
    total = len(items)
    print(f"Found {total} frames to process.")

    num_workers = max(1, min(10, cpu_count()))
    print(f"Starting parallel processing with {num_workers} worker processes...")

    t0 = time.time()
    processed_count = 0
    chunksize = 20

    with Pool(processes=num_workers) as pool:
        for res in pool.imap_unordered(process_single_frame, items, chunksize=chunksize):
            if res:
                processed_count += 1
            if processed_count % 1000 == 0 or processed_count == total:
                elapsed = time.time() - t0
                rate = processed_count / elapsed if elapsed > 0 else 0
                remaining = (total - processed_count) / rate if rate > 0 else 0
                print(f"[{processed_count:5d}/{total}] ({processed_count/total*100:5.1f}%) | "
                      f"Elapsed: {elapsed:5.1f}s | Rate: {rate:4.1f} frames/s | ETA: {remaining:5.1f}s", flush=True)

    total_time = time.time() - t0
    print(f"\nProcessing complete! Processed {processed_count}/{total} frames in {total_time:.1f}s ({total_time/60:.2f} mins).")


if __name__ == "__main__":
    main()
