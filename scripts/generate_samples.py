"""
Generate verification samples for ROI-DMS partitions with real, unscaled pixel sizes.
Methods:
1. Cropped method (400x400 crop around driver)
2. Collage method (overlapping 3-tile composite)
3. Baseline method (native 640x640)

Sizes:
- 1280x1280 upscaled (bilinear)
- 640x640 (native baseline, 640 collage, 640 upscaled from 400 for crop)
- 320x320 downscaled (bilinear)
- 160x160 downscaled (bilinear)
- 80x80 downscaled (bilinear)
"""

import os
from pathlib import Path
from PIL import Image
from partition_pipeline import (
    SPLITS, RESOLUTIONS, METHODS,
    process_image, parse_yolo_labels, transform_labels, to_yolo_lines, draw_bounding_boxes
)

WORKSPACE_ROOT = Path("c:/Dev/repos/Public repos/research/ROI-DMS")
RAW_IMAGES_DIR = WORKSPACE_ROOT / "data/processed/RGB/images"
RAW_LABELS_DIR = WORKSPACE_ROOT / "data/processed/RGB/labels"
PARTITIONS_DIR = WORKSPACE_ROOT / "partitions"
SAMPLES_DIR = WORKSPACE_ROOT / "samples/real_scale_samples"

# Focus on the user's primary sample: subject_05_video_01_frame_0048 (drinking water bottle)
# Also include subject_01_video_01_frame_0112 (phone use) and subject_02_video_01_frame_0048 (drinking)
SAMPLES = [
    {
        "split": "test",
        "subject": "subject_05",
        "video": "video_01",
        "frame": "subject_05_video_01_frame_0048",
        "activity": "drinking"
    },
    {
        "split": "train",
        "subject": "subject_01",
        "video": "video_01",
        "frame": "subject_01_video_01_frame_0112",
        "activity": "phone_use"
    },
    {
        "split": "val",
        "subject": "subject_02",
        "video": "video_01",
        "frame": "subject_02_video_01_frame_0048",
        "activity": "drinking"
    }
]


def run():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    catalog = []

    for s_info in SAMPLES:
        split = s_info["split"]
        sub = s_info["subject"]
        vid = s_info["video"]
        stem = s_info["frame"]
        act = s_info["activity"]

        raw_img_path = RAW_IMAGES_DIR / sub / vid / f"{stem}.jpg"
        raw_lbl_path = RAW_LABELS_DIR / sub / vid / f"{stem}.txt"

        if not raw_img_path.exists():
            continue

        raw_img = Image.open(raw_img_path).convert("RGB")
        raw_lbl_text = raw_lbl_path.read_text(encoding="utf-8") if raw_lbl_path.exists() else ""
        parsed_boxes = parse_yolo_labels(raw_lbl_text, raw_img.width, raw_img.height)

        for method in METHODS:
            # Deterministically transform boxes for this method once
            transformed_boxes = transform_labels(parsed_boxes, method)
            yolo_lines = to_yolo_lines(transformed_boxes)

            for res_w, res_h in RESOLUTIONS:
                res_str = f"{res_w}x{res_h}"

                # 1. Output in official partition folder
                part_img_dir = PARTITIONS_DIR / method / res_str / split / "images"
                part_lbl_dir = PARTITIONS_DIR / method / res_str / split / "labels"
                part_img_dir.mkdir(parents=True, exist_ok=True)
                part_lbl_dir.mkdir(parents=True, exist_ok=True)

                # Process image at GENUINE pixel dimensions with Bilinear interpolation
                processed_img = process_image(raw_img, method, (res_w, res_h))
                part_img_path = part_img_dir / f"{stem}.jpg"
                processed_img.save(part_img_path, quality=95)

                part_lbl_path = part_lbl_dir / f"{stem}.txt"
                part_lbl_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

                # 2. Output genuine, unscaled annotated image in samples directory
                annotated_img = draw_bounding_boxes(processed_img, transformed_boxes)
                # Verify that annotated_img genuinely has (res_w, res_h)
                assert annotated_img.size == (res_w, res_h), f"Size mismatch: {annotated_img.size} != {(res_w, res_h)}"

                # Save raw and annotated sample
                vis_img_path = SAMPLES_DIR / f"{split}_{method}_{res_str}_{stem}_annotated.png"
                raw_sample_path = SAMPLES_DIR / f"{split}_{method}_{res_str}_{stem}_raw.jpg"
                annotated_img.save(vis_img_path)
                processed_img.save(raw_sample_path, quality=95)

                catalog.append({
                    "split": split,
                    "stem": stem,
                    "activity": act,
                    "method": method,
                    "resolution": res_str,
                    "width": res_w,
                    "height": res_h,
                    "raw_img_bytes": raw_sample_path.stat().st_size,
                    "annotated_png_bytes": vis_img_path.stat().st_size,
                    "num_boxes": len(transformed_boxes),
                    "part_img_path": str(part_img_path),
                    "vis_img_path": str(vis_img_path),
                    "raw_sample_path": str(raw_sample_path)
                })

    print(f"Generated {len(catalog)} real-scale partition sample images.")
    return catalog


if __name__ == "__main__":
    run()
