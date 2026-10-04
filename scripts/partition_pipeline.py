"""
ROI-DMS Partition & Preprocessing Pipeline
=========================================
Strictly deterministic, mathematically verified pipeline for:
- Methods: baseline, cropped, collage
- Resolutions: 1280x1280, 640x640, 320x320, 160x160, 80x80
- Algorithm: Bilinear interpolation (PIL.Image.Resampling.BILINEAR)
- Exact bounding box projection & strict tile boundary containment
"""

from pathlib import Path
from PIL import Image, ImageDraw

# Dataset Subject Partition (8:3:3)
SPLITS = {
    "train": [
        "subject_01", "subject_04", "subject_06", "subject_07",
        "subject_08", "subject_09", "subject_13", "subject_14"
    ],
    "val": [
        "subject_02", "subject_03", "subject_11"
    ],
    "test": [
        "subject_05", "subject_10", "subject_12"
    ]
}

RESOLUTIONS = [
    (1280, 1280),
    (640, 640),
    (320, 320),
    (160, 160),
    (80, 80)
]

METHODS = ["baseline", "cropped", "collage"]

CLASS_NAMES = {
    0: "yawning",
    1: "hand_over_mouth",
    2: "drinking",
    3: "phone_use"
}

CLASS_COLORS = {
    0: "#00FF00",  # Green
    1: "#00E5FF",  # Cyan
    2: "#FFD600",  # Yellow
    3: "#FF3D00"   # Red / Orange
}

# Cropped ROI parameters (docs/crop.md Section B: 400px box)
# X min: 140, X max: 540, Y min: 120, Y max: 520
CROP_BOX = (140, 120, 540, 520)

# Overlapping collage parameters (docs/crop.md Section A)
# Each entry is ((source_x1, source_y1, source_x2, source_y2), (dest_x1, dest_y1, dest_x2, dest_y2), tile_name)
COLLAGE_TILES = [
    # Tile 1: Head strip (X: [173, 493], Y: [0, 640]) -> Dest Left half (X: [0, 320], Y: [0, 640])
    ((173, 0, 493, 640), (0, 0, 320, 640), "head_strip"),
    # Tile 2: Right interaction box (X: [320, 640], Y: [215, 535]) -> Dest Top-right (X: [320, 640], Y: [0, 320])
    ((320, 215, 640, 535), (320, 0, 640, 320), "right_box"),
    # Tile 3: Left interaction box (X: [0, 320], Y: [215, 535]) -> Dest Bottom-right (X: [320, 640], Y: [320, 640])
    ((0, 215, 320, 535), (320, 320, 640, 640), "left_box")
]


def process_image(img: Image.Image, method: str, target_size: tuple[int, int]) -> Image.Image:
    """
    Deterministically processes a 640x640 PIL Image with the chosen method
    and resizes using strict Bilinear Interpolation.
    """
    if method == "baseline":
        processed = img
    elif method == "cropped":
        # 400x400 crop around driver
        processed = img.crop(CROP_BOX)
    elif method == "collage":
        # 640x640 3-tile canvas
        canvas = Image.new("RGB", (640, 640))
        for (sx1, sy1, sx2, sy2), (dx1, dy1, dx2, dy2), _ in COLLAGE_TILES:
            patch = img.crop((sx1, sy1, sx2, sy2))
            canvas.paste(patch, (dx1, dy1))
        processed = canvas
    else:
        raise ValueError(f"Unknown method: {method}")

    # Bilinear interpolation for all resizes
    if processed.size != target_size:
        processed = processed.resize(target_size, Image.Resampling.BILINEAR)

    return processed


def parse_yolo_labels(label_text: str, img_w: int = 640, img_h: int = 640) -> list[dict]:
    """Parses YOLO normalized labels into absolute pixel coordinates."""
    boxes = []
    for line in label_text.strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        abs_xc = xc * img_w
        abs_yc = yc * img_h
        abs_w = w * img_w
        abs_h = h * img_h
        x1 = abs_xc - abs_w / 2.0
        y1 = abs_yc - abs_h / 2.0
        x2 = abs_xc + abs_w / 2.0
        y2 = abs_yc + abs_h / 2.0
        boxes.append({
            "class_id": cls_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "orig_xc": xc, "orig_yc": yc, "orig_w": w, "orig_h": h,
            "orig_area": abs_w * abs_h
        })
    return boxes


def transform_labels(boxes: list[dict], method: str) -> list[dict]:
    """
    Deterministically transforms bounding boxes for the chosen method.
    Returns a list of dicts with:
    - 'class_id': int
    - 'xc', 'yc', 'w', 'h': normalized [0, 1] YOLO coords
    - 'x1', 'y1', 'x2', 'y2': normalized [0, 1] corners
    - 'tile_bounds': normalized [0, 1] bounds of the tile (for clipping visual strokes)
    """
    transformed = []

    if method == "baseline":
        for b in boxes:
            transformed.append({
                "class_id": b["class_id"],
                "xc": b["orig_xc"],
                "yc": b["orig_yc"],
                "w": b["orig_w"],
                "h": b["orig_h"],
                "x1": b["orig_xc"] - b["orig_w"] / 2.0,
                "y1": b["orig_yc"] - b["orig_h"] / 2.0,
                "x2": b["orig_xc"] + b["orig_w"] / 2.0,
                "y2": b["orig_yc"] + b["orig_h"] / 2.0,
                "tile_bounds": (0.0, 0.0, 1.0, 1.0)
            })
        return transformed

    if method == "cropped":
        cx1, cy1, cx2, cy2 = CROP_BOX
        crop_w = float(cx2 - cx1)
        crop_h = float(cy2 - cy1)

        for b in boxes:
            # Deterministic intersection with crop window
            ix1 = max(b["x1"], float(cx1))
            iy1 = max(b["y1"], float(cy1))
            ix2 = min(b["x2"], float(cx2))
            iy2 = min(b["y2"], float(cy2))

            # Must have non-empty visible intersection (>= 2 pixels)
            if (ix2 - ix1) >= 2.0 and (iy2 - iy1) >= 2.0:
                new_x1 = (ix1 - cx1) / crop_w
                new_y1 = (iy1 - cy1) / crop_h
                new_x2 = (ix2 - cx1) / crop_w
                new_y2 = (iy2 - cy1) / crop_h

                # Strict clamping to [0, 1]
                new_x1 = max(0.0, min(1.0, new_x1))
                new_y1 = max(0.0, min(1.0, new_y1))
                new_x2 = max(0.0, min(1.0, new_x2))
                new_y2 = max(0.0, min(1.0, new_y2))

                w_norm = new_x2 - new_x1
                h_norm = new_y2 - new_y1
                xc_norm = new_x1 + w_norm / 2.0
                yc_norm = new_y1 + h_norm / 2.0

                transformed.append({
                    "class_id": b["class_id"],
                    "xc": xc_norm,
                    "yc": yc_norm,
                    "w": w_norm,
                    "h": h_norm,
                    "x1": new_x1,
                    "y1": new_y1,
                    "x2": new_x2,
                    "y2": new_y2,
                    "tile_bounds": (0.0, 0.0, 1.0, 1.0)
                })
        return transformed

    if method == "collage":
        canvas_w = 640.0
        canvas_h = 640.0

        for b in boxes:
            for (sx1, sy1, sx2, sy2), (dx1, dy1, dx2, dy2), tile_name in COLLAGE_TILES:
                # Deterministic intersection with source tile
                ix1 = max(b["x1"], float(sx1))
                iy1 = max(b["y1"], float(sy1))
                ix2 = min(b["x2"], float(sx2))
                iy2 = min(b["y2"], float(sy2))

                # Check if visible in this tile (>= 2 pixels in both dimensions)
                if (ix2 - ix1) >= 2.0 and (iy2 - iy1) >= 2.0:
                    # Map source intersection directly to destination coordinates
                    dest_x1 = (ix1 - sx1) + dx1
                    dest_y1 = (iy1 - sy1) + dy1
                    dest_x2 = (ix2 - sx1) + dx1
                    dest_y2 = (iy2 - sy1) + dy1

                    # Enforce strict tile containment
                    dest_x1 = max(float(dx1), min(float(dx2), dest_x1))
                    dest_y1 = max(float(dy1), min(float(dy2), dest_y1))
                    dest_x2 = max(float(dx1), min(float(dx2), dest_x2))
                    dest_y2 = max(float(dy1), min(float(dy2), dest_y2))

                    # Normalize relative to 640x640 canvas
                    x1_norm = dest_x1 / canvas_w
                    y1_norm = dest_y1 / canvas_h
                    x2_norm = dest_x2 / canvas_w
                    y2_norm = dest_y2 / canvas_h

                    w_norm = x2_norm - x1_norm
                    h_norm = y2_norm - y1_norm
                    xc_norm = x1_norm + w_norm / 2.0
                    yc_norm = y1_norm + h_norm / 2.0

                    tile_x1_norm = dx1 / canvas_w
                    tile_y1_norm = dy1 / canvas_h
                    tile_x2_norm = dx2 / canvas_w
                    tile_y2_norm = dy2 / canvas_h

                    transformed.append({
                        "class_id": b["class_id"],
                        "xc": xc_norm,
                        "yc": yc_norm,
                        "w": w_norm,
                        "h": h_norm,
                        "x1": x1_norm,
                        "y1": y1_norm,
                        "x2": x2_norm,
                        "y2": y2_norm,
                        "tile_name": tile_name,
                        "tile_bounds": (tile_x1_norm, tile_y1_norm, tile_x2_norm, tile_y2_norm)
                    })
        return transformed

    return transformed


def to_yolo_lines(transformed_boxes: list[dict]) -> list[str]:
    """Converts transformed box dicts into standard YOLO lines."""
    lines = []
    for b in transformed_boxes:
        lines.append(f"{b['class_id']} {b['xc']:.6f} {b['yc']:.6f} {b['w']:.6f} {b['h']:.6f}")
    return lines


def draw_bounding_boxes(img: Image.Image, transformed_boxes: list[dict]) -> Image.Image:
    """
    Draws bounding boxes and class labels onto a copy of the image.
    Ensures box strokes are strictly contained within their tile bounds so they never bleed.
    Adapts stroke width and font size to the image resolution.
    """
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    # Responsive stroke thickness
    if w <= 80:
        stroke = 1
    elif w <= 160:
        stroke = 1
    elif w <= 320:
        stroke = 2
    elif w <= 640:
        stroke = 2
    else:
        stroke = 3

    for b in transformed_boxes:
        cls_id = b["class_id"]
        # Convert normalized to pixel coordinates
        px1 = b["x1"] * w
        py1 = b["y1"] * h
        px2 = b["x2"] * w
        py2 = b["y2"] * h

        # Tile boundary limits in pixels
        tx1, ty1, tx2, ty2 = b["tile_bounds"]
        ptx1 = tx1 * w
        pty1 = ty1 * h
        ptx2 = tx2 * w
        pty2 = ty2 * h

        # Ensure stroke stays strictly inside tile boundary
        draw_x1 = max(ptx1, px1)
        draw_y1 = max(pty1, py1)
        draw_x2 = min(ptx2 - 1, px2)
        draw_y2 = min(pty2 - 1, py2)

        if draw_x2 > draw_x1 and draw_y2 > draw_y1:
            color = CLASS_COLORS.get(cls_id, "#FF0000")
            draw.rectangle([draw_x1, draw_y1, draw_x2, draw_y2], outline=color, width=stroke)

            # Draw label text if resolution is 320 or higher
            if w >= 320:
                label_text = CLASS_NAMES.get(cls_id, str(cls_id))
                text_y = max(draw_y1 + 2, draw_y1 - 12)
                draw.text((draw_x1 + 3, text_y), label_text, fill=color)

    return annotated
