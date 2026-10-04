"""
Generate Ultralytics YOLO dataset configuration YAMLs for all 15 partitions.
Paths are formatted relative to the repository root for flawless Ultralytics resolution.
"""

from pathlib import Path

METHODS = ["baseline", "cropped", "collage"]
RESOLUTIONS = ["1280x1280", "640x640", "320x320", "160x160", "80x80"]

CONFIGS_DIR = Path("configs/partitions")
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE = """# YOLO Dataset Configuration for {method} @ {resolution}
# Generated for ROI-DMS research benchmark

path: partitions/{method}/{resolution}
train: train/images
val: val/images
test: test/images

names:
  0: yawning
  1: hand_over_mouth
  2: drinking
  3: phone_use
"""

def generate_configs():
    created = []
    for method in METHODS:
        for res in RESOLUTIONS:
            cfg_path = CONFIGS_DIR / f"{method}_{res}.yaml"
            content = TEMPLATE.format(method=method, resolution=res)
            cfg_path.write_text(content, encoding="utf-8")
            created.append(cfg_path)
    print(f"Generated {len(created)} partition dataset configs in configs/partitions/")

if __name__ == "__main__":
    generate_configs()
