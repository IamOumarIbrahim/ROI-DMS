
# Table 1 Baselines
| Input size | Source | Resize method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | GFLOPs | Latency (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1280×1280 | Synthetic upscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 640×640 | Native (baseline) | — | ... | ... | ... | ... | ... | ... | ... | ... |
| 320×320 | Downscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 160×160 | Downscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 80×80 | Downscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |

# Table 2 Cropped ROI (400x400)
| Input size | Source | Resize method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | GFLOPs | Latency (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1280×1280 | Synthetic upscale from 400 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 640×640 | Synthetic upscale from 400 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 320×320 | Downscale from 400 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 160×160 | Downscale from 400 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 80×80 | Downscale from 400 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |


# Table 3 Overlapping collage (640x640)
| Input size | Source | Resize method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | GFLOPs | Latency (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1280×1280 | Synthetic upscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 640×640 | Collage from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 320×320 | Downscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 160×160 | Downscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |
| 80×80 | Downscale from 640 | Bilinear | ... | ... | ... | ... | ... | ... | ... | ... |