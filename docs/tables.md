# Empirical Benchmark Tables

## Table 1: Baselines
| Input size | Source | Resize method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | GFLOPs | Latency (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| 640×640 | Native (baseline) | — | 0.882 ± 0.020 | 0.846 ± 0.016 | 0.863 ± 0.002 | 0.889 ± 0.019 | 0.548 ± 0.003 | 6.4 | 9.80 ± 0.16 | 2.59 |
| 320×320 | Downscale from 640 | Bilinear | 0.913 ± 0.019 | 0.813 ± 0.030 | 0.859 ± 0.009 | 0.886 ± 0.008 | 0.560 ± 0.004 | 1.6 | 10.35 ± 1.04 | 2.59 |
| 160×160 | Downscale from 640 | Bilinear | 0.875 ± 0.009 | 0.814 ± 0.011 | 0.844 ± 0.003 | 0.837 ± 0.002 | 0.504 ± 0.000 | 0.4 | 9.67 ± 0.82 | 2.59 |
| 80×80 | Downscale from 640 | Bilinear | 0.889 ± 0.018 | 0.704 ± 0.022 | 0.786 ± 0.016 | 0.772 ± 0.027 | 0.437 ± 0.016 | 0.1 | 9.88 ± 0.22 | 2.59 |

## Table 2: Cropped ROI (400×400)
| Input size | Source | Resize method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | GFLOPs | Latency (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| 640×640 | Synthetic upscale from 400 | Bilinear | 0.879 ± 0.024 | 0.816 ± 0.028 | 0.846 ± 0.020 | 0.850 ± 0.001 | 0.571 ± 0.010 | 6.4 | 10.02 ± 0.27 | 2.59 |
| 320×320 | Downscale from 400 | Bilinear | 0.865 ± 0.030 | 0.864 ± 0.014 | 0.864 ± 0.009 | 0.885 ± 0.004 | 0.608 ± 0.016 | 1.6 | 9.42 ± 0.65 | 2.59 |
| 160×160 | Downscale from 400 | Bilinear | 0.886 ± 0.014 | 0.817 ± 0.022 | 0.850 ± 0.013 | 0.845 ± 0.010 | 0.566 ± 0.017 | 0.4 | 9.80 ± 0.38 | 2.59 |
| 80×80 | Downscale from 400 | Bilinear | 0.871 ± 0.041 | 0.730 ± 0.058 | 0.791 ± 0.019 | 0.813 ± 0.011 | 0.514 ± 0.008 | 0.1 | 10.46 ± 0.63 | 2.59 |

## Table 3: Overlapping Collage (640×640)
| Input size | Source | Resize method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | GFLOPs | Latency (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|
| 640×640 | Collage from 640 | Bilinear | 0.853 ± 0.012 | 0.811 ± 0.013 | 0.832 ± 0.012 | 0.805 ± 0.016 | 0.526 ± 0.012 | 6.4 | 10.45 ± 0.38 | 2.59 |
| 320×320 | Downscale from 640 | Bilinear | 0.868 ± 0.011 | 0.826 ± 0.015 | 0.847 ± 0.006 | 0.819 ± 0.006 | 0.538 ± 0.006 | 1.6 | 11.03 ± 0.48 | 2.59 |
| 160×160 | Downscale from 640 | Bilinear | 0.838 ± 0.008 | 0.698 ± 0.030 | 0.761 ± 0.020 | 0.716 ± 0.012 | 0.452 ± 0.011 | 0.4 | 9.98 ± 0.22 | 2.59 |
| 80×80 | Downscale from 640 | Bilinear | 0.701 ± 0.079 | 0.584 ± 0.012 | 0.635 ± 0.035 | 0.520 ± 0.047 | 0.294 ± 0.015 | 0.1 | 9.96 ± 0.42 | 2.59 |

## Table 4: Detection Accuracy and Spatial Preprocessing Overhead
Empirical benchmark measured across 5,000 images (Mean ± Std ms) on NVIDIA RTX 4060 host.
- **Baseline**: 0.00 ms (native hardware capture at resolution $R$).
- **Cropped**: $640 \to 400$ ROI crop followed by bilinear resampling to resolution $R$.
- **Collage**: Pure 3-patch spatial tiling on native resolution $R$ image (no downscale included).

| Method | Input | Accuracy | Added Latency (ms) |
|---|---|---|---|
| **Baseline** | 640 | 0.767 ± 0.008 | 0.00 ± 0.00 |
| | 320 | **0.769 ± 0.006** | 0.00 ± 0.00 |
| | 160 | 0.728 ± 0.002 | 0.00 ± 0.00 |
| | 80 | 0.665 ± 0.023 | 0.00 ± 0.00 |
| **Cropped** | 640* | 0.756 ± 0.012 | 2.76 ± 0.50 |
| | 320 | **0.786 ± 0.004** | 1.15 ± 0.27 |
| | 160 | 0.754 ± 0.014 | 0.67 ± 0.17 |
| | 80 | 0.706 ± 0.007 | 0.52 ± 0.14 |
| **Collage** | 640 | 0.721 ± 0.016 | 0.57 ± 0.16 |
| | 320 | **0.735 ± 0.001** | 0.15 ± 0.06 |
| | 160 | 0.643 ± 0.017 | 0.05 ± 0.05 |
| | 80 | 0.483 ± 0.040 | 0.03 ± 0.02 |