# ROI-DMS: Spatial ROI Compression for Driver Monitoring Systems

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLO11n](https://img.shields.io/badge/model-YOLO11n-green.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An empirical investigation into **input-space spatial compression** for safety-critical Driver Monitoring Systems (DMS) on the Driver Monitoring Dataset (DMD). We evaluate whether spatial restructuring and region-of-interest cropping preserve fine-grained driver distraction detection across extreme resolution scaling ($640 \times 640 \rightarrow 80 \times 80$) while reducing computational footprint.

---

## Key Findings

1. **Cropped ROI Dominates Low-Bitrate Regimes**:
   At extreme downsampling ($80 \times 80$), **Cropped ROI achieves 0.813 mAP@0.5** compared to Baseline's **0.772** (+4.1 points) and Collage's **0.520** (+29.3 points).
2. **Subtle Facial Signal Preservation**:
   On **Yawning** detection at $80 \times 80$, Cropped ROI achieves **0.733 mAP@0.5** vs Baseline's **0.496** (+23.7 points), proving that removing irrelevant interior vehicle pixels preserves feature density for mouth/eye kinematics.
3. **Collage Boundary Penalty**:
   Stitching driver interaction patches introduces artificial seam boundaries across hand/head trajectories, causing catastrophic degradation at small resolutions ($80 \times 80$ drops to 0.256 mAP on Yawning).
4. **$320 \times 320$\,px Universal Peak Accuracy**:
   Across all three framing methods, $320 \times 320$\,px achieves superior composite accuracy over native $640 \times 640$\,px. Cropped at $320 \times 320$\,px achieves peak overall accuracy (**0.786**) while boosting throughput by **+25.7 FPS** (+31%) over native 640 Baseline.
5. **Negligible Preprocessing Latency**:
   On-the-fly ROI extraction takes only **0.56–1.00 ms** on CPU host, delivering sustained end-to-end pipeline throughput of **>110 FPS** on an NVIDIA RTX 4060.

---

## Experimental Setup & Rigor

* **Model**: Ultralytics `YOLO11n` (2.59M parameters, 6.4 GFLOPs at 640×640).
* **Partitions Tested**:
  * **Baseline**: Full $640 \times 640$ frame downscaled bilinearly ($640$, $320$, $160$, $80$).
  * **Cropped ROI**: Fixed $400 \times 400$ driver-centric bounding region ($X \in [140, 540], Y \in [120, 520]$).
  * **Collage**: 3-strip canonical interaction composite (Head strip + Left/Right interaction quadrants).
* **Disjoint Subject Split (8:3:3)**:
  * **Train (8 subjects)**: `subject_01`, `04`, `06`, `07`, `08`, `09`, `13`, `14` (8,984 images)
  * **Val (3 subjects)**: `subject_02`, `03`, `11` (3,423 images)
  * **Test (3 subjects)**: `subject_05`, `10`, `12` (3,213 images)
  * *Strictly zero subject overlap across splits.*
* **Zero Data Leakage Calibration**:
  * Confidence threshold $\tau^*$ is calibrated out-of-sample on the `val` split to maximize macro F1, frozen, and evaluated blind on the held-out `test` split.
* **Seeds**: 3 independent seeds ($0, 1, 2$) per configuration ($36$ models trained to 100 epochs with AMP, batch size 16).

---

## Benchmark Results

All metrics reported on the **held-out test split** across 3 seeds ($\text{Mean} \pm \text{Std}$).

### Table 1: Baselines
| Input size | Source | Resize method | Precision ($\tau^*$) | Recall ($\tau^*$) | F1 ($\tau^*$) | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | E2E Pipeline (ms) | E2E FPS | Preprocess (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **640×640** | Native (baseline) | — | 0.882 ± 0.020 | 0.846 ± 0.016 | 0.863 ± 0.002 | 0.889 ± 0.019 | 0.548 ± 0.003 | 9.80 ± 0.16 | 12.03 ± 0.21 | 83.1 ± 1.5 | 0.00 ± 0.00 | 2.59 |
| **320×320** | Downscale from 640 | Bilinear | 0.913 ± 0.019 | 0.813 ± 0.030 | 0.859 ± 0.009 | 0.886 ± 0.008 | 0.560 ± 0.004 | 10.35 ± 1.04 | 10.83 ± 0.26 | 92.4 ± 2.3 | 1.75 ± 0.13 | 2.59 |
| **160×160** | Downscale from 640 | Bilinear | 0.875 ± 0.009 | 0.814 ± 0.011 | 0.844 ± 0.003 | 0.837 ± 0.002 | 0.504 ± 0.000 | 9.67 ± 0.82 | 8.51 ± 0.31 | 117.7 ± 4.1 | 1.18 ± 0.02 | 2.59 |
| **80×80** | Downscale from 640 | Bilinear | 0.889 ± 0.018 | 0.704 ± 0.022 | 0.786 ± 0.016 | 0.772 ± 0.027 | 0.437 ± 0.016 | 9.88 ± 0.22 | 9.11 ± 0.24 | 109.9 ± 2.9 | 1.04 ± 0.02 | 2.59 |

### Table 2: Cropped ROI (400×400)
| Input size | Source | Resize method | Precision ($\tau^*$) | Recall ($\tau^*$) | F1 ($\tau^*$) | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | E2E Pipeline (ms) | E2E FPS | Preprocess (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **640×640** | Upscale from 400 | Bilinear | 0.879 ± 0.024 | 0.816 ± 0.028 | 0.846 ± 0.020 | 0.850 ± 0.001 | 0.571 ± 0.010 | 10.02 ± 0.27 | 15.10 ± 0.28 | 66.2 ± 1.3 | 2.69 ± 0.07 | 2.59 |
| **320×320** | Downscale from 400 | Bilinear | 0.865 ± 0.030 | 0.864 ± 0.014 | 0.864 ± 0.009 | 0.885 ± 0.004 | 0.608 ± 0.016 | 9.42 ± 0.65 | 9.21 ± 0.39 | 108.8 ± 4.7 | 1.00 ± 0.03 | 2.59 |
| **160×160** | Downscale from 400 | Bilinear | 0.886 ± 0.014 | 0.817 ± 0.022 | 0.850 ± 0.013 | 0.845 ± 0.010 | 0.566 ± 0.017 | 9.80 ± 0.38 | 8.90 ± 0.30 | 112.5 ± 3.8 | 0.66 ± 0.02 | 2.59 |
| **80×80** | Downscale from 400 | Bilinear | 0.871 ± 0.041 | 0.730 ± 0.058 | 0.791 ± 0.019 | **0.813 ± 0.011** | **0.514 ± 0.008** | 10.46 ± 0.63 | 8.80 ± 0.19 | **113.7 ± 2.4** | 0.56 ± 0.00 | 2.59 |

### Table 3: Overlapping Collage (640×640)
| Input size | Source | Resize method | Precision ($\tau^*$) | Recall ($\tau^*$) | F1 ($\tau^*$) | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | E2E Pipeline (ms) | E2E FPS | Preprocess (ms) | Params (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **640×640** | Collage from 640 | Bilinear | 0.853 ± 0.012 | 0.811 ± 0.013 | 0.832 ± 0.012 | 0.805 ± 0.016 | 0.526 ± 0.012 | 10.45 ± 0.38 | 13.13 ± 0.59 | 76.3 ± 3.4 | 0.76 ± 0.01 | 2.59 |
| **320×320** | Downscale from 640 | Bilinear | 0.868 ± 0.011 | 0.826 ± 0.015 | 0.847 ± 0.006 | 0.819 ± 0.006 | 0.538 ± 0.006 | 11.03 ± 0.48 | 11.60 ± 0.64 | 86.4 ± 4.6 | 2.35 ± 0.04 | 2.59 |
| **160×160** | Downscale from 640 | Bilinear | 0.838 ± 0.008 | 0.698 ± 0.030 | 0.761 ± 0.020 | 0.716 ± 0.012 | 0.452 ± 0.011 | 9.98 ± 0.22 | 11.38 ± 0.33 | 87.9 ± 2.5 | 1.95 ± 0.06 | 2.59 |
| **80×80** | Downscale from 640 | Bilinear | 0.701 ± 0.079 | 0.584 ± 0.012 | 0.635 ± 0.035 | 0.520 ± 0.047 | 0.294 ± 0.015 | 9.96 ± 0.42 | 10.25 ± 0.22 | 97.6 ± 2.1 | 1.77 ± 0.05 | 2.59 |

### Table 4: Per-Class mAP@0.5 Breakdown
| Method & Resolution | Yawning | Hand Over Mouth | Drinking | Phone Use | Macro Mean |
|---|---|---|---|---|---|
| **Baseline @ 640×640** | 0.870 ± 0.070 | 0.895 ± 0.024 | 0.934 ± 0.031 | 0.860 ± 0.025 | **0.889 ± 0.019** |
| **Baseline @ 320×320** | 0.879 ± 0.013 | 0.876 ± 0.017 | **0.937 ± 0.010** | 0.853 ± 0.033 | 0.886 ± 0.008 |
| **Baseline @ 160×160** | 0.713 ± 0.048 | 0.880 ± 0.014 | 0.928 ± 0.031 | 0.827 ± 0.021 | 0.837 ± 0.002 |
| **Baseline @ 80×80** | 0.496 ± 0.071 | 0.840 ± 0.024 | 0.938 ± 0.026 | 0.812 ± 0.012 | 0.772 ± 0.027 |
| **Cropped @ 640×640** | 0.801 ± 0.026 | 0.907 ± 0.036 | 0.883 ± 0.014 | 0.810 ± 0.031 | 0.850 ± 0.001 |
| **Cropped @ 320×320** | **0.934 ± 0.012** | **0.909 ± 0.048** | 0.872 ± 0.023 | 0.825 ± 0.030 | 0.885 ± 0.004 |
| **Cropped @ 160×160** | **0.907 ± 0.018** | 0.856 ± 0.035 | 0.830 ± 0.020 | 0.787 ± 0.013 | **0.845 ± 0.010** |
| **Cropped @ 80×80** | **0.733 ± 0.043** | **0.872 ± 0.038** | 0.823 ± 0.009 | **0.823 ± 0.017** | **0.813 ± 0.011** |
| **Collage @ 640×640** | 0.787 ± 0.040 | 0.806 ± 0.042 | 0.871 ± 0.008 | 0.755 ± 0.011 | 0.805 ± 0.016 |
| **Collage @ 320×320** | 0.808 ± 0.022 | 0.834 ± 0.020 | 0.873 ± 0.016 | 0.759 ± 0.005 | 0.819 ± 0.006 |
| **Collage @ 160×160** | 0.547 ± 0.031 | 0.721 ± 0.035 | 0.829 ± 0.030 | 0.767 ± 0.006 | 0.716 ± 0.012 |
| **Collage @ 80×80** | 0.256 ± 0.072 | 0.511 ± 0.051 | 0.575 ± 0.042 | 0.736 ± 0.026 | 0.520 ± 0.047 |

### Table 5: Accuracy vs Throughput Pareto Trade-off
*Composite Accuracy denotes $\frac{1}{3}(\text{F1} + \text{mAP}_{50} + \text{mAP}_{50\text{--}95})$. $\Delta\text{FPS}$ relative to native 640 Baseline ($83.1\,\text{FPS}$). Batch size 1 on RTX 4060.*

| Method | Resolution | Composite Accuracy | E2E Latency (ms) | E2E FPS | $\Delta$ FPS (vs Baseline 640) |
|---|---|---|---|---|---|
| **Baseline** | 640×640 | 0.767 ± 0.008 | 12.03 ± 0.21 | 83.1 ± 1.5 | --- (Ref) |
| | 320×320 | **0.769 ± 0.006** | 10.83 ± 0.26 | 92.4 ± 2.3 | +9.3 |
| | 160×160 | 0.728 ± 0.002 | 8.51 ± 0.31 | 117.7 ± 4.1 | +34.6 |
| | 80×80 | 0.665 ± 0.023 | 9.11 ± 0.24 | 109.9 ± 2.9 | +26.8 |
| **Cropped** | 640×640* | 0.756 ± 0.012 | 15.10 ± 0.28 | 66.2 ± 1.3 | -16.9 |
| | 320×320 | **0.786 ± 0.004** | 9.21 ± 0.39 | 108.8 ± 4.7 | **+25.7** |
| | 160×160 | 0.754 ± 0.014 | 8.90 ± 0.30 | 112.5 ± 3.8 | +29.4 |
| | 80×80 | 0.706 ± 0.007 | 8.80 ± 0.19 | 113.7 ± 2.4 | +30.6 |
| **Collage** | 640×640 | 0.721 ± 0.016 | 13.13 ± 0.59 | 76.3 ± 3.4 | -6.8 |
| | 320×320 | **0.735 ± 0.001** | 11.60 ± 0.64 | 86.4 ± 4.6 | +3.3 |
| | 160×160 | 0.643 ± 0.017 | 11.38 ± 0.33 | 87.9 ± 2.5 | +4.8 |
| | 80×80 | 0.483 ± 0.040 | 10.25 ± 0.22 | 97.6 ± 2.1 | +14.5 |

---

## Repository Structure

```text
ROI-DMS/
├── configs/partitions/        # YAML dataset configs for Ultralytics
├── docs/                      # Original specifications and crop parameters
├── images/crop/               # Visual diagrams (FIGURE_1, FIGURE_2)
├── results/                   # Evaluation summaries and state
│   ├── progress.json          # Complete 36-run tracking log
│   ├── summary_table.csv      # Model-level macro metrics & latency
│   ├── per_class_summary.csv  # Class-level precision, recall, mAP, F1
│   └── rebenchmarked_fps.csv  # Sustained hardware benchmark verification log
├── scripts/
│   ├── partition_pipeline.py  # Deterministic crop, collage & bbox projection
│   ├── process_all_partitions.py # Batch processor for all resolution subsets
│   ├── benchmark_gpu.py       # Two-tier latency benchmark (Inference + E2E)
│   ├── rebenchmark_fps.py     # Batch FPS & latency re-verification orchestrator
│   └── train_eval_benchmark.py # Training, calibration, testing & reporting orchestrator
├── results_timestamp.md       # Compiled benchmark markdown tables
└── README.md
```

---

## Reproducing Results

### 1. Requirements
```bash
pip install ultralytics torch torchvision pandas numpy pillow
```

### 2. Generate Dataset Partitions
Generate the deterministic multi-resolution partitions from the raw DMD dataset:
```bash
python scripts/process_all_partitions.py
```

### 3. Run Training, Validation, and Evaluation
Run the complete experimental pipeline across all 3 seeds and 4 resolutions:
```bash
python scripts/train_eval_benchmark.py --resolutions 640x640 320x320 160x160 80x80 --seeds 0 1 2 --phase all --device 0
```

### 4. Standalone Benchmark & Report Generation
To re-evaluate existing trained models and regenerate the markdown tables:
```bash
python scripts/train_eval_benchmark.py --resolutions 640x640 320x320 160x160 80x80 --phase test_only --device 0
```

### 5. Re-Run Throughput & Latency Benchmarks
To run a batch benchmark across all 36 models on GPU:
```bash
python scripts/rebenchmark_fps.py
```
Outputs are written to `results/summary_table.csv`, `results/rebenchmarked_fps.csv`, and `results_timestamp.md`.

---

## License
MIT License.
