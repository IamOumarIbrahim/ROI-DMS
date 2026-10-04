"""
ROI-DMS: Rigorous Training, Out-of-Sample Calibration, and Evaluation Pipeline
=============================================================================
Self-contained research runner for input-side Driver Monitoring System (DMS) optimization.

Protocol Specifications:
- Model: YOLO11n
- Seeds: [0, 1, 2]
- Methods: baseline, cropped, collage
- Resolutions: 640x640, 320x320, 160x160, 80x80, 1280x1280
- Strict Two-Stage Architecture:
    * Stage 1: All 45 models trained and validated first.
    * Stage 2: Once training & validation is set, only testing/benchmarking happens at a time.
    * Stage 3: Auto-populate markdown tables in root as results_timestamp.md.
- Strict Control Variables:
    * Pretrained weights: models/pretrained/yolo11n.pt
    * Epochs: 100
    * Batch Size: 16
    * Device: cuda:0 (NVIDIA GeForce RTX 4060)
    * AMP: True
    * Early stopping: Disabled (patience=0)
- Zero Data Leakage:
    * Training: train split
    * Checkpoint selection: validation split (best.pt)
    * Threshold Calibration: tau* tuned strictly on validation split
    * Final Benchmarking: isolated held-out test split at frozen tau*, tau=0.25, tau=0.50
- Comprehensive Per-Class Logging:
    * yawning, hand_over_mouth, drinking, phone_use
"""

import os
import sys
import json
import time
import datetime
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from benchmark_gpu import benchmark_model_gpu, benchmark_pipeline_gpu

WORKSPACE_ROOT = Path("c:/Dev/repos/Public repos/research/ROI-DMS")
PRETRAINED_WEIGHTS = WORKSPACE_ROOT / "models/pretrained/yolo11n.pt"
CONFIGS_DIR = WORKSPACE_ROOT / "configs/partitions"
RUNS_DIR = WORKSPACE_ROOT / "runs/detect"
RESULTS_DIR = WORKSPACE_ROOT / "results"
PROGRESS_FILE = RESULTS_DIR / "progress.json"
SUMMARY_CSV = RESULTS_DIR / "summary_table.csv"
PER_CLASS_CSV = RESULTS_DIR / "per_class_summary.csv"

MODEL_NAME = "yolo11n"
SEEDS = [0, 1, 2]
METHODS = ["baseline", "cropped", "collage"]
# 640x640 prioritized first for the core comparison, then resolution sweeps
RESOLUTIONS = ["640x640", "320x320", "160x160", "80x80", "1280x1280"]

EPOCHS = 100
BATCH_SIZE = 16

CLASS_NAMES = {
    0: "yawning",
    1: "hand_over_mouth",
    2: "drinking",
    3: "phone_use"
}


def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        try:
            return json.loads(PROGRESS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(progress: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2), encoding="utf-8")


def append_summary_csv(entry: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    flat_entry = {k: v for k, v in entry.items() if not isinstance(v, (dict, list))}
    df = pd.DataFrame([flat_entry])
    if not SUMMARY_CSV.exists():
        df.to_csv(SUMMARY_CSV, index=False)
    else:
        df.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)


def append_per_class_csv(rows: list):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not PER_CLASS_CSV.exists():
        df.to_csv(PER_CLASS_CSV, index=False)
    else:
        df.to_csv(PER_CLASS_CSV, mode="a", header=False, index=False)


def calibrate_threshold_on_val(model: YOLO, cfg_file: str, imgsz: int, device: int) -> float:
    """
    Calibrates confidence threshold tau* strictly on the validation split.
    Uses Ultralytics PR curves on 'val' to select tau* that maximizes overall F1.
    Guarantees that test data is NEVER inspected during threshold selection.
    """
    val_res = model.val(
        data=cfg_file,
        split="val",
        imgsz=imgsz,
        conf=0.001,
        device=device,
        verbose=False
    )
    
    try:
        f1_curve = val_res.box.f1_curve  # shape (num_classes, 1000)
        mean_f1 = np.mean(f1_curve, axis=0)
        best_idx = int(np.argmax(mean_f1))
        tau_star = round(float(best_idx / 1000.0), 3)
        tau_star = max(0.05, min(0.85, tau_star))
    except Exception:
        tau_star = 0.25
        
    return tau_star


def extract_per_class_metrics(val_res) -> dict:
    """
    Extracts precision, recall, f1, ap50, and ap50-95 per class.
    """
    out = {}
    ap_cls = list(val_res.box.ap_class_index) if hasattr(val_res.box, "ap_class_index") else []
    for cid, cname in CLASS_NAMES.items():
        if cid in ap_cls:
            idx = ap_cls.index(cid)
            p = float(val_res.box.p[idx]) if hasattr(val_res.box, "p") and len(val_res.box.p) > idx else 0.0
            r = float(val_res.box.r[idx]) if hasattr(val_res.box, "r") and len(val_res.box.r) > idx else 0.0
            f1 = float(val_res.box.f1[idx]) if hasattr(val_res.box, "f1") and len(val_res.box.f1) > idx else (
                2 * p * r / (p + r + 1e-8) if (p + r) > 0 else 0.0
            )
            ap50 = float(val_res.box.all_ap[idx, 0]) if hasattr(val_res.box, "all_ap") and len(val_res.box.all_ap) > idx else 0.0
            ap50_95 = float(val_res.box.all_ap[idx].mean()) if hasattr(val_res.box, "all_ap") and len(val_res.box.all_ap) > idx else 0.0
        else:
            p, r, f1, ap50, ap50_95 = 0.0, 0.0, 0.0, 0.0, 0.0
            
        out[cname] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "map50": round(ap50, 4),
            "map50_95": round(ap50_95, 4)
        }
    return out


def train_single_run(method: str, resolution: str, seed: int, device: int = 0) -> dict:
    """
    Stage 1: Trains a model on 'train' and validates on 'val' to produce best.pt.
    """
    run_key = f"{MODEL_NAME}_{method}_{resolution}_seed{seed}"
    res_w = int(resolution.split("x")[0])
    cfg_file = str(CONFIGS_DIR / f"{method}_{resolution}.yaml")
    
    project_dir = RUNS_DIR / MODEL_NAME / f"{method}_{resolution}"
    run_name = f"seed_{seed}"
    expected_best = project_dir / run_name / "weights/best.pt"

    print(f"\n{'='*70}")
    print(f"[STAGE 1 - TRAINING] Run: {run_key}")
    print(f"Model: {MODEL_NAME} | Method: {method} | Resolution: {resolution} | Seed: {seed}")
    print(f"Pretrained weights: {PRETRAINED_WEIGHTS}")
    print(f"Config: {cfg_file}")
    print(f"{'='*70}\n")

    t_start = time.time()

    if not expected_best.exists():
        model = YOLO(str(PRETRAINED_WEIGHTS))
        _ = model.train(
            data=cfg_file,
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=res_w,
            patience=0,
            amp=True,
            seed=seed,
            device=device,
            project=str(project_dir.resolve()),
            name=run_name,
            exist_ok=True,
            verbose=False
        )
    else:
        print(f"Existing best weights found at {expected_best}. Skipping re-training.")

    train_time = round(time.time() - t_start, 1)
    return {
        "run_key": run_key,
        "status": "trained",
        "best_weights": str(expected_best),
        "train_time_s": train_time
    }


def evaluate_single_run(method: str, resolution: str, seed: int, device: int = 0) -> dict:
    """
    Stage 2: Strictly isolated test evaluation & GPU latency benchmarking (one at a time).
    """
    run_key = f"{MODEL_NAME}_{method}_{resolution}_seed{seed}"
    res_w = int(resolution.split("x")[0])
    cfg_file = str(CONFIGS_DIR / f"{method}_{resolution}.yaml")
    
    project_dir = RUNS_DIR / MODEL_NAME / f"{method}_{resolution}"
    run_name = f"seed_{seed}"
    expected_best = project_dir / run_name / "weights/best.pt"

    if not expected_best.exists():
        raise FileNotFoundError(f"Cannot evaluate {run_key}: weights not found at {expected_best}")

    print(f"\n{'='*70}")
    print(f"[STAGE 2 - ISOLATED TESTING & BENCHMARKING] Run: {run_key}")
    print(f"Model: {MODEL_NAME} | Method: {method} | Resolution: {resolution} | Seed: {seed}")
    print(f"Weights: {expected_best}")
    print(f"{'='*70}\n")

    eval_model = YOLO(str(expected_best))

    # 1. Strict Out-of-Sample Calibration (Validation ONLY)
    print("Calibrating confidence threshold tau* strictly on validation split...")
    tau_star = calibrate_threshold_on_val(eval_model, cfg_file, res_w, device)
    print(f"Calibrated tau* = {tau_star:.3f} on validation set.")

    # 2. Held-Out Test Evaluation at Frozen tau*
    print(f"Evaluating held-out test split at frozen tau* = {tau_star:.3f}...")
    test_calibrated = eval_model.val(
        data=cfg_file,
        split="test",
        conf=tau_star,
        imgsz=res_w,
        device=device,
        verbose=False
    )

    # 3. Held-Out Test Evaluation at Fixed Standard Thresholds (tau=0.25 and tau=0.50)
    print("Evaluating held-out test split at fixed tau=0.25...")
    test_tau25 = eval_model.val(
        data=cfg_file,
        split="test",
        conf=0.25,
        imgsz=res_w,
        device=device,
        verbose=False
    )

    print("Evaluating held-out test split at fixed tau=0.50...")
    test_tau50 = eval_model.val(
        data=cfg_file,
        split="test",
        conf=0.50,
        imgsz=res_w,
        device=device,
        verbose=False
    )

    pc_calibrated = extract_per_class_metrics(test_calibrated)
    pc_tau25 = extract_per_class_metrics(test_tau25)
    pc_tau50 = extract_per_class_metrics(test_tau50)

    map50 = float(test_tau25.box.map50)
    map50_95 = float(test_tau25.box.map)

    # 4. Isolated GPU Latency Benchmarking (Two-Tier Evaluation)
    print("Benchmarking single-stream GPU forward latency (Tier 1: Model Forward)...")
    bench_tier1 = benchmark_model_gpu(str(expected_best), imgsz=res_w, device=f"cuda:{device}")

    print("Benchmarking on-the-fly End-to-End pipeline latency (Tier 2: Pipeline)...")
    bench_tier2 = benchmark_pipeline_gpu(str(expected_best), method=method, imgsz=res_w, device=f"cuda:{device}")

    entry = {
        "run_key": run_key,
        "status": "evaluated",
        "model": MODEL_NAME,
        "method": method,
        "resolution": resolution,
        "seed": seed,
        # Calibrated macro metrics
        "tau_star": tau_star,
        "precision_tau_star": round(float(test_calibrated.box.mp), 4),
        "recall_tau_star": round(float(test_calibrated.box.mr), 4),
        "f1_tau_star": round(2 * float(test_calibrated.box.mp) * float(test_calibrated.box.mr) / (float(test_calibrated.box.mp) + float(test_calibrated.box.mr) + 1e-8), 4),
        # Fixed tau=0.25 macro metrics
        "precision_tau25": round(float(test_tau25.box.mp), 4),
        "recall_tau25": round(float(test_tau25.box.mr), 4),
        "f1_tau25": round(2 * float(test_tau25.box.mp) * float(test_tau25.box.mr) / (float(test_tau25.box.mp) + float(test_tau25.box.mr) + 1e-8), 4),
        # Fixed tau=0.50 macro metrics
        "precision_tau50": round(float(test_tau50.box.mp), 4),
        "recall_tau50": round(float(test_tau50.box.mr), 4),
        "f1_tau50": round(2 * float(test_tau50.box.mp) * float(test_tau50.box.mr) / (float(test_tau50.box.mp) + float(test_tau50.box.mr) + 1e-8), 4),
        # Benchmark metrics
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "latency_ms": bench_tier1["mean_ms"],
        "fps": bench_tier1["fps"],
        "e2e_latency_ms": bench_tier2["e2e_mean_ms"],
        "e2e_fps": bench_tier2["e2e_fps"],
        "preprocess_ms": bench_tier2["preprocess_ms"],
        "params_m": bench_tier1["params_m"],
        "best_weights": str(expected_best),
        "per_class": pc_calibrated
    }

    # Flatten per-class metrics into main table
    for cname in CLASS_NAMES.values():
        entry[f"map50_{cname}"] = pc_tau25[cname]["map50"]
        entry[f"map50_95_{cname}"] = pc_tau25[cname]["map50_95"]
        entry[f"f1_tau_star_{cname}"] = pc_calibrated[cname]["f1"]
        entry[f"recall_tau_star_{cname}"] = pc_calibrated[cname]["recall"]
        entry[f"prec_tau_star_{cname}"] = pc_calibrated[cname]["precision"]
        entry[f"f1_tau25_{cname}"] = pc_tau25[cname]["f1"]
        entry[f"f1_tau50_{cname}"] = pc_tau50[cname]["f1"]

    per_class_rows = []
    for cname in CLASS_NAMES.values():
        per_class_rows.append({
            "run_key": run_key,
            "model": MODEL_NAME,
            "method": method,
            "resolution": resolution,
            "seed": seed,
            "class_name": cname,
            "map50": pc_tau25[cname]["map50"],
            "map50_95": pc_tau25[cname]["map50_95"],
            "tau_star": tau_star,
            "prec_tau_star": pc_calibrated[cname]["precision"],
            "recall_tau_star": pc_calibrated[cname]["recall"],
            "f1_tau_star": pc_calibrated[cname]["f1"],
            "prec_tau25": pc_tau25[cname]["precision"],
            "recall_tau25": pc_tau25[cname]["recall"],
            "f1_tau25": pc_tau25[cname]["f1"],
            "prec_tau50": pc_tau50[cname]["precision"],
            "recall_tau50": pc_tau50[cname]["recall"],
            "f1_tau50": pc_tau50[cname]["f1"]
        })

    append_per_class_csv(per_class_rows)
    return entry


def generate_results_markdown() -> str:
    """
    Stage 3: Compiles clean markdown summary tables into results_timestamp.md in the root directory.
    Matches the schema of docs/tables.md.
    """
    if not SUMMARY_CSV.exists():
        print(f"No summary CSV found at {SUMMARY_CSV}")
        return ""

    df = pd.read_csv(SUMMARY_CSV)
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    md = []
    md.append(f"# ROI-DMS Experimental Benchmark Results")
    md.append(f"**Generated**: {now_str}  ")
    md.append(f"**Model**: YOLO11n | **Hardware**: NVIDIA GeForce RTX 4060 8GB | **Calibration Split**: Validation (Zero Leakage)  \n")

    # Helper for formatting mean +/- std
    def format_metric(sub_df, col, decimals=4):
        vals = sub_df[col].dropna().values
        if len(vals) == 0:
            return "—"
        if len(vals) == 1:
            return f"{vals[0]:.{decimals}f}"
        return f"{np.mean(vals):.{decimals}f} ± {np.std(vals):.{decimals}f}"

    source_descriptions = {
        "baseline": {
            "1280x1280": "Synthetic upscale from 640",
            "640x640": "Native (baseline)",
            "320x320": "Downscale from 640",
            "160x160": "Downscale from 640",
            "80x80": "Downscale from 640"
        },
        "cropped": {
            "1280x1280": "Synthetic upscale from 400",
            "640x640": "Synthetic upscale from 400",
            "320x320": "Downscale from 400",
            "160x160": "Downscale from 400",
            "80x80": "Downscale from 400"
        },
        "collage": {
            "1280x1280": "Synthetic upscale from 640",
            "640x640": "Collage from 640",
            "320x320": "Downscale from 640",
            "160x160": "Downscale from 640",
            "80x80": "Downscale from 640"
        }
    }

    # Generate Tables 1, 2, 3 as defined in docs/tables.md
    method_titles = {
        "baseline": "Table 1 Baselines",
        "cropped": "Table 2 Cropped ROI (400x400)",
        "collage": "Table 3 Overlapping collage (640x640)"
    }

    order_res = ["640x640", "320x320", "160x160", "80x80"]

    for method, title in method_titles.items():
        md.append(f"## {title}")
        md.append("| Input size | Source | Resize method | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | E2E Pipeline (ms) | E2E FPS | Preprocess (ms) | Params (M) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        
        m_df = df[df["method"] == method]
        for res in order_res:
            r_df = m_df[m_df["resolution"] == res]
            src = source_descriptions[method].get(res, "—")
            resize = "—" if (method == "baseline" and res == "640x640") else "Bilinear"
            
            p_str = format_metric(r_df, "precision_tau_star", 3)
            r_str = format_metric(r_df, "recall_tau_star", 3)
            f1_str = format_metric(r_df, "f1_tau_star", 3)
            map50_str = format_metric(r_df, "map50", 3)
            map95_str = format_metric(r_df, "map50_95", 3)
            lat_str = format_metric(r_df, "latency_ms", 2)
            e2e_str = format_metric(r_df, "e2e_latency_ms", 2) if "e2e_latency_ms" in df.columns else "—"
            fps_str = format_metric(r_df, "e2e_fps", 1) if "e2e_fps" in df.columns else format_metric(r_df, "fps", 1)
            pre_str = format_metric(r_df, "preprocess_ms", 2) if "preprocess_ms" in df.columns else "—"
            par_str = format_metric(r_df, "params_m", 2)
            
            md.append(f"| {res} | {src} | {resize} | {p_str} | {r_str} | {f1_str} | {map50_str} | {map95_str} | {lat_str} | {e2e_str} | {fps_str} | {pre_str} | {par_str} |")
        md.append("")

    # Table 4: Core 640x640 Comparison
    md.append("## Table 4: Core 640x640 Method Comparison (Priority Evaluation)")
    md.append("| Method | Seeds Evaluated | Precision (tau*) | Recall (tau*) | F1 (tau*) | mAP@0.5 | mAP@0.5:0.95 | Inference (ms) | E2E Pipeline (ms) | E2E FPS | Preprocess (ms) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    df_640 = df[df["resolution"] == "640x640"]
    for m in METHODS:
        m_sub = df_640[df_640["method"] == m]
        n_seeds = len(m_sub)
        p_str = format_metric(m_sub, "precision_tau_star", 3)
        r_str = format_metric(m_sub, "recall_tau_star", 3)
        f1_str = format_metric(m_sub, "f1_tau_star", 3)
        map50_str = format_metric(m_sub, "map50", 3)
        map95_str = format_metric(m_sub, "map50_95", 3)
        lat_str = format_metric(m_sub, "latency_ms", 2)
        e2e_str = format_metric(m_sub, "e2e_latency_ms", 2) if "e2e_latency_ms" in df.columns else "—"
        fps_str = format_metric(m_sub, "e2e_fps", 1) if "e2e_fps" in df.columns else format_metric(m_sub, "fps", 1)
        pre_str = format_metric(m_sub, "preprocess_ms", 2) if "preprocess_ms" in df.columns else "—"
        md.append(f"| **{m.capitalize()}** | {n_seeds}/3 | {p_str} | {r_str} | {f1_str} | {map50_str} | {map95_str} | {lat_str} | {e2e_str} | {fps_str} | {pre_str} |")
    md.append("")

    # Table 5: Out-of-Sample Threshold Calibration Comparison
    md.append("## Table 5: Out-of-Sample Calibration Impact (Frozen tau* vs Fixed Thresholds)")
    md.append("| Method & Resolution | tau* (Mean) | F1 @ tau* | F1 @ tau=0.25 | F1 @ tau=0.50 | Delta F1 (tau* - tau=0.25) |")
    md.append("|---|---|---|---|---|---|")
    for m in METHODS:
        for res in order_res:
            sub = df[(df["method"] == m) & (df["resolution"] == res)]
            if len(sub) == 0:
                continue
            tau_mean = format_metric(sub, "tau_star", 3)
            f1_star = format_metric(sub, "f1_tau_star", 3)
            f1_25 = format_metric(sub, "f1_tau25", 3)
            f1_50 = format_metric(sub, "f1_tau50", 3)
            diff = np.mean(sub["f1_tau_star"].values) - np.mean(sub["f1_tau25"].values)
            md.append(f"| {m.capitalize()} @ {res} | {tau_mean} | {f1_star} | {f1_25} | {f1_50} | {diff:+.3f} |")
    md.append("")

    # Table 6: Per-Class Performance Summary (AP@0.5)
    md.append("## Table 6: Per-Class Detection Performance (mAP@0.5)")
    md.append("| Method & Resolution | Yawning | Hand Over Mouth | Drinking | Phone Use | Macro Mean |")
    md.append("|---|---|---|---|---|---|")
    for m in METHODS:
        for res in order_res:
            sub = df[(df["method"] == m) & (df["resolution"] == res)]
            if len(sub) == 0:
                continue
            yaw = format_metric(sub, "map50_yawning", 3)
            hom = format_metric(sub, "map50_hand_over_mouth", 3)
            drk = format_metric(sub, "map50_drinking", 3)
            phn = format_metric(sub, "map50_phone_use", 3)
            mac = format_metric(sub, "map50", 3)
            md.append(f"| {m.capitalize()} @ {res} | {yaw} | {hom} | {drk} | {phn} | {mac} |")
    md.append("")

    content = "\n".join(md)
    
    # Save both results_timestamp.md and timestamped copy in root
    out_file_static = WORKSPACE_ROOT / "results_timestamp.md"
    out_file_dynamic = WORKSPACE_ROOT / f"results_{timestamp_tag}.md"
    
    out_file_static.write_text(content, encoding="utf-8")
    out_file_dynamic.write_text(content, encoding="utf-8")
    print(f"\n[SUMMARY] Saved results table to {out_file_static} and {out_file_dynamic}")
    return content


def main():
    parser = argparse.ArgumentParser(description="ROI-DMS Training & Evaluation Pipeline")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--resolutions", nargs="+", default=RESOLUTIONS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--phase", choices=["all", "train_only", "test_only", "report_only"], default="all")
    args = parser.parse_args()

    if args.phase == "report_only":
        generate_results_markdown()
        return

    progress = load_progress()
    total_configs = len(args.methods) * len(args.resolutions) * len(args.seeds)

    print("=" * 70)
    print(f"ROI-DMS EXPERIMENTAL PIPELINE: Phase = {args.phase.upper()}")
    print(f"Total configurations in queue: {total_configs}")
    print(f"Model: {MODEL_NAME}")
    print(f"Seeds: {args.seeds}")
    print(f"Resolutions: {args.resolutions}")
    print(f"Methods: {args.methods}")
    print("=" * 70)

    # -------------------------------------------------------------
    # STAGE 1: Train & Validate ALL models first
    # -------------------------------------------------------------
    if args.phase in ["all", "train_only"]:
        print("\n>>> STARTING STAGE 1: TRAINING & VALIDATION ACROSS ALL MODELS <<<")
        for resolution in args.resolutions:
            for method in args.methods:
                for seed in args.seeds:
                    run_key = f"{MODEL_NAME}_{method}_{resolution}_seed{seed}"
                    
                    # Check if already trained
                    if run_key in progress and progress[run_key].get("status") in ["trained", "evaluated"]:
                        print(f"[{run_key}] Training already completed. Skipping.")
                        continue

                    try:
                        res = train_single_run(method, resolution, seed, device=args.device)
                        progress[run_key] = res
                        save_progress(progress)
                        print(f"[{run_key}] Training finished in {res['train_time_s']}s.")
                    except Exception as e:
                        print(f"ERROR during training of {run_key}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()

        print("\n>>> STAGE 1 COMPLETE: All models have finished training & validation! <<<\n")

    # -------------------------------------------------------------
    # STAGE 2: Once training & validation is set, ensure ONLY testing happens at a time
    # -------------------------------------------------------------
    if args.phase in ["all", "test_only"]:
        print("\n>>> STARTING STAGE 2: ISOLATED TESTING & LATENCY BENCHMARKING <<<")
        print("Note: Ensuring strictly ONE test / benchmark runs at a time on dedicated GPU.\n")
        
        for resolution in args.resolutions:
            for method in args.methods:
                for seed in args.seeds:
                    run_key = f"{MODEL_NAME}_{method}_{resolution}_seed{seed}"
                    
                    if run_key in progress and progress[run_key].get("status") == "evaluated":
                        print(f"[{run_key}] Already evaluated. Skipping.")
                        continue

                    try:
                        entry = evaluate_single_run(method, resolution, seed, device=args.device)
                        progress[run_key] = entry
                        save_progress(progress)
                        append_summary_csv(entry)
                        
                        # Dynamically update the summary markdown tables after each evaluation
                        generate_results_markdown()
                        
                        print(f"\n[EVAL DONE] {run_key}:")
                        print(f"  Calibrated tau*={entry['tau_star']} -> Recall={entry['recall_tau_star']}, Precision={entry['precision_tau_star']}, F1={entry['f1_tau_star']}")
                        print(f"  Fixed tau=0.25       -> Recall={entry['recall_tau25']}, Precision={entry['precision_tau25']}, F1={entry['f1_tau25']}")
                        print(f"  mAP50={entry['map50']} | mAP50-95={entry['map50_95']} | Latency={entry['latency_ms']} ms ({entry['fps']} FPS)")
                    except Exception as e:
                        print(f"ERROR during evaluation of {run_key}: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()

        print("\n>>> STAGE 2 COMPLETE: All models evaluated and benchmarked! <<<\n")

    # -------------------------------------------------------------
    # STAGE 3: Final Results Markdown Generation
    # -------------------------------------------------------------
    print("\n>>> GENERATING FINAL SUMMARY TABLES IN results_timestamp.md <<<")
    generate_results_markdown()
    print("Pipeline execution fully finished!")


if __name__ == "__main__":
    main()
