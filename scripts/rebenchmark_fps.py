"""
Re-benchmark all 36 models on GPU for FPS & Latency verification.
Runs both Tier 1 (Pure GPU Forward) and Tier 2 (End-to-End Pipeline).
Saves detailed results to results/rebenchmarked_fps.csv.
"""

import sys
import time
import torch
import pandas as pd
from pathlib import Path

# Insert scripts dir
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_gpu import benchmark_model_gpu, benchmark_pipeline_gpu

def main():
    df_orig = pd.read_csv('results/summary_table.csv')
    res_list = ['640x640', '320x320', '160x160', '80x80']
    df_subset = df_orig[df_orig['resolution'].isin(res_list)].copy()
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0)})")
    
    results = []
    
    total = len(df_subset)
    for idx, (_, row) in enumerate(df_subset.iterrows(), 1):
        run_key = row['run_key']
        method = row['method']
        res_str = row['resolution']
        imgsz = int(res_str.split('x')[0])
        seed = row['seed']
        weights_path = row['best_weights']
        
        print(f"[{idx}/{total}] Benchmarking {run_key} ({method} @ {imgsz}px, seed {seed})...")
        
        # Tier 1: Pure forward pass
        t1 = benchmark_model_gpu(weights_path, imgsz=imgsz, warmup=20, runs=100, device=device)
        
        # Tier 2: E2E Pipeline
        t2 = benchmark_pipeline_gpu(weights_path, method=method, imgsz=imgsz, warmup=15, runs=50, device=device)
        
        results.append({
            "run_key": run_key,
            "method": method,
            "resolution": res_str,
            "imgsz": imgsz,
            "seed": seed,
            "weights_path": weights_path,
            # Tier 1 metrics
            "gpu_forward_ms": t1["mean_ms"],
            "gpu_forward_median_ms": t1["median_ms"],
            "gpu_forward_fps": t1["fps"],
            # Tier 2 metrics
            "preprocess_ms": t2["preprocess_ms"],
            "e2e_ms": t2["e2e_mean_ms"],
            "e2e_median_ms": t2["e2e_median_ms"],
            "e2e_fps": t2["e2e_fps"],
            # Original recorded for comparison
            "orig_latency_ms": row["latency_ms"],
            "orig_fps": row["fps"],
            "orig_preprocess_ms": row["preprocess_ms"],
            "orig_e2e_latency_ms": row["e2e_latency_ms"],
            "orig_e2e_fps": row["e2e_fps"]
        })
        
    df_res = pd.DataFrame(results)
    out_csv = Path("results/rebenchmarked_fps.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved re-benchmarked results to {out_csv}")
    
    # Aggregated summary
    grouped = df_res.groupby(['method', 'resolution']).agg({
        'e2e_fps': ['mean', 'std'],
        'e2e_ms': ['mean', 'std'],
        'preprocess_ms': ['mean', 'std'],
        'gpu_forward_fps': ['mean', 'std'],
        'gpu_forward_ms': ['mean', 'std'],
        'orig_e2e_fps': ['mean', 'std']
    })
    print("\n" + "="*80)
    print("RE-BENCHMARKED SUMMARY (Mean +- Std over 3 seeds):")
    print("="*80)
    print(grouped.to_string())

if __name__ == "__main__":
    main()
