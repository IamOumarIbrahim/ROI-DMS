"""
GPU Latency and GFLOPs Benchmarking Tool
========================================
Measures exact inference latency (synchronized CUDA events) and sustained FPS
on the active GPU (NVIDIA GeForce RTX 4060) under controlled conditions.
"""

import time
import torch
from pathlib import Path
from ultralytics import YOLO


def benchmark_model_gpu(model_path: str, imgsz: int = 640, batch_size: int = 1, warmup: int = 20, runs: int = 100, device: str = "cuda:0") -> dict:
    """
    Benchmarks single-stream GPU forward latency with CUDA events.
    Returns:
        dict with latency_mean_ms, latency_median_ms, fps, gflops, params_m
    """
    from ultralytics.utils.checks import check_imgsz
    imgsz = int(check_imgsz(imgsz, stride=32))
    
    model = YOLO(model_path)
    model.to(device)
    
    # Dummy tensor
    dummy = torch.randn(batch_size, 3, imgsz, imgsz, device=device)
    
    # Warmup
    for _ in range(warmup):
        _ = model.model(dummy)
    torch.cuda.synchronize()
    
    # Timing with CUDA Events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    
    for i in range(runs):
        start_events[i].record()
        _ = model.model(dummy)
        end_events[i].record()
        
    torch.cuda.synchronize()
    
    timings = [s.elapsed_time(e) for s, e in zip(start_events, end_events)] # ms
    timings.sort()
    
    mean_ms = sum(timings) / len(timings)
    median_ms = timings[len(timings) // 2]
    fps = 1000.0 / mean_ms
    
    # Model parameters
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    return {
        "model": str(model_path),
        "imgsz": imgsz,
        "device": torch.cuda.get_device_name(0),
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "fps": round(fps, 1),
        "params_m": round(params_m, 2)
    }


def benchmark_pipeline_gpu(model_path: str, method: str, imgsz: int = 640, warmup: int = 15, runs: int = 50, device: str = "cuda:0", sample_image_path: str = None) -> dict:
    """
    Benchmarks real-time on-the-fly End-to-End pipeline latency:
    Raw 640x640 Frame -> Crop/Stitch/Resize -> Tensor Transfer -> Forward Pass -> NMS
    """
    import sys
    from PIL import Image
    from ultralytics.utils.checks import check_imgsz
    imgsz = int(check_imgsz(imgsz, stride=32))
    sys.path.insert(0, str(Path(__file__).parent))
    from partition_pipeline import process_image

    if sample_image_path is None or not Path(sample_image_path).exists():
        sample_image_path = "data/processed/RGB/images/subject_01/video_01/subject_01_video_01_frame_0001.jpg"
    
    raw_img = Image.open(sample_image_path).convert("RGB")
    model = YOLO(model_path)
    
    # Warmup
    for _ in range(warmup):
        p = process_image(raw_img, method, (imgsz, imgsz))
        _ = model(p, imgsz=imgsz, device=device, verbose=False)
    torch.cuda.synchronize()
    
    pre_times = []
    e2e_times = []
    
    for _ in range(runs):
        t0 = time.perf_counter()
        p = process_image(raw_img, method, (imgsz, imgsz))
        t1 = time.perf_counter()
        _ = model(p, imgsz=imgsz, device=device, verbose=False)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        pre_times.append((t1 - t0) * 1000.0)
        e2e_times.append((t2 - t0) * 1000.0)
        
    e2e_times.sort()
    pre_times.sort()
    
    mean_e2e = sum(e2e_times) / len(e2e_times)
    median_e2e = e2e_times[len(e2e_times) // 2]
    mean_pre = sum(pre_times) / len(pre_times)
    fps_e2e = 1000.0 / mean_e2e
    
    return {
        "e2e_mean_ms": round(mean_e2e, 3),
        "e2e_median_ms": round(median_e2e, 3),
        "e2e_fps": round(fps_e2e, 1),
        "preprocess_ms": round(mean_pre, 3)
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark model latency on GPU")
    parser.add_argument("--model", type=str, default="models/pretrained/yolo11n.pt")
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    
    res_tier1 = benchmark_model_gpu(args.model, imgsz=args.imgsz)
    print("Tier 1 (GPU Forward):", res_tier1)
    res_tier2 = benchmark_pipeline_gpu(args.model, method=args.method, imgsz=args.imgsz)
    print("Tier 2 (End-to-End Pipeline):", res_tier2)
