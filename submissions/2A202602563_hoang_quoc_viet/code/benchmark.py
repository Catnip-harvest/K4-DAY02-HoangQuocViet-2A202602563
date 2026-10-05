"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo được cài đặt ở đây:
  - warmup: bỏ `warmup` (mặc định 10) lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU mỗi lần đo
  - `iters` (mặc định 100, tối thiểu 50) lần đo, báo cáo p50, p95, p99 và mean
  - ghi GPU, dtype, batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý/đọc ảnh: chỉ đo forward của model với tensor đã nằm sẵn trên GPU
"""
from __future__ import annotations

import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây) sau warmup; sync() trước và sau mỗi lần đo."""
    if iters < 50:
        raise ValueError("cần >= 50 lần đo (GUIDE 4.1)")
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "n": iters, "warmup": warmup}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False, k_views: int = 1,
                   view_fn=None) -> dict:
    """Độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast FP16) | "fp16" (model.half()).
    k_views/view_fn: để đo TTA thật (mỗi lần đo chạy đủ K view rồi gộp softmax), xem tta_latency.
    """
    import copy

    import torch
    dev = torch.device(device)
    m = copy.deepcopy(model).to(dev).eval()
    m = m.to(memory_format=torch.channels_last)
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev).contiguous(memory_format=torch.channels_last)
    if dtype == "fp16":
        m = m.half()
        x = x.half()
    elif dtype not in ("fp32", "amp"):
        raise ValueError(f"dtype không hợp lệ: {dtype}")

    def forward():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16,
                                                    enabled=dtype == "amp" and dev.type == "cuda"):
            if view_fn is None:
                return m(x)
            views = view_fn(x)
            return torch.stack([m(v.contiguous(memory_format=torch.channels_last)).float().softmax(-1)
                                for v in views]).mean(0)

    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    r = bench(forward, warmup, iters, sync)
    gpu = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu"
    out = {"gpu": gpu, "dtype": dtype, "batch": batch_size, "img_size": img_size, "fused_bn": fused_bn,
           "k_views": k_views, **{k: r[k] for k in ("p50", "p95", "p99", "mean", "n", "warmup")},
           "images_per_s": batch_size / (r["p50"] / 1000.0), "torch": torch.__version__,
           "preprocessing_included": False}
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out


def tta_latency(model, k_views: int, view_fn=None, **kw) -> dict:
    """Độ trễ thật của TTA K view (chạy đủ K forward + gộp softmax trong mỗi lần đo).

    Nếu view_fn None: lặp lại cùng ảnh K lần (chi phí xấp xỉ K view cùng kích thước).
    So sánh p50 với K * p50 của 1 view để kiểm tra giả định "chi phí tuyến tính theo K".
    """
    if view_fn is None:
        def view_fn(x):
            return [x] * k_views
    return latency_report(model, k_views=k_views, view_fn=view_fn, **kw)
