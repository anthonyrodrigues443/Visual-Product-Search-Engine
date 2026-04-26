"""Benchmark the production inference pipeline component-by-component.

Anthony's evaluate.py computes a single "ms/query" number; this script
breaks that down into the three pieces a production engineer cares about:

    1. CLIP forward pass (the dominant cost)
    2. Color + spatial feature extraction (CPU-only NumPy)
    3. FAISS index search (sub-ms in practice)

Plus it computes the latency *distribution* (p50/p95/p99) and throughput,
which the test_benchmarks pytest contract verifies in CI.

Output:
    results/benchmark_report.json
    results/benchmark_latency.png

Usage:
    python scripts/benchmark_inference.py --n-queries 200 --gallery-size 300
    python scripts/benchmark_inference.py --use-stub      # skip CLIP download
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import train  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    extract_color_palette,
    extract_hsv_histogram,
    extract_spatial_color_grid,
)


def make_random_image(rng, size=224):
    return Image.fromarray((rng.rand(size, size, 3) * 255).astype(np.uint8), mode="RGB")


def percentiles(values, ps=(50, 95, 99)):
    s = sorted(values)
    n = len(s)
    return {f"p{p}": round(s[min(int(n * p / 100), n - 1)], 3) for p in ps}


def benchmark_block(name, fn, n_warmup=5, n_iter=50):
    for _ in range(n_warmup):
        fn()
    latencies_ms = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "name": name,
        "n_iter": n_iter,
        "mean_ms": round(mean(latencies_ms), 3),
        "median_ms": round(median(latencies_ms), 3),
        **percentiles(latencies_ms),
        "latencies": latencies_ms,
    }


def maybe_load_clip(use_stub: bool):
    if use_stub:
        from tests.test_integration import _StubCLIP, _stub_preprocess  # type: ignore

        return _StubCLIP(seed=0), _stub_preprocess, "stub-768D"

    cfg = train.load_config()
    model, pp = train._load_clip(cfg)
    return model, pp, cfg["model"]["backbone"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--gallery-size", type=int, default=300)
    parser.add_argument("--use-stub", action="store_true",
                        help="Skip downloading the real CLIP backbone")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(0)

    print(f"[1/4] Loading model ({'stub' if args.use_stub else 'real CLIP'})...")
    clip_model, clip_preprocess, model_name = maybe_load_clip(args.use_stub)
    cfg = train.load_config()

    print(f"[2/4] Generating {args.gallery_size} gallery + {args.n_queries} query images...")
    gallery_images = [make_random_image(rng) for _ in range(args.gallery_size)]
    query_images = [make_random_image(rng) for _ in range(args.n_queries)]

    print("[3/4] Building gallery index...")
    t0 = time.perf_counter()
    g_feats = train.extract_features(gallery_images, clip_model, clip_preprocess, cfg)
    g_fused = train.fuse_features(g_feats, cfg)
    g_normed = g_fused / (np.linalg.norm(g_fused, axis=1, keepdims=True) + 1e-8)
    index = faiss.IndexFlatIP(g_normed.shape[1])
    index.add(np.ascontiguousarray(g_normed, dtype=np.float32))
    build_time_s = time.perf_counter() - t0
    print(f"    built {g_normed.shape[1]}D index for {args.gallery_size} items in {build_time_s:.1f}s")

    print("[4/4] Running per-component benchmarks...")
    sample_query = query_images[0]
    import torch

    def clip_one():
        with torch.no_grad():
            t = clip_preprocess(sample_query).unsqueeze(0)
            clip_model.encode_image(t)

    def color_one():
        rgb = extract_color_palette(sample_query, bins_per_channel=cfg["features"]["color"]["rgb_bins"])
        hsv = extract_hsv_histogram(sample_query, bins=cfg["features"]["color"]["hsv_bins"])
        return np.concatenate([rgb, hsv])

    def spatial_one():
        return extract_spatial_color_grid(
            sample_query,
            grid_rows=cfg["features"]["spatial"]["grid_rows"],
            grid_cols=cfg["features"]["spatial"]["grid_cols"],
            bins=cfg["features"]["spatial"]["bins"],
        )

    q_feats_one = train.extract_features([sample_query], clip_model, clip_preprocess, cfg)
    q_fused_one = train.fuse_features(q_feats_one, cfg)
    q_normed_one = q_fused_one / (np.linalg.norm(q_fused_one, axis=1, keepdims=True) + 1e-8)
    q_normed_one = np.ascontiguousarray(q_normed_one.astype(np.float32))

    def faiss_one():
        index.search(q_normed_one, 20)

    blocks = [
        benchmark_block("clip_forward_single", clip_one, n_warmup=2, n_iter=20),
        benchmark_block("color_features_single", color_one),
        benchmark_block("spatial_features_single", spatial_one),
        benchmark_block("faiss_search_single", faiss_one, n_warmup=10, n_iter=200),
    ]

    end_to_end_ms = blocks[0]["median_ms"] + blocks[1]["median_ms"] + blocks[2]["median_ms"] + blocks[3]["median_ms"]

    report = {
        "model": model_name,
        "gallery_size": args.gallery_size,
        "n_queries": args.n_queries,
        "feature_dim": int(g_normed.shape[1]),
        "build_time_s": round(build_time_s, 2),
        "components": [{k: v for k, v in b.items() if k != "latencies"} for b in blocks],
        "estimated_end_to_end_ms": round(end_to_end_ms, 2),
        "estimated_throughput_qps": round(1000.0 / end_to_end_ms, 1),
    }

    report_path = args.output / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {report_path}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    names = [b["name"].replace("_single", "").replace("_", " ") for b in blocks]
    medians = [b["median_ms"] for b in blocks]
    p99s = [b["p99"] for b in blocks]

    x = np.arange(len(names))
    ax[0].bar(x - 0.2, medians, 0.4, label="median", color="#2E86AB")
    ax[0].bar(x + 0.2, p99s, 0.4, label="p99", color="#A23B72")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, rotation=20, ha="right")
    ax[0].set_ylabel("Latency (ms)")
    ax[0].set_title(f"Per-component latency ({model_name}, gallery={args.gallery_size})")
    ax[0].legend()
    ax[0].grid(axis="y", alpha=0.3)

    faiss_lats = blocks[3]["latencies"]
    ax[1].hist(faiss_lats, bins=30, color="#2E86AB", edgecolor="white")
    ax[1].axvline(blocks[3]["median_ms"], color="red", linestyle="--", label=f"p50={blocks[3]['median_ms']:.2f}ms")
    ax[1].axvline(blocks[3]["p99"], color="orange", linestyle="--", label=f"p99={blocks[3]['p99']:.2f}ms")
    ax[1].set_xlabel("FAISS search latency (ms)")
    ax[1].set_ylabel("Count")
    ax[1].set_title(f"FAISS Flat search distribution ({len(faiss_lats)} runs)")
    ax[1].legend()
    ax[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plot_path = args.output / "benchmark_latency.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    print(f"Plot:   {plot_path}")

    print("\nSummary:")
    for b in blocks:
        print(f"  {b['name']:<28s}  median={b['median_ms']:>7.3f}ms  p99={b['p99']:>7.3f}ms")
    print(f"\n  estimated end-to-end: {end_to_end_ms:.1f}ms/query  ->  ~{1000/end_to_end_ms:.0f} qps")


if __name__ == "__main__":
    main()
