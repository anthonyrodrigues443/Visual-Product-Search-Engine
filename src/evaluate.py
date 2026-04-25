"""
Evaluation script for Visual Product Search Engine.

Runs Recall@K evaluation on the full query set using pre-computed embeddings.
Reports per-category breakdown and ablation comparison.

Usage:
  python -m src.evaluate [--n-eval 300] [--k 20]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def recall_at_k(indices, q_pids, g_pids, k):
    hits = sum(q_pids[i] in g_pids[indices[i][:k]] for i in range(len(indices)))
    return hits / len(indices)


def evaluate_pipeline(
    q_text, g_text, q_color, g_color, q_cats, g_cats, q_pids, g_pids,
    w_text=0.80, k=20
):
    results = []
    for i in range(len(q_text)):
        cat = q_cats[i]
        cidx = np.where(g_cats == cat)[0]
        text_s = g_text[cidx] @ q_text[i]
        color_s = g_color[cidx] @ q_color[i]
        fused = w_text * text_s + (1 - w_text) * color_s
        order = np.argsort(-fused)[:k]
        results.append(cidx[order])
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Visual Product Search")
    parser.add_argument("--n-eval", type=int, default=300,
                        help="Number of gallery products to use for evaluation")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--w-text", type=float, default=0.80)
    parser.add_argument("--output", default="results/eval_phase6.json")
    args = parser.parse_args()

    cache = PROJECT_ROOT / "data" / "processed" / "emb_cache"
    proc = PROJECT_ROOT / "data" / "processed"

    gallery_df = pd.read_csv(proc / "gallery.csv")
    query_df = pd.read_csv(proc / "query.csv")

    eval_pids = gallery_df["product_id"].values[:args.n_eval]
    g_df = gallery_df[gallery_df["product_id"].isin(eval_pids)].reset_index(drop=True)
    q_df = query_df[query_df["product_id"].isin(eval_pids)].reset_index(drop=True)

    print(f"Eval set: {len(g_df)} gallery, {len(q_df)} queries, {args.n_eval} products")

    def load_normed(name):
        emb = np.load(cache / f"{name}.npy").astype(np.float32)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)

    g_text = load_normed("clip_b32_text_gallery")[g_df.index]
    q_text = load_normed("clip_b32_text_query")[q_df.index]
    g_color = load_normed("color48_gallery")[g_df.index]
    q_color = load_normed("color48_query")[q_df.index]

    g_cats = g_df["category2"].values
    q_cats = q_df["category2"].values
    g_pids = g_df["product_id"].values
    q_pids = q_df["product_id"].values

    t0 = time.perf_counter()
    indices = evaluate_pipeline(
        q_text, g_text, q_color, g_color, q_cats, g_cats, q_pids, g_pids,
        w_text=args.w_text, k=args.k
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    metrics = {
        f"R@{k}": recall_at_k(indices, q_pids, g_pids, k)
        for k in [1, 5, 10, 20]
    }

    print("\nOverall Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    print(f"  Latency: {elapsed_ms:.1f}ms total ({elapsed_ms / len(q_df):.2f}ms/query)")

    # Per-category breakdown
    per_cat = {}
    for cat in g_df["category2"].unique():
        cat_q_mask = q_cats == cat
        cat_q_idx = np.where(cat_q_mask)[0]
        if len(cat_q_idx) == 0:
            continue
        cat_indices = [indices[i] for i in cat_q_idx]
        cat_q_pids = q_pids[cat_q_idx]
        r1 = recall_at_k(cat_indices, cat_q_pids, g_pids, 1)
        r5 = recall_at_k(cat_indices, cat_q_pids, g_pids, 5)
        per_cat[cat] = {"R@1": round(r1, 4), "R@5": round(r5, 4), "n_queries": int(sum(cat_q_mask))}

    print("\nPer-Category R@1:")
    for cat, vals in sorted(per_cat.items(), key=lambda x: -x[1]["R@1"]):
        print(f"  {cat:15s}: R@1={vals['R@1']:.4f}  R@5={vals['R@5']:.4f}  n={vals['n_queries']}")

    out = {
        "overall": metrics,
        "per_category": per_cat,
        "config": {"w_text": args.w_text, "k": args.k, "n_eval": args.n_eval},
        "latency_total_ms": elapsed_ms,
        "latency_per_query_ms": elapsed_ms / len(q_df),
        "pipeline": "cat-filter + color-hist-48D + CLIP-B32-text",
    }

    out_path = PROJECT_ROOT / args.output
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
