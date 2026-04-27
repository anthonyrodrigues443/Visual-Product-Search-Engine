"""Recompute every number the Streamlit UI claims, against the real cached
embeddings, and dump a side-by-side comparison.

Outputs results/ui_number_verification.json so the next pass can diff.
Run with: KMP_DUPLICATE_LIB_OK=TRUE python scripts/verify_ui_numbers.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.search_engine import ProductSearchEngine

ROOT = Path(__file__).parent.parent
CACHE = ROOT / "data" / "processed" / "emb_cache"


def normed(name):
    e = np.load(CACHE / f"{name}.npy").astype(np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)


def recall_at_k(retrieved_pids, gold_pids, k):
    return float(np.mean([gold in retrieved_pids[i, :k] for i, gold in enumerate(gold_pids)]))


def search_visual_batch(g_visual, g_color, g_cats, g_pids,
                         q_visual, q_color, q_cats, w_visual, k=20,
                         use_cat_filter=True):
    """Vectorised reproduction of ProductSearchEngine._search."""
    n = len(q_cats)
    out = np.full((n, k), "", dtype=object)
    for i in range(n):
        if use_cat_filter:
            mask = g_cats == q_cats[i]
        else:
            mask = np.ones(len(g_cats), dtype=bool)
        cidx = np.where(mask)[0]
        if len(cidx) == 0:
            continue
        vs = g_visual[cidx] @ q_visual[i] if np.any(q_visual[i] != 0) else np.zeros(len(cidx))
        cs = g_color[cidx] @ q_color[i] if np.any(q_color[i] != 0) else np.zeros(len(cidx))
        combined = w_visual * vs + (1 - w_visual) * cs
        order = np.argsort(-combined)[:k]
        topk = cidx[order]
        for j, idx in enumerate(topk):
            out[i, j] = g_pids[idx]
    return out


def main():
    e = ProductSearchEngine().load_gallery()
    g_visual = e.g_visual
    g_color = e.g_color
    g_cats = e.g_cats
    g_pids = e.g_pids

    qdf = pd.read_csv(ROOT / "data" / "processed" / "query.csv")
    qdf["category2"] = qdf["category2"].fillna("unknown")
    q_visual = normed("clip_b32_query")
    q_color = normed("color48_query")
    q_cats = qdf["category2"].values
    q_pids = qdf["product_id"].values

    n = len(qdf)
    print(f"Queries: {n}, Gallery: {len(g_pids)}\n")

    report = {"n_queries": n, "n_gallery": len(g_pids), "checks": {}}

    # ---- 1. Production champion: CLIP B/32 + cat filter + color a=0.4 ----
    label = "champion (cat + CLIP B/32 + color a=0.4)"
    t0 = time.perf_counter()
    retrieved = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                     q_visual, q_color, q_cats, w_visual=0.4, k=20)
    elapsed = time.perf_counter() - t0
    r = {f"R@{k}": recall_at_k(retrieved, q_pids, k) for k in (1, 5, 10, 20)}
    print(f"{label}")
    for k, v in r.items():
        print(f"  {k} = {v:.4f}")
    print(f"  ({elapsed:.2f}s)")
    report["checks"]["champion"] = {"label": label, **r,
                                     "ui_claim": {"R@1": 0.683, "R@5": 0.862, "R@10": 0.913, "R@20": 0.941}}

    # ---- 2. Per-category R@1 (sidebar bars) ----
    print("\nper-category R@1 (champion):")
    cats_seen = sorted(set(q_cats))
    per_cat = {}
    for cat in cats_seen:
        mask = q_cats == cat
        if mask.sum() == 0:
            continue
        sub_recall = recall_at_k(retrieved[mask], q_pids[mask], 1)
        per_cat[cat] = round(sub_recall, 4)
        print(f"  {cat:15s} {sub_recall:.4f}  (n={int(mask.sum())})")
    report["checks"]["per_category_r1"] = per_cat

    ui_per_cat = {
        "suiting": 1.000, "jackets": 0.794, "denim": 0.741,
        "sweaters": 0.722, "shirts": 0.715, "pants": 0.668,
        "sweatshirts": 0.638, "tees": 0.633, "shorts": 0.495,
    }
    report["checks"]["per_category_ui_claim"] = ui_per_cat

    # ---- 3. Ablation ----
    ablation = {}

    # Full system already computed
    ablation["full"] = {"R@1": report["checks"]["champion"]["R@1"]}

    # No category filter
    print("\nablation: remove category filter")
    r_nocat = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                    q_visual, q_color, q_cats, w_visual=0.4, k=10,
                                    use_cat_filter=False)
    ablation["no_cat_filter"] = {"R@1": recall_at_k(r_nocat, q_pids, 1)}
    print(f"  R@1 = {ablation['no_cat_filter']['R@1']:.4f}")

    # No color (CLIP only)
    print("\nablation: remove color (CLIP image only, with cat filter)")
    r_nocolor = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                      q_visual, np.zeros_like(q_color), q_cats,
                                      w_visual=1.0, k=10)
    ablation["no_color"] = {"R@1": recall_at_k(r_nocolor, q_pids, 1)}
    print(f"  R@1 = {ablation['no_color']['R@1']:.4f}")

    # No CLIP image (color only, with cat filter)
    print("\nablation: remove CLIP image (color only, with cat filter)")
    r_noclip = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                     np.zeros_like(q_visual), q_color, q_cats,
                                     w_visual=0.0, k=10)
    ablation["no_clip"] = {"R@1": recall_at_k(r_noclip, q_pids, 1)}
    print(f"  R@1 = {ablation['no_clip']['R@1']:.4f}")

    # Color-only (no cat filter, no CLIP)
    print("\nablation: color-only (no cat filter)")
    r_coloronly = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                        np.zeros_like(q_visual), q_color, q_cats,
                                        w_visual=0.0, k=10, use_cat_filter=False)
    ablation["color_only_no_cat"] = {"R@1": recall_at_k(r_coloronly, q_pids, 1)}
    print(f"  R@1 = {ablation['color_only_no_cat']['R@1']:.4f}")

    # CLIP B/32 bare (no cat, no color)
    print("\nablation: CLIP B/32 bare (no cat, no color)")
    r_clipbare = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                       q_visual, np.zeros_like(q_color), q_cats,
                                       w_visual=1.0, k=10, use_cat_filter=False)
    ablation["clip_bare"] = {"R@1": recall_at_k(r_clipbare, q_pids, 1)}
    print(f"  R@1 = {ablation['clip_bare']['R@1']:.4f}")

    report["checks"]["ablation"] = ablation
    report["checks"]["ablation_ui_claim"] = {
        "full": 0.6826,
        "no_cat_filter": 0.5934,
        "no_color": 0.5687,
        "no_clip": 0.3380,        # UI claims this is "color-only" = 0.338
        "color_only_no_cat": None,  # UI doesn't separate this
    }

    # ---- 4. Other leaderboard rows ----
    print("\nleaderboard sweep:")
    leaderboard = {}

    # CLIP B/32 + color a=0.5 (no cat) — Phase 2 row
    r_p2 = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                q_visual, q_color, q_cats, w_visual=0.5, k=10,
                                use_cat_filter=False)
    leaderboard["clip_b32_color_0.5_no_cat"] = {
        "R@1": recall_at_k(r_p2, q_pids, 1),
        "R@5": recall_at_k(r_p2, q_pids, 5),
        "R@10": recall_at_k(r_p2, q_pids, 10),
    }
    print(f"  CLIP B/32 + color a=0.5 (no cat) R@1={leaderboard['clip_b32_color_0.5_no_cat']['R@1']:.4f}  ui=0.576")

    # CLIP B/32 + cat filter, no color
    r_p3a = search_visual_batch(g_visual, g_color, g_cats, g_pids,
                                  q_visual, np.zeros_like(q_color), q_cats,
                                  w_visual=1.0, k=10, use_cat_filter=True)
    leaderboard["clip_b32_cat_no_color"] = {"R@1": recall_at_k(r_p3a, q_pids, 1)}
    print(f"  CLIP B/32 + cat (no color)       R@1={leaderboard['clip_b32_cat_no_color']['R@1']:.4f}  ui=0.569")

    report["checks"]["leaderboard"] = leaderboard

    # ---- Save report ----
    out_path = ROOT / "results" / "ui_number_verification.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
