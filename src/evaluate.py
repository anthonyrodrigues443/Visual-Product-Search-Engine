"""Evaluate the visual product search system on the DeepFashion test split.

Computes Recall@K, per-category breakdowns, and timing benchmarks.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.train import _load_clip, extract_features, fuse_features, load_config

PROJECT_ROOT = Path(__file__).parent.parent


def recall_at_k(
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    retrieved_indices: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    results = {}
    for k in ks:
        correct = sum(
            1
            for qi, qp in enumerate(query_pids)
            if qp in gallery_pids[retrieved_indices[qi, :k]]
        )
        results[f"R@{k}"] = round(correct / len(query_pids), 4)
    return results


def per_category_recall(
    query_categories: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    retrieved_indices: np.ndarray,
    k: int = 1,
) -> dict[str, float]:
    cats = {}
    for cat in np.unique(query_categories):
        mask = query_categories == cat
        qp = query_pids[mask]
        qi = np.where(mask)[0]
        correct = sum(1 for i, p in zip(qi, qp) if p in gallery_pids[retrieved_indices[i, :k]])
        cats[cat] = round(correct / len(qp), 4) if len(qp) > 0 else 0.0
    return cats


def category_filtered_search(
    gallery_features: np.ndarray,
    query_features: np.ndarray,
    gallery_cats: np.ndarray,
    query_cats: np.ndarray,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    n_queries = len(query_features)
    indices = np.full((n_queries, k), -1, dtype=np.int64)
    scores = np.zeros((n_queries, k), dtype=np.float32)

    gn = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)
    qn = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)

    for cat in np.unique(query_cats):
        q_mask = query_cats == cat
        g_mask = gallery_cats == cat
        if g_mask.sum() == 0:
            continue

        g_idx = np.where(g_mask)[0]
        q_idx = np.where(q_mask)[0]
        cat_k = min(k, len(g_idx))

        sub_index = faiss.IndexFlatIP(gn.shape[1])
        sub_index.add(np.ascontiguousarray(gn[g_idx], dtype=np.float32))
        D, I = sub_index.search(np.ascontiguousarray(qn[q_idx], dtype=np.float32), cat_k)

        for qi_local, qi_global in enumerate(q_idx):
            for rank in range(cat_k):
                indices[qi_global, rank] = g_idx[I[qi_local, rank]]
                scores[qi_global, rank] = D[qi_local, rank]

    return scores, indices


def evaluate(
    gallery_images: list[Image.Image],
    gallery_pids: list[str],
    gallery_cats: list[str],
    query_images: list[Image.Image],
    query_pids: list[str],
    query_cats: list[str],
    config_path: str | None = None,
    use_category_filter: bool = True,
) -> dict:
    cfg = load_config(config_path)
    k = cfg["retrieval"]["top_k"]

    clip_model, clip_preprocess = _load_clip(cfg)

    print("Extracting gallery features...")
    t0 = time.time()
    g_feats = extract_features(gallery_images, clip_model, clip_preprocess, cfg)
    g_fused = fuse_features(g_feats, cfg)
    gallery_time = time.time() - t0

    print("Extracting query features...")
    t0 = time.time()
    q_feats = extract_features(query_images, clip_model, clip_preprocess, cfg)
    q_fused = fuse_features(q_feats, cfg)
    query_time = time.time() - t0

    g_pids = np.array(gallery_pids)
    q_pids = np.array(query_pids)
    g_cats_arr = np.array(gallery_cats)
    q_cats_arr = np.array(query_cats)

    if use_category_filter:
        t0 = time.time()
        D, I = category_filtered_search(g_fused, q_fused, g_cats_arr, q_cats_arr, k)
        search_time = time.time() - t0
    else:
        gn = g_fused / (np.linalg.norm(g_fused, axis=1, keepdims=True) + 1e-8)
        qn = q_fused / (np.linalg.norm(q_fused, axis=1, keepdims=True) + 1e-8)
        t0 = time.time()
        index = faiss.IndexFlatIP(gn.shape[1])
        index.add(np.ascontiguousarray(gn, dtype=np.float32))
        D, I = index.search(np.ascontiguousarray(qn, dtype=np.float32), k)
        search_time = time.time() - t0

    overall = recall_at_k(q_pids, g_pids, I)
    per_cat = per_category_recall(q_cats_arr, q_pids, g_pids, I, k=1)

    ms_per_query = (query_time + search_time) / len(query_images) * 1000

    results = {
        "overall": overall,
        "per_category": per_cat,
        "timing": {
            "gallery_extraction_s": round(gallery_time, 1),
            "query_extraction_s": round(query_time, 1),
            "search_s": round(search_time, 3),
            "ms_per_query": round(ms_per_query, 1),
        },
        "config": {
            "n_gallery": len(gallery_images),
            "n_queries": len(query_images),
            "feature_dim": g_fused.shape[1],
            "category_filter": use_category_filter,
            "top_k": k,
        },
    }

    print(f"\nResults (category_filter={use_category_filter}):")
    for metric, val in overall.items():
        print(f"  {metric}: {val:.4f}")
    print(f"  Latency: {ms_per_query:.1f} ms/query")

    return results


if __name__ == "__main__":
    import argparse

    from src.data_pipeline import (
        create_retrieval_splits,
        download_deepfashion_images,
        download_deepfashion_metadata,
    )

    parser = argparse.ArgumentParser(description="Evaluate visual search")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--no-category-filter", action="store_true")
    parser.add_argument("--output", default=None, help="Path to save results JSON")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = download_deepfashion_metadata(max_items=args.max_items)
    _, gallery_df, query_df = create_retrieval_splits(df, test_frac=cfg["dataset"]["test_frac"])

    eval_pids = gallery_df["product_id"].values[: cfg["dataset"]["eval_products"]]
    g_df = gallery_df[gallery_df["product_id"].isin(eval_pids)]
    q_df = query_df[query_df["product_id"].isin(eval_pids)]

    img_dir = download_deepfashion_images(g_df)
    download_deepfashion_images(q_df)

    def load_images(df_slice, img_dir):
        images, pids, cats = [], [], []
        for _, row in df_slice.iterrows():
            p = img_dir / f"{row['item_id']}.jpg"
            if p.exists():
                images.append(Image.open(p).convert("RGB"))
                pids.append(row["product_id"])
                cats.append(row["category2"])
        return images, pids, cats

    g_imgs, g_pids, g_cats = load_images(g_df, img_dir)
    q_imgs, q_pids, q_cats = load_images(q_df, img_dir)

    results = evaluate(
        g_imgs, g_pids, g_cats,
        q_imgs, q_pids, q_cats,
        config_path=args.config,
        use_category_filter=not args.no_category_filter,
    )

    out_path = args.output or str(PROJECT_ROOT / "results" / "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
