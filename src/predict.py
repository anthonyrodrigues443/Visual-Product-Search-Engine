"""Run visual product search: given a query image, return top-K matches.

Loads a pre-built FAISS index and gallery metadata, extracts features
from the query image, and returns ranked results with similarity scores.
"""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image

from src.train import _load_clip, extract_features, load_config

PROJECT_ROOT = Path(__file__).parent.parent


class VisualSearchEngine:
    def __init__(self, index_dir: str | Path | None = None, config_path: str | None = None):
        self.cfg = load_config(config_path)
        if index_dir is None:
            index_dir = PROJECT_ROOT / "models"
        index_dir = Path(index_dir)

        self.index = faiss.read_index(str(index_dir / "gallery.index"))
        self.gallery_features = np.load(index_dir / "gallery_features.npy")
        with open(index_dir / "gallery_metadata.json") as f:
            self.metadata = json.load(f)

        self.product_ids = self.metadata["product_ids"]
        self.categories = self.metadata["categories"]
        self.clip_model, self.clip_preprocess = _load_clip(self.cfg)

    def search(
        self,
        query_image: Image.Image,
        top_k: int = 20,
        category: str | None = None,
    ) -> list[dict]:
        feats = extract_features(
            [query_image], self.clip_model, self.clip_preprocess, self.cfg
        )
        fused = self._fuse(feats)
        fused_normed = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)

        if category and self.cfg["retrieval"]["category_filter"]:
            return self._category_filtered_search(fused_normed[0], category, top_k)

        scores, indices = self.index.search(
            np.ascontiguousarray(fused_normed, dtype=np.float32), top_k
        )
        return self._format_results(indices[0], scores[0])

    def _category_filtered_search(
        self, query_vec: np.ndarray, category: str, top_k: int
    ) -> list[dict]:
        cat_mask = [i for i, c in enumerate(self.categories) if c == category]
        if not cat_mask:
            scores, indices = self.index.search(query_vec.reshape(1, -1), top_k)
            return self._format_results(indices[0], scores[0])

        cat_features = self.gallery_features[cat_mask]
        k = min(top_k, len(cat_mask))

        sub_index = faiss.IndexFlatIP(cat_features.shape[1])
        sub_index.add(np.ascontiguousarray(cat_features, dtype=np.float32))
        scores, local_indices = sub_index.search(query_vec.reshape(1, -1).astype(np.float32), k)

        global_indices = [cat_mask[i] for i in local_indices[0]]
        return self._format_results(np.array(global_indices), scores[0])

    def _fuse(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        weights = [
            self.cfg["fusion"]["w_clip"],
            self.cfg["fusion"]["w_color"],
            self.cfg["fusion"]["w_spatial"],
        ]
        arrays = [feats["clip"], feats["color"], feats["spatial"]]
        parts = []
        for arr, w in zip(arrays, weights):
            normed = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
            parts.append(normed * w)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _format_results(self, indices: np.ndarray, scores: np.ndarray) -> list[dict]:
        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            idx = int(idx)
            if idx < 0:
                continue
            results.append({
                "rank": rank + 1,
                "product_id": self.product_ids[idx],
                "category": self.categories[idx],
                "score": round(float(score), 4),
                "gallery_index": idx,
            })
        return results


def search_single(
    query_path: str,
    index_dir: str | None = None,
    top_k: int = 20,
    category: str | None = None,
) -> list[dict]:
    engine = VisualSearchEngine(index_dir=index_dir)
    img = Image.open(query_path).convert("RGB")
    return engine.search(img, top_k=top_k, category=category)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visual product search")
    parser.add_argument("query", help="Path to query image")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--category", default=None)
    parser.add_argument("--index-dir", default=None)
    args = parser.parse_args()

    results = search_single(args.query, args.index_dir, args.top_k, args.category)
    for r in results:
        print(f"  #{r['rank']:2d}  {r['product_id']:<30s}  {r['category']:<15s}  score={r['score']:.4f}")
