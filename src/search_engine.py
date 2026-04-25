"""
Production search engine for Visual Product Search.

Best pipeline from Phase 5 ablation: category filter + color histogram + text embeddings.
No CLIP visual embeddings — ablation showed removing them IMPROVES R@1 by +1.35pp.

Pipeline:
  1. Category filter (hard constraint)
  2. Score = w_text * text_sim(q, g) + (1 - w_text) * color_sim(q, g)
  3. Return top-K results with per-component scores

R@1=0.920 on DeepFashion In-Shop (300 gallery, 1027 queries)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "emb_cache"
DATA_PROC = PROJECT_ROOT / "data" / "processed"


@dataclass
class SearchResult:
    rank: int
    item_id: str
    product_id: str
    category: str
    color: str
    description: str
    text_score: float
    color_score: float
    combined_score: float
    image_path: Optional[Path] = None


@dataclass
class SearchResponse:
    results: list[SearchResult]
    query_category: str
    n_gallery_candidates: int
    latency_ms: float
    pipeline: str = "cat-filter + text-embed + color-hist"


class ProductSearchEngine:
    """Production-grade visual product search using text + color features.

    Loads pre-computed gallery embeddings from disk. For queries, extracts
    color histograms from uploaded images and text embeddings from descriptions.
    """

    # Best hyperparameters from Phase 5 grid search
    DEFAULT_W_TEXT = 0.80
    DEFAULT_K = 20

    CATEGORIES = ["denim", "jackets", "pants", "shirts", "shorts",
                  "suiting", "sweaters", "sweatshirts", "tees"]

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        data_proc: Path = DATA_PROC,
        image_dir: Optional[Path] = None,
        w_text: float = DEFAULT_W_TEXT,
        device: str = "cpu",
    ):
        self.cache_dir = Path(cache_dir)
        self.data_proc = Path(data_proc)
        self.image_dir = Path(image_dir) if image_dir else PROJECT_ROOT / "data" / "raw" / "images"
        self.w_text = w_text
        self.device = device
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._gallery_loaded = False

    def load_gallery(self) -> "ProductSearchEngine":
        """Load pre-computed gallery embeddings and metadata."""
        self.gallery_df = pd.read_csv(self.data_proc / "gallery.csv")
        self.gallery_df["category2"] = self.gallery_df["category2"].fillna("unknown")

        self.g_text = self._load_normed("clip_b32_text_gallery")
        self.g_color = self._load_normed("color48_gallery")
        self.g_cats = self.gallery_df["category2"].values
        self.g_pids = self.gallery_df["product_id"].values
        self.g_item_ids = self.gallery_df["item_id"].values

        self._gallery_loaded = True
        return self

    def _load_normed(self, name: str) -> np.ndarray:
        embs = np.load(self.cache_dir / f"{name}.npy").astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.maximum(norms, 1e-8)

    def _ensure_clip(self):
        if self._clip_model is not None:
            return
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess
        self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text description using CLIP B/32."""
        self._ensure_clip()
        import torch
        tokens = self._clip_tokenizer([text]).to(self.device)
        with torch.no_grad():
            feat = self._clip_model.encode_text(tokens).float().cpu().numpy()
        feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
        return feat[0]

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Embed an image using CLIP B/32 (used only for optional visual signal)."""
        self._ensure_clip()
        import torch
        img_tensor = self._clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._clip_model.encode_image(img_tensor).float().cpu().numpy()
        feat = feat / np.maximum(np.linalg.norm(feat, axis=1, keepdims=True), 1e-8)
        return feat[0]

    def extract_color(self, img: Image.Image, bins: int = 8) -> np.ndarray:
        """Extract 48D RGB color histogram (8 bins × 3 channels)."""
        from src.feature_engineering import extract_color_palette
        return extract_color_palette(img, bins_per_channel=bins)

    def search_by_text(
        self,
        description: str,
        category: Optional[str] = None,
        k: int = DEFAULT_K,
    ) -> SearchResponse:
        """Search by text description only (no image required)."""
        assert self._gallery_loaded, "Call load_gallery() first"
        t0 = time.perf_counter()

        q_text = self.embed_text(description)
        q_color = np.zeros(48, dtype=np.float32)  # no color signal without image

        return self._search(
            q_text=q_text,
            q_color=q_color,
            category=category,
            k=k,
            t0=t0,
            w_text=1.0,  # pure text when no image
        )

    def search_by_image_and_text(
        self,
        img: Image.Image,
        description: str,
        category: Optional[str] = None,
        k: int = DEFAULT_K,
    ) -> SearchResponse:
        """Search using both image (color) and text description."""
        assert self._gallery_loaded, "Call load_gallery() first"
        t0 = time.perf_counter()

        q_text = self.embed_text(description)
        q_color = self.extract_color(img)
        q_color = q_color / np.maximum(np.linalg.norm(q_color), 1e-8)

        return self._search(
            q_text=q_text,
            q_color=q_color,
            category=category,
            k=k,
            t0=t0,
            w_text=self.w_text,
        )

    def search_by_precomputed(
        self,
        q_text: np.ndarray,
        q_color: np.ndarray,
        category: Optional[str] = None,
        k: int = DEFAULT_K,
    ) -> SearchResponse:
        """Search using pre-computed embeddings (for batch eval / demo from cache)."""
        assert self._gallery_loaded, "Call load_gallery() first"
        t0 = time.perf_counter()
        return self._search(
            q_text=q_text,
            q_color=q_color,
            category=category,
            k=k,
            t0=t0,
            w_text=self.w_text,
        )

    def _search(
        self,
        q_text: np.ndarray,
        q_color: np.ndarray,
        category: Optional[str],
        k: int,
        t0: float,
        w_text: float,
    ) -> SearchResponse:
        if category and category in self.CATEGORIES:
            mask = self.g_cats == category
            cidx = np.where(mask)[0]
        else:
            cidx = np.arange(len(self.g_cats))
            category = "all"

        g_text_c = self.g_text[cidx]
        g_color_c = self.g_color[cidx]

        text_scores = g_text_c @ q_text
        color_scores = g_color_c @ q_color if np.any(q_color != 0) else np.zeros(len(cidx))

        combined = w_text * text_scores + (1 - w_text) * color_scores
        top_k_local = np.argsort(-combined)[:k]
        top_k_global = cidx[top_k_local]

        results = []
        for rank, (local_i, global_i) in enumerate(zip(top_k_local, top_k_global)):
            row = self.gallery_df.iloc[global_i]
            img_path = self.image_dir / f"{row['item_id']}.jpg"
            results.append(SearchResult(
                rank=rank + 1,
                item_id=row["item_id"],
                product_id=row["product_id"],
                category=row["category2"],
                color=str(row.get("color", "")),
                description=str(row.get("description", "")),
                text_score=float(text_scores[local_i]),
                color_score=float(color_scores[local_i]),
                combined_score=float(combined[local_i]),
                image_path=img_path if img_path.exists() else None,
            ))

        latency_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            results=results,
            query_category=category,
            n_gallery_candidates=len(cidx),
            latency_ms=latency_ms,
        )

    @property
    def gallery_size(self) -> int:
        return len(self.gallery_df) if self._gallery_loaded else 0

    def get_performance_summary(self) -> dict:
        return {
            "best_r1": 0.920,
            "best_r5": 0.990,
            "best_r10": 0.990,
            "best_r20": 0.990,
            "pipeline": "cat-filter + color-hist-48D + CLIP-B32-text",
            "w_text": self.w_text,
            "gallery_size": self.gallery_size,
            "n_queries_eval": 1027,
            "note": "Ablation: removing CLIP visual IMPROVES R@1 by +1.35pp",
        }
