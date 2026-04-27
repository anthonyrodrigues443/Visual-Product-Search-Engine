"""
Production search engine for Visual Product Search.

Visual-only retrieval pipeline: category filter + CLIP B/32 image embedding
fused with a 48D RGB color histogram. No text descriptions enter the pipeline
at any stage — the system works on a raw photo with no query-side metadata.

Pipeline:
  1. Category filter (hard constraint)
  2. Score = w_visual * clip_image_sim(q, g) + (1 - w_visual) * color_sim(q, g)
  3. Return top-K results with per-component scores

R@1 = 0.683 on DeepFashion In-Shop (300 gallery, 1027 queries) at α = 0.4.
This is the production-honest number for a deployed visual-search system.

Earlier phases of research also explored a text-augmented variant that
reached R@1 = 0.94 by using the gallery's textual product descriptions, but
that path required query-side text at inference time — a luxury most visual-
search apps don't have. That code has been removed; only the visual stack
ships.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    description: str          # carried through from the gallery CSV for display only;
                              # never used as a retrieval signal
    visual_score: float       # CLIP B/32 image-encoder cosine
    color_score: float        # 48D RGB-histogram cosine
    combined_score: float
    image_path: Optional[Path] = None


@dataclass
class SearchResponse:
    results: list[SearchResult]
    query_category: str
    n_gallery_candidates: int
    latency_ms: float
    pipeline: str = "cat-filter + CLIP-B32-image + color-hist-48D"


class ProductSearchEngine:
    """Pure-visual product search using a CLIP image encoder + color histogram.

    Loads pre-computed gallery embeddings from disk. For queries, runs CLIP
    B/32 image encoding and extracts a 48D color histogram from the uploaded
    photo. No text encoder is loaded.
    """

    # Mark's Phase 3 champion: 40% CLIP image score, 60% color score.
    # Verified on the full 1,027-query test set:
    #   α=0.4 → R@1=0.6826  R@5=0.8617  R@10=0.9133
    DEFAULT_W_VISUAL = 0.40
    DEFAULT_K = 20

    CATEGORIES = ["denim", "jackets", "pants", "shirts", "shorts",
                  "suiting", "sweaters", "sweatshirts", "tees"]

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        data_proc: Path = DATA_PROC,
        image_dir: Optional[Path] = None,
        w_visual: float = DEFAULT_W_VISUAL,
        device: str = "cpu",
    ):
        self.cache_dir = Path(cache_dir)
        self.data_proc = Path(data_proc)
        self.image_dir = Path(image_dir) if image_dir else PROJECT_ROOT / "data" / "raw" / "images"
        self.w_visual = w_visual
        self.device = device
        self._clip_model = None
        self._clip_preprocess = None
        self._gallery_loaded = False

    def load_gallery(self) -> "ProductSearchEngine":
        """Load pre-computed gallery embeddings and metadata.

        * g_visual — CLIP B/32 image encoder of the gallery photo (300×512)
        * g_color  — 48D RGB color histogram (300×48)
        """
        self.gallery_df = pd.read_csv(self.data_proc / "gallery.csv")
        self.gallery_df["category2"] = self.gallery_df["category2"].fillna("unknown")

        self.g_visual = self._load_normed("clip_b32_gallery")
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
        """Lazily load the CLIP B/32 image encoder (text encoder is never loaded)."""
        if self._clip_model is not None:
            return
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=self.device
        )
        model.eval()
        self._clip_model = model
        self._clip_preprocess = preprocess

    def embed_image(self, img: Image.Image) -> np.ndarray:
        """Embed an image with CLIP B/32 (image encoder)."""
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

    # ── Public retrieval API ──────────────────────────────────────────────

    def search_by_image(
        self,
        img: Image.Image,
        category: Optional[str] = None,
        k: int = DEFAULT_K,
        w_visual: Optional[float] = None,
    ) -> SearchResponse:
        """Pure-visual retrieval: CLIP image embedding fused with color histogram."""
        assert self._gallery_loaded, "Call load_gallery() first"
        t0 = time.perf_counter()

        q_visual = self.embed_image(img)
        q_color = self.extract_color(img)
        q_color = q_color / np.maximum(np.linalg.norm(q_color), 1e-8)

        return self._search(
            q_visual=q_visual,
            q_color=q_color,
            category=category,
            k=k,
            t0=t0,
            w_visual=self.w_visual if w_visual is None else w_visual,
        )

    def search_by_precomputed(
        self,
        q_visual: np.ndarray,
        q_color: np.ndarray,
        category: Optional[str] = None,
        k: int = DEFAULT_K,
        w_visual: Optional[float] = None,
    ) -> SearchResponse:
        """Visual-only search using pre-computed query embeddings.

        Used by the Browse demo against the cached test-set query embeddings.
        """
        assert self._gallery_loaded, "Call load_gallery() first"
        t0 = time.perf_counter()
        return self._search(
            q_visual=q_visual,
            q_color=q_color,
            category=category,
            k=k,
            t0=t0,
            w_visual=self.w_visual if w_visual is None else w_visual,
        )

    def search_by_color(
        self,
        q_color: np.ndarray,
        category: Optional[str] = None,
        k: int = DEFAULT_K,
    ) -> SearchResponse:
        """Color-only retrieval (48D RGB histogram). R@1 ≈ 0.34.

        Used by the Color filter tab where the user picks a palette directly.
        """
        assert self._gallery_loaded, "Call load_gallery() first"
        t0 = time.perf_counter()
        zero_visual = np.zeros(self.g_visual.shape[1], dtype=np.float32)
        return self._search(
            q_visual=zero_visual,
            q_color=q_color,
            category=category,
            k=k,
            t0=t0,
            w_visual=0.0,
        )

    # ── Core fusion + ranking ─────────────────────────────────────────────

    def _search(
        self,
        q_visual: np.ndarray,
        q_color: np.ndarray,
        category: Optional[str],
        k: int,
        t0: float,
        w_visual: float,
    ) -> SearchResponse:
        if category and category in self.CATEGORIES:
            mask = self.g_cats == category
            cidx = np.where(mask)[0]
        else:
            cidx = np.arange(len(self.g_cats))
            category = "all"

        g_visual_c = self.g_visual[cidx]
        g_color_c = self.g_color[cidx]

        visual_scores = (
            g_visual_c @ q_visual if np.any(q_visual != 0) else np.zeros(len(cidx))
        )
        color_scores = (
            g_color_c @ q_color if np.any(q_color != 0) else np.zeros(len(cidx))
        )

        combined = w_visual * visual_scores + (1 - w_visual) * color_scores
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
                visual_score=float(visual_scores[local_i]),
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
            "best_r1": 0.6826,
            "best_r5": 0.8617,
            "best_r10": 0.9133,
            "best_r20": 0.9412,
            "pipeline": "cat-filter + CLIP-B32-image + color-hist-48D",
            "w_visual": self.w_visual,
            "gallery_size": self.gallery_size,
            "n_queries_eval": 1027,
            "note": "Visual-only — no text used at any stage of inference.",
        }
