"""End-to-end integration tests for the visual search pipeline.

Anthony's existing test suite covers unit tests for individual components
(data splits, feature extractors, recall metrics). These tests validate that
those components compose correctly through the full pipeline:

    train.build_index  ->  predict.VisualSearchEngine  ->  evaluate.recall_at_k

To stay fast and CI-friendly, the CLIP backbone is replaced with a deterministic
random-projection stub. All other code paths (feature concatenation, fusion
weighting, FAISS index build/load round-trip, category filtering, result
formatting) execute exactly as in production.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np
import pytest
from PIL import Image

from src import predict, train
from src.evaluate import category_filtered_search, recall_at_k


# ---------------------------------------------------------------------------
# Stubs / fixtures
# ---------------------------------------------------------------------------


class _StubCLIP:
    """Deterministic stand-in for open_clip.ViT-L/14.

    Returns a 768D embedding derived from the input tensor's mean pixel value
    plus a fixed random projection. Same image always produces the same vector,
    different images produce different vectors — sufficient to exercise the
    fusion/index/search pipeline without a 1.5 GB model download.
    """

    EMBED_DIM = 768

    def __init__(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        self._proj = rng.randn(3, self.EMBED_DIM).astype(np.float32) * 0.1

    def eval(self):
        return self

    def encode_image(self, batch):
        import torch

        x = batch.detach().cpu().numpy()  # (N, 3, H, W)
        per_channel_mean = x.mean(axis=(2, 3))  # (N, 3)
        feats = per_channel_mean @ self._proj  # (N, 768)
        return torch.from_numpy(feats)


def _stub_preprocess(img: Image.Image):
    import torch

    arr = np.asarray(img.convert("RGB").resize((32, 32))).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


class _patch_clip:
    """Patch _load_clip in both train.py and predict.py.

    predict.py does `from src.train import _load_clip`, which binds the function
    into predict's namespace at import time. Patching only train._load_clip
    leaves predict's binding intact and falls through to the real 1.5GB
    CLIP ViT-L/14 download. So we patch both names.
    """

    def __enter__(self):
        from src import predict as _predict_mod  # local import to avoid circular hits

        stub = _StubCLIP(seed=42)
        loader = lambda cfg: (stub, _stub_preprocess)
        self._patches = [
            patch.object(train, "_load_clip", loader),
            patch.object(_predict_mod, "_load_clip", loader),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *args):
        for p in self._patches:
            p.stop()


def _make_solid_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (64, 64), color=color)


@pytest.fixture
def tiny_gallery():
    """6 distinct solid-color images across 2 categories."""
    return {
        "images": [
            _make_solid_image((255, 0, 0)),
            _make_solid_image((0, 255, 0)),
            _make_solid_image((0, 0, 255)),
            _make_solid_image((255, 255, 0)),
            _make_solid_image((255, 0, 255)),
            _make_solid_image((0, 255, 255)),
        ],
        "ids": ["red_001", "green_001", "blue_001", "yellow_001", "magenta_001", "cyan_001"],
        "cats": ["shirts", "shirts", "shirts", "pants", "pants", "pants"],
    }


@pytest.fixture
def built_index(tiny_gallery, tmp_path):
    """Build a real FAISS index in tmp_path using the stub CLIP backbone."""
    with _patch_clip():
        out_dir = train.build_index(
            tiny_gallery["images"],
            tiny_gallery["ids"],
            tiny_gallery["cats"],
            output_dir=tmp_path,
        )
    return out_dir


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIndexBuildPersistRoundTrip:
    def test_writes_three_artifacts(self, built_index):
        for name in ("gallery.index", "gallery_features.npy", "gallery_metadata.json"):
            assert (built_index / name).exists(), f"missing artifact: {name}"

    def test_metadata_matches_gallery(self, built_index, tiny_gallery):
        with open(built_index / "gallery_metadata.json") as f:
            md = json.load(f)
        assert md["product_ids"] == tiny_gallery["ids"]
        assert md["categories"] == tiny_gallery["cats"]
        assert md["n_gallery"] == len(tiny_gallery["images"])

    def test_index_dimension_matches_metadata(self, built_index):
        index = faiss.read_index(str(built_index / "gallery.index"))
        with open(built_index / "gallery_metadata.json") as f:
            md = json.load(f)
        assert index.d == md["feature_dim"]
        # CLIP (768) + color (48) + spatial (192) = 1008
        assert index.d == 768 + 48 + 192

    def test_features_normalized(self, built_index):
        feats = np.load(built_index / "gallery_features.npy")
        norms = np.linalg.norm(feats, axis=1)
        np.testing.assert_allclose(norms, np.ones(len(feats)), atol=1e-5)


class TestSearchEngineEndToEnd:
    def test_self_query_returns_self_at_rank_1(self, built_index, tiny_gallery):
        with _patch_clip():
            engine = predict.VisualSearchEngine(index_dir=built_index)
            for img, pid in zip(tiny_gallery["images"], tiny_gallery["ids"]):
                results = engine.search(img, top_k=3)
                assert results[0]["product_id"] == pid, (
                    f"expected {pid} at rank 1, got {results[0]['product_id']}"
                )

    def test_results_are_ordered_by_score(self, built_index, tiny_gallery):
        with _patch_clip():
            engine = predict.VisualSearchEngine(index_dir=built_index)
            results = engine.search(tiny_gallery["images"][0], top_k=6)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates_results(self, built_index, tiny_gallery):
        with _patch_clip():
            engine = predict.VisualSearchEngine(index_dir=built_index)
            assert len(engine.search(tiny_gallery["images"][0], top_k=2)) == 2
            assert len(engine.search(tiny_gallery["images"][0], top_k=5)) == 5

    def test_result_schema(self, built_index, tiny_gallery):
        with _patch_clip():
            engine = predict.VisualSearchEngine(index_dir=built_index)
            results = engine.search(tiny_gallery["images"][0], top_k=3)
        for r in results:
            assert set(r.keys()) == {"rank", "product_id", "category", "score", "gallery_index"}
            assert isinstance(r["rank"], int)
            assert isinstance(r["score"], float)
            assert isinstance(r["product_id"], str)


class TestCategoryFilterEndToEnd:
    def test_filter_restricts_to_category(self, built_index, tiny_gallery):
        with _patch_clip():
            engine = predict.VisualSearchEngine(index_dir=built_index)
            # Query is a "shirt" — restrict gallery to pants only
            results = engine.search(tiny_gallery["images"][0], top_k=5, category="pants")
        assert len(results) == 3, "only 3 pants in gallery"
        for r in results:
            assert r["category"] == "pants"

    def test_unknown_category_falls_back_to_global(self, built_index, tiny_gallery):
        with _patch_clip():
            engine = predict.VisualSearchEngine(index_dir=built_index)
            results = engine.search(tiny_gallery["images"][0], top_k=5, category="nonexistent")
        assert len(results) == 5  # global search still returns top-K


class TestEvaluatePipelineEndToEnd:
    """Recall@K must be 100% when querying with the exact gallery images."""

    def test_perfect_recall_for_self_query(self, tiny_gallery):
        # Skip the index build — exercise the eval path directly with stub features
        with _patch_clip():
            cfg = train.load_config()
            stub_model, stub_pp = train._load_clip(cfg)
            feats = train.extract_features(
                tiny_gallery["images"], stub_model, stub_pp, cfg
            )
            fused = train.fuse_features(feats, cfg)

        gn = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
        index = faiss.IndexFlatIP(gn.shape[1])
        index.add(np.ascontiguousarray(gn, dtype=np.float32))
        _, retrieved = index.search(np.ascontiguousarray(gn, dtype=np.float32), 1)

        pids = np.array(tiny_gallery["ids"])
        result = recall_at_k(pids, pids, retrieved, ks=(1,))
        assert result["R@1"] == 1.0


class TestCategoryFilteredSearchIntegration:
    """category_filtered_search (eval) and VisualSearchEngine (predict) must agree."""

    def test_eval_and_predict_agree_under_category_filter(self, built_index, tiny_gallery):
        gallery_feats = np.load(built_index / "gallery_features.npy")
        gallery_cats = np.array(tiny_gallery["cats"])

        with _patch_clip():
            engine = predict.VisualSearchEngine(index_dir=built_index)
            # First query (red_001) is in shirts — restrict to shirts
            engine_results = engine.search(tiny_gallery["images"][0], top_k=3, category="shirts")
            engine_top1_idx = engine_results[0]["gallery_index"]

        # Query feature reproduces the same vector that was stored at index 0
        query_vec = gallery_feats[0:1]
        _, eval_indices = category_filtered_search(
            gallery_feats,
            query_vec,
            gallery_cats,
            np.array(["shirts"]),
            k=3,
        )
        assert eval_indices[0, 0] == engine_top1_idx
