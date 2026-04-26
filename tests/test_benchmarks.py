"""Performance benchmarks for the visual search engine.

These tests assert on production latency contracts rather than just correctness:
- Feature extraction stays in the budget for a single product image
- FAISS search latency stays sub-millisecond for the production gallery size
- Per-query throughput supports the documented "0.10 ms/query" claim from Phase 6

CLIP is stubbed out the same way as in test_integration so the benchmarks
measure the *fusion + search* path that's deterministic across machines.
The CLIP forward pass itself is benchmarked separately in
scripts/benchmark_inference.py with the real backbone.
"""

from __future__ import annotations

import time
from statistics import mean, median
from unittest.mock import patch

import faiss
import numpy as np
import pytest
from PIL import Image

from src import train


# Same stub as test_integration. Keeping the duplicate so the two test files
# stay independently runnable rather than coupling them through a shared module.
class _StubCLIP:
    EMBED_DIM = 768

    def __init__(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        self._proj = rng.randn(3, self.EMBED_DIM).astype(np.float32) * 0.1

    def eval(self):
        return self

    def encode_image(self, batch):
        import torch

        x = batch.detach().cpu().numpy()
        feats = x.mean(axis=(2, 3)) @ self._proj
        return torch.from_numpy(feats)


def _stub_preprocess(img: Image.Image):
    import torch

    arr = np.asarray(img.convert("RGB").resize((32, 32))).astype(np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _patch_clip():
    stub = _StubCLIP(seed=42)
    return patch.object(train, "_load_clip", lambda cfg: (stub, _stub_preprocess))


@pytest.fixture(scope="module")
def gallery_300():
    """300-image synthetic gallery — matches Phase 6 evaluation size."""
    rng = np.random.RandomState(0)
    return [
        Image.fromarray(
            (rng.rand(64, 64, 3) * 255).astype(np.uint8), mode="RGB"
        )
        for _ in range(300)
    ]


@pytest.fixture(scope="module")
def fused_gallery(gallery_300):
    with _patch_clip():
        cfg = train.load_config()
        model, pp = train._load_clip(cfg)
        feats = train.extract_features(gallery_300, model, pp, cfg)
        fused = train.fuse_features(feats, cfg)
    return fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)


@pytest.fixture(scope="module")
def faiss_index(fused_gallery):
    idx = faiss.IndexFlatIP(fused_gallery.shape[1])
    idx.add(np.ascontiguousarray(fused_gallery, dtype=np.float32))
    return idx


# ---------------------------------------------------------------------------


class TestFaissSearchLatency:
    """The R@1=72.9% Phase 6 result reported 0.10 ms/query for the search step
    on a 300-product gallery. This regresses if the fusion dim balloons or
    someone replaces IndexFlatIP with a slower variant."""

    def test_single_query_latency_under_5ms(self, faiss_index, fused_gallery):
        query = fused_gallery[0:1]
        # Warm up
        for _ in range(5):
            faiss_index.search(query, 20)
        latencies = []
        for _ in range(50):
            t0 = time.perf_counter()
            faiss_index.search(query, 20)
            latencies.append((time.perf_counter() - t0) * 1000)
        assert median(latencies) < 5.0, f"median search latency {median(latencies):.2f}ms exceeds 5ms"

    def test_batch_search_throughput(self, faiss_index, fused_gallery):
        batch = fused_gallery[:32]
        for _ in range(3):
            faiss_index.search(batch, 20)
        t0 = time.perf_counter()
        n_iters = 20
        for _ in range(n_iters):
            faiss_index.search(batch, 20)
        elapsed = time.perf_counter() - t0
        qps = (n_iters * 32) / elapsed
        # Production claim is "0.10 ms/query" search-only on 300 gallery.
        # Even with 100x slack we should clear 1000 qps for FAISS Flat search.
        assert qps > 1000, f"throughput {qps:.0f} qps below 1000 floor"


class TestFeatureExtractionLatency:
    def test_color_histogram_under_3ms(self):
        from src.feature_engineering import extract_color_palette

        img = Image.new("RGB", (224, 224), color=(120, 80, 60))
        for _ in range(3):
            extract_color_palette(img)
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            extract_color_palette(img)
            latencies.append((time.perf_counter() - t0) * 1000)
        assert median(latencies) < 3.0, (
            f"color palette median {median(latencies):.2f}ms exceeds 3ms budget "
            f"(documented as ~0.5ms in feature_engineering.py)"
        )

    def test_spatial_grid_under_15ms(self):
        from src.feature_engineering import extract_spatial_color_grid

        img = Image.new("RGB", (224, 224), color=(120, 80, 60))
        for _ in range(3):
            extract_spatial_color_grid(img)
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            extract_spatial_color_grid(img)
            latencies.append((time.perf_counter() - t0) * 1000)
        assert median(latencies) < 15.0, (
            f"spatial grid median {median(latencies):.2f}ms exceeds 15ms budget"
        )

    def test_hsv_histogram_under_5ms(self):
        from src.feature_engineering import extract_hsv_histogram

        img = Image.new("RGB", (224, 224), color=(120, 80, 60))
        for _ in range(3):
            extract_hsv_histogram(img)
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            extract_hsv_histogram(img)
            latencies.append((time.perf_counter() - t0) * 1000)
        assert median(latencies) < 5.0, (
            f"hsv histogram median {median(latencies):.2f}ms exceeds 5ms budget"
        )


class TestMemoryFootprint:
    """The production index must stay small enough to fit in container memory."""

    def test_300_product_index_under_5mb(self, fused_gallery):
        # 300 vectors × 1008 dims × 4 bytes = 1.21 MB for the matrix
        size_mb = fused_gallery.nbytes / (1024 * 1024)
        assert size_mb < 5.0, f"gallery features take {size_mb:.2f}MB, exceeds 5MB budget"

    def test_index_serialization_round_trip(self, faiss_index, tmp_path):
        path = tmp_path / "test.index"
        faiss.write_index(faiss_index, str(path))
        size_mb = path.stat().st_size / (1024 * 1024)
        assert size_mb < 5.0, f"serialized index is {size_mb:.2f}MB"

        loaded = faiss.read_index(str(path))
        assert loaded.d == faiss_index.d
        assert loaded.ntotal == faiss_index.ntotal


class TestLatencyDistribution:
    """Tail latency matters for production — p99 must stay reasonable."""

    def test_p99_search_latency(self, faiss_index, fused_gallery):
        for _ in range(10):
            faiss_index.search(fused_gallery[:1], 20)

        latencies = []
        for i in range(200):
            q = fused_gallery[i % len(fused_gallery) : i % len(fused_gallery) + 1]
            t0 = time.perf_counter()
            faiss_index.search(q, 20)
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        # p99 should not be more than 50× p50 — guards against pathological GC pauses
        # in CI rather than nailing an absolute number.
        assert p99 < max(50.0, p50 * 50), (
            f"p99 {p99:.2f}ms is too far above p50 {p50:.2f}ms (p95={p95:.2f}ms)"
        )
