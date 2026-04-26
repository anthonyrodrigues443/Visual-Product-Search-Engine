"""Tests for the FastAPI service in api.py.

Uses TestClient + a stubbed VisualSearchEngine so we exercise the HTTP layer
(routing, validation, schema serialisation) without touching CLIP weights.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


class _StubEngine:
    def __init__(self):
        self.cfg = {
            "model": {"backbone": "ViT-L-14"},
            "fusion": {"w_clip": 1.0, "w_color": 1.0, "w_spatial": 0.25},
        }
        self.product_ids = [f"prod_{i:03d}" for i in range(6)]
        self.categories = ["shirts", "shirts", "shirts", "pants", "pants", "pants"]
        self.gallery_features = np.zeros((6, 1008), dtype=np.float32)

    def search(self, image, top_k=10, category=None):
        if category == "raises":
            raise RuntimeError("boom")
        n = min(top_k, 3 if category in self.categories else 6)
        return [
            {
                "rank": i + 1,
                "product_id": self.product_ids[i],
                "category": category or self.categories[i],
                "score": round(1.0 - i * 0.05, 4),
                "gallery_index": i,
            }
            for i in range(n)
        ]


@pytest.fixture
def client():
    import api as api_module

    with patch.object(api_module, "_get_engine", lambda: _StubEngine()):
        api_module._engine = _StubEngine()
        with TestClient(api_module.app) as c:
            yield c
        api_module._engine = None


def _png_bytes(color=(200, 50, 50)):
    img = Image.new("RGB", (32, 32), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["engine_loaded"] is True


class TestInfo:
    def test_info_reports_engine_metadata(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["model_backbone"] == "ViT-L-14"
        assert body["feature_dim"] == 1008
        assert body["n_gallery"] == 6
        assert body["fusion_weights"]["w_clip"] == 1.0


class TestCategories:
    def test_categories_lists_unique_with_counts(self, client):
        r = client.get("/categories")
        assert r.status_code == 200
        body = r.json()
        assert body["categories"] == ["pants", "shirts"]
        assert body["counts"] == {"shirts": 3, "pants": 3}


class TestSearch:
    def test_search_returns_ranked_results(self, client):
        r = client.post(
            "/search",
            files={"image": ("test.png", _png_bytes(), "image/png")},
            data={"top_k": 5},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["n_results"] == 5
        assert body["query_filename"] == "test.png"
        assert body["category_filter_applied"] is False
        ranks = [hit["rank"] for hit in body["results"]]
        assert ranks == sorted(ranks)
        scores = [hit["score"] for hit in body["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_category_filter(self, client):
        r = client.post(
            "/search",
            files={"image": ("q.jpg", _png_bytes(), "image/jpeg")},
            data={"top_k": 10, "category": "shirts"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["category_filter_applied"] is True
        for hit in body["results"]:
            assert hit["category"] == "shirts"

    def test_search_rejects_non_image(self, client):
        r = client.post(
            "/search",
            files={"image": ("notes.txt", b"hello world", "text/plain")},
            data={"top_k": 5},
        )
        assert r.status_code == 400
        assert "image" in r.json()["detail"].lower()

    def test_search_rejects_corrupt_image_bytes(self, client):
        r = client.post(
            "/search",
            files={"image": ("broken.png", b"not actually an image", "image/png")},
            data={"top_k": 5},
        )
        assert r.status_code == 400

    def test_search_top_k_validation(self, client):
        r = client.post(
            "/search",
            files={"image": ("q.png", _png_bytes(), "image/png")},
            data={"top_k": 0},
        )
        assert r.status_code == 422  # FastAPI validation error

        r = client.post(
            "/search",
            files={"image": ("q.png", _png_bytes(), "image/png")},
            data={"top_k": 1000},
        )
        assert r.status_code == 422

    def test_search_engine_exception_returns_500(self, client):
        r = client.post(
            "/search",
            files={"image": ("q.png", _png_bytes(), "image/png")},
            data={"top_k": 5, "category": "raises"},
        )
        assert r.status_code == 500
        assert "boom" in r.json()["detail"]


class TestRoot:
    def test_root_redirects_to_docs(self, client):
        r = client.get("/", follow_redirects=False)
        assert r.status_code in (301, 302, 307)
        assert "/docs" in r.headers["location"]
