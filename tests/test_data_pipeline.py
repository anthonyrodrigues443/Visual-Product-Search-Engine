"""Tests for data pipeline: metadata loading, split creation, image utilities."""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import create_retrieval_splits


@pytest.fixture
def sample_df():
    rows = []
    for pid in range(20):
        for view in range(4):
            rows.append({
                "index": pid * 4 + view,
                "item_id": f"product_{pid:03d}_view{view}" + ("_front" if view == 0 else ""),
                "product_id": f"product_{pid:03d}",
                "category1": "CLOTHING",
                "category2": "shirts" if pid < 10 else "pants",
                "category3": "casual",
                "color": "black",
                "description": f"A product {pid}",
            })
    return pd.DataFrame(rows)


class TestCreateRetrievalSplits:
    def test_split_sizes(self, sample_df):
        train_df, gallery_df, query_df = create_retrieval_splits(sample_df, test_frac=0.2, seed=42)
        n_products = sample_df["product_id"].nunique()
        n_test = int(n_products * 0.2)
        assert gallery_df["product_id"].nunique() == n_test
        assert train_df["product_id"].nunique() == n_products - n_test

    def test_no_overlap(self, sample_df):
        train_df, gallery_df, query_df = create_retrieval_splits(sample_df, test_frac=0.2, seed=42)
        train_pids = set(train_df["product_id"])
        test_pids = set(gallery_df["product_id"])
        assert train_pids.isdisjoint(test_pids)

    def test_gallery_one_per_product(self, sample_df):
        _, gallery_df, _ = create_retrieval_splits(sample_df, test_frac=0.2, seed=42)
        counts = gallery_df.groupby("product_id").size()
        assert (counts == 1).all()

    def test_gallery_prefers_front_view(self, sample_df):
        _, gallery_df, _ = create_retrieval_splits(sample_df, test_frac=0.5, seed=42)
        for _, row in gallery_df.iterrows():
            assert "front" in row["item_id"].lower()

    def test_query_products_match_gallery(self, sample_df):
        _, gallery_df, query_df = create_retrieval_splits(sample_df, test_frac=0.2, seed=42)
        gallery_pids = set(gallery_df["product_id"])
        query_pids = set(query_df["product_id"])
        assert query_pids.issubset(gallery_pids)

    def test_deterministic(self, sample_df):
        _, g1, q1 = create_retrieval_splits(sample_df, test_frac=0.2, seed=42)
        _, g2, q2 = create_retrieval_splits(sample_df, test_frac=0.2, seed=42)
        assert list(g1["product_id"]) == list(g2["product_id"])
        assert list(q1["product_id"]) == list(q2["product_id"])

    def test_different_seeds_differ(self, sample_df):
        _, g1, _ = create_retrieval_splits(sample_df, test_frac=0.2, seed=42)
        _, g2, _ = create_retrieval_splits(sample_df, test_frac=0.2, seed=99)
        assert list(g1["product_id"]) != list(g2["product_id"])
