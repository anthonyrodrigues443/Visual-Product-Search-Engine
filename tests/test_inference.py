"""Tests for retrieval evaluation metrics and category-filtered search."""

import numpy as np
import pytest

from src.evaluate import category_filtered_search, per_category_recall, recall_at_k


class TestRecallAtK:
    def test_perfect_retrieval(self):
        query_pids = np.array(["a", "b", "c"])
        gallery_pids = np.array(["a", "b", "c", "d", "e"])
        indices = np.array([[0, 1, 2], [1, 0, 2], [2, 3, 4]])
        result = recall_at_k(query_pids, gallery_pids, indices, ks=(1, 3))
        assert result["R@1"] == 1.0
        assert result["R@3"] == 1.0

    def test_zero_retrieval(self):
        query_pids = np.array(["a", "b"])
        gallery_pids = np.array(["c", "d", "e"])
        indices = np.array([[0, 1, 2], [1, 0, 2]])
        result = recall_at_k(query_pids, gallery_pids, indices, ks=(1,))
        assert result["R@1"] == 0.0

    def test_partial_retrieval(self):
        query_pids = np.array(["a", "b"])
        gallery_pids = np.array(["a", "c", "d"])
        indices = np.array([[0, 1, 2], [1, 0, 2]])  # first correct, second wrong
        result = recall_at_k(query_pids, gallery_pids, indices, ks=(1,))
        assert result["R@1"] == 0.5

    def test_r5_higher_than_r1(self):
        query_pids = np.array(["a", "b"])
        gallery_pids = np.array(["a", "b", "c", "d", "e"])
        indices = np.array([[2, 3, 4, 0, 1], [3, 4, 2, 1, 0]])  # correct at rank 4/5
        result = recall_at_k(query_pids, gallery_pids, indices, ks=(1, 5))
        assert result["R@5"] >= result["R@1"]

    def test_output_is_rounded(self):
        query_pids = np.array(["a", "b", "c"])
        gallery_pids = np.array(["a", "b", "d"])
        indices = np.array([[0, 1, 2], [1, 0, 2], [2, 0, 1]])
        result = recall_at_k(query_pids, gallery_pids, indices, ks=(1,))
        assert isinstance(result["R@1"], float)
        assert len(str(result["R@1"]).split(".")[-1]) <= 4


class TestPerCategoryRecall:
    def test_per_category(self):
        cats = np.array(["shirts", "shirts", "pants", "pants"])
        q_pids = np.array(["a", "b", "c", "d"])
        g_pids = np.array(["a", "b", "c", "d", "e"])
        indices = np.array([[0, 1], [2, 0], [2, 3], [3, 4]])
        result = per_category_recall(cats, q_pids, g_pids, indices, k=1)
        assert result["shirts"] == 0.5  # a correct, b wrong
        assert result["pants"] == 1.0  # both correct


class TestCategoryFilteredSearch:
    def test_filters_by_category(self):
        np.random.seed(42)
        dim = 128
        g_feats = np.ascontiguousarray(np.random.randn(50, dim).astype(np.float32))
        q_feats = np.ascontiguousarray(np.random.randn(10, dim).astype(np.float32))
        g_cats = np.array(["shirts"] * 25 + ["pants"] * 25)
        q_cats = np.array(["shirts"] * 5 + ["pants"] * 5)

        _, indices = category_filtered_search(g_feats, q_feats, g_cats, q_cats, k=5)

        for qi in range(5):  # shirts queries
            for rank in range(5):
                if indices[qi, rank] >= 0:
                    assert g_cats[indices[qi, rank]] == "shirts"
        for qi in range(5, 10):  # pants queries
            for rank in range(5):
                if indices[qi, rank] >= 0:
                    assert g_cats[indices[qi, rank]] == "pants"

    def test_output_shape(self):
        dim = 128
        g_feats = np.ascontiguousarray(np.random.randn(40, dim).astype(np.float32))
        q_feats = np.ascontiguousarray(np.random.randn(10, dim).astype(np.float32))
        g_cats = np.array(["a"] * 20 + ["b"] * 20)
        q_cats = np.array(["a"] * 5 + ["b"] * 5)

        scores, indices = category_filtered_search(g_feats, q_feats, g_cats, q_cats, k=5)
        assert scores.shape == (10, 5)
        assert indices.shape == (10, 5)

    def test_k_larger_than_category(self):
        dim = 128
        g_feats = np.ascontiguousarray(np.random.randn(30, dim).astype(np.float32))
        q_feats = np.ascontiguousarray(np.random.randn(4, dim).astype(np.float32))
        g_cats = np.array(["a"] * 3 + ["b"] * 27)
        q_cats = np.array(["a"] * 4)

        scores, indices = category_filtered_search(g_feats, q_feats, g_cats, q_cats, k=10)
        for qi in range(4):
            valid = indices[qi][indices[qi] >= 0]
            assert len(valid) == 3  # only 3 items in category "a"
