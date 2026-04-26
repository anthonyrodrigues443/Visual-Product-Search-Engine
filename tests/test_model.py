"""Tests for feature extraction, fusion, and FAISS index operations."""

import numpy as np
import pytest
from PIL import Image

from src.feature_engineering import (
    extract_color_palette,
    extract_hsv_histogram,
    extract_spatial_color_grid,
)
from src.train import fuse_features, load_config


@pytest.fixture
def red_image():
    return Image.new("RGB", (128, 128), (255, 0, 0))


@pytest.fixture
def blue_image():
    return Image.new("RGB", (128, 128), (0, 0, 255))


@pytest.fixture
def cfg():
    return load_config()


class TestColorPalette:
    def test_output_shape(self, red_image):
        feat = extract_color_palette(red_image, bins_per_channel=8)
        assert feat.shape == (24,)

    def test_normalized(self, red_image):
        feat = extract_color_palette(red_image, bins_per_channel=8)
        assert abs(feat.sum() - 1.0) < 1e-5

    def test_red_dominant(self, red_image):
        feat = extract_color_palette(red_image, bins_per_channel=8)
        r_bins = feat[:8]
        assert r_bins[-1] > 0.3  # last bin (high red) dominates

    def test_different_images_differ(self, red_image, blue_image):
        r_feat = extract_color_palette(red_image)
        b_feat = extract_color_palette(blue_image)
        assert not np.allclose(r_feat, b_feat)

    def test_dtype(self, red_image):
        feat = extract_color_palette(red_image)
        assert feat.dtype == np.float32


class TestHSVHistogram:
    def test_output_shape(self, red_image):
        feat = extract_hsv_histogram(red_image, bins=8)
        assert feat.shape == (24,)

    def test_normalized(self, red_image):
        feat = extract_hsv_histogram(red_image, bins=8)
        assert abs(feat.sum() - 1.0) < 1e-5

    def test_different_bins(self, red_image):
        feat4 = extract_hsv_histogram(red_image, bins=4)
        feat8 = extract_hsv_histogram(red_image, bins=8)
        assert feat4.shape == (12,)
        assert feat8.shape == (24,)


class TestSpatialColorGrid:
    def test_output_shape(self, red_image):
        feat = extract_spatial_color_grid(red_image, grid_rows=4, grid_cols=4, bins=4)
        assert feat.shape == (192,)

    def test_uniform_image_has_equal_regions(self, red_image):
        feat = extract_spatial_color_grid(red_image, grid_rows=4, grid_cols=4, bins=4)
        region_dim = 4 * 3  # bins * channels
        regions = feat.reshape(16, region_dim)
        for i in range(1, 16):
            assert np.allclose(regions[0], regions[i], atol=1e-5)


class TestFeatureFusion:
    def test_output_dim(self, cfg):
        clip = np.random.randn(5, 768).astype(np.float32)
        color = np.random.randn(5, 48).astype(np.float32)
        spatial = np.random.randn(5, 192).astype(np.float32)
        fused = fuse_features({"clip": clip, "color": color, "spatial": spatial}, cfg)
        assert fused.shape == (5, 768 + 48 + 192)

    def test_weighting_applied(self, cfg):
        clip = np.ones((1, 768), dtype=np.float32)
        color = np.ones((1, 48), dtype=np.float32)
        spatial = np.ones((1, 192), dtype=np.float32)
        fused = fuse_features({"clip": clip, "color": color, "spatial": spatial}, cfg)
        clip_block = fused[0, :768]
        spatial_block = fused[0, 768 + 48:]
        clip_norm = np.linalg.norm(clip_block)
        spatial_norm = np.linalg.norm(spatial_block)
        assert clip_norm / spatial_norm == pytest.approx(
            cfg["fusion"]["w_clip"] / cfg["fusion"]["w_spatial"], rel=0.01
        )

    def test_dtype(self, cfg):
        feats = {
            "clip": np.random.randn(3, 768).astype(np.float32),
            "color": np.random.randn(3, 48).astype(np.float32),
            "spatial": np.random.randn(3, 192).astype(np.float32),
        }
        fused = fuse_features(feats, cfg)
        assert fused.dtype == np.float32
