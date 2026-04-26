"""Build a visual product search index from a gallery of images.

Extracts CLIP ViT-L/14 + color + spatial features, concatenates with
Optuna-tuned weights, and persists the FAISS index + metadata to disk.
"""

import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from src.feature_engineering import (
    extract_color_palette,
    extract_hsv_histogram,
    extract_spatial_color_grid,
)

PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path: str | None = None) -> dict:
    if path is None:
        path = PROJECT_ROOT / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def _load_clip(cfg: dict):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["model"]["backbone"], pretrained=cfg["model"]["pretrained"]
    )
    model.eval()
    return model, preprocess


def extract_features(
    images: list[Image.Image],
    clip_model,
    clip_preprocess,
    cfg: dict,
    batch_size: int = 32,
) -> dict[str, np.ndarray]:
    clip_feats = []
    batch = []
    for img in tqdm(images, desc="CLIP", leave=False):
        batch.append(clip_preprocess(img))
        if len(batch) >= batch_size:
            with torch.no_grad():
                f = clip_model.encode_image(torch.stack(batch))
            clip_feats.append(f.cpu().float().numpy())
            batch = []
    if batch:
        with torch.no_grad():
            f = clip_model.encode_image(torch.stack(batch))
        clip_feats.append(f.cpu().float().numpy())
    clip_arr = np.vstack(clip_feats)

    color_feats = []
    spatial_feats = []
    for img in tqdm(images, desc="Color+Spatial", leave=False):
        rgb = extract_color_palette(img, bins_per_channel=cfg["features"]["color"]["rgb_bins"])
        hsv = extract_hsv_histogram(img, bins=cfg["features"]["color"]["hsv_bins"])
        color_feats.append(np.concatenate([rgb, hsv]))
        spatial_feats.append(
            extract_spatial_color_grid(
                img,
                grid_rows=cfg["features"]["spatial"]["grid_rows"],
                grid_cols=cfg["features"]["spatial"]["grid_cols"],
                bins=cfg["features"]["spatial"]["bins"],
            )
        )

    return {
        "clip": clip_arr,
        "color": np.array(color_feats, dtype=np.float32),
        "spatial": np.array(spatial_feats, dtype=np.float32),
    }


def fuse_features(feats: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    weights = [cfg["fusion"]["w_clip"], cfg["fusion"]["w_color"], cfg["fusion"]["w_spatial"]]
    arrays = [feats["clip"], feats["color"], feats["spatial"]]
    parts = []
    for arr, w in zip(arrays, weights):
        normed = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
        parts.append(normed * w)
    return np.concatenate(parts, axis=1).astype(np.float32)


def build_index(
    gallery_images: list[Image.Image],
    gallery_ids: list[str],
    gallery_categories: list[str],
    output_dir: str | Path | None = None,
    config_path: str | None = None,
) -> Path:
    cfg = load_config(config_path)
    if output_dir is None:
        output_dir = PROJECT_ROOT / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building index for {len(gallery_images)} gallery images...")
    t0 = time.time()

    clip_model, clip_preprocess = _load_clip(cfg)
    feats = extract_features(gallery_images, clip_model, clip_preprocess, cfg)
    fused = fuse_features(feats, cfg)

    fused_normed = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)
    index = faiss.IndexFlatIP(fused_normed.shape[1])
    index.add(np.ascontiguousarray(fused_normed))

    faiss.write_index(index, str(output_dir / "gallery.index"))
    np.save(output_dir / "gallery_features.npy", fused_normed)

    metadata = {
        "product_ids": gallery_ids,
        "categories": gallery_categories,
        "n_gallery": len(gallery_images),
        "feature_dim": int(fused_normed.shape[1]),
        "build_time_s": round(time.time() - t0, 1),
        "config": cfg,
    }
    with open(output_dir / "gallery_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Index built: {fused_normed.shape[1]}D, {len(gallery_images)} vectors, {time.time()-t0:.1f}s")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build visual search index")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--output", default=None, help="Output directory for index")
    parser.add_argument("--max-items", type=int, default=None, help="Limit gallery size")
    args = parser.parse_args()

    from src.data_pipeline import (
        create_retrieval_splits,
        download_deepfashion_images,
        download_deepfashion_metadata,
    )

    cfg = load_config(args.config)
    print("Downloading metadata...")
    df = download_deepfashion_metadata(max_items=args.max_items)
    print("Creating splits...")
    _, gallery_df, _ = create_retrieval_splits(df, test_frac=cfg["dataset"]["test_frac"])

    print("Downloading gallery images...")
    img_dir = download_deepfashion_images(gallery_df)

    images = []
    ids = []
    cats = []
    for _, row in gallery_df.iterrows():
        img_path = img_dir / f"{row['item_id']}.jpg"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            images.append(img)
            ids.append(row["product_id"])
            cats.append(row["category2"])

    build_index(images, ids, cats, output_dir=args.output, config_path=args.config)
