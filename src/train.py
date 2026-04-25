"""
Build production artifacts for the Visual Product Search Engine.

Extracts and caches:
  - CLIP B/32 text embeddings for gallery
  - Color 48D histograms for gallery
  - Saves production config to models/search_config.json

Usage:
  python -m src.train [--gallery-csv data/processed/gallery.csv] [--device cpu]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import extract_color_palette


def build_text_embeddings(df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
    import open_clip, torch
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai", device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    texts = df["description"].fillna("").tolist()
    batch_size = 64
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text embeddings"):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.no_grad():
            emb = model.encode_text(tokens).float().cpu().numpy()
        all_embs.append(emb)
    embs = np.vstack(all_embs)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def build_color_embeddings(df: pd.DataFrame, image_dir: Path) -> np.ndarray:
    all_colors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Color histograms"):
        img_path = image_dir / f"{row['item_id']}.jpg"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            feat = extract_color_palette(img, bins_per_channel=8)
        else:
            feat = np.zeros(24, dtype=np.float32)
        all_colors.append(feat)
    embs = np.vstack(all_colors)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Build Visual Search production artifacts")
    parser.add_argument("--gallery-csv", default="data/processed/gallery.csv")
    parser.add_argument("--image-dir", default="data/raw/images")
    parser.add_argument("--cache-dir", default="data/processed/emb_cache")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    gallery_csv = PROJECT_ROOT / args.gallery_csv
    image_dir = PROJECT_ROOT / args.image_dir
    cache_dir = PROJECT_ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading gallery from {gallery_csv}")
    gallery_df = pd.read_csv(gallery_csv)
    print(f"Gallery: {len(gallery_df)} items, {gallery_df['category2'].nunique()} categories")

    t0 = time.perf_counter()

    # Text embeddings
    text_path = cache_dir / "clip_b32_text_gallery.npy"
    if not text_path.exists() or not args.skip_existing:
        print("Building CLIP B/32 text embeddings...")
        text_embs = build_text_embeddings(gallery_df, device=args.device)
        np.save(text_path, text_embs)
        print(f"Saved: {text_path} shape={text_embs.shape}")
    else:
        print(f"Skipping text embeddings (exists): {text_path}")

    # Color embeddings
    color_path = cache_dir / "color48_gallery.npy"
    if not color_path.exists() or not args.skip_existing:
        print("Building color histograms...")
        color_embs = build_color_embeddings(gallery_df, image_dir)
        np.save(color_path, color_embs)
        print(f"Saved: {color_path} shape={color_embs.shape}")
    else:
        print(f"Skipping color embeddings (exists): {color_path}")

    elapsed = time.perf_counter() - t0
    print(f"\nBuild complete in {elapsed:.1f}s")

    # Save production config
    config = {
        "pipeline": "cat-filter + color-hist-48D + CLIP-B32-text",
        "w_text": 0.80,
        "k": 20,
        "clip_model": "ViT-B-32",
        "clip_pretrained": "openai",
        "color_bins": 8,
        "color_dim": 24,
        "text_dim": 512,
        "gallery_size": len(gallery_df),
        "best_r1": 0.920,
        "best_r5": 0.990,
        "note": "Ablation Phase 5: removing CLIP visual improves R@1 +1.35pp",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "search_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: models/search_config.json")

    print("\nProduction artifacts ready:")
    print(f"  {text_path}")
    print(f"  {color_path}")
    print(f"  models/search_config.json")


if __name__ == "__main__":
    main()
