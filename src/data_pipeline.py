import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def download_deepfashion_metadata(max_items=None):
    """Download DeepFashion In-Shop metadata (no images) as a DataFrame."""
    from datasets import load_dataset

    ds = load_dataset("Marqo/deepfashion-inshop", split="data", streaming=True)

    records = []
    for i, ex in enumerate(tqdm(ds, desc="Loading metadata")):
        if max_items and i >= max_items:
            break
        item_id = ex["item_ID"]
        parts = item_id.rsplit("_", 2)
        product_id = parts[0] if len(parts) >= 3 else item_id

        records.append({
            "index": i,
            "item_id": item_id,
            "product_id": product_id,
            "category1": ex["category1"],
            "category2": ex["category2"],
            "category3": ex["category3"],
            "color": ex["color"],
            "description": ex["text"] if ex["text"] else "",
        })

    df = pd.DataFrame(records)
    return df


def download_deepfashion_images(df, save_dir=None, max_items=None):
    """Download images for items in df, saving to disk."""
    from datasets import load_dataset

    if save_dir is None:
        save_dir = DATA_RAW / "images"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Marqo/deepfashion-inshop", split="data", streaming=True)

    indices_needed = set(df["index"].values[:max_items] if max_items else df["index"].values)
    saved = 0

    for i, ex in enumerate(tqdm(ds, desc="Downloading images", total=max(indices_needed) + 1)):
        if i not in indices_needed:
            continue

        img = ex["image"]
        item_id = ex["item_ID"]
        img_path = save_dir / f"{item_id}.jpg"

        if not img_path.exists():
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path, "JPEG", quality=90)

        saved += 1
        if saved >= len(indices_needed):
            break

    return save_dir


def load_metadata(path=None):
    """Load cached metadata CSV."""
    if path is None:
        path = DATA_PROCESSED / "metadata.csv"
    return pd.read_csv(path)


def create_retrieval_splits(df, test_frac=0.2, seed=42):
    """Split products into gallery and query sets for retrieval evaluation.

    Gallery: one image per product (the 'front' view if available).
    Query: remaining images of test products.
    """
    rng = np.random.RandomState(seed)

    products = df["product_id"].unique()
    rng.shuffle(products)

    n_test = int(len(products) * test_frac)
    test_products = set(products[:n_test])
    train_products = set(products[n_test:])

    train_df = df[df["product_id"].isin(train_products)].copy()
    test_df = df[df["product_id"].isin(test_products)].copy()

    gallery_rows = []
    query_rows = []

    for pid, group in test_df.groupby("product_id"):
        front = group[group["item_id"].str.contains("front", case=False)]
        if len(front) > 0:
            gallery_rows.append(front.iloc[0])
            remaining = group.drop(front.index[0])
        else:
            gallery_rows.append(group.iloc[0])
            remaining = group.iloc[1:]
        for _, row in remaining.iterrows():
            query_rows.append(row)

    gallery_df = pd.DataFrame(gallery_rows)
    query_df = pd.DataFrame(query_rows)

    return train_df, gallery_df, query_df
