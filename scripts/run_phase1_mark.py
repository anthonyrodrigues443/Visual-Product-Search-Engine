"""
Phase 1 Mark — Complementary Experiment: Color Palette Features + EfficientNet-B0

Building on Anthony's Phase 1 (ResNet50 baseline, R@1=30.7%, jackets hardest at 13.9%):

Hypothesis: The 0.048 similarity-separation gap has two causes:
  1. CNN architecture: ResNet50 global-average-pooling collapses spatial info
  2. Missing color signal: ResNet50 embeds color indirectly via texture — explicit
     color palette features should help categories like jackets where hue is a
     primary consumer discriminator

Experiments:
  1.M.1  EfficientNet-B0 (ImageNet) + FAISS           — different backbone baseline
  1.M.2  Color-only retrieval (palette + HSV hist)    — how much does color alone do?
  1.M.3  ResNet50 + Color (augmented embedding)       — can we stack Anthony's model?
  1.M.4  EfficientNet-B0 + Color (augmented)          — best single-session combo
  1.M.5  Color re-ranking of ResNet50 top-20          — inference-time add-on
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS = PROJECT_ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

from src.data_pipeline import create_retrieval_splits
from src.feature_engineering import (
    extract_color_palette,
    extract_hsv_histogram,
    augment_embedding_with_color,
    color_rerank,
)

# ===== CONSTANTS =====
EVAL_PRODUCTS = 300
SEED = 42


def compute_recall_at_k(query_pids, gallery_pids, retrieved_indices, k):
    correct = 0
    for i in range(len(query_pids)):
        top_k_pids = gallery_pids[retrieved_indices[i, :k]]
        if query_pids[i] in top_k_pids:
            correct += 1
    return correct / len(query_pids)


def eval_recall(query_pids, gallery_pids, indices, label=""):
    r1  = compute_recall_at_k(query_pids, gallery_pids, indices, 1)
    r5  = compute_recall_at_k(query_pids, gallery_pids, indices, 5)
    r10 = compute_recall_at_k(query_pids, gallery_pids, indices, 10)
    r20 = compute_recall_at_k(query_pids, gallery_pids, indices, 20)
    print(f"  {label:40s}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}  R@20={r20:.4f}")
    sys.stdout.flush()
    return {"recall@1": r1, "recall@5": r5, "recall@10": r10, "recall@20": r20}


def faiss_search(gallery_feats, query_feats, k=20):
    import faiss
    gallery_f32 = np.ascontiguousarray(gallery_feats, dtype=np.float32)
    query_f32   = np.ascontiguousarray(query_feats,   dtype=np.float32)

    # L2-normalize for cosine search via inner product
    gallery_norms = np.linalg.norm(gallery_f32, axis=1, keepdims=True) + 1e-8
    query_norms   = np.linalg.norm(query_f32,   axis=1, keepdims=True) + 1e-8
    gallery_normed = gallery_f32 / gallery_norms
    query_normed   = query_f32   / query_norms

    dim = gallery_normed.shape[1]
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(dim)
    index.add(gallery_normed)
    distances, indices = index.search(query_normed, k)
    return distances, indices


def per_category_recall(query_pids, query_cats, gallery_pids, indices, k=1):
    cats = {}
    for cat in sorted(set(query_cats)):
        mask = query_cats == cat
        if mask.sum() < 5:
            continue
        r = compute_recall_at_k(query_pids[mask], gallery_pids, indices[mask], k)
        cats[cat] = {"recall@1": r, "n_queries": int(mask.sum())}
    return cats


# ===== 1. LOAD / DOWNLOAD METADATA =====
print("=" * 65)
print("PHASE 1 MARK — COLOR PALETTE FEATURES + EFFICIENTNET-B0")
print("=" * 65)
print("\n--- 1. Loading metadata ---")
sys.stdout.flush()

meta_path = DATA_PROCESSED / 'metadata.csv'
if meta_path.exists():
    df = pd.read_csv(meta_path)
    print(f"  Loaded metadata from cache: {len(df):,} images, {df['product_id'].nunique():,} products")
else:
    print("  Downloading metadata from HuggingFace (Marqo/deepfashion-inshop)...")
    from datasets import load_dataset
    ds = load_dataset("Marqo/deepfashion-inshop", split="data", streaming=True)
    records = []
    for i, ex in enumerate(tqdm(ds, desc="  Loading")):
        item_id = ex["item_ID"]
        parts   = item_id.rsplit("_", 2)
        product_id = parts[0] if len(parts) >= 3 else item_id
        records.append({
            "index":      i,
            "item_id":    item_id,
            "product_id": product_id,
            "category1":  ex["category1"],
            "category2":  ex["category2"],
            "category3":  ex["category3"],
            "color":      ex["color"],
            "description": ex["text"] if ex["text"] else "",
        })
    df = pd.DataFrame(records)
    df.to_csv(meta_path, index=False)
    print(f"  Saved {len(df):,} rows -> {meta_path}")
sys.stdout.flush()

# ===== 2. RETRIEVAL SPLITS =====
print("\n--- 2. Retrieval splits ---")
train_df, gallery_df, query_df = create_retrieval_splits(df, test_frac=0.2, seed=SEED)

# Deterministic 300-product eval slice
rng = np.random.RandomState(SEED)
all_products = gallery_df['product_id'].values
eval_products = all_products[:EVAL_PRODUCTS]

eval_gallery = gallery_df[gallery_df['product_id'].isin(eval_products)].reset_index(drop=True)
eval_query   = query_df[query_df['product_id'].isin(eval_products)].reset_index(drop=True)
print(f"  Gallery: {len(eval_gallery):,}  Query: {len(eval_query):,}")
sys.stdout.flush()

all_eval_idx = sorted(set(
    eval_gallery['index'].astype(int).tolist() +
    eval_query['index'].astype(int).tolist()
))

# ===== 3. DOWNLOAD IMAGES (with disk cache) =====
IMG_CACHE = PROJECT_ROOT / 'data' / 'raw' / 'images'
IMG_CACHE.mkdir(parents=True, exist_ok=True)

# Build index->item_id map for disk cache lookup
idx_to_item = {}
for _, row in eval_gallery.iterrows():
    idx_to_item[int(row['index'])] = row['item_id']
for _, row in eval_query.iterrows():
    idx_to_item[int(row['index'])] = row['item_id']

# Check which images are already on disk
cached_indices = {idx for idx, item_id in idx_to_item.items()
                  if (IMG_CACHE / f"{item_id}.jpg").exists()}
missing_indices = set(all_eval_idx) - cached_indices

print(f"\n--- 3. Images: {len(cached_indices)} cached, {len(missing_indices)} to download ---")
sys.stdout.flush()

images_by_index = {}

# Load cached images from disk
for idx in cached_indices:
    img_path = IMG_CACHE / f"{idx_to_item[idx]}.jpg"
    images_by_index[idx] = Image.open(img_path).convert('RGB')

# Download only missing images
if missing_indices:
    from datasets import load_dataset as _load_ds
    ds_stream = _load_ds("Marqo/deepfashion-inshop", split="data", streaming=True)
    needed = set(missing_indices)
    for i, ex in enumerate(tqdm(ds_stream, total=max(needed)+1, desc="  Downloading missing")):
        if i in needed:
            img = ex['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images_by_index[i] = img
            item_id = idx_to_item.get(i)
            if item_id:
                img.save(IMG_CACHE / f"{item_id}.jpg", "JPEG", quality=90)
            needed.discard(i)
            if not needed:
                break

print(f"  Ready: {len(images_by_index)} images total")
sys.stdout.flush()


# ===== 4. EXPERIMENT 1.M.1: EfficientNet-B0 + FAISS =====
print("\n" + "=" * 65)
print("EXPERIMENT 1.M.1 — EfficientNet-B0 (ImageNet) + FAISS")
print("=" * 65)
sys.stdout.flush()

import torch
import torchvision.transforms as T
import torchvision.models as tvm

device = 'cpu'

eff_model = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
eff_model.classifier = torch.nn.Identity()
eff_model = eff_model.to(device)
eff_model.eval()
print(f"  EfficientNet-B0 loaded — embedding dim: 1280")
sys.stdout.flush()

eff_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def extract_cnn_features(model, images_dict, indices, transform, batch_size=32):
    features, valid_idx = [], []
    batch, bidx = [], []
    for idx in tqdm(indices, desc="  Extracting CNN features"):
        idx_int = int(idx)
        if idx_int not in images_dict:
            continue
        batch.append(transform(images_dict[idx_int]))
        bidx.append(idx_int)
        if len(batch) >= batch_size:
            feats = model(torch.stack(batch).to(device)).cpu().numpy()
            features.append(feats)
            valid_idx.extend(bidx)
            batch, bidx = [], []
    if batch:
        feats = model(torch.stack(batch).to(device)).cpu().numpy()
        features.append(feats)
        valid_idx.extend(bidx)
    return np.vstack(features), valid_idx


eff_gallery_feats, eff_gal_idx = extract_cnn_features(
    eff_model, images_by_index, eval_gallery['index'].values, eff_transform)
eff_query_feats, eff_qry_idx = extract_cnn_features(
    eff_model, images_by_index, eval_query['index'].values, eff_transform)
print(f"  Gallery: {eff_gallery_feats.shape}  Query: {eff_query_feats.shape}")
sys.stdout.flush()

gal_valid_df = eval_gallery[eval_gallery['index'].isin(eff_gal_idx)].reset_index(drop=True)
qry_valid_df = eval_query[eval_query['index'].isin(eff_qry_idx)].reset_index(drop=True)

gallery_pids = gal_valid_df['product_id'].values
query_pids   = qry_valid_df['product_id'].values
query_cats   = qry_valid_df['category2'].values

eff_dist, eff_idx = faiss_search(eff_gallery_feats, eff_query_feats)
results_eff = eval_recall(query_pids, gallery_pids, eff_idx, label="EfficientNet-B0")
results_eff_per_cat = per_category_recall(query_pids, query_cats, gallery_pids, eff_idx)

# Free EfficientNet model
del eff_model
gc.collect()


# ===== 5. EXPERIMENT 1.M.2: Color-only Retrieval =====
print("\n" + "=" * 65)
print("EXPERIMENT 1.M.2 — Color-only Retrieval (Palette + HSV Histogram)")
print("=" * 65)
sys.stdout.flush()

def extract_all_color_features(images_dict, indices):
    palette_feats, hsv_feats, valid_idx = [], [], []
    for idx in tqdm(indices, desc="  Extracting color features"):
        idx_int = int(idx)
        if idx_int not in images_dict:
            continue
        img = images_dict[idx_int]
        palette_feats.append(extract_color_palette(img))
        hsv_feats.append(extract_hsv_histogram(img, bins=8))
        valid_idx.append(idx_int)
    return np.array(palette_feats), np.array(hsv_feats), valid_idx


gal_palette, gal_hsv, gal_color_idx = extract_all_color_features(
    images_by_index, eval_gallery['index'].values)
qry_palette, qry_hsv, qry_color_idx = extract_all_color_features(
    images_by_index, eval_query['index'].values)

# Combined color feature: palette(20D) + HSV_hist(24D) = 44D
gal_color_feat = np.hstack([gal_palette, gal_hsv])
qry_color_feat = np.hstack([qry_palette, qry_hsv])
print(f"  Color feature dim: {gal_color_feat.shape[1]}D (RGB hist 24D + HSV hist 24D)")
sys.stdout.flush()

# Align valid sets
gal_color_set = set(gal_color_idx)
qry_color_set = set(qry_color_idx)
gal_aligned_mask = [int(idx) in gal_color_set for idx in gal_valid_df['index'].values]
qry_aligned_mask = [int(idx) in qry_color_set for idx in qry_valid_df['index'].values]

gal_color_feat_aligned = gal_color_feat[:sum(gal_aligned_mask)]
qry_color_feat_aligned = qry_color_feat[:sum(qry_aligned_mask)]
gallery_pids_c = gallery_pids[:sum(gal_aligned_mask)]
query_pids_c   = query_pids[:sum(qry_aligned_mask)]
query_cats_c   = query_cats[:sum(qry_aligned_mask)]

color_dist, color_idx = faiss_search(gal_color_feat_aligned, qry_color_feat_aligned)
results_color = eval_recall(query_pids_c, gallery_pids_c, color_idx, label="Color-only (palette+HSV)")
results_color_per_cat = per_category_recall(query_pids_c, query_cats_c, gallery_pids_c, color_idx)


# ===== 6. EXPERIMENT 1.M.3: ResNet50 + Color (augmented) =====
print("\n" + "=" * 65)
print("EXPERIMENT 1.M.3 — ResNet50 + Color (augmented embedding)")
print("=" * 65)
sys.stdout.flush()

resnet = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

resnet_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

gal_resnet_feats, gal_rn_idx = extract_cnn_features(
    resnet, images_by_index, eval_gallery['index'].values, resnet_transform)
qry_resnet_feats, qry_rn_idx = extract_cnn_features(
    resnet, images_by_index, eval_query['index'].values, resnet_transform)
print(f"  ResNet50 features: gallery={gal_resnet_feats.shape}  query={qry_resnet_feats.shape}")
sys.stdout.flush()

# Augment: ResNet50(2048D) + normalized_color(44D * 0.3) → augmented
COLOR_WEIGHT = 0.3
n_gal = min(len(gal_resnet_feats), len(gal_color_feat))
n_qry = min(len(qry_resnet_feats), len(qry_color_feat))

gal_aug = np.array([
    augment_embedding_with_color(gal_resnet_feats[i], gal_color_feat[i], COLOR_WEIGHT)
    for i in range(n_gal)
])
qry_aug = np.array([
    augment_embedding_with_color(qry_resnet_feats[i], qry_color_feat[i], COLOR_WEIGHT)
    for i in range(n_qry)
])
print(f"  Augmented dim: {gal_aug.shape[1]}D (ResNet50 1D + color 0.3D)")
sys.stdout.flush()

gallery_pids_aug = gallery_pids[:n_gal]
query_pids_aug   = query_pids[:n_qry]
query_cats_aug   = query_cats[:n_qry]

aug_dist, aug_idx = faiss_search(gal_aug, qry_aug)
results_aug = eval_recall(query_pids_aug, gallery_pids_aug, aug_idx, label="ResNet50 + Color (augmented)")
results_aug_per_cat = per_category_recall(query_pids_aug, query_cats_aug, gallery_pids_aug, aug_idx)


# ===== 7. EXPERIMENT 1.M.4: EfficientNet-B0 + Color =====
print("\n" + "=" * 65)
print("EXPERIMENT 1.M.4 — EfficientNet-B0 + Color (augmented)")
print("=" * 65)
sys.stdout.flush()

eff_model2 = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
eff_model2.classifier = torch.nn.Identity()
eff_model2 = eff_model2.to(device)
eff_model2.eval()

eff2_gal_feats, _ = extract_cnn_features(
    eff_model2, images_by_index, eval_gallery['index'].values, eff_transform)
eff2_qry_feats, _ = extract_cnn_features(
    eff_model2, images_by_index, eval_query['index'].values, eff_transform)

n_gal2 = min(len(eff2_gal_feats), len(gal_color_feat))
n_qry2 = min(len(eff2_qry_feats), len(qry_color_feat))

eff2_gal_aug = np.array([
    augment_embedding_with_color(eff2_gal_feats[i], gal_color_feat[i], COLOR_WEIGHT)
    for i in range(n_gal2)
])
eff2_qry_aug = np.array([
    augment_embedding_with_color(eff2_qry_feats[i], qry_color_feat[i], COLOR_WEIGHT)
    for i in range(n_qry2)
])
print(f"  Augmented dim: {eff2_gal_aug.shape[1]}D")
sys.stdout.flush()

gallery_pids_eff2 = gallery_pids[:n_gal2]
query_pids_eff2   = query_pids[:n_qry2]
query_cats_eff2   = query_cats[:n_qry2]

eff2_dist, eff2_idx = faiss_search(eff2_gal_aug, eff2_qry_aug)
results_eff2 = eval_recall(query_pids_eff2, gallery_pids_eff2, eff2_idx, label="EfficientNet-B0 + Color (augmented)")
results_eff2_per_cat = per_category_recall(query_pids_eff2, query_cats_eff2, gallery_pids_eff2, eff2_idx)

del eff_model2
gc.collect()


# ===== 8. EXPERIMENT 1.M.5: Color Re-ranking of ResNet50 =====
print("\n" + "=" * 65)
print("EXPERIMENT 1.M.5 — Color Re-ranking of ResNet50 top-20")
print("=" * 65)
print("  (test alpha=0.7 and alpha=0.5 blend weights)")
sys.stdout.flush()

# Get raw ResNet50 search first (using augmented but checking re-rank approach)
rn_dist_raw, rn_idx_raw = faiss_search(gal_resnet_feats[:n_gal], qry_resnet_feats[:n_qry])

print("\n  Baseline ResNet50-only:")
results_resnet = eval_recall(query_pids_aug, gallery_pids_aug, rn_idx_raw, label="ResNet50 baseline (this session)")

# Re-rank at alpha=0.7 (70% CNN, 30% color)
rr_idx_07 = color_rerank(
    qry_color_feat[0],  # placeholder — we loop per query
    gal_color_feat[:n_gal],
    rn_idx_raw, rn_dist_raw,
    top_k_rerank=20, alpha=0.7,
)
# Proper vectorized re-rank
print("\n  Re-ranking with alpha=0.7 (70% CNN + 30% color):")
reranked_07 = np.zeros_like(rn_idx_raw)
reranked_05 = np.zeros_like(rn_idx_raw)
gal_colors_for_rr = gal_color_feat[:n_gal]
gal_colors_normed = gal_colors_for_rr / (np.linalg.norm(gal_colors_for_rr, axis=1, keepdims=True) + 1e-8)
for i in range(len(rn_idx_raw)):
    cand_idx = rn_idx_raw[i]
    cnn_scores = rn_dist_raw[i]
    qc = qry_color_feat[i] / (np.linalg.norm(qry_color_feat[i]) + 1e-8)
    color_scores = gal_colors_normed[cand_idx] @ qc

    blend_07 = 0.7 * cnn_scores + 0.3 * color_scores
    blend_05 = 0.5 * cnn_scores + 0.5 * color_scores
    reranked_07[i] = cand_idx[np.argsort(-blend_07)]
    reranked_05[i] = cand_idx[np.argsort(-blend_05)]

results_rr07 = eval_recall(query_pids_aug, gallery_pids_aug, reranked_07, label="ResNet50 + re-rank alpha=0.7")
results_rr07_per_cat = per_category_recall(query_pids_aug, query_cats_aug, gallery_pids_aug, reranked_07)

print("\n  Re-ranking with alpha=0.5 (50% CNN + 50% color):")
results_rr05 = eval_recall(query_pids_aug, gallery_pids_aug, reranked_05, label="ResNet50 + re-rank alpha=0.5")
results_rr05_per_cat = per_category_recall(query_pids_aug, query_cats_aug, gallery_pids_aug, reranked_05)


# ===== 9. SUMMARY TABLE =====
print("\n" + "=" * 65)
print("SUMMARY TABLE — ALL EXPERIMENTS")
print("=" * 65)
sys.stdout.flush()

# Anthony's ResNet50 result for reference (from metrics.json)
anthony_resnet = {"recall@1": 0.3067, "recall@5": 0.4927, "recall@10": 0.5901, "recall@20": 0.6913}

experiments = [
    ("Anthony: ResNet50 (baseline)",    anthony_resnet),
    ("1.M.1: EfficientNet-B0",          results_eff),
    ("1.M.2: Color-only (44D)",         results_color),
    ("1.M.5: ResNet50 baseline (ours)", results_resnet),
    ("1.M.3: ResNet50 + Color",         results_aug),
    ("1.M.4: EfficientNet-B0 + Color",  results_eff2),
    ("1.M.5a: ResNet50 + re-rank alpha=0.7",results_rr07),
    ("1.M.5b: ResNet50 + re-rank alpha=0.5",results_rr05),
]

print(f"{'Experiment':<42} {'R@1':>7} {'R@5':>7} {'R@10':>8} {'R@20':>8}")
print("-" * 76)
for name, r in experiments:
    print(f"  {name:<40} {r['recall@1']:>7.4f} {r['recall@5']:>7.4f} {r['recall@10']:>8.4f} {r['recall@20']:>8.4f}")
sys.stdout.flush()

# Per-category jackets comparison
print("\n--- JACKETS R@1 (hardest category per Anthony: 13.9%) ---")
for name, per_cat in [
    ("Anthony: ResNet50", {"jackets": {"recall@1": 0.1392}}),
    ("EfficientNet-B0", results_eff_per_cat),
    ("ResNet50 + Color", results_aug_per_cat),
    ("EfficientNet-B0 + Color", results_eff2_per_cat),
    ("ResNet50 + re-rank alpha=0.7", results_rr07_per_cat),
]:
    jk = per_cat.get("jackets", {})
    r1 = jk.get("recall@1", float("nan"))
    n  = jk.get("n_queries", "?")
    print(f"  {name:<38} jackets R@1={r1:.4f}  (n={n})")
sys.stdout.flush()


# ===== 10. PLOTS =====
print("\n--- Generating plots ---")
sys.stdout.flush()

# 10a. Summary bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
labels = ["Anthony\nResNet50", "Eff-B0", "Color\nonly", "RN50+\nColor", "Eff-B0+\nColor", "RN50+\nRerank\nalpha=0.7"]
r1_vals  = [anthony_resnet["recall@1"],  results_eff["recall@1"],  results_color["recall@1"],
            results_aug["recall@1"],      results_eff2["recall@1"], results_rr07["recall@1"]]
r10_vals = [anthony_resnet["recall@10"], results_eff["recall@10"], results_color["recall@10"],
            results_aug["recall@10"],     results_eff2["recall@10"], results_rr07["recall@10"]]

colors_bar = ['#95a5a6', '#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
x = np.arange(len(labels))
w = 0.35
axes[0].bar(x - w/2, r1_vals,  w, label='R@1',  color=colors_bar, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[0].bar(x + w/2, r10_vals, w, label='R@10', color=colors_bar, alpha=0.5,  edgecolor='black', linewidth=0.5, hatch='//')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel('Recall')
axes[0].set_title('R@1 and R@10 by Approach', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].set_ylim(0, 0.85)
for xi, r1, r10 in zip(x, r1_vals, r10_vals):
    axes[0].text(xi - w/2, r1 + 0.01, f'{r1:.3f}', ha='center', fontsize=7, fontweight='bold')
    axes[0].text(xi + w/2, r10 + 0.01, f'{r10:.3f}', ha='center', fontsize=7)

# 10b. Per-category heatmap for top 3 experiments
cats_order = ['shirts', 'sweaters', 'denim', 'tees', 'pants', 'shorts', 'sweatshirts', 'jackets']
exp_cats = {
    "ResNet50\n(Anthony)": {"shirts": 0.3884, "sweaters": 0.3784, "denim": 0.3506, "tees": 0.3525,
                             "pants": 0.3056, "shorts": 0.2532, "sweatshirts": 0.2441, "jackets": 0.1392},
    "Eff-B0":        {k: results_eff_per_cat.get(k, {}).get("recall@1", np.nan) for k in cats_order},
    "RN50+Color":    {k: results_aug_per_cat.get(k, {}).get("recall@1", np.nan) for k in cats_order},
    "Eff-B0+Color":  {k: results_eff2_per_cat.get(k, {}).get("recall@1", np.nan) for k in cats_order},
    "RN50+Rerank":   {k: results_rr07_per_cat.get(k, {}).get("recall@1", np.nan) for k in cats_order},
}
heat_data = pd.DataFrame(exp_cats, index=cats_order)
sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
            vmin=0.1, vmax=0.5, linewidths=0.5)
axes[1].set_title('Per-Category R@1 (sorted easiest→hardest)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=25)

plt.suptitle('Phase 1 Mark: Color Features vs CNN-only Baselines', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(RESULTS / 'phase1_mark_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/phase1_mark_comparison.png")

# 10c. Similarity separation comparison
# Compute separation for best model (ResNet50 + Color)
correct_sims_aug, incorrect_sims_aug = [], []
K = 20
for i in range(len(query_pids_aug)):
    for j in range(K):
        gal_j = aug_idx[i, j]
        if gal_j < len(gallery_pids_aug):
            sim = float(aug_dist[i, j]) if j < aug_dist.shape[1] else 0.0
            if query_pids_aug[i] == gallery_pids_aug[gal_j]:
                correct_sims_aug.append(sim)
            else:
                incorrect_sims_aug.append(sim)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(correct_sims_aug,   bins=40, alpha=0.7, color='green', density=True,
             label=f'Correct (n={len(correct_sims_aug)})')
axes[0].hist(incorrect_sims_aug, bins=40, alpha=0.7, color='red',   density=True,
             label=f'Incorrect (n={len(incorrect_sims_aug)})')
axes[0].set_xlabel('Cosine Similarity')
axes[0].set_ylabel('Density')
sep_aug = np.mean(correct_sims_aug) - np.mean(incorrect_sims_aug)
axes[0].set_title(f'ResNet50+Color Similarity Dist.\nSeparation={sep_aug:.4f}', fontsize=12, fontweight='bold')
axes[0].legend()

# Jackets: per-category improvement bar
jk_r1s = [0.1392,
           results_eff_per_cat.get('jackets', {}).get('recall@1', np.nan),
           results_aug_per_cat.get('jackets', {}).get('recall@1', np.nan),
           results_eff2_per_cat.get('jackets', {}).get('recall@1', np.nan),
           results_rr07_per_cat.get('jackets', {}).get('recall@1', np.nan)]
jk_labels = ['ResNet50\n(Anthony)', 'Eff-B0', 'RN50\n+Color', 'Eff-B0\n+Color', 'RN50+\nRerank alpha=0.7']
jk_colors = ['#95a5a6', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
axes[1].bar(jk_labels, jk_r1s, color=jk_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Recall@1 (Jackets)')
axes[1].set_title('Jackets R@1 — Hardest Category\nCan Color Features Close the Gap?', fontsize=12, fontweight='bold')
axes[1].axhline(0.3067, color='gray', linestyle='--', label='Overall R@1 baseline')
axes[1].legend(fontsize=9)
for xi, val in enumerate(jk_r1s):
    if not np.isnan(val):
        axes[1].text(xi, val + 0.005, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS / 'phase1_mark_jackets_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/phase1_mark_jackets_analysis.png")
sys.stdout.flush()


# ===== 11. SAVE METRICS =====
print("\n--- Saving metrics ---")
sys.stdout.flush()

metrics_path = RESULTS / 'metrics.json'
with open(metrics_path, 'r') as f:
    all_metrics = json.load(f)

all_metrics['phase1_mark'] = {
    "description": "Mark Phase 1 complementary: color palette + EfficientNet-B0",
    "eval_products": EVAL_PRODUCTS,
    "color_feature_dim_rgb_hsv": int(gal_color_feat.shape[1]),
    "color_weight": COLOR_WEIGHT,
    "experiments": {
        "efficientnet_b0":          results_eff,
        "color_only_44d":           results_color,
        "resnet50_color_augmented": results_aug,
        "efficientnet_b0_color":    results_eff2,
        "resnet50_rerank_alpha07":  results_rr07,
        "resnet50_rerank_alpha05":  results_rr05,
    },
    "per_category": {
        "efficientnet_b0":          results_eff_per_cat,
        "resnet50_color_augmented": results_aug_per_cat,
        "efficientnet_b0_color":    results_eff2_per_cat,
        "resnet50_rerank_alpha07":  results_rr07_per_cat,
    },
    "similarity_separation_aug": {
        "correct_mean":   float(np.mean(correct_sims_aug)) if correct_sims_aug else None,
        "incorrect_mean": float(np.mean(incorrect_sims_aug)) if incorrect_sims_aug else None,
        "separation":     float(sep_aug) if correct_sims_aug else None,
    }
}

with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"  Saved: {metrics_path}")
sys.stdout.flush()


# ===== 12. FINAL SUMMARY =====
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

best_r1 = max([results_eff["recall@1"], results_aug["recall@1"],
               results_eff2["recall@1"], results_rr07["recall@1"]])
best_label = [name for name, r in experiments if r["recall@1"] == best_r1][0]

print(f"\n  Best overall R@1: {best_r1:.4f} ({best_label})")
print(f"  Anthony's ResNet50 R@1: {anthony_resnet['recall@1']:.4f}")
delta = best_r1 - anthony_resnet["recall@1"]
print(f"  Delta vs Anthony baseline: {delta:+.4f}")

anthony_jacket = 0.1392
best_jacket_r1 = max(
    results_eff_per_cat.get("jackets", {}).get("recall@1", 0),
    results_aug_per_cat.get("jackets", {}).get("recall@1", 0),
    results_eff2_per_cat.get("jackets", {}).get("recall@1", 0),
    results_rr07_per_cat.get("jackets", {}).get("recall@1", 0),
)
print(f"\n  Anthony jacket R@1:  {anthony_jacket:.4f}")
print(f"  Best Mark jacket R@1: {best_jacket_r1:.4f} (Δ={best_jacket_r1 - anthony_jacket:+.4f})")

print(f"\n  Similarity separation (ResNet50+Color): {sep_aug:.4f}")
print(f"  Anthony separation (ResNet50 only):     0.0485")
print(f"  Separation delta: {sep_aug - 0.0485:+.4f}")

print("\n  Files saved:")
for fp in sorted(RESULTS.glob('phase1_mark*')):
    print(f"    {fp.name}")

print("\nPHASE 1 MARK COMPLETE")
sys.stdout.flush()
os._exit(0)
