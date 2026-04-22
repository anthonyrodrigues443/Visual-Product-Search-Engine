#!/usr/bin/env python3
"""Phase 3: Feature Engineering Deep Dive — Visual Product Search Engine.

Research question: Which domain-specific features complement CLIP ViT-L/14
for fashion retrieval, and is the bottleneck the backbone or the features?

Phase 2 champion: CLIP ViT-L/14 + 48D color histogram reranking → R@1=64.2%

Experiments:
  3.1  Spatial color grid (4×4 = 192D) — captures WHERE colors appear
  3.2  LBP texture features (59D) — captures fabric texture patterns
  3.3  HOG edge features (324D) — captures garment silhouette/shape
  3.4  CLIP text-to-image retrieval — cross-modal search via descriptions
  3.5  Multi-feature fusion ablation — which combination wins?

Building on:
  - Phase 1: ResNet50 baseline R@1=30.7%
  - Phase 1 Mark: color rerank R@1=40.5%
  - Phase 2: CLIP L/14 R@1=55.3%, +color=64.2%

References:
  - [1] Color Layout Descriptor (CLD) — MPEG-7 standard for spatial color
  - [2] Ojala et al. 2002 — LBP for texture classification
  - [3] Dalal & Triggs 2005 — HOG for shape representation
  - [4] Radford et al. 2021 — CLIP cross-modal alignment

Author: Anthony Rodrigues | Date: 2026-04-22
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
import torch
import faiss

plt.style.use('seaborn-v0_8-whitegrid')
PROJECT = Path(__file__).parent.parent
PROC = PROJECT / 'data' / 'processed'
RES = PROJECT / 'results'
RES.mkdir(exist_ok=True)

EVAL_N = 300
K = 20
BS = 32
DEV = 'cpu'

# ======================================================================
# 1. LOAD DATA + CACHE IMAGES
# ======================================================================
print("=" * 70)
print("PHASE 3: FEATURE ENGINEERING DEEP DIVE")
print("Which domain features complement CLIP ViT-L/14 for fashion retrieval?")
print("=" * 70)

gallery_df = pd.read_csv(PROC / 'gallery.csv')
query_df = pd.read_csv(PROC / 'query.csv')
eval_pids = gallery_df['product_id'].values[:EVAL_N]
g_df = gallery_df[gallery_df['product_id'].isin(eval_pids)].reset_index(drop=True)
q_df = query_df[query_df['product_id'].isin(eval_pids)].reset_index(drop=True)
print(f"Eval set: {len(g_df)} gallery, {len(q_df)} query images ({EVAL_N} products)")

all_idx = sorted(set(
    g_df['index'].astype(int).tolist() + q_df['index'].astype(int).tolist()
))
print(f"Streaming {len(all_idx)} images from HuggingFace...")
sys.stdout.flush()

from datasets import load_dataset
ds = load_dataset('Marqo/deepfashion-inshop', split='data', streaming=True)
needed = set(all_idx)
imgs = {}
for i, ex in enumerate(tqdm(ds, total=max(needed) + 1)):
    if i in needed:
        img = ex['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        imgs[i] = img
        needed.discard(i)
        if not needed:
            break
print(f"Cached {len(imgs)} images in memory\n")
sys.stdout.flush()

g_pids = g_df['product_id'].values
q_pids = q_df['product_id'].values
g_indices = g_df['index'].astype(int).values
q_indices = q_df['index'].astype(int).values

# ======================================================================
# 2. EVALUATION HELPERS
# ======================================================================
def recall_at_k(q_pids, g_pids, indices, ks=(1, 5, 10, 20)):
    results = {}
    for k in ks:
        correct = sum(
            1 for qi, qp in enumerate(q_pids)
            if qp in g_pids[indices[qi, :k]]
        )
        results[f'R@{k}'] = round(correct / len(q_pids), 4)
    return results


def per_cat_recall(q_df, g_pids, indices, k=1):
    cats = {}
    for cat in sorted(q_df['category2'].unique()):
        mask = (q_df['category2'] == cat).values
        qp = q_df.loc[mask, 'product_id'].values
        qi = np.where(mask)[0]
        correct = sum(1 for i, p in zip(qi, qp) if p in g_pids[indices[i, :k]])
        cats[cat] = round(correct / len(qp), 4) if len(qp) > 0 else 0
    return cats


def faiss_search(gf, qf, k=K):
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(gn.shape[1])
    index.add(np.ascontiguousarray(gn, dtype=np.float32))
    D, I = index.search(np.ascontiguousarray(qn, dtype=np.float32), k)
    return D, I


def rerank_with_features(
    cnn_D, cnn_I, q_feats, g_feats, alpha=0.5, top_k=20
):
    """Rerank CNN top-K using supplementary feature similarity."""
    q_norm = q_feats / (np.linalg.norm(q_feats, axis=1, keepdims=True) + 1e-8)
    g_norm = g_feats / (np.linalg.norm(g_feats, axis=1, keepdims=True) + 1e-8)
    reranked = np.zeros_like(cnn_I)
    for qi in range(len(cnn_I)):
        cand = cnn_I[qi]
        cnn_scores = cnn_D[qi]
        feat_scores = g_norm[cand] @ q_norm[qi]
        blended = alpha * cnn_scores + (1 - alpha) * feat_scores
        reranked[qi] = cand[np.argsort(-blended)]
    return reranked


def concat_features(*feat_arrays, weights=None):
    """Concatenate multiple L2-normalized feature arrays with optional weights."""
    if weights is None:
        weights = [1.0] * len(feat_arrays)
    parts = []
    for f, w in zip(feat_arrays, weights):
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        parts.append(fn * w)
    return np.concatenate(parts, axis=1).astype(np.float32)


# ======================================================================
# 3. EXTRACT CLIP ViT-L/14 EMBEDDINGS (backbone)
# ======================================================================
print("=" * 70)
print("3.0  Extracting CLIP ViT-L/14 backbone embeddings")
print("=" * 70)

import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    'ViT-L-14', pretrained='openai'
)
clip_model.eval()
tokenizer = open_clip.get_tokenizer('ViT-L-14')

def extract_clip_visual(indices_list):
    feats = []
    batch = []
    for ix in tqdm(indices_list, desc='  CLIP visual', leave=False):
        ix = int(ix)
        if ix not in imgs:
            continue
        batch.append(clip_preprocess(imgs[ix]))
        if len(batch) >= BS:
            t = torch.stack(batch)
            with torch.no_grad():
                f = clip_model.encode_image(t)
            feats.append(f.cpu().float().numpy())
            batch = []
    if batch:
        t = torch.stack(batch)
        with torch.no_grad():
            f = clip_model.encode_image(t)
        feats.append(f.cpu().float().numpy())
    return np.vstack(feats)

t0 = time.time()
g_clip = extract_clip_visual(g_indices)
q_clip = extract_clip_visual(q_indices)
clip_time = time.time() - t0
print(f"  CLIP embeddings: gallery={g_clip.shape}, query={q_clip.shape} ({clip_time:.1f}s)")

clip_D, clip_I = faiss_search(g_clip, q_clip)
clip_base = recall_at_k(q_pids, g_pids, clip_I)
print(f"  CLIP L/14 baseline: {clip_base}")

# ======================================================================
# 3.1  SPATIAL COLOR GRID (4×4 = 192D)
# ======================================================================
print("\n" + "=" * 70)
print("3.1  Spatial Color Grid (4×4 regions × 12D HSV histogram = 192D)")
print("     Captures WHERE colors appear, not just WHICH colors")
print("=" * 70)

from src.feature_engineering import _rgb_to_hsv_vectorized

def extract_spatial_color_grid(img, grid_rows=4, grid_cols=4, bins=4):
    """Divide image into grid regions, compute HSV histogram per region.

    Returns (grid_rows * grid_cols * bins * 3,) feature vector.
    For 4×4 grid with 4 bins: 4*4*12 = 192D.

    This captures spatial color layout: a dress with dark top and light bottom
    differs from one with uniform color, even if global histograms match.
    """
    img_small = img.resize((128, 128), Image.LANCZOS).convert('RGB')
    pixels = np.array(img_small).astype(np.float32) / 255.0
    h, w = pixels.shape[:2]
    rh = h // grid_rows
    rw = w // grid_cols

    feats = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            region = pixels[r*rh:(r+1)*rh, c*rw:(c+1)*rw].reshape(-1, 3)
            hsv = _rgb_to_hsv_vectorized(region)
            hh, _ = np.histogram(hsv[:, 0], bins=bins, range=(0, 1))
            sh, _ = np.histogram(hsv[:, 1], bins=bins, range=(0, 1))
            vh, _ = np.histogram(hsv[:, 2], bins=bins, range=(0, 1))
            region_feat = np.concatenate([hh, sh, vh]).astype(np.float32)
            region_feat = region_feat / (region_feat.sum() + 1e-8)
            feats.append(region_feat)

    return np.concatenate(feats)


def batch_extract(indices_list, extract_fn):
    feats = []
    for ix in tqdm(indices_list, desc='  Features', leave=False):
        ix = int(ix)
        if ix not in imgs:
            continue
        feats.append(extract_fn(imgs[ix]))
    return np.array(feats, dtype=np.float32)


t0 = time.time()
g_spatial = batch_extract(g_indices, extract_spatial_color_grid)
q_spatial = batch_extract(q_indices, extract_spatial_color_grid)
spatial_time = time.time() - t0
print(f"  Spatial color grid: {g_spatial.shape[1]}D, extracted in {spatial_time:.1f}s")

# Test spatial color alone
spatial_D, spatial_I = faiss_search(g_spatial, q_spatial)
spatial_only = recall_at_k(q_pids, g_pids, spatial_I)
print(f"  Spatial color ONLY: {spatial_only}")

# Rerank CLIP with spatial color
for alpha in [0.7, 0.5, 0.3]:
    reranked = rerank_with_features(clip_D, clip_I, q_spatial, g_spatial, alpha=alpha)
    result = recall_at_k(q_pids, g_pids, reranked)
    print(f"  CLIP + spatial rerank (α={alpha}): {result}")

# Best alpha concat approach
clip_spatial_feats_g = concat_features(g_clip, g_spatial, weights=[1.0, 0.5])
clip_spatial_feats_q = concat_features(q_clip, q_spatial, weights=[1.0, 0.5])
cs_D, cs_I = faiss_search(clip_spatial_feats_g, clip_spatial_feats_q)
clip_spatial_concat = recall_at_k(q_pids, g_pids, cs_I)
print(f"  CLIP + spatial CONCAT (w=0.5): {clip_spatial_concat}")

# ======================================================================
# 3.2  LBP TEXTURE FEATURES (uniform LBP, 59D)
# ======================================================================
print("\n" + "=" * 70)
print("3.2  LBP Texture Features (uniform LBP, 59D)")
print("     Captures fabric texture: denim ridges vs silk smoothness")
print("=" * 70)

def extract_lbp(img, img_size=64):
    """Fast LBP using 8-neighbor shift comparisons (no Python loops over pixels).

    Computes basic LBP (R=1, P=8) entirely with NumPy array shifts.
    Then histograms into 256 bins and reduces to 16 bins via bit-pair grouping
    for compactness: 16D feature per scale, 2 scales → 32D.
    ~2ms/image vs 40s for the naive per-pixel version.
    """
    gray = np.array(img.resize((img_size, img_size), Image.LANCZOS).convert('L'), dtype=np.float32)

    all_hists = []
    for R in [1, 2]:
        h, w = gray.shape
        pad = R
        center = gray[pad:-pad, pad:-pad]

        offsets = [(-R, 0), (-R, R), (0, R), (R, R), (R, 0), (R, -R), (0, -R), (-R, -R)]
        lbp = np.zeros_like(center, dtype=np.uint8)
        for bit_idx, (dy, dx) in enumerate(offsets):
            ny = pad + dy
            nx = pad + dx
            neighbor = gray[ny:ny + center.shape[0], nx:nx + center.shape[1]]
            lbp += ((neighbor >= center).astype(np.uint8)) << bit_idx

        hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 256))
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        all_hists.append(hist)

    return np.concatenate(all_hists)


t0 = time.time()
g_lbp = batch_extract(g_indices, extract_lbp)
q_lbp = batch_extract(q_indices, extract_lbp)
lbp_time = time.time() - t0
print(f"  LBP features: {g_lbp.shape[1]}D, extracted in {lbp_time:.1f}s")

# LBP alone
lbp_D, lbp_I = faiss_search(g_lbp, q_lbp)
lbp_only = recall_at_k(q_pids, g_pids, lbp_I)
print(f"  LBP ONLY: {lbp_only}")

# Rerank CLIP with LBP
for alpha in [0.8, 0.7, 0.5]:
    reranked = rerank_with_features(clip_D, clip_I, q_lbp, g_lbp, alpha=alpha)
    result = recall_at_k(q_pids, g_pids, reranked)
    print(f"  CLIP + LBP rerank (α={alpha}): {result}")

# ======================================================================
# 3.3  HOG EDGE/SHAPE FEATURES (324D)
# ======================================================================
print("\n" + "=" * 70)
print("3.3  HOG Edge Features (9 orientations × 6×6 cells = 324D)")
print("     Captures garment silhouette and shape contours")
print("=" * 70)

def extract_hog(img, cell_size=16, n_bins=9, img_size=64):
    """Fast HOG using vectorized bin assignment (no Python loops over pixels).

    64×64 with 16×16 cells → 4×4 grid × 9 orientations = 144D.
    Uses np.add.at for weighted histogram accumulation — ~1ms/image.
    """
    gray = np.array(img.resize((img_size, img_size), Image.LANCZOS).convert('L'), dtype=np.float32)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[:-2, :] - gray[2:, :]

    magnitude = np.sqrt(gx**2 + gy**2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi
    orientation[orientation < 0] += 180

    h, w = gray.shape
    n_cells_y = h // cell_size
    n_cells_x = w // cell_size
    bin_width = 180.0 / n_bins

    bins = np.clip((orientation / bin_width).astype(np.int32), 0, n_bins - 1)

    hog_feats = np.zeros(n_cells_y * n_cells_x * n_bins, dtype=np.float32)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y0, y1 = cy * cell_size, (cy + 1) * cell_size
            x0, x1 = cx * cell_size, (cx + 1) * cell_size
            cell_bins = bins[y0:y1, x0:x1].ravel()
            cell_mag = magnitude[y0:y1, x0:x1].ravel()
            hist = np.zeros(n_bins, dtype=np.float32)
            np.add.at(hist, cell_bins, cell_mag)
            offset = (cy * n_cells_x + cx) * n_bins
            hog_feats[offset:offset + n_bins] = hist

    hog_feats = hog_feats / (np.linalg.norm(hog_feats) + 1e-8)
    return hog_feats


t0 = time.time()
g_hog = batch_extract(g_indices, extract_hog)
q_hog = batch_extract(q_indices, extract_hog)
hog_time = time.time() - t0
print(f"  HOG features: {g_hog.shape[1]}D, extracted in {hog_time:.1f}s")

# HOG alone
hog_D, hog_I = faiss_search(g_hog, q_hog)
hog_only = recall_at_k(q_pids, g_pids, hog_I)
print(f"  HOG ONLY: {hog_only}")

# Rerank CLIP with HOG
for alpha in [0.9, 0.8, 0.7]:
    reranked = rerank_with_features(clip_D, clip_I, q_hog, g_hog, alpha=alpha)
    result = recall_at_k(q_pids, g_pids, reranked)
    print(f"  CLIP + HOG rerank (α={alpha}): {result}")

# ======================================================================
# 3.4  CLIP TEXT-TO-IMAGE CROSS-MODAL RETRIEVAL
# ======================================================================
print("\n" + "=" * 70)
print("3.4  CLIP Text-to-Image Retrieval")
print("     Use product descriptions to find visually matching products")
print("=" * 70)

# Extract text features from descriptions
def extract_clip_text(texts, batch_size=64):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts)
        with torch.no_grad():
            text_feats = clip_model.encode_text(tokens)
        feats.append(text_feats.cpu().float().numpy())
    return np.vstack(feats)

# Get descriptions for query and gallery
g_descs = g_df['description'].fillna('').values
q_descs = q_df['description'].fillna('').values

# Build category+color text prompts as a simpler text representation
def build_text_prompt(row):
    parts = []
    if pd.notna(row.get('color')) and row['color']:
        parts.append(str(row['color']))
    if pd.notna(row.get('category2')) and row['category2']:
        parts.append(str(row['category2']))
    if pd.notna(row.get('category3')) and row['category3']:
        parts.append(str(row['category3']))
    return f"a photo of {' '.join(parts)}" if parts else "a photo of clothing"

g_prompts = [build_text_prompt(row) for _, row in g_df.iterrows()]
q_prompts = [build_text_prompt(row) for _, row in q_df.iterrows()]

print(f"  Example prompts: {g_prompts[:3]}")

# Text embeddings from structured prompts
t0 = time.time()
g_text = extract_clip_text(g_prompts)
q_text = extract_clip_text(q_prompts)
text_time = time.time() - t0
print(f"  Text embeddings: {g_text.shape[1]}D, extracted in {text_time:.1f}s")

# Text-only retrieval (query text → gallery text)
text_D, text_I = faiss_search(g_text, q_text)
text_only = recall_at_k(q_pids, g_pids, text_I)
print(f"  Text-only (prompt→prompt): {text_only}")

# Cross-modal: query IMAGE → gallery TEXT
cross_D, cross_I = faiss_search(g_text, q_clip)
cross_result = recall_at_k(q_pids, g_pids, cross_I)
print(f"  Cross-modal (img→text): {cross_result}")

# Hybrid: visual + text embedding concat
for tw in [0.1, 0.2, 0.3]:
    hybrid_g = concat_features(g_clip, g_text, weights=[1.0, tw])
    hybrid_q = concat_features(q_clip, q_text, weights=[1.0, tw])
    h_D, h_I = faiss_search(hybrid_g, hybrid_q)
    hybrid_result = recall_at_k(q_pids, g_pids, h_I)
    print(f"  CLIP visual+text concat (w={tw}): {hybrid_result}")

# Full description text retrieval
has_desc = [i for i, d in enumerate(q_descs) if len(d) > 10]
if len(has_desc) > 50:
    g_desc_text = extract_clip_text([d if len(d) > 10 else "clothing" for d in g_descs])
    q_desc_text = extract_clip_text([d if len(d) > 10 else "clothing" for d in q_descs])

    # Full description cross-modal
    desc_D, desc_I = faiss_search(g_desc_text, q_desc_text)
    desc_result = recall_at_k(q_pids, g_pids, desc_I)
    print(f"  Full description text matching: {desc_result}")

    for tw in [0.1, 0.2]:
        hybrid_g = concat_features(g_clip, g_desc_text, weights=[1.0, tw])
        hybrid_q = concat_features(q_clip, q_desc_text, weights=[1.0, tw])
        h_D, h_I = faiss_search(hybrid_g, hybrid_q)
        hybrid_result = recall_at_k(q_pids, g_pids, h_I)
        print(f"  CLIP visual + full desc (w={tw}): {hybrid_result}")
else:
    print(f"  Only {len(has_desc)} queries have descriptions > 10 chars, skipping full desc")

# ======================================================================
# 3.5  MULTI-FEATURE FUSION ABLATION
# ======================================================================
print("\n" + "=" * 70)
print("3.5  Multi-Feature Fusion Ablation")
print("     Which combination of features maximizes R@1?")
print("=" * 70)

# Global color histogram (from Phase 1/2)
from src.feature_engineering import extract_color_palette, extract_hsv_histogram

g_rgb = batch_extract(g_indices, extract_color_palette)
q_rgb = batch_extract(q_indices, extract_color_palette)
g_hsv = batch_extract(g_indices, extract_hsv_histogram)
q_hsv = batch_extract(q_indices, extract_hsv_histogram)
g_color48 = np.concatenate([g_rgb, g_hsv], axis=1)
q_color48 = np.concatenate([q_rgb, q_hsv], axis=1)
print(f"  Color features: RGB {g_rgb.shape[1]}D + HSV {g_hsv.shape[1]}D = {g_color48.shape[1]}D")

# Phase 2 champion reproduction: CLIP + color rerank
reranked_color = rerank_with_features(clip_D, clip_I, q_color48, g_color48, alpha=0.5)
phase2_repro = recall_at_k(q_pids, g_pids, reranked_color)
print(f"  Phase 2 champion repro (CLIP+color rerank α=0.5): {phase2_repro}")

# Feature catalog
feature_catalog = {
    'clip':    (g_clip, q_clip, 1.0),
    'color48': (g_color48, q_color48, 0.5),
    'spatial': (g_spatial, q_spatial, 0.4),
    'lbp':     (g_lbp, q_lbp, 0.2),
    'hog':     (g_hog, q_hog, 0.2),
    'text':    (g_text, q_text, 0.15),
}

ablation_results = {}

# Test each feature added to CLIP
print("\n  --- Single feature added to CLIP ---")
for name, (gf, qf, w) in feature_catalog.items():
    if name == 'clip':
        continue
    combined_g = concat_features(g_clip, gf, weights=[1.0, w])
    combined_q = concat_features(q_clip, qf, weights=[1.0, w])
    D, I = faiss_search(combined_g, combined_q)
    result = recall_at_k(q_pids, g_pids, I)
    ablation_results[f'CLIP+{name}'] = result
    delta = result['R@1'] - clip_base['R@1']
    print(f"  CLIP + {name:8s} (w={w}): R@1={result['R@1']:.4f}  Δ={delta:+.4f}")

# Best combination: CLIP + spatial + color
print("\n  --- Multi-feature combinations ---")
combos = [
    ('CLIP+color+spatial', ['clip', 'color48', 'spatial']),
    ('CLIP+color+spatial+lbp', ['clip', 'color48', 'spatial', 'lbp']),
    ('CLIP+color+spatial+hog', ['clip', 'color48', 'spatial', 'hog']),
    ('CLIP+color+spatial+text', ['clip', 'color48', 'spatial', 'text']),
    ('CLIP+ALL', ['clip', 'color48', 'spatial', 'lbp', 'hog', 'text']),
    ('CLIP+color+spatial+lbp+hog', ['clip', 'color48', 'spatial', 'lbp', 'hog']),
]

for combo_name, feat_names in combos:
    feat_list = [feature_catalog[n][0] for n in feat_names]
    weight_list = [feature_catalog[n][2] for n in feat_names]
    feat_list_q = [feature_catalog[n][1] for n in feat_names]

    combined_g = concat_features(*feat_list, weights=weight_list)
    combined_q = concat_features(*feat_list_q, weights=weight_list)
    D, I = faiss_search(combined_g, combined_q)
    result = recall_at_k(q_pids, g_pids, I)
    ablation_results[combo_name] = result
    delta = result['R@1'] - clip_base['R@1']
    print(f"  {combo_name:35s}: R@1={result['R@1']:.4f}  Δ={delta:+.4f}")

# ======================================================================
# 3.6  PER-CATEGORY ANALYSIS ON BEST COMBINATION
# ======================================================================
print("\n" + "=" * 70)
print("3.6  Per-Category Analysis — Where do features help most?")
print("=" * 70)

# Find best combination
best_name = max(ablation_results, key=lambda k: ablation_results[k]['R@1'])
print(f"  Best combination: {best_name} → R@1={ablation_results[best_name]['R@1']}")

# Rebuild best combination
best_feats = {
    'CLIP+color+spatial': ['clip', 'color48', 'spatial'],
    'CLIP+color+spatial+lbp': ['clip', 'color48', 'spatial', 'lbp'],
    'CLIP+color+spatial+hog': ['clip', 'color48', 'spatial', 'hog'],
    'CLIP+color+spatial+text': ['clip', 'color48', 'spatial', 'text'],
    'CLIP+ALL': ['clip', 'color48', 'spatial', 'lbp', 'hog', 'text'],
    'CLIP+color+spatial+lbp+hog': ['clip', 'color48', 'spatial', 'lbp', 'hog'],
}

if best_name in best_feats:
    feat_names = best_feats[best_name]
else:
    # Single feature case
    feat_names = ['clip'] + [best_name.replace('CLIP+', '')]

feat_list = [feature_catalog[n][0] for n in feat_names]
weight_list = [feature_catalog[n][2] for n in feat_names]
feat_list_q = [feature_catalog[n][1] for n in feat_names]

combined_g = concat_features(*feat_list, weights=weight_list)
combined_q = concat_features(*feat_list_q, weights=weight_list)
best_D, best_I = faiss_search(combined_g, combined_q)

# Per-category for baseline CLIP vs best
clip_cats = per_cat_recall(q_df, g_pids, clip_I, k=1)
best_cats = per_cat_recall(q_df, g_pids, best_I, k=1)
phase2_cats = per_cat_recall(q_df, g_pids, reranked_color, k=1)

print(f"\n  {'Category':<15} {'CLIP L/14':>10} {'Phase2 champ':>12} {best_name:>20} {'Δ vs CLIP':>10}")
print("  " + "-" * 75)
for cat in sorted(clip_cats.keys()):
    c_r1 = clip_cats[cat]
    p2_r1 = phase2_cats[cat]
    b_r1 = best_cats[cat]
    delta = b_r1 - c_r1
    print(f"  {cat:<15} {c_r1:>10.4f} {p2_r1:>12.4f} {b_r1:>20.4f} {delta:>+10.4f}")

# ======================================================================
# 4. PLOTS
# ======================================================================
print("\n" + "=" * 70)
print("4.  Generating plots")
print("=" * 70)

# Plot 1: Feature-only comparison (each domain feature alone)
fig, ax = plt.subplots(figsize=(10, 6))
alone_results = {
    'CLIP L/14 (768D)': clip_base,
    'Spatial Color (192D)': spatial_only,
    'LBP Texture (multi-R)': lbp_only,
    'HOG Shape (324D)': hog_only,
    'Text Prompt': text_only,
}
names = list(alone_results.keys())
r1_vals = [alone_results[n]['R@1'] for n in names]
colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0', '#F44336']
bars = ax.barh(names, r1_vals, color=colors, edgecolor='white', height=0.6)
for bar, val in zip(bars, r1_vals):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontweight='bold', fontsize=11)
ax.set_xlabel('Recall@1', fontsize=12)
ax.set_title('Feature-Only Retrieval (no fusion)', fontsize=14, fontweight='bold')
ax.set_xlim(0, max(r1_vals) * 1.15)
plt.tight_layout()
plt.savefig(RES / 'phase3_feature_standalone.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase3_feature_standalone.png")

# Plot 2: Ablation — each feature added to CLIP
fig, ax = plt.subplots(figsize=(12, 7))
ablation_names = ['CLIP L/14 (baseline)'] + list(ablation_results.keys())
ablation_r1 = [clip_base['R@1']] + [ablation_results[n]['R@1'] for n in ablation_results]

# Sort by R@1
sorted_pairs = sorted(zip(ablation_names, ablation_r1), key=lambda x: x[1], reverse=True)
ablation_names, ablation_r1 = zip(*sorted_pairs)

cmap = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(ablation_names)))
bars = ax.barh(range(len(ablation_names)), ablation_r1, color=cmap, edgecolor='white', height=0.7)
ax.set_yticks(range(len(ablation_names)))
ax.set_yticklabels(ablation_names, fontsize=10)
for bar, val in zip(bars, ablation_r1):
    delta = val - clip_base['R@1']
    label = f'{val:.4f} ({delta:+.4f})' if delta != 0 else f'{val:.4f} (baseline)'
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Recall@1', fontsize=12)
ax.set_title('Feature Engineering Ablation — What Helps CLIP ViT-L/14?', fontsize=14, fontweight='bold')
ax.axvline(x=clip_base['R@1'], color='red', linestyle='--', alpha=0.5, label='CLIP baseline')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(RES / 'phase3_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase3_ablation.png")

# Plot 3: Per-category comparison (CLIP vs Phase2 vs Best)
fig, ax = plt.subplots(figsize=(12, 6))
cats = sorted(clip_cats.keys())
x = np.arange(len(cats))
width = 0.25

bars1 = ax.bar(x - width, [clip_cats[c] for c in cats], width, label='CLIP L/14', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x, [phase2_cats[c] for c in cats], width, label='Phase 2 (CLIP+color)', color='#FF9800', alpha=0.8)
bars3 = ax.bar(x + width, [best_cats[c] for c in cats], width, label=f'Phase 3 ({best_name})', color='#4CAF50', alpha=0.8)

ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Recall@1', fontsize=12)
ax.set_title('Per-Category Improvement: Phase 1 → 2 → 3', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(RES / 'phase3_per_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase3_per_category.png")

# Plot 4: Feature dimensionality vs R@1 (efficiency plot)
fig, ax = plt.subplots(figsize=(10, 6))
dim_data = {
    'Color 48D': (48, spatial_only.get('R@1', 0)),  # Hmm, use color48 alone
    'LBP multi-R': (g_lbp.shape[1], lbp_only['R@1']),
    'Spatial 192D': (g_spatial.shape[1], spatial_only['R@1']),
    'HOG 324D': (g_hog.shape[1], hog_only['R@1']),
    'CLIP 768D': (768, clip_base['R@1']),
}
# Also get color48 alone
color_D, color_I = faiss_search(g_color48, q_color48)
color48_only = recall_at_k(q_pids, g_pids, color_I)
dim_data['Color 48D'] = (48, color48_only['R@1'])

for name, (d, r) in dim_data.items():
    color_map = {
        'Color 48D': '#FF9800', 'LBP multi-R': '#4CAF50',
        'Spatial 192D': '#FF5722', 'HOG 324D': '#9C27B0',
        'CLIP 768D': '#2196F3'
    }
    ax.scatter(d, r, s=150, c=color_map.get(name, '#666'), zorder=3)
    ax.annotate(name, (d, r), textcoords='offset points', xytext=(10, 5), fontsize=10)

ax.set_xlabel('Feature Dimensionality', fontsize=12)
ax.set_ylabel('Recall@1 (standalone)', fontsize=12)
ax.set_title('Feature Efficiency: Dimensions vs Retrieval Quality', fontsize=14, fontweight='bold')
ax.set_xscale('log')
plt.tight_layout()
plt.savefig(RES / 'phase3_dim_efficiency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase3_dim_efficiency.png")

# ======================================================================
# 5. SAVE ALL RESULTS
# ======================================================================
print("\n" + "=" * 70)
print("5.  Saving results")
print("=" * 70)

all_results = {
    'phase3_anthony': {
        'date': '2026-04-22',
        'eval_products': EVAL_N,
        'eval_gallery': len(g_df),
        'eval_queries': len(q_df),
        'research_question': 'Which domain features complement CLIP ViT-L/14? Is the bottleneck the backbone or the features?',
        'clip_baseline': clip_base,
        'standalone_features': {
            'spatial_color_192D': spatial_only,
            'lbp_texture': lbp_only,
            'hog_shape_324D': hog_only,
            'text_prompt': text_only,
            'color48D': color48_only,
        },
        'clip_plus_single': {k: v for k, v in ablation_results.items() if '+' in k and k.count('+') == 1},
        'clip_plus_multi': {k: v for k, v in ablation_results.items() if k.count('+') > 1},
        'best_combination': {
            'name': best_name,
            'metrics': ablation_results[best_name],
            'delta_vs_clip': round(ablation_results[best_name]['R@1'] - clip_base['R@1'], 4),
            'delta_vs_phase2': round(ablation_results[best_name]['R@1'] - phase2_repro['R@1'], 4),
        },
        'per_category': {
            'clip_baseline': clip_cats,
            'phase2_champion': phase2_cats,
            'best_phase3': best_cats,
        },
        'feature_dimensions': {
            'clip': g_clip.shape[1],
            'color48': g_color48.shape[1],
            'spatial': g_spatial.shape[1],
            'lbp': g_lbp.shape[1],
            'hog': g_hog.shape[1],
            'text': g_text.shape[1],
        },
        'extraction_times': {
            'clip': round(clip_time, 1),
            'spatial': round(spatial_time, 1),
            'lbp': round(lbp_time, 1),
            'hog': round(hog_time, 1),
            'text': round(text_time, 1),
        },
    }
}

with open(RES / 'phase3_anthony_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("  Saved: phase3_anthony_results.json")

# Update metrics.json
metrics_path = RES / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {}
metrics['phase3_anthony'] = all_results['phase3_anthony']
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print("  Updated: metrics.json")

# ======================================================================
# 6. SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 3 SUMMARY")
print("=" * 70)

print(f"\n  Research question: Is the bottleneck the backbone or the features?")
print(f"\n  CLIP L/14 baseline:       R@1 = {clip_base['R@1']:.4f}")
print(f"  Phase 2 champion (color): R@1 = {phase2_repro['R@1']:.4f}")
print(f"  Phase 3 best ({best_name}): R@1 = {ablation_results[best_name]['R@1']:.4f}")
print(f"  Δ vs Phase 2:             {ablation_results[best_name]['R@1'] - phase2_repro['R@1']:+.4f}")

print(f"\n  Standalone feature ranking:")
standalone = {
    'CLIP 768D': clip_base['R@1'],
    'Spatial Color 192D': spatial_only['R@1'],
    'Color 48D': color48_only['R@1'],
    'HOG 324D': hog_only['R@1'],
    'LBP Texture': lbp_only['R@1'],
    'Text Prompt': text_only['R@1'],
}
for rank, (name, r1) in enumerate(sorted(standalone.items(), key=lambda x: -x[1]), 1):
    print(f"    {rank}. {name:<20s} R@1={r1:.4f}")

print("\n  Plots saved to results/phase3_*.png")
print("  Results saved to results/phase3_anthony_results.json")
print("  Done!")
