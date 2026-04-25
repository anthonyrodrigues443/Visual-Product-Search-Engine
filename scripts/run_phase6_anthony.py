#!/usr/bin/env python3
"""Phase 6: Explainability & Model Understanding — Visual Product Search

Research questions:
  1. Per-query feature attribution: For each query, which component (CLIP, color,
     spatial, category filter) is responsible for finding the correct product?
  2. Embedding space structure: How well do categories cluster? Where do they overlap?
  3. Failure mode taxonomy: What visual patterns characterize the 278 failures?
  4. Per-category feature reliance: Do different categories depend on different features?
  5. Similarity score decomposition: What distinguishes success from failure at the
     individual similarity-component level?
  6. Category filter error analysis: When does category filtering help vs hurt?

Building on Phase 5 champion: CLIP L/14 + color + spatial + category filter → R@1=0.7293

Author: Anthony Rodrigues | Date: 2026-04-25
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
from collections import defaultdict, Counter
import torch
import faiss
import requests

plt.style.use('seaborn-v0_8-whitegrid')
PROJECT = Path(__file__).parent.parent
PROC = PROJECT / 'data' / 'processed'
RES = PROJECT / 'results'
RES.mkdir(exist_ok=True)

EVAL_N = 300
K = 50
BS = 32

# ======================================================================
# 1. DATA + IMAGES (same as Phase 5)
# ======================================================================
print("=" * 70)
print("PHASE 6: EXPLAINABILITY & MODEL UNDERSTANDING (Anthony)")
print("Why does our visual search work — and where does it break?")
print("=" * 70)

gallery_df = pd.read_csv(PROC / 'gallery.csv')
query_df = pd.read_csv(PROC / 'query.csv')
eval_pids = gallery_df['product_id'].values[:EVAL_N]
g_df = gallery_df[gallery_df['product_id'].isin(eval_pids)].reset_index(drop=True)
q_df = query_df[query_df['product_id'].isin(eval_pids)].reset_index(drop=True)
print(f"Eval set: {len(g_df)} gallery, {len(q_df)} query ({EVAL_N} products)")

all_idx = sorted(set(
    g_df['index'].astype(int).tolist() + q_df['index'].astype(int).tolist()
))
print(f"Streaming {len(all_idx)} images from HuggingFace...")
sys.stdout.flush()

from datasets import load_dataset

def stream_images_with_retry(all_idx, max_retries=5):
    needed = set(all_idx)
    imgs = {}
    for attempt in range(max_retries):
        if not needed:
            break
        print(f"  Attempt {attempt+1}/{max_retries}: {len(needed)} images remaining...")
        sys.stdout.flush()
        try:
            ds = load_dataset('Marqo/deepfashion-inshop', split='data', streaming=True)
            for i, ex in enumerate(tqdm(ds, total=max(needed) + 1, disable=(attempt > 0))):
                if i in needed:
                    img = ex['image']
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    imgs[i] = img
                    needed.discard(i)
                    if not needed:
                        break
        except (requests.exceptions.ReadTimeout, Exception) as e:
            print(f"  Retry {attempt+1}: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(3 * (attempt + 1))
    return imgs

imgs = stream_images_with_retry(all_idx)
print(f"Cached {len(imgs)} images in memory\n")
sys.stdout.flush()

g_pids = g_df['product_id'].values
q_pids = q_df['product_id'].values
g_indices = g_df['index'].astype(int).values
q_indices = q_df['index'].astype(int).values
g_cats = g_df['category2'].values
q_cats = q_df['category2'].values

# ======================================================================
# 2. HELPERS
# ======================================================================
def recall_at_k(q_pids, g_pids, indices, ks=(1, 5, 10, 20)):
    results = {}
    for k in ks:
        correct = sum(1 for qi, qp in enumerate(q_pids) if qp in g_pids[indices[qi, :k]])
        results[f'R@{k}'] = round(correct / len(q_pids), 4)
    return results

def faiss_search(gf, qf, k=K):
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(gn.shape[1])
    index.add(np.ascontiguousarray(gn, dtype=np.float32))
    D, I = index.search(np.ascontiguousarray(qn, dtype=np.float32), k)
    return D, I

def concat_features(*feat_arrays, weights=None):
    if weights is None:
        weights = [1.0] * len(feat_arrays)
    parts = []
    for f, w in zip(feat_arrays, weights):
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        parts.append(fn * w)
    return np.concatenate(parts, axis=1).astype(np.float32)

def category_filtered_search(vis_g, vis_q, g_cats, q_cats, k=K):
    indices = np.zeros((len(vis_q), k), dtype=np.int64)
    scores = np.zeros((len(vis_q), k), dtype=np.float32)
    gn = vis_g / (np.linalg.norm(vis_g, axis=1, keepdims=True) + 1e-8)
    qn = vis_q / (np.linalg.norm(vis_q, axis=1, keepdims=True) + 1e-8)

    for cat in np.unique(q_cats):
        q_mask = (q_cats == cat)
        g_mask = (g_cats == cat)
        if g_mask.sum() == 0:
            continue
        g_idx = np.where(g_mask)[0]
        q_idx = np.where(q_mask)[0]
        cat_g = gn[g_idx]
        cat_q = qn[q_idx]
        cat_k = min(k, len(g_idx))
        faiss.omp_set_num_threads(1)
        index = faiss.IndexFlatIP(cat_g.shape[1])
        index.add(np.ascontiguousarray(cat_g, dtype=np.float32))
        D, I = index.search(np.ascontiguousarray(cat_q, dtype=np.float32), cat_k)
        for qi_local, qi_global in enumerate(q_idx):
            for rank in range(cat_k):
                indices[qi_global, rank] = g_idx[I[qi_local, rank]]
                scores[qi_global, rank] = D[qi_local, rank]
            if cat_k < k:
                remaining = np.setdiff1d(np.arange(len(gn)), g_idx)
                if len(remaining) > 0:
                    rem_scores = qn[qi_global] @ gn[remaining].T
                    top_rem = np.argsort(-rem_scores)[:k - cat_k]
                    for j, idx in enumerate(top_rem):
                        indices[qi_global, cat_k + j] = remaining[idx]
                        scores[qi_global, cat_k + j] = rem_scores[idx]
    return scores, indices

# ======================================================================
# 3. EXTRACT FEATURES (same pipeline as Phase 5)
# ======================================================================
print("=" * 70)
print("3. Feature Extraction")
print("=" * 70)

import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
clip_model.eval()

def extract_clip_visual(indices_list):
    feats = []
    batch = []
    for ix in tqdm(indices_list, desc='  CLIP visual', leave=False):
        ix = int(ix)
        if ix not in imgs: continue
        batch.append(clip_preprocess(imgs[ix]))
        if len(batch) >= BS:
            with torch.no_grad():
                f = clip_model.encode_image(torch.stack(batch))
            feats.append(f.cpu().float().numpy())
            batch = []
    if batch:
        with torch.no_grad():
            f = clip_model.encode_image(torch.stack(batch))
        feats.append(f.cpu().float().numpy())
    return np.vstack(feats)

def batch_extract(indices_list, extract_fn):
    feats = []
    for ix in tqdm(indices_list, desc='  Features', leave=False):
        ix = int(ix)
        if ix not in imgs: continue
        feats.append(extract_fn(imgs[ix]))
    return np.array(feats, dtype=np.float32)

from src.feature_engineering import (
    extract_color_palette, extract_hsv_histogram, _rgb_to_hsv_vectorized
)

def extract_spatial_color_grid(img, grid_rows=4, grid_cols=4, bins=4):
    img_small = img.resize((128, 128), Image.LANCZOS).convert('RGB')
    pixels = np.array(img_small).astype(np.float32) / 255.0
    h, w = pixels.shape[:2]
    rh, rw = h // grid_rows, w // grid_cols
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

t0 = time.time()
g_clip = extract_clip_visual(g_indices)
q_clip = extract_clip_visual(q_indices)
print(f"  CLIP: {g_clip.shape} ({time.time()-t0:.1f}s)")

t0 = time.time()
g_rgb = batch_extract(g_indices, extract_color_palette)
q_rgb = batch_extract(q_indices, extract_color_palette)
g_hsv = batch_extract(g_indices, extract_hsv_histogram)
q_hsv = batch_extract(q_indices, extract_hsv_histogram)
g_color = np.concatenate([g_rgb, g_hsv], axis=1)
q_color = np.concatenate([q_rgb, q_hsv], axis=1)
print(f"  Color: {g_color.shape[1]}D ({time.time()-t0:.1f}s)")

t0 = time.time()
g_spatial = batch_extract(g_indices, extract_spatial_color_grid)
q_spatial = batch_extract(q_indices, extract_spatial_color_grid)
print(f"  Spatial: {g_spatial.shape[1]}D ({time.time()-t0:.1f}s)")

# Phase 5 champion weights
W_CLIP, W_COLOR, W_SPATIAL = 1.0, 1.0, 0.25

# Build combined features
vis_g = concat_features(g_clip, g_color, g_spatial, weights=[W_CLIP, W_COLOR, W_SPATIAL])
vis_q = concat_features(q_clip, q_color, q_spatial, weights=[W_CLIP, W_COLOR, W_SPATIAL])

# Run champion pipeline (with category filter)
cat_D, cat_I = category_filtered_search(vis_g, vis_q, g_cats, q_cats)
cat_result = recall_at_k(q_pids, g_pids, cat_I)
print(f"\n  Champion (Visual + Cat Filter): {cat_result}")

# Also run without category filter for comparison
vis_D, vis_I = faiss_search(vis_g, vis_q)
vis_result = recall_at_k(q_pids, g_pids, vis_I)
print(f"  Without cat filter:             {vis_result}")

# CLIP-only baseline
clip_D, clip_I = faiss_search(g_clip, q_clip)
clip_result = recall_at_k(q_pids, g_pids, clip_I)
print(f"  CLIP-only baseline:             {clip_result}")

all_results = {}

# ======================================================================
# EXPERIMENT 6.1: PER-QUERY FEATURE ATTRIBUTION
# ======================================================================
print("\n" + "=" * 70)
print("6.1  Per-Query Feature Attribution")
print("     For each query: which component rescued it?")
print("=" * 70)

# For each query, compute R@1 with each system:
#   1. CLIP only
#   2. CLIP + color
#   3. CLIP + color + spatial
#   4. CLIP + color + spatial + cat filter (champion)

# Normalize each feature set
def norm_feats(f):
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)

g_clip_n = norm_feats(g_clip)
q_clip_n = norm_feats(q_clip)
g_color_n = norm_feats(g_color) * W_COLOR
q_color_n = norm_feats(q_color) * W_COLOR
g_spatial_n = norm_feats(g_spatial) * W_SPATIAL
q_spatial_n = norm_feats(q_spatial) * W_SPATIAL

# Per-query cosine similarity to correct gallery item
def per_query_correct_rank(g_feats, q_feats, q_pids, g_pids):
    """For each query, find the rank of the correct product."""
    D, I = faiss_search(g_feats, q_feats, k=K)
    ranks = []
    for qi in range(len(q_pids)):
        qp = q_pids[qi]
        rank = K + 1  # not found
        for r in range(min(K, I.shape[1])):
            if g_pids[I[qi, r]] == qp:
                rank = r + 1
                break
        ranks.append(rank)
    return np.array(ranks)

# Compute ranks for each system
clip_ranks = per_query_correct_rank(g_clip, q_clip, q_pids, g_pids)

cc_g = concat_features(g_clip, g_color, weights=[W_CLIP, W_COLOR])
cc_q = concat_features(q_clip, q_color, weights=[W_CLIP, W_COLOR])
clip_color_ranks = per_query_correct_rank(cc_g, cc_q, q_pids, g_pids)

full_ranks = per_query_correct_rank(vis_g, vis_q, q_pids, g_pids)

# For category-filtered, compute ranks from cat_I
cat_ranks = np.full(len(q_pids), K + 1, dtype=int)
for qi in range(len(q_pids)):
    qp = q_pids[qi]
    for r in range(min(K, cat_I.shape[1])):
        if g_pids[cat_I[qi, r]] == qp:
            cat_ranks[qi] = r + 1
            break

# Attribution: which component RESCUED this query (moved it to rank 1)?
attribution = []
for qi in range(len(q_pids)):
    cr = clip_ranks[qi]
    ccr = clip_color_ranks[qi]
    fr = full_ranks[qi]
    catr = cat_ranks[qi]

    if catr == 1:
        if cr == 1:
            rescuer = 'CLIP alone'
        elif ccr == 1:
            rescuer = 'Color rescued'
        elif fr == 1:
            rescuer = 'Spatial rescued'
        else:
            rescuer = 'Cat filter rescued'
    else:
        rescuer = 'Failed (all systems)'

    attribution.append({
        'query_idx': qi,
        'category': q_cats[qi],
        'clip_rank': int(cr),
        'clip_color_rank': int(ccr),
        'full_rank': int(fr),
        'cat_filter_rank': int(catr),
        'rescuer': rescuer,
    })

attr_df = pd.DataFrame(attribution)
rescuer_counts = attr_df['rescuer'].value_counts()

print("\n  Per-query attribution (who made it R@1=1?):")
for rescuer, count in rescuer_counts.items():
    pct = count / len(attr_df) * 100
    print(f"    {rescuer:<25} {count:>5} ({pct:>5.1f}%)")

# Per-category attribution
print("\n  Per-category attribution breakdown:")
cat_attr = attr_df.groupby('category')['rescuer'].value_counts().unstack(fill_value=0)
print(cat_attr.to_string())

all_results['attribution'] = {
    'counts': rescuer_counts.to_dict(),
    'per_category': {cat: row.to_dict() for cat, row in cat_attr.iterrows()},
    'total_queries': len(attr_df),
}

# ======================================================================
# EXPERIMENT 6.2: SIMILARITY SCORE DECOMPOSITION
# ======================================================================
print("\n" + "=" * 70)
print("6.2  Similarity Score Decomposition")
print("     For each query: decompose the similarity to the correct product")
print("     into CLIP, color, and spatial components")
print("=" * 70)

# For each query, find the correct gallery item and compute component-wise similarities
correct_gallery_idx = {}
for qi in range(len(q_pids)):
    qp = q_pids[qi]
    for gi in range(len(g_pids)):
        if g_pids[gi] == qp:
            correct_gallery_idx[qi] = gi
            break

decomp_records = []
for qi in range(len(q_pids)):
    if qi not in correct_gallery_idx:
        continue
    gi = correct_gallery_idx[qi]

    # Component-wise cosine similarities
    clip_sim = float(q_clip_n[qi] @ g_clip_n[gi])
    color_sim = float(q_color_n[qi] @ g_color_n[gi])
    spatial_sim = float(q_spatial_n[qi] @ g_spatial_n[gi])

    # Combined similarity (weighted sum in the concatenated space is proportional to this)
    combined_sim = float(vis_q[qi] @ vis_g[gi] / (
        np.linalg.norm(vis_q[qi]) * np.linalg.norm(vis_g[gi]) + 1e-8))

    is_success = (cat_ranks[qi] == 1)

    decomp_records.append({
        'query_idx': qi,
        'category': q_cats[qi],
        'clip_sim': clip_sim,
        'color_sim': color_sim,
        'spatial_sim': spatial_sim,
        'combined_sim': combined_sim,
        'champion_rank': int(cat_ranks[qi]),
        'success': is_success,
    })

decomp_df = pd.DataFrame(decomp_records)

# Success vs failure similarity comparison
success_mask = decomp_df['success']
print("\n  Mean similarity to correct product (success vs failure):")
print(f"  {'Component':<15} {'Success':>10} {'Failure':>10} {'Gap':>10}")
print("  " + "-" * 50)
for comp in ['clip_sim', 'color_sim', 'spatial_sim', 'combined_sim']:
    s_mean = decomp_df.loc[success_mask, comp].mean()
    f_mean = decomp_df.loc[~success_mask, comp].mean()
    gap = s_mean - f_mean
    print(f"  {comp:<15} {s_mean:>10.4f} {f_mean:>10.4f} {gap:>+10.4f}")

# Which component has the largest gap? That's the most discriminative
gaps = {}
for comp in ['clip_sim', 'color_sim', 'spatial_sim']:
    s_mean = decomp_df.loc[success_mask, comp].mean()
    f_mean = decomp_df.loc[~success_mask, comp].mean()
    gaps[comp] = s_mean - f_mean

most_discriminative = max(gaps, key=gaps.get)
print(f"\n  Most discriminative component: {most_discriminative} (gap={gaps[most_discriminative]:+.4f})")

# Per-category similarity profiles
print("\n  Per-category mean CLIP sim (success):")
cat_clip_sim = decomp_df[decomp_df['success']].groupby('category')['clip_sim'].mean().sort_values(ascending=False)
for cat, sim in cat_clip_sim.items():
    print(f"    {cat:<15} {sim:.4f}")

all_results['similarity_decomposition'] = {
    'success_means': {comp: float(decomp_df.loc[success_mask, comp].mean())
                      for comp in ['clip_sim', 'color_sim', 'spatial_sim', 'combined_sim']},
    'failure_means': {comp: float(decomp_df.loc[~success_mask, comp].mean())
                      for comp in ['clip_sim', 'color_sim', 'spatial_sim', 'combined_sim']},
    'gaps': {k: float(v) for k, v in gaps.items()},
    'most_discriminative': most_discriminative,
}

# ======================================================================
# EXPERIMENT 6.3: FAILURE MODE TAXONOMY
# ======================================================================
print("\n" + "=" * 70)
print("6.3  Failure Mode Taxonomy")
print("     Categorize the 278 failures by failure pattern")
print("=" * 70)

failure_records = []
for qi in range(len(q_pids)):
    if cat_ranks[qi] == 1:
        continue  # skip successes

    gi_correct = correct_gallery_idx.get(qi, -1)
    if gi_correct < 0:
        continue

    # What did the model retrieve instead?
    gi_retrieved = cat_I[qi, 0]
    retrieved_cat = g_cats[gi_retrieved]
    query_cat = q_cats[qi]

    # Component-wise sims to correct vs retrieved
    clip_sim_correct = float(q_clip_n[qi] @ g_clip_n[gi_correct])
    clip_sim_retrieved = float(q_clip_n[qi] @ g_clip_n[gi_retrieved])
    color_sim_correct = float(q_color_n[qi] @ g_color_n[gi_correct])
    color_sim_retrieved = float(q_color_n[qi] @ g_color_n[gi_retrieved])
    spatial_sim_correct = float(q_spatial_n[qi] @ g_spatial_n[gi_correct])
    spatial_sim_retrieved = float(q_spatial_n[qi] @ g_spatial_n[gi_retrieved])

    # Classify failure mode
    clip_margin = clip_sim_correct - clip_sim_retrieved
    color_margin = color_sim_correct - color_sim_retrieved
    spatial_margin = spatial_sim_correct - spatial_sim_retrieved

    if query_cat != retrieved_cat:
        mode = 'Cross-category confusion'
    elif clip_margin < -0.02 and color_margin > 0.02:
        mode = 'CLIP wrong, color right'
    elif clip_margin > 0.02 and color_margin < -0.02:
        mode = 'Color wrong, CLIP right'
    elif clip_margin < -0.02 and color_margin < -0.02:
        mode = 'Both CLIP and color wrong'
    elif abs(clip_margin) < 0.02 and abs(color_margin) < 0.02:
        mode = 'Ambiguous (margins < 0.02)'
    else:
        mode = 'Mixed signals'

    failure_records.append({
        'query_idx': qi,
        'query_cat': query_cat,
        'retrieved_cat': retrieved_cat,
        'champion_rank': int(cat_ranks[qi]),
        'clip_margin': clip_margin,
        'color_margin': color_margin,
        'spatial_margin': spatial_margin,
        'failure_mode': mode,
    })

fail_df = pd.DataFrame(failure_records)

mode_counts = fail_df['failure_mode'].value_counts()
print("\n  Failure mode taxonomy:")
for mode, count in mode_counts.items():
    pct = count / len(fail_df) * 100
    print(f"    {mode:<35} {count:>5} ({pct:>5.1f}%)")

# Per-category failure rates
cat_fail_rate = {}
for cat in sorted(q_df['category2'].unique()):
    n_total = (q_cats == cat).sum()
    n_fail = len(fail_df[fail_df['query_cat'] == cat])
    cat_fail_rate[cat] = round(n_fail / max(n_total, 1), 4)
    print(f"  {cat:<15} fail rate: {cat_fail_rate[cat]:.2%} ({n_fail}/{n_total})")

# Where do failures land in the ranking?
rank_dist = fail_df['champion_rank'].describe()
print(f"\n  Failure rank distribution: median={rank_dist['50%']:.0f}, "
      f"75th={rank_dist['75%']:.0f}, max={rank_dist['max']:.0f}")

all_results['failure_taxonomy'] = {
    'mode_counts': mode_counts.to_dict(),
    'per_category_fail_rate': cat_fail_rate,
    'n_failures': len(fail_df),
    'failure_rank_median': float(rank_dist['50%']),
    'failure_rank_75th': float(rank_dist['75%']),
}

# ======================================================================
# EXPERIMENT 6.4: CATEGORY FILTER IMPACT ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("6.4  Category Filter Impact — Help vs Hurt")
print("=" * 70)

# Compare per-query: with vs without category filter
cat_filter_impact = []
for qi in range(len(q_pids)):
    rank_with = int(cat_ranks[qi])
    rank_without = int(full_ranks[qi])
    delta = rank_without - rank_with  # positive = filter helped

    cat_filter_impact.append({
        'query_idx': qi,
        'category': q_cats[qi],
        'rank_with_filter': rank_with,
        'rank_without_filter': rank_without,
        'delta_rank': delta,
        'filter_helped': delta > 0,
        'filter_hurt': delta < 0,
        'filter_neutral': delta == 0,
    })

cf_df = pd.DataFrame(cat_filter_impact)

n_helped = cf_df['filter_helped'].sum()
n_hurt = cf_df['filter_hurt'].sum()
n_neutral = cf_df['filter_neutral'].sum()
print(f"\n  Category filter impact:")
print(f"    Helped:  {n_helped:>5} ({n_helped/len(cf_df)*100:>5.1f}%)")
print(f"    Hurt:    {n_hurt:>5} ({n_hurt/len(cf_df)*100:>5.1f}%)")
print(f"    Neutral: {n_neutral:>5} ({n_neutral/len(cf_df)*100:>5.1f}%)")

# Median rank improvement when helped
helped_deltas = cf_df.loc[cf_df['filter_helped'], 'delta_rank']
hurt_deltas = cf_df.loc[cf_df['filter_hurt'], 'delta_rank']
print(f"    Median rank improvement when helped: {helped_deltas.median():.0f}")
if len(hurt_deltas) > 0:
    print(f"    Median rank degradation when hurt:  {hurt_deltas.median():.0f}")

# Per-category
print("\n  Per-category filter impact (% queries helped):")
for cat in sorted(q_df['category2'].unique()):
    cat_mask = (cf_df['category'] == cat)
    n_cat = cat_mask.sum()
    n_cat_helped = cf_df.loc[cat_mask, 'filter_helped'].sum()
    n_cat_hurt = cf_df.loc[cat_mask, 'filter_hurt'].sum()
    print(f"    {cat:<15} helped={n_cat_helped:>4}/{n_cat:<4} "
          f"hurt={n_cat_hurt:>3}/{n_cat:<4}")

all_results['category_filter_impact'] = {
    'n_helped': int(n_helped),
    'n_hurt': int(n_hurt),
    'n_neutral': int(n_neutral),
    'total': len(cf_df),
    'median_improvement': float(helped_deltas.median()),
    'median_degradation': float(hurt_deltas.median()) if len(hurt_deltas) > 0 else 0,
}

# ======================================================================
# EXPERIMENT 6.5: EMBEDDING SPACE STRUCTURE (t-SNE)
# ======================================================================
print("\n" + "=" * 70)
print("6.5  Embedding Space Structure (t-SNE)")
print("     How well do categories cluster in the embedding space?")
print("=" * 70)

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Run t-SNE on gallery embeddings (300 items — fast enough)
print("  Running t-SNE on gallery CLIP embeddings...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
g_clip_2d = tsne.fit_transform(g_clip_n)

# Also run on combined embeddings
print("  Running t-SNE on gallery combined embeddings...")
tsne2 = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
g_combined_2d = tsne2.fit_transform(vis_g)

# Silhouette scores
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
g_cat_labels = le.fit_transform(g_cats)

sil_clip = silhouette_score(g_clip_n, g_cat_labels)
sil_combined = silhouette_score(vis_g, g_cat_labels)
print(f"\n  Silhouette score (CLIP only):  {sil_clip:.4f}")
print(f"  Silhouette score (combined):   {sil_combined:.4f}")
print(f"  Color+spatial adds to clustering: {sil_combined - sil_clip:+.4f}")

# Per-category silhouette
from sklearn.metrics import silhouette_samples
sil_samples_clip = silhouette_samples(g_clip_n, g_cat_labels)
sil_samples_combined = silhouette_samples(vis_g, g_cat_labels)

print("\n  Per-category silhouette (combined):")
for cat_id, cat_name in enumerate(le.classes_):
    mask = (g_cat_labels == cat_id)
    cat_sil = sil_samples_combined[mask].mean()
    cat_sil_clip = sil_samples_clip[mask].mean()
    delta = cat_sil - cat_sil_clip
    print(f"    {cat_name:<15} combined={cat_sil:>7.4f}  CLIP={cat_sil_clip:>7.4f}  Δ={delta:>+7.4f}")

all_results['embedding_structure'] = {
    'silhouette_clip': float(sil_clip),
    'silhouette_combined': float(sil_combined),
    'silhouette_delta': float(sil_combined - sil_clip),
    'tsne_clip': g_clip_2d.tolist(),
    'tsne_combined': g_combined_2d.tolist(),
    'gallery_cats': g_cats.tolist(),
}

# ======================================================================
# EXPERIMENT 6.6: RANK IMPROVEMENT ANATOMY
# ======================================================================
print("\n" + "=" * 70)
print("6.6  Rank Improvement Anatomy")
print("     How does each component change the rank of the correct product?")
print("=" * 70)

# Build a table: for each query, show rank under each system
rank_anatomy = pd.DataFrame({
    'category': q_cats,
    'CLIP_rank': clip_ranks,
    'CLIP+color_rank': clip_color_ranks,
    'Full_rank': full_ranks,
    'Champion_rank': cat_ranks,
})

# Mean rank per system
print("\n  Mean rank of correct product (lower = better):")
for col in ['CLIP_rank', 'CLIP+color_rank', 'Full_rank', 'Champion_rank']:
    mean_rank = rank_anatomy[col].mean()
    median_rank = rank_anatomy[col].median()
    r1 = (rank_anatomy[col] == 1).mean()
    print(f"    {col:<20} mean={mean_rank:>6.2f}  median={median_rank:>5.1f}  R@1={r1:.4f}")

# Per-category mean rank
print("\n  Per-category mean rank improvement (CLIP → Champion):")
for cat in sorted(q_df['category2'].unique()):
    mask = (rank_anatomy['category'] == cat)
    clip_mr = rank_anatomy.loc[mask, 'CLIP_rank'].mean()
    champ_mr = rank_anatomy.loc[mask, 'Champion_rank'].mean()
    delta = clip_mr - champ_mr
    print(f"    {cat:<15} CLIP={clip_mr:>6.2f} → Champion={champ_mr:>6.2f}  improvement={delta:>+6.2f}")

all_results['rank_anatomy'] = {
    'mean_ranks': {col: float(rank_anatomy[col].mean())
                   for col in ['CLIP_rank', 'CLIP+color_rank', 'Full_rank', 'Champion_rank']},
    'r1_rates': {col: float((rank_anatomy[col] == 1).mean())
                 for col in ['CLIP_rank', 'CLIP+color_rank', 'Full_rank', 'Champion_rank']},
}

# ======================================================================
# 7. PLOTS
# ======================================================================
print("\n" + "=" * 70)
print("7.  Generating plots")
print("=" * 70)

fig = plt.figure(figsize=(22, 18))

# --- Plot 1: Per-query attribution pie/bar chart ---
ax1 = fig.add_subplot(3, 3, 1)
rescuer_names = list(rescuer_counts.index)
rescuer_vals = list(rescuer_counts.values)
colors_pie = ['#4CAF50', '#FF9800', '#2196F3', '#9C27B0', '#F44336'][:len(rescuer_names)]
ax1.barh(range(len(rescuer_names)), rescuer_vals, color=colors_pie, edgecolor='white')
ax1.set_yticks(range(len(rescuer_names)))
ax1.set_yticklabels([n[:25] for n in rescuer_names], fontsize=8)
for i, (name, val) in enumerate(zip(rescuer_names, rescuer_vals)):
    pct = val / len(attr_df) * 100
    ax1.text(val + 5, i, f'{val} ({pct:.1f}%)', va='center', fontsize=8, fontweight='bold')
ax1.set_xlabel('Queries', fontsize=10)
ax1.set_title('6.1: Who Rescued Each Query?', fontsize=11, fontweight='bold')

# --- Plot 2: Similarity decomposition (success vs failure) ---
ax2 = fig.add_subplot(3, 3, 2)
comps = ['clip_sim', 'color_sim', 'spatial_sim']
comp_labels = ['CLIP', 'Color', 'Spatial']
x = np.arange(len(comps))
width = 0.35
s_means = [decomp_df.loc[success_mask, c].mean() for c in comps]
f_means = [decomp_df.loc[~success_mask, c].mean() for c in comps]
bars1 = ax2.bar(x - width/2, s_means, width, label='Success', color='#4CAF50', alpha=0.85)
bars2 = ax2.bar(x + width/2, f_means, width, label='Failure', color='#F44336', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(comp_labels, fontsize=10)
ax2.set_ylabel('Mean cosine sim to correct product', fontsize=9)
ax2.set_title('6.2: Similarity Decomposition', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9)
for bar, val in zip(bars1, s_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=8)
for bar, val in zip(bars2, f_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', fontsize=8)

# --- Plot 3: Failure mode taxonomy ---
ax3 = fig.add_subplot(3, 3, 3)
mode_names = list(mode_counts.index)
mode_vals = list(mode_counts.values)
colors_modes = ['#F44336', '#FF9800', '#9C27B0', '#2196F3', '#4CAF50', '#607D8B'][:len(mode_names)]
wedges, texts, autotexts = ax3.pie(mode_vals, labels=None, autopct='%1.1f%%',
                                     colors=colors_modes, startangle=90,
                                     textprops={'fontsize': 8})
ax3.legend([n[:30] for n in mode_names], loc='center left', bbox_to_anchor=(-0.3, 0.5),
           fontsize=7, frameon=False)
ax3.set_title('6.3: Failure Mode Taxonomy', fontsize=11, fontweight='bold')

# --- Plot 4: Category filter impact ---
ax4 = fig.add_subplot(3, 3, 4)
cats_sorted = sorted(q_df['category2'].unique())
helped_pcts = []
hurt_pcts = []
for cat in cats_sorted:
    cat_mask = (cf_df['category'] == cat)
    n_cat = cat_mask.sum()
    helped_pcts.append(cf_df.loc[cat_mask, 'filter_helped'].sum() / max(n_cat, 1) * 100)
    hurt_pcts.append(cf_df.loc[cat_mask, 'filter_hurt'].sum() / max(n_cat, 1) * 100)
x = np.arange(len(cats_sorted))
ax4.bar(x, helped_pcts, color='#4CAF50', alpha=0.8, label='Helped')
ax4.bar(x, [-h for h in hurt_pcts], color='#F44336', alpha=0.8, label='Hurt')
ax4.set_xticks(x)
ax4.set_xticklabels(cats_sorted, rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('% of queries', fontsize=10)
ax4.axhline(y=0, color='black', linewidth=0.5)
ax4.set_title('6.4: Category Filter Impact', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)

# --- Plot 5: t-SNE CLIP embeddings ---
ax5 = fig.add_subplot(3, 3, 5)
unique_cats = sorted(np.unique(g_cats))
cat_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cats)))
for i, cat in enumerate(unique_cats):
    mask = (g_cats == cat)
    ax5.scatter(g_clip_2d[mask, 0], g_clip_2d[mask, 1],
                c=[cat_colors[i]], label=cat, s=20, alpha=0.7, edgecolors='none')
ax5.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.8)
ax5.set_title(f'6.5a: t-SNE CLIP (sil={sil_clip:.3f})', fontsize=11, fontweight='bold')
ax5.set_xticks([]); ax5.set_yticks([])

# --- Plot 6: t-SNE Combined embeddings ---
ax6 = fig.add_subplot(3, 3, 6)
for i, cat in enumerate(unique_cats):
    mask = (g_cats == cat)
    ax6.scatter(g_combined_2d[mask, 0], g_combined_2d[mask, 1],
                c=[cat_colors[i]], label=cat, s=20, alpha=0.7, edgecolors='none')
ax6.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.8)
ax6.set_title(f'6.5b: t-SNE Combined (sil={sil_combined:.3f})', fontsize=11, fontweight='bold')
ax6.set_xticks([]); ax6.set_yticks([])

# --- Plot 7: Per-category failure rate ---
ax7 = fig.add_subplot(3, 3, 7)
cats_by_fail = sorted(cat_fail_rate.keys(), key=lambda c: cat_fail_rate[c], reverse=True)
fail_rates = [cat_fail_rate[c] * 100 for c in cats_by_fail]
bar_colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(cats_by_fail)))
bars = ax7.barh(range(len(cats_by_fail)), fail_rates, color=bar_colors, edgecolor='white')
ax7.set_yticks(range(len(cats_by_fail)))
ax7.set_yticklabels(cats_by_fail, fontsize=9)
for bar, val in zip(bars, fail_rates):
    ax7.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
ax7.set_xlabel('Failure Rate (%)', fontsize=10)
ax7.set_title('6.3b: Per-Category Failure Rate', fontsize=11, fontweight='bold')

# --- Plot 8: Rank improvement anatomy ---
ax8 = fig.add_subplot(3, 3, 8)
systems = ['CLIP_rank', 'CLIP+color_rank', 'Full_rank', 'Champion_rank']
system_labels = ['CLIP', '+Color', '+Spatial', '+Cat Filter']
mean_ranks = [rank_anatomy[s].mean() for s in systems]
r1_rates = [(rank_anatomy[s] == 1).mean() * 100 for s in systems]
ax8_twin = ax8.twinx()
ax8.bar(range(len(systems)), mean_ranks, color='#2196F3', alpha=0.7, label='Mean Rank')
ax8_twin.plot(range(len(systems)), r1_rates, 'o-', color='#FF5722', linewidth=2,
              markersize=8, label='R@1 %')
ax8.set_xticks(range(len(systems)))
ax8.set_xticklabels(system_labels, fontsize=9)
ax8.set_ylabel('Mean Rank (lower=better)', fontsize=9, color='#2196F3')
ax8_twin.set_ylabel('R@1 (%)', fontsize=9, color='#FF5722')
ax8.set_title('6.6: Rank Improvement Anatomy', fontsize=11, fontweight='bold')
lines1, labels1 = ax8.get_legend_handles_labels()
lines2, labels2 = ax8_twin.get_legend_handles_labels()
ax8.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# --- Plot 9: Per-category similarity heatmap ---
ax9 = fig.add_subplot(3, 3, 9)
cat_sim_data = {}
for cat in sorted(q_df['category2'].unique()):
    mask = (decomp_df['category'] == cat) & decomp_df['success']
    if mask.sum() > 0:
        cat_sim_data[cat] = {
            'CLIP': decomp_df.loc[mask, 'clip_sim'].mean(),
            'Color': decomp_df.loc[mask, 'color_sim'].mean(),
            'Spatial': decomp_df.loc[mask, 'spatial_sim'].mean(),
        }
if cat_sim_data:
    sim_matrix = pd.DataFrame(cat_sim_data).T
    sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax9,
                cbar_kws={'shrink': 0.6}, annot_kws={'fontsize': 7})
    ax9.set_title('6.2b: Mean Sim by Category (Success)', fontsize=11, fontweight='bold')
    ax9.set_xticklabels(ax9.get_xticklabels(), fontsize=9)
    ax9.set_yticklabels(ax9.get_yticklabels(), fontsize=8, rotation=0)

plt.suptitle('Phase 6: Explainability & Model Understanding (Anthony)\n'
             'Visual Product Search — Why Does It Work?',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(RES / 'phase6_anthony_explainability.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase6_anthony_explainability.png")

# Additional plot: similarity distributions
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
for i, (comp, label) in enumerate(zip(comps, comp_labels)):
    ax = axes2[i]
    success_vals = decomp_df.loc[success_mask, comp].values
    failure_vals = decomp_df.loc[~success_mask, comp].values
    ax.hist(success_vals, bins=30, alpha=0.6, color='#4CAF50', label='Success', density=True)
    ax.hist(failure_vals, bins=30, alpha=0.6, color='#F44336', label='Failure', density=True)
    ax.axvline(x=success_vals.mean(), color='#2E7D32', linestyle='--', linewidth=2)
    ax.axvline(x=failure_vals.mean(), color='#C62828', linestyle='--', linewidth=2)
    ax.set_xlabel(f'{label} Cosine Similarity', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{label} Sim Distribution\n(gap={success_vals.mean()-failure_vals.mean():+.4f})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)

plt.suptitle('Phase 6: Similarity Distributions — Success vs Failure',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RES / 'phase6_anthony_sim_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase6_anthony_sim_distributions.png")

# ======================================================================
# 8. SAVE RESULTS
# ======================================================================
print("\n" + "=" * 70)
print("8.  Saving results")
print("=" * 70)

# Remove non-serializable items
save_results = {}
for k, v in all_results.items():
    if k == 'embedding_structure':
        # Skip the large t-SNE arrays from JSON
        save_results[k] = {kk: vv for kk, vv in v.items()
                           if kk not in ('tsne_clip', 'tsne_combined', 'gallery_cats')}
    else:
        save_results[k] = v

with open(RES / 'phase6_anthony_results.json', 'w') as f:
    json.dump(save_results, f, indent=2, default=str)
print("  Saved: phase6_anthony_results.json")

# Update metrics.json
metrics_path = RES / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {}
metrics['phase6_anthony'] = save_results
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2, default=str)
print("  Updated: metrics.json")

# ======================================================================
# 9. SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 6 SUMMARY (Anthony)")
print("=" * 70)

print(f"\n  6.1 Attribution:")
for r, c in rescuer_counts.items():
    print(f"    {r}: {c} queries ({c/len(attr_df)*100:.1f}%)")

print(f"\n  6.2 Similarity decomposition:")
print(f"    Most discriminative: {most_discriminative} (gap={gaps[most_discriminative]:+.4f})")

print(f"\n  6.3 Failure taxonomy:")
for m, c in mode_counts.head(3).items():
    print(f"    {m}: {c} ({c/len(fail_df)*100:.1f}%)")

print(f"\n  6.4 Category filter:")
print(f"    Helped {n_helped} queries, hurt {n_hurt}, neutral {n_neutral}")

print(f"\n  6.5 Embedding structure:")
print(f"    Silhouette: CLIP={sil_clip:.4f} → Combined={sil_combined:.4f} (Δ={sil_combined-sil_clip:+.4f})")

print(f"\n  6.6 Rank anatomy:")
for s, l in zip(systems, system_labels):
    mr = rank_anatomy[s].mean()
    r1 = (rank_anatomy[s] == 1).mean()
    print(f"    {l}: mean rank={mr:.2f}, R@1={r1:.4f}")

print("\n  Done!")
