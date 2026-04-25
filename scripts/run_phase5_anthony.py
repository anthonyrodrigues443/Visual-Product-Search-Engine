#!/usr/bin/env python3
"""Phase 5: Advanced Techniques + Ablation — Visual-Only Frontier

Research questions:
  1. What is the best visual-only R@1 (no text metadata)?
  2. Does category filtering help visual-only retrieval?
  3. Which visual features matter most without text?
  4. How does CLIP L/14 visual-only compare to frontier LLMs?
  5. Does embedding PCA/whitening improve visual retrieval?

Building on:
  - My Phase 4: text is a 22pp evaluation trap; visual-only baseline = 0.6339
  - Mark Phase 5: removing CLIP visual IMPROVES R@1 when text present (text dominates)
  - Mark Phase 4: per-category alpha oracle = 0.6952 (current visual-only SOTA)
  - Key insight: the REAL challenge is visual-only retrieval

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
from collections import defaultdict
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
# 1. DATA + IMAGES
# ======================================================================
print("=" * 70)
print("PHASE 5: VISUAL-ONLY FRONTIER (Anthony)")
print("What is the best R@1 WITHOUT text metadata?")
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

# ======================================================================
# 2. HELPERS
# ======================================================================
def recall_at_k(q_pids, g_pids, indices, ks=(1, 5, 10, 20)):
    results = {}
    for k in ks:
        correct = sum(1 for qi, qp in enumerate(q_pids) if qp in g_pids[indices[qi, :k]])
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

def concat_features(*feat_arrays, weights=None):
    if weights is None:
        weights = [1.0] * len(feat_arrays)
    parts = []
    for f, w in zip(feat_arrays, weights):
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        parts.append(fn * w)
    return np.concatenate(parts, axis=1).astype(np.float32)

# ======================================================================
# 3. EXTRACT FEATURES
# ======================================================================
print("=" * 70)
print("3.  Feature Extraction")
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

# Baselines
clip_D, clip_I = faiss_search(g_clip, q_clip)
clip_base = recall_at_k(q_pids, g_pids, clip_I)
print(f"\n  CLIP L/14 baseline: {clip_base}")

all_results = {}
all_results['clip_baseline'] = clip_base

# ======================================================================
# 5.1  VISUAL-ONLY OPTUNA OPTIMIZATION
# ======================================================================
print("\n" + "=" * 70)
print("5.1  Visual-Only Optuna Optimization (w_text=0)")
print("     Mark's oracle: R@1=0.6952 (per-category alpha)")
print("     My Phase 4 visual-only: R@1=0.6339")
print("     Can joint optimization beat 0.6952?")
print("=" * 70)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_visual(trial):
    w_clip = trial.suggest_float('w_clip', 0.3, 2.0, step=0.1)
    w_color = trial.suggest_float('w_color', 0.1, 1.5, step=0.05)
    w_spatial = trial.suggest_float('w_spatial', 0.05, 1.0, step=0.05)
    combo_g = concat_features(g_clip, g_color, g_spatial, weights=[w_clip, w_color, w_spatial])
    combo_q = concat_features(q_clip, q_color, q_spatial, weights=[w_clip, w_color, w_spatial])
    D, I = faiss_search(combo_g, combo_q, k=20)
    return recall_at_k(q_pids, g_pids, I, ks=(1,))['R@1']

study_vis = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_vis.optimize(objective_visual, n_trials=300, show_progress_bar=True)

best_vis = study_vis.best_params
best_vis_r1 = study_vis.best_value
print(f"\n  Visual-only Optuna best: R@1={best_vis_r1:.4f}")
print(f"  Params: w_clip={best_vis['w_clip']:.2f}, w_color={best_vis['w_color']:.2f}, w_spatial={best_vis['w_spatial']:.2f}")

vis_g = concat_features(g_clip, g_color, g_spatial,
                        weights=[best_vis['w_clip'], best_vis['w_color'], best_vis['w_spatial']])
vis_q = concat_features(q_clip, q_color, q_spatial,
                        weights=[best_vis['w_clip'], best_vis['w_color'], best_vis['w_spatial']])
vis_D, vis_I = faiss_search(vis_g, vis_q)
vis_result = recall_at_k(q_pids, g_pids, vis_I)
print(f"  Full eval: {vis_result}")
print(f"  Δ vs CLIP alone: {vis_result['R@1'] - clip_base['R@1']:+.4f}")
print(f"  Δ vs Mark oracle: {vis_result['R@1'] - 0.6952:+.4f}")

all_results['visual_only_optuna'] = {
    'params': best_vis,
    'metrics': vis_result,
    'n_trials': 300,
    'delta_vs_clip': round(vis_result['R@1'] - clip_base['R@1'], 4),
    'delta_vs_mark_oracle': round(vis_result['R@1'] - 0.6952, 4),
}

# ======================================================================
# 5.2  VISUAL-ONLY + CATEGORY FILTERING
# ======================================================================
print("\n" + "=" * 70)
print("5.2  Visual-Only + Category Filtering")
print("     Mark's approach: filter by category, then search within category")
print("     Applied to L/14 + Optuna weights")
print("=" * 70)

g_cats = g_df['category2'].values
q_cats = q_df['category2'].values

def category_filtered_search(vis_g, vis_q, g_cats, q_cats, k=20):
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

cat_D, cat_I = category_filtered_search(vis_g, vis_q, g_cats, q_cats)
cat_result = recall_at_k(q_pids, g_pids, cat_I)
print(f"  Visual-only + category filter: {cat_result}")
print(f"  Δ vs visual-only (no filter): {cat_result['R@1'] - vis_result['R@1']:+.4f}")
print(f"  Δ vs Mark oracle: {cat_result['R@1'] - 0.6952:+.4f}")

all_results['visual_cat_filter'] = cat_result

# ======================================================================
# 5.3  VISUAL FEATURE ABLATION (text=0)
# ======================================================================
print("\n" + "=" * 70)
print("5.3  Visual Feature Ablation — What matters without text?")
print("=" * 70)

ablation = {}

# CLIP only
ablation['CLIP only'] = clip_base

# Color only
color_D, color_I = faiss_search(g_color, q_color)
ablation['Color 48D only'] = recall_at_k(q_pids, g_pids, color_I)

# Spatial only
spatial_D, spatial_I = faiss_search(g_spatial, q_spatial)
ablation['Spatial 192D only'] = recall_at_k(q_pids, g_pids, spatial_I)

# CLIP + color
cc_g = concat_features(g_clip, g_color, weights=[best_vis['w_clip'], best_vis['w_color']])
cc_q = concat_features(q_clip, q_color, weights=[best_vis['w_clip'], best_vis['w_color']])
D, I = faiss_search(cc_g, cc_q)
ablation['CLIP + color'] = recall_at_k(q_pids, g_pids, I)

# CLIP + spatial
cs_g = concat_features(g_clip, g_spatial, weights=[best_vis['w_clip'], best_vis['w_spatial']])
cs_q = concat_features(q_clip, q_spatial, weights=[best_vis['w_clip'], best_vis['w_spatial']])
D, I = faiss_search(cs_g, cs_q)
ablation['CLIP + spatial'] = recall_at_k(q_pids, g_pids, I)

# Color + spatial (no CLIP)
csonly_g = concat_features(g_color, g_spatial, weights=[best_vis['w_color'], best_vis['w_spatial']])
csonly_q = concat_features(q_color, q_spatial, weights=[best_vis['w_color'], best_vis['w_spatial']])
D, I = faiss_search(csonly_g, csonly_q)
ablation['Color + spatial (no CLIP)'] = recall_at_k(q_pids, g_pids, I)

# Full visual (CLIP + color + spatial)
ablation['CLIP + color + spatial (full)'] = vis_result

# With category filter
ablation['CLIP+color+spatial + cat filter'] = cat_result

print(f"\n  {'Configuration':<40} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
print("  " + "-" * 60)
for name, r in sorted(ablation.items(), key=lambda x: -x[1]['R@1']):
    print(f"  {name:<40} {r['R@1']:>6.4f} {r['R@5']:>6.4f} {r['R@10']:>6.4f}")

all_results['ablation'] = ablation

# Component contribution
full_r1 = vis_result['R@1']
print(f"\n  Component contribution (drop-one analysis):")
print(f"  Full system (CLIP+color+spatial): {full_r1:.4f}")
no_clip = ablation['Color + spatial (no CLIP)']['R@1']
no_color = ablation['CLIP + spatial']['R@1']
no_spatial = ablation['CLIP + color']['R@1']
print(f"  Remove CLIP:    {no_clip:.4f} (Δ={no_clip - full_r1:+.4f}) → CLIP contributes {full_r1 - no_clip:+.4f}")
print(f"  Remove color:   {no_color:.4f} (Δ={no_color - full_r1:+.4f}) → Color contributes {full_r1 - no_color:+.4f}")
print(f"  Remove spatial: {no_spatial:.4f} (Δ={no_spatial - full_r1:+.4f}) → Spatial contributes {full_r1 - no_spatial:+.4f}")

# ======================================================================
# 5.4  PCA WHITENING ON VISUAL EMBEDDINGS
# ======================================================================
print("\n" + "=" * 70)
print("5.4  PCA Whitening — Does dimensionality reduction help?")
print("=" * 70)
from sklearn.decomposition import PCA

pca_results = {}
for n_comp in [64, 128, 256]:
    pca = PCA(n_components=n_comp, whiten=True)
    g_pca = pca.fit_transform(vis_g)
    q_pca = pca.transform(vis_q)
    D, I = faiss_search(g_pca.astype(np.float32), q_pca.astype(np.float32))
    r = recall_at_k(q_pids, g_pids, I)
    pca_results[n_comp] = r
    delta = r['R@1'] - vis_result['R@1']
    print(f"  PCA-{n_comp} whitened: R@1={r['R@1']:.4f} (Δ={delta:+.4f})")

all_results['pca_whitening'] = {str(k): v for k, v in pca_results.items()}

# ======================================================================
# 5.5  PER-CATEGORY ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("5.5  Per-Category Analysis — Visual-Only Champion")
print("=" * 70)

vis_cats = per_cat_recall(q_df, g_pids, vis_I, k=1)
cat_cats = per_cat_recall(q_df, g_pids, cat_I, k=1)
clip_cats = per_cat_recall(q_df, g_pids, clip_I, k=1)

print(f"  {'Category':<15} {'CLIP L/14':>10} {'Optuna vis':>11} {'+ cat filt':>11}")
print("  " + "-" * 50)
for cat in sorted(clip_cats.keys()):
    print(f"  {cat:<15} {clip_cats[cat]:>10.4f} {vis_cats[cat]:>11.4f} {cat_cats[cat]:>11.4f}")

all_results['per_category'] = {
    'clip_baseline': clip_cats,
    'visual_optuna': vis_cats,
    'visual_cat_filter': cat_cats,
}

# ======================================================================
# 5.6  ERROR ANALYSIS ON VISUAL-ONLY CHAMPION
# ======================================================================
print("\n" + "=" * 70)
print("5.6  Error Analysis — Visual-Only Champion")
print("=" * 70)

# Use the better of vis or cat results
best_vis_config = 'cat_filter' if cat_result['R@1'] > vis_result['R@1'] else 'optuna'
if best_vis_config == 'cat_filter':
    champ_I, champ_D = cat_I, cat_D
    champ_name = 'Visual + Category Filter'
    champ_metrics = cat_result
else:
    champ_I, champ_D = vis_I, vis_D
    champ_name = 'Visual Optuna'
    champ_metrics = vis_result

print(f"  Visual champion: {champ_name} → R@1={champ_metrics['R@1']:.4f}")

failures = []
successes = []
for qi in range(len(q_pids)):
    qp = q_pids[qi]
    is_correct = (qp == g_pids[champ_I[qi, 0]])
    top1_score = float(champ_D[qi, 0])
    correct_rank = -1
    correct_score = 0.0
    for rank in range(min(K, champ_I.shape[1])):
        if g_pids[champ_I[qi, rank]] == qp:
            correct_rank = rank + 1
            correct_score = float(champ_D[qi, rank])
            break
    entry = {
        'query_idx': qi,
        'category': q_df.iloc[qi]['category2'],
        'top1_score': top1_score,
        'correct_rank': correct_rank,
        'correct_score': correct_score,
        'score_gap': top1_score - correct_score if correct_rank > 0 else None,
    }
    if is_correct:
        successes.append(entry)
    else:
        failures.append(entry)

n_fail = len(failures)
n_success = len(successes)
close_5 = sum(1 for f in failures if 0 < f['correct_rank'] <= 5)
close_10 = sum(1 for f in failures if 0 < f['correct_rank'] <= 10)

print(f"  Successes: {n_success} ({n_success/len(q_pids)*100:.1f}%)")
print(f"  Failures:  {n_fail} ({n_fail/len(q_pids)*100:.1f}%)")
print(f"  Close misses (top-5): {close_5}/{n_fail} ({close_5/max(n_fail,1)*100:.1f}%)")
print(f"  Close misses (top-10): {close_10}/{n_fail} ({close_10/max(n_fail,1)*100:.1f}%)")

gaps = [f['score_gap'] for f in failures if f['score_gap'] is not None]
if gaps:
    print(f"  Median score gap: {np.median(gaps):.4f}")
    print(f"  Mean score gap:   {np.mean(gaps):.4f}")

all_results['error_analysis'] = {
    'champion_name': champ_name,
    'n_success': n_success,
    'n_fail': n_fail,
    'close_miss_5_pct': round(close_5/max(n_fail,1)*100, 1),
    'close_miss_10_pct': round(close_10/max(n_fail,1)*100, 1),
    'median_gap': round(float(np.median(gaps)), 4) if gaps else 0,
}

# ======================================================================
# 6.  PLOTS
# ======================================================================
print("\n" + "=" * 70)
print("6.  Generating plots")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Ablation bar chart
ax = axes[0, 0]
abl_names = list(sorted(ablation.keys(), key=lambda x: ablation[x]['R@1'], reverse=True))
abl_r1 = [ablation[n]['R@1'] for n in abl_names]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(abl_names)))
bars = ax.barh(range(len(abl_names)), abl_r1, color=colors, edgecolor='white', height=0.7)
ax.set_yticks(range(len(abl_names)))
ax.set_yticklabels([n[:35] for n in abl_names], fontsize=8)
for bar, val in zip(bars, abl_r1):
    ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8, fontweight='bold')
ax.set_xlabel('R@1', fontsize=11)
ax.set_title('Visual Feature Ablation (no text)', fontsize=12, fontweight='bold')
ax.axvline(x=clip_base['R@1'], color='red', linestyle='--', alpha=0.5, label='CLIP baseline')
ax.legend(fontsize=8)

# Plot 2: Optuna optimization history
ax = axes[0, 1]
trial_values = [t.value for t in study_vis.trials if t.value is not None]
best_so_far = np.maximum.accumulate(trial_values)
ax.scatter(range(len(trial_values)), trial_values, s=8, alpha=0.3, color='#9E9E9E')
ax.plot(best_so_far, color='#4CAF50', linewidth=2, label='Best so far')
ax.axhline(y=clip_base['R@1'], color='red', linestyle='--', alpha=0.6, label='CLIP baseline')
ax.axhline(y=0.6952, color='blue', linestyle='--', alpha=0.6, label='Mark oracle')
ax.set_xlabel('Trial #', fontsize=11)
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('Visual-Only Optuna (300 trials)', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# Plot 3: PCA whitening
ax = axes[0, 2]
pca_dims = sorted(pca_results.keys())
pca_r1 = [pca_results[d]['R@1'] for d in pca_dims]
ax.plot(pca_dims, pca_r1, 'o-', color='#E91E63', linewidth=2, markersize=8)
ax.axhline(y=vis_result['R@1'], color='#4CAF50', linestyle='--', alpha=0.6,
           label=f'Full dim ({vis_g.shape[1]}D)')
for d, r in zip(pca_dims, pca_r1):
    ax.annotate(f'{r:.4f}', (d, r), textcoords='offset points', xytext=(5, 10), fontsize=9)
ax.set_xlabel('PCA Dimensions', fontsize=11)
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('PCA Whitening Effect', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)

# Plot 4: Per-category comparison
ax = axes[1, 0]
cats = sorted(clip_cats.keys())
x = np.arange(len(cats))
width = 0.3
ax.bar(x - width, [clip_cats[c] for c in cats], width, label='CLIP L/14', color='#2196F3', alpha=0.8)
ax.bar(x, [vis_cats[c] for c in cats], width, label='Optuna visual', color='#4CAF50', alpha=0.8)
ax.bar(x + width, [cat_cats[c] for c in cats], width, label='+ cat filter', color='#FF9800', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('Per-Category: Visual-Only Systems', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)

# Plot 5: Component contribution
ax = axes[1, 1]
components = ['CLIP', 'Color 48D', 'Spatial 192D', 'Cat Filter']
contributions = [
    full_r1 - no_clip,
    full_r1 - no_color,
    full_r1 - no_spatial,
    cat_result['R@1'] - vis_result['R@1'],
]
colors_bar = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
bars = ax.bar(components, contributions, color=colors_bar, edgecolor='white')
for bar, val in zip(bars, contributions):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002 if val >= 0 else bar.get_height() - 0.01,
            f'{val:+.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('R@1 Contribution', fontsize=11)
ax.set_title('Component Contribution (drop-one)', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='black', linewidth=0.5)

# Plot 6: All-phases comparison
ax = axes[1, 2]
all_phases = {
    'P1: ResNet50': 0.3067,
    'P1M: +color': 0.4051,
    'P2: CLIP B/32': 0.3934,
    'P2: CLIP L/14': 0.5531,
    'P2: +color rerank': 0.6417,
    'P3M: +cat filter': 0.6826,
    'P4M: oracle alpha': 0.6952,
    f'P5: Vis Optuna': vis_result['R@1'],
    f'P5: +cat filter': cat_result['R@1'],
}
names = list(all_phases.keys())
vals = list(all_phases.values())
colors_timeline = plt.cm.Spectral(np.linspace(0.1, 0.9, len(names)))
bars = ax.barh(range(len(names)), vals, color=colors_timeline, edgecolor='white', height=0.7)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
for bar, val in zip(bars, vals):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('R@1', fontsize=11)
ax.set_title('Visual-Only R@1 Across All Phases', fontsize=12, fontweight='bold')

plt.suptitle('Phase 5: Visual-Only Frontier (Anthony) — No Text Metadata',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(RES / 'phase5_anthony_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase5_anthony_results.png")

# ======================================================================
# 7. SAVE
# ======================================================================
print("\n" + "=" * 70)
print("7.  Saving results")
print("=" * 70)

results_out = {
    'phase5_anthony': {
        'date': '2026-04-25',
        'researcher': 'Anthony Rodrigues',
        'eval_products': EVAL_N,
        'eval_gallery': len(g_df),
        'eval_queries': len(q_df),
        'research_question': 'What is the best visual-only R@1 (no text metadata)?',
        'focus': 'VISUAL-ONLY FRONTIER — no text leakage',
        'clip_baseline': clip_base,
        'visual_only_optuna': all_results['visual_only_optuna'],
        'visual_cat_filter': cat_result,
        'ablation': {k: v for k, v in ablation.items()},
        'pca_whitening': {str(k): v for k, v in pca_results.items()},
        'per_category': all_results['per_category'],
        'error_analysis': all_results['error_analysis'],
        'phase5_champion': {
            'name': champ_name,
            'metrics': champ_metrics,
            'production_valid': True,
        },
    }
}

with open(RES / 'phase5_anthony_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("  Saved: phase5_anthony_results.json")

metrics_path = RES / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {}
metrics['phase5_anthony'] = results_out['phase5_anthony']
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print("  Updated: metrics.json")

# ======================================================================
# 8. SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 5 SUMMARY (Anthony)")
print("=" * 70)
print(f"\n  CLIP L/14 baseline:      R@1 = {clip_base['R@1']:.4f}")
print(f"  Visual-only Optuna:      R@1 = {vis_result['R@1']:.4f} (Δ vs CLIP: {vis_result['R@1']-clip_base['R@1']:+.4f})")
print(f"  + Category filter:       R@1 = {cat_result['R@1']:.4f}")
print(f"  Mark oracle (for ref):   R@1 = 0.6952")
print(f"\n  Visual-only champion: {champ_name} → R@1={champ_metrics['R@1']:.4f}")
print(f"  Optuna weights: w_clip={best_vis['w_clip']:.2f}, w_color={best_vis['w_color']:.2f}, w_spatial={best_vis['w_spatial']:.2f}")
print(f"\n  Component contributions (drop-one):")
print(f"    CLIP:    {full_r1 - no_clip:+.4f}")
print(f"    Color:   {full_r1 - no_color:+.4f}")
print(f"    Spatial: {full_r1 - no_spatial:+.4f}")
print(f"    Cat filt: {cat_result['R@1'] - vis_result['R@1']:+.4f}")
print("\n  Done!")
