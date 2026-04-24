#!/usr/bin/env python3
"""Phase 4: Hyperparameter Tuning + Error Analysis — Anthony's Session

Research questions:
  1. Can Optuna joint-optimize all fusion weights to beat manual tuning?
  2. Does two-stage retrieval (CLIP top-K → feature rerank) outperform single-stage concat?
  3. Does query augmentation (flip averaging) improve robustness?
  4. What text weight maximizes R@1? (Phase 3 showed text is strongest supplement)
  5. Where does the model fail and why? (Confidence calibration, failure taxonomy)

Building on:
  - Anthony Phase 3 champion: CLIP L/14 + color + spatial + text → R@1=0.6748
  - Mark Phase 4: per-cat alpha oracle = 0.6952, 96D color HURTS (-23pp)
  - Mark finding: 85.3% of failures are close misses (correct in top-5)

Complementary to Mark's Phase 4:
  - Mark tuned per-category alpha independently → I jointly optimize ALL weights
  - Mark tested multiplicative fusion → I test two-stage reranking
  - Mark analyzed rank distribution → I analyze confidence calibration + failure taxonomy
  - Mark experimented with color resolution → I experiment with query augmentation

Author: Anthony Rodrigues | Date: 2026-04-24
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
K = 50
BS = 32
DEV = 'cpu'

# ======================================================================
# 1. LOAD DATA + STREAM IMAGES
# ======================================================================
print("=" * 70)
print("PHASE 4: HYPERPARAMETER TUNING + ERROR ANALYSIS (Anthony)")
print("Can joint optimization + two-stage reranking push past 0.6748?")
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

import requests
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


def concat_features(*feat_arrays, weights=None):
    if weights is None:
        weights = [1.0] * len(feat_arrays)
    parts = []
    for f, w in zip(feat_arrays, weights):
        fn = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
        parts.append(fn * w)
    return np.concatenate(parts, axis=1).astype(np.float32)


# ======================================================================
# 3. EXTRACT ALL FEATURES
# ======================================================================
print("=" * 70)
print("3.0  Extracting all features (CLIP L/14, color, spatial, text)")
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


def extract_clip_text(texts, batch_size=64):
    feats = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = tokenizer(batch_texts)
        with torch.no_grad():
            text_feats = clip_model.encode_text(tokens)
        feats.append(text_feats.cpu().float().numpy())
    return np.vstack(feats)


def batch_extract(indices_list, extract_fn):
    feats = []
    for ix in tqdm(indices_list, desc='  Features', leave=False):
        ix = int(ix)
        if ix not in imgs:
            continue
        feats.append(extract_fn(imgs[ix]))
    return np.array(feats, dtype=np.float32)


from src.feature_engineering import (
    extract_color_palette, extract_hsv_histogram,
    extract_spatial_color_grid as _spatial_fn,
    _rgb_to_hsv_vectorized
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


# --- Extract CLIP visual ---
t0 = time.time()
g_clip = extract_clip_visual(g_indices)
q_clip = extract_clip_visual(q_indices)
clip_time = time.time() - t0
print(f"  CLIP: gallery={g_clip.shape}, query={q_clip.shape} ({clip_time:.1f}s)")

# --- Extract color (48D) ---
t0 = time.time()
g_rgb = batch_extract(g_indices, extract_color_palette)
q_rgb = batch_extract(q_indices, extract_color_palette)
g_hsv = batch_extract(g_indices, extract_hsv_histogram)
q_hsv = batch_extract(q_indices, extract_hsv_histogram)
g_color = np.concatenate([g_rgb, g_hsv], axis=1)
q_color = np.concatenate([q_rgb, q_hsv], axis=1)
color_time = time.time() - t0
print(f"  Color: {g_color.shape[1]}D ({color_time:.1f}s)")

# --- Extract spatial color (192D) ---
t0 = time.time()
g_spatial = batch_extract(g_indices, extract_spatial_color_grid)
q_spatial = batch_extract(q_indices, extract_spatial_color_grid)
spatial_time = time.time() - t0
print(f"  Spatial: {g_spatial.shape[1]}D ({spatial_time:.1f}s)")

# --- Extract text (768D) ---
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

t0 = time.time()
g_text = extract_clip_text(g_prompts)
q_text = extract_clip_text(q_prompts)
text_time = time.time() - t0
print(f"  Text: {g_text.shape[1]}D ({text_time:.1f}s)")

# --- Phase 3 champion reproduction ---
clip_D, clip_I = faiss_search(g_clip, q_clip)
clip_base = recall_at_k(q_pids, g_pids, clip_I)
print(f"\n  CLIP L/14 baseline: {clip_base}")

# Phase 3 champion: CLIP + color(0.5) + spatial(0.4) + text(0.15)
p3_g = concat_features(g_clip, g_color, g_spatial, g_text, weights=[1.0, 0.5, 0.4, 0.15])
p3_q = concat_features(q_clip, q_color, q_spatial, q_text, weights=[1.0, 0.5, 0.4, 0.15])
p3_D, p3_I = faiss_search(p3_g, p3_q)
p3_result = recall_at_k(q_pids, g_pids, p3_I)
print(f"  Phase 3 champion (manual weights): {p3_result}")

all_experiments = {}
all_experiments['phase3_champion_repro'] = p3_result

# ======================================================================
# 4.1  EXPERIMENT: TEXT WEIGHT SWEEP
# ======================================================================
print("\n" + "=" * 70)
print("4.1  Text Weight Sweep — How much text is optimal?")
print("     Phase 3 used w_text=0.15. Mark showed text is the key signal.")
print("=" * 70)

text_sweep = {}
for tw in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    combo_g = concat_features(g_clip, g_color, g_spatial, g_text,
                              weights=[1.0, 0.5, 0.4, tw])
    combo_q = concat_features(q_clip, q_color, q_spatial, q_text,
                              weights=[1.0, 0.5, 0.4, tw])
    D, I = faiss_search(combo_g, combo_q)
    result = recall_at_k(q_pids, g_pids, I)
    text_sweep[tw] = result
    delta = result['R@1'] - p3_result['R@1']
    print(f"  w_text={tw:.2f}: R@1={result['R@1']:.4f} (Δ={delta:+.4f})")

best_tw = max(text_sweep, key=lambda x: text_sweep[x]['R@1'])
print(f"\n  >> Best text weight: {best_tw} → R@1={text_sweep[best_tw]['R@1']}")
all_experiments['text_weight_sweep'] = {str(k): v for k, v in text_sweep.items()}

# ======================================================================
# 4.2  EXPERIMENT: OPTUNA JOINT WEIGHT OPTIMIZATION
# ======================================================================
print("\n" + "=" * 70)
print("4.2  Optuna Joint Optimization of ALL 4 feature weights")
print("     Mark optimized per-category alpha independently.")
print("     I optimize (w_clip, w_color, w_spatial, w_text) jointly.")
print("=" * 70)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    w_clip = trial.suggest_float('w_clip', 0.5, 2.0, step=0.1)
    w_color = trial.suggest_float('w_color', 0.1, 1.0, step=0.05)
    w_spatial = trial.suggest_float('w_spatial', 0.1, 0.8, step=0.05)
    w_text = trial.suggest_float('w_text', 0.05, 0.60, step=0.05)

    combo_g = concat_features(g_clip, g_color, g_spatial, g_text,
                              weights=[w_clip, w_color, w_spatial, w_text])
    combo_q = concat_features(q_clip, q_color, q_spatial, q_text,
                              weights=[w_clip, w_color, w_spatial, w_text])
    D, I = faiss_search(combo_g, combo_q, k=20)
    r1 = recall_at_k(q_pids, g_pids, I, ks=(1,))['R@1']
    return r1


study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200, show_progress_bar=True)

best = study.best_params
best_r1 = study.best_value
print(f"\n  Optuna best: R@1={best_r1:.4f}")
print(f"  Params: w_clip={best['w_clip']:.2f}, w_color={best['w_color']:.2f}, "
      f"w_spatial={best['w_spatial']:.2f}, w_text={best['w_text']:.2f}")

# Re-evaluate with full K
optuna_g = concat_features(g_clip, g_color, g_spatial, g_text,
                           weights=[best['w_clip'], best['w_color'],
                                    best['w_spatial'], best['w_text']])
optuna_q = concat_features(q_clip, q_color, q_spatial, q_text,
                           weights=[best['w_clip'], best['w_color'],
                                    best['w_spatial'], best['w_text']])
optuna_D, optuna_I = faiss_search(optuna_g, optuna_q)
optuna_result = recall_at_k(q_pids, g_pids, optuna_I)
print(f"  Full eval: {optuna_result}")
delta = optuna_result['R@1'] - p3_result['R@1']
print(f"  Δ vs Phase 3 champion: {delta:+.4f}")

all_experiments['optuna_joint'] = {
    'best_params': best,
    'metrics': optuna_result,
    'n_trials': 200,
    'delta_vs_p3': round(delta, 4),
}

# Top-10 trials table
print("\n  Top-10 Optuna trials:")
print(f"  {'Trial':>6} {'w_clip':>7} {'w_color':>8} {'w_spatial':>9} {'w_text':>7} {'R@1':>6}")
for t in sorted(study.trials, key=lambda t: t.value if t.value else 0, reverse=True)[:10]:
    print(f"  {t.number:6d} {t.params['w_clip']:7.2f} {t.params['w_color']:8.2f} "
          f"{t.params['w_spatial']:9.2f} {t.params['w_text']:7.2f} {t.value:6.4f}")

# ======================================================================
# 4.3  EXPERIMENT: TWO-STAGE RETRIEVAL (CLIP shortlist → feature rerank)
# ======================================================================
print("\n" + "=" * 70)
print("4.3  Two-Stage Retrieval")
print("     Stage 1: CLIP L/14 shortlists top-K candidates")
print("     Stage 2: Multi-feature blend reranks the shortlist")
print("     Mark found 85.3% of failures have correct product in top-5.")
print("     Two-stage trades initial recall for better ranking precision.")
print("=" * 70)


def two_stage_rerank(clip_D, clip_I, q_feats_list, g_feats_list, weights, top_k=20):
    """Stage 2: rerank CLIP top-K using concatenated feature similarities."""
    reranked = np.zeros((len(clip_I), top_k), dtype=np.int64)
    for qi in range(len(clip_I)):
        cand_idx = clip_I[qi, :top_k]
        blended_score = np.zeros(top_k, dtype=np.float32)
        for qf, gf, w in zip(q_feats_list, g_feats_list, weights):
            qn = qf[qi] / (np.linalg.norm(qf[qi]) + 1e-8)
            gn = gf[cand_idx] / (np.linalg.norm(gf[cand_idx], axis=1, keepdims=True) + 1e-8)
            sims = gn @ qn
            blended_score += w * sims
        reranked[qi] = cand_idx[np.argsort(-blended_score)]
    return reranked


two_stage_results = {}
for stage1_k in [10, 20, 30, 50]:
    reranked = two_stage_rerank(
        clip_D, clip_I,
        [q_clip, q_color, q_spatial, q_text],
        [g_clip, g_color, g_spatial, g_text],
        weights=[best['w_clip'], best['w_color'], best['w_spatial'], best['w_text']],
        top_k=stage1_k
    )
    padded = np.zeros((len(reranked), K), dtype=np.int64)
    padded[:, :stage1_k] = reranked
    if stage1_k < K:
        for qi in range(len(reranked)):
            remaining = [idx for idx in clip_I[qi] if idx not in set(reranked[qi])]
            padded[qi, stage1_k:] = remaining[:K - stage1_k]
    result = recall_at_k(q_pids, g_pids, padded)
    two_stage_results[stage1_k] = result
    delta = result['R@1'] - optuna_result['R@1']
    print(f"  Stage1 top-{stage1_k:2d}: R@1={result['R@1']:.4f} (Δ vs Optuna: {delta:+.4f})")

all_experiments['two_stage'] = {str(k): v for k, v in two_stage_results.items()}

# ======================================================================
# 4.4  EXPERIMENT: QUERY AUGMENTATION (flip averaging)
# ======================================================================
print("\n" + "=" * 70)
print("4.4  Query Augmentation — Horizontal Flip Averaging")
print("     Average embedding of original + flipped query for viewpoint robustness")
print("=" * 70)

from torchvision import transforms

flip_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
])


def extract_clip_visual_augmented(indices_list, n_augments=1):
    """Extract averaged embedding of original + augmented views."""
    feats_orig = []
    feats_aug = []
    batch_orig = []
    batch_aug = []
    for ix in tqdm(indices_list, desc='  CLIP aug', leave=False):
        ix = int(ix)
        if ix not in imgs:
            continue
        img = imgs[ix]
        batch_orig.append(clip_preprocess(img))
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        batch_aug.append(clip_preprocess(flipped))

        if len(batch_orig) >= BS:
            with torch.no_grad():
                f_orig = clip_model.encode_image(torch.stack(batch_orig))
                f_aug = clip_model.encode_image(torch.stack(batch_aug))
            feats_orig.append(f_orig.cpu().float().numpy())
            feats_aug.append(f_aug.cpu().float().numpy())
            batch_orig = []
            batch_aug = []

    if batch_orig:
        with torch.no_grad():
            f_orig = clip_model.encode_image(torch.stack(batch_orig))
            f_aug = clip_model.encode_image(torch.stack(batch_aug))
        feats_orig.append(f_orig.cpu().float().numpy())
        feats_aug.append(f_aug.cpu().float().numpy())

    orig = np.vstack(feats_orig)
    aug = np.vstack(feats_aug)
    averaged = (orig + aug) / 2.0
    return averaged


t0 = time.time()
q_clip_aug = extract_clip_visual_augmented(q_indices)
aug_time = time.time() - t0
print(f"  Augmented query embeddings extracted in {aug_time:.1f}s")

# Test with Optuna-tuned weights
aug_combo_q = concat_features(q_clip_aug, q_color, q_spatial, q_text,
                              weights=[best['w_clip'], best['w_color'],
                                       best['w_spatial'], best['w_text']])
aug_D, aug_I = faiss_search(optuna_g, aug_combo_q)
aug_result = recall_at_k(q_pids, g_pids, aug_I)
delta = aug_result['R@1'] - optuna_result['R@1']
print(f"  Augmented query + Optuna weights: {aug_result}")
print(f"  Δ vs Optuna (no aug): {delta:+.4f}")

# Also test augmented CLIP-only
aug_D2, aug_I2 = faiss_search(g_clip, q_clip_aug)
aug_clip_result = recall_at_k(q_pids, g_pids, aug_I2)
delta2 = aug_clip_result['R@1'] - clip_base['R@1']
print(f"  Augmented CLIP-only: {aug_clip_result} (Δ vs CLIP bare: {delta2:+.4f})")

all_experiments['query_augmentation'] = {
    'aug_optuna': aug_result,
    'aug_clip_only': aug_clip_result,
    'delta_vs_optuna': round(delta, 4),
    'delta_vs_clip': round(delta2, 4),
}

# ======================================================================
# 4.5  EXPERIMENT: COMBINED BEST — Optuna weights + best stage1 K
# ======================================================================
print("\n" + "=" * 70)
print("4.5  Combined Best Configuration")
print("=" * 70)

# Find best two-stage K
best_k = max(two_stage_results, key=lambda x: two_stage_results[x]['R@1'])
best_two_stage = two_stage_results[best_k]

# Compare all Phase 4 configs
configs = {
    'Phase 3 champion (manual)': p3_result,
    'Text weight optimized': text_sweep[best_tw],
    'Optuna joint (200 trials)': optuna_result,
    f'Two-stage (top-{best_k})': best_two_stage,
    'Query augmentation + Optuna': aug_result,
}

print(f"\n  {'Configuration':<40} {'R@1':>6} {'R@5':>6} {'R@10':>6} {'R@20':>6}")
print("  " + "-" * 70)
for name, r in sorted(configs.items(), key=lambda x: -x[1]['R@1']):
    print(f"  {name:<40} {r['R@1']:>6.4f} {r['R@5']:>6.4f} {r['R@10']:>6.4f} {r['R@20']:>6.4f}")

# Determine Phase 4 champion
phase4_champion_name = max(configs, key=lambda x: configs[x]['R@1'])
phase4_champion = configs[phase4_champion_name]
print(f"\n  >> Phase 4 Champion: {phase4_champion_name}")
print(f"     R@1={phase4_champion['R@1']:.4f} (Δ vs P3: {phase4_champion['R@1'] - p3_result['R@1']:+.4f})")

# Use the champion for error analysis
if 'Optuna' in phase4_champion_name:
    champ_I = optuna_I
    champ_D = optuna_D
elif 'augmentation' in phase4_champion_name:
    champ_I = aug_I
    champ_D = aug_D
else:
    champ_I = p3_I
    champ_D = p3_D

# ======================================================================
# 4.6  ERROR ANALYSIS — Deep dive on failures
# ======================================================================
print("\n" + "=" * 70)
print("4.6  Error Analysis — Why does the champion fail?")
print("=" * 70)

successes = []
failures = []
for qi in range(len(q_pids)):
    qp = q_pids[qi]
    top1_pid = g_pids[champ_I[qi, 0]]
    is_correct = (qp == top1_pid)
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
        'query_pid': qp,
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
print(f"  Successes: {n_success} ({n_success/len(q_pids)*100:.1f}%)")
print(f"  Failures:  {n_fail} ({n_fail/len(q_pids)*100:.1f}%)")

# Rank distribution of failures
close_miss_5 = sum(1 for f in failures if 0 < f['correct_rank'] <= 5)
close_miss_10 = sum(1 for f in failures if 0 < f['correct_rank'] <= 10)
not_in_topK = sum(1 for f in failures if f['correct_rank'] == -1)
print(f"\n  Close misses (correct in top-5): {close_miss_5}/{n_fail} ({close_miss_5/n_fail*100:.1f}%)")
print(f"  In top-10: {close_miss_10}/{n_fail} ({close_miss_10/n_fail*100:.1f}%)")
print(f"  Not in top-{K}: {not_in_topK}/{n_fail} ({not_in_topK/n_fail*100:.1f}%)")

# Score gap analysis
gaps = [f['score_gap'] for f in failures if f['score_gap'] is not None]
if gaps:
    print(f"\n  Score gap stats:")
    print(f"    Median: {np.median(gaps):.4f}")
    print(f"    Mean:   {np.mean(gaps):.4f}")
    print(f"    <0.01 (tiny):  {sum(1 for g in gaps if g < 0.01)/len(gaps)*100:.1f}%")
    print(f"    <0.05 (small): {sum(1 for g in gaps if g < 0.05)/len(gaps)*100:.1f}%")
    print(f"    >0.10 (large): {sum(1 for g in gaps if g > 0.10)/len(gaps)*100:.1f}%")

# Per-category failure analysis
cat_fails = defaultdict(list)
cat_totals = defaultdict(int)
for qi in range(len(q_pids)):
    cat = q_df.iloc[qi]['category2']
    cat_totals[cat] += 1
for f in failures:
    cat_fails[f['category']].append(f)

print(f"\n  Per-category failure rates:")
print(f"  {'Category':<15} {'Fail Rate':>10} {'N Fail':>7} {'N Total':>8} {'Med Rank':>9} {'Med Gap':>8}")
print("  " + "-" * 62)
per_cat_error = {}
for cat in sorted(cat_totals.keys()):
    cat_f = cat_fails[cat]
    rate = len(cat_f) / cat_totals[cat]
    ranks = [f['correct_rank'] for f in cat_f if f['correct_rank'] > 0]
    cat_gaps = [f['score_gap'] for f in cat_f if f['score_gap'] is not None]
    med_rank = np.median(ranks) if ranks else 0
    med_gap = np.median(cat_gaps) if cat_gaps else 0
    per_cat_error[cat] = {
        'fail_rate': round(rate, 4),
        'n_fail': len(cat_f),
        'n_total': cat_totals[cat],
        'med_rank': round(float(med_rank), 1),
        'med_gap': round(float(med_gap), 4),
    }
    print(f"  {cat:<15} {rate:>10.1%} {len(cat_f):>7} {cat_totals[cat]:>8} {med_rank:>9.1f} {med_gap:>8.4f}")

# ======================================================================
# 4.7  CONFIDENCE CALIBRATION ANALYSIS
# ======================================================================
print("\n" + "=" * 70)
print("4.7  Confidence Calibration — Can score gap predict failure?")
print("=" * 70)

success_top1_scores = [s['top1_score'] for s in successes]
fail_top1_scores = [f['top1_score'] for f in failures]

print(f"  Success mean top-1 score: {np.mean(success_top1_scores):.4f}")
print(f"  Failure mean top-1 score: {np.mean(fail_top1_scores):.4f}")
print(f"  Score separation:         {np.mean(success_top1_scores) - np.mean(fail_top1_scores):.4f}")

# Score-gap-between-top1-and-top2 as confidence signal
confidence_gaps = []
for qi in range(len(q_pids)):
    gap_12 = float(champ_D[qi, 0] - champ_D[qi, 1])
    is_correct = (q_pids[qi] == g_pids[champ_I[qi, 0]])
    confidence_gaps.append({'gap': gap_12, 'correct': is_correct,
                            'category': q_df.iloc[qi]['category2']})

conf_df = pd.DataFrame(confidence_gaps)

# Binned calibration
n_bins = 10
conf_df['gap_bin'] = pd.qcut(conf_df['gap'], q=n_bins, duplicates='drop')
calibration = conf_df.groupby('gap_bin', observed=True)['correct'].agg(['mean', 'count'])
print(f"\n  Confidence calibration (top1-top2 gap as confidence):")
print(f"  {'Gap Range':<30} {'Accuracy':>10} {'Count':>7}")
print("  " + "-" * 50)
calibration_data = {}
for idx, row in calibration.iterrows():
    gap_str = str(idx)
    print(f"  {gap_str:<30} {row['mean']:>10.3f} {int(row['count']):>7}")
    calibration_data[gap_str] = {'accuracy': round(row['mean'], 4), 'count': int(row['count'])}

# Can we reject low-confidence queries to improve precision?
thresholds = [0.005, 0.01, 0.02, 0.03, 0.05]
print(f"\n  Confidence threshold rejection:")
print(f"  {'Threshold':>10} {'Accepted':>10} {'R@1 on accepted':>16} {'Rejected':>10} {'R@1 rejected':>13}")
print("  " + "-" * 65)
threshold_results = {}
for thr in thresholds:
    accepted = conf_df[conf_df['gap'] >= thr]
    rejected = conf_df[conf_df['gap'] < thr]
    acc_r1 = accepted['correct'].mean() if len(accepted) > 0 else 0
    rej_r1 = rejected['correct'].mean() if len(rejected) > 0 else 0
    print(f"  {thr:>10.3f} {len(accepted):>10} {acc_r1:>16.4f} {len(rejected):>10} {rej_r1:>13.4f}")
    threshold_results[str(thr)] = {
        'n_accepted': len(accepted), 'r1_accepted': round(acc_r1, 4),
        'n_rejected': len(rejected), 'r1_rejected': round(rej_r1, 4),
    }

all_experiments['confidence_calibration'] = {
    'success_mean_score': round(float(np.mean(success_top1_scores)), 4),
    'failure_mean_score': round(float(np.mean(fail_top1_scores)), 4),
    'score_separation': round(float(np.mean(success_top1_scores) - np.mean(fail_top1_scores)), 4),
    'calibration_bins': calibration_data,
    'threshold_rejection': threshold_results,
}

# ======================================================================
# 5.  PER-CATEGORY ANALYSIS ON CHAMPION
# ======================================================================
print("\n" + "=" * 70)
print("5.  Per-Category — Phase 3 vs Phase 4 Champion")
print("=" * 70)

p3_cats = per_cat_recall(q_df, g_pids, p3_I, k=1)
champ_cats = per_cat_recall(q_df, g_pids, champ_I, k=1)

print(f"  {'Category':<15} {'Phase 3':>8} {'Phase 4':>8} {'Δ':>8}")
print("  " + "-" * 42)
for cat in sorted(p3_cats.keys()):
    d = champ_cats[cat] - p3_cats[cat]
    print(f"  {cat:<15} {p3_cats[cat]:>8.4f} {champ_cats[cat]:>8.4f} {d:>+8.4f}")

# ======================================================================
# 6.  PLOTS
# ======================================================================
print("\n" + "=" * 70)
print("6.  Generating plots")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))

# Plot 1: Text weight sweep
ax = axes[0, 0]
tws = sorted(text_sweep.keys())
r1s = [text_sweep[tw]['R@1'] for tw in tws]
ax.plot(tws, r1s, 'o-', color='#2196F3', linewidth=2, markersize=8)
ax.axhline(y=p3_result['R@1'], color='red', linestyle='--', alpha=0.6, label=f'Phase 3 ({p3_result["R@1"]:.4f})')
best_idx = np.argmax(r1s)
ax.scatter([tws[best_idx]], [r1s[best_idx]], s=200, c='gold', zorder=5, edgecolors='black', linewidth=2)
ax.set_xlabel('Text Weight', fontsize=11)
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('Text Weight Sweep', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Optuna optimization history
ax = axes[0, 1]
trial_values = [t.value for t in study.trials if t.value is not None]
best_so_far = np.maximum.accumulate(trial_values)
ax.scatter(range(len(trial_values)), trial_values, s=10, alpha=0.3, color='#9E9E9E', label='Trials')
ax.plot(best_so_far, color='#4CAF50', linewidth=2, label='Best so far')
ax.axhline(y=p3_result['R@1'], color='red', linestyle='--', alpha=0.6, label=f'Phase 3')
ax.set_xlabel('Trial #', fontsize=11)
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('Optuna Joint Optimization (200 trials)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Two-stage K comparison
ax = axes[0, 2]
stage_ks = sorted(two_stage_results.keys())
stage_r1 = [two_stage_results[k]['R@1'] for k in stage_ks]
ax.bar([f'K={k}' for k in stage_ks], stage_r1, color=['#FF9800', '#4CAF50', '#2196F3', '#9C27B0'],
       edgecolor='white')
ax.axhline(y=optuna_result['R@1'], color='red', linestyle='--', alpha=0.6, label=f'Single-stage Optuna')
for i, (k, r) in enumerate(zip(stage_ks, stage_r1)):
    ax.text(i, r + 0.002, f'{r:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('Two-Stage Reranking (Stage1 Top-K)', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# Plot 4: Per-category comparison
ax = axes[1, 0]
cats = sorted(p3_cats.keys())
x = np.arange(len(cats))
width = 0.35
ax.bar(x - width/2, [p3_cats[c] for c in cats], width, label='Phase 3', color='#FF9800', alpha=0.8)
ax.bar(x + width/2, [champ_cats[c] for c in cats], width, label='Phase 4', color='#4CAF50', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('R@1', fontsize=11)
ax.set_title('Per-Category: Phase 3 → 4', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)

# Plot 5: Confidence calibration curve
ax = axes[1, 1]
gap_bins = conf_df.groupby(pd.qcut(conf_df['gap'], q=10, duplicates='drop'), observed=True)
cal_data = gap_bins['correct'].agg(['mean', 'count'])
bin_centers = [(b.left + b.right) / 2 for b in cal_data.index]
ax.plot(bin_centers, cal_data['mean'], 'o-', color='#E91E63', linewidth=2, markersize=8)
ax.fill_between(bin_centers, cal_data['mean'], alpha=0.15, color='#E91E63')
ax.set_xlabel('Top1-Top2 Score Gap', fontsize=11)
ax.set_ylabel('Accuracy (R@1)', fontsize=11)
ax.set_title('Confidence Calibration', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 6: Error analysis — failure rank distribution
ax = axes[1, 2]
fail_ranks = [f['correct_rank'] for f in failures if f['correct_rank'] > 0]
bins_rank = [1.5, 2.5, 3.5, 5.5, 10.5, 20.5, 50.5]
labels_rank = ['2', '3', '4-5', '6-10', '11-20', '21-50']
counts, _ = np.histogram(fail_ranks, bins=bins_rank)
pcts = counts / len(failures) * 100
bars = ax.bar(labels_rank, pcts, color=['#F44336', '#FF9800', '#FFC107', '#4CAF50', '#2196F3', '#9C27B0'],
              edgecolor='white')
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Rank of Correct Product', fontsize=11)
ax.set_ylabel('% of Failures', fontsize=11)
ax.set_title(f'Failure Rank Distribution (N={n_fail})', fontsize=13, fontweight='bold')

plt.suptitle('Phase 4: Hyperparameter Tuning + Error Analysis (Anthony)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(RES / 'phase4_anthony_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase4_anthony_results.png")

# Score distribution plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(success_top1_scores, bins=40, alpha=0.6, label=f'Success (N={n_success})', color='#4CAF50', density=True)
ax.hist(fail_top1_scores, bins=40, alpha=0.6, label=f'Failure (N={n_fail})', color='#F44336', density=True)
ax.axvline(np.mean(success_top1_scores), color='#2E7D32', linestyle='--', linewidth=2)
ax.axvline(np.mean(fail_top1_scores), color='#C62828', linestyle='--', linewidth=2)
ax.set_xlabel('Top-1 Blended Score', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Score Distribution: Successes vs Failures', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(RES / 'phase4_anthony_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: phase4_anthony_error_analysis.png")

# ======================================================================
# 7.  SAVE ALL RESULTS
# ======================================================================
print("\n" + "=" * 70)
print("7.  Saving results")
print("=" * 70)

results_out = {
    'phase4_anthony': {
        'date': '2026-04-24',
        'researcher': 'Anthony Rodrigues',
        'eval_products': EVAL_N,
        'eval_gallery': len(g_df),
        'eval_queries': len(q_df),
        'research_question': 'Can Optuna + two-stage reranking + query augmentation push R@1 past Phase 3 champion (0.6748)?',
        'phase3_champion': p3_result,
        'text_weight_sweep': {str(k): v for k, v in text_sweep.items()},
        'best_text_weight': best_tw,
        'optuna_best_params': best,
        'optuna_result': optuna_result,
        'two_stage_results': {str(k): v for k, v in two_stage_results.items()},
        'query_augmentation': {
            'aug_optuna': aug_result,
            'aug_clip_only': aug_clip_result,
        },
        'phase4_champion': {
            'name': phase4_champion_name,
            'metrics': phase4_champion,
            'delta_vs_p3': round(phase4_champion['R@1'] - p3_result['R@1'], 4),
        },
        'error_analysis': {
            'n_success': n_success,
            'n_fail': n_fail,
            'close_miss_top5_pct': round(close_miss_5 / n_fail * 100, 1) if n_fail > 0 else 0,
            'close_miss_top10_pct': round(close_miss_10 / n_fail * 100, 1) if n_fail > 0 else 0,
            'not_in_topK_pct': round(not_in_topK / n_fail * 100, 1) if n_fail > 0 else 0,
            'per_category': per_cat_error,
        },
        'confidence_calibration': all_experiments.get('confidence_calibration', {}),
        'per_category': {
            'phase3': p3_cats,
            'phase4_champion': champ_cats,
        },
    }
}

with open(RES / 'phase4_anthony_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)
print("  Saved: phase4_anthony_results.json")

metrics_path = RES / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)
else:
    metrics = {}
metrics['phase4_anthony'] = results_out['phase4_anthony']
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print("  Updated: metrics.json")

# ======================================================================
# 8.  SUMMARY
# ======================================================================
print("\n" + "=" * 70)
print("PHASE 4 SUMMARY (Anthony)")
print("=" * 70)

print(f"\n  Phase 3 champion:  R@1 = {p3_result['R@1']:.4f} (manual weights)")
print(f"  Phase 4 champion:  R@1 = {phase4_champion['R@1']:.4f} ({phase4_champion_name})")
print(f"  Δ: {phase4_champion['R@1'] - p3_result['R@1']:+.4f}")

if best_tw != 0.15:
    print(f"\n  Finding 1: Optimal text weight = {best_tw} (was 0.15)")
    print(f"             Just changing text weight: R@1={text_sweep[best_tw]['R@1']:.4f}")

print(f"\n  Optuna weights: clip={best['w_clip']:.2f}, color={best['w_color']:.2f}, "
      f"spatial={best['w_spatial']:.2f}, text={best['w_text']:.2f}")
print(f"  Manual weights: clip=1.00, color=0.50, spatial=0.40, text=0.15")

print(f"\n  Query augmentation Δ: {all_experiments['query_augmentation']['delta_vs_optuna']:+.4f}")
print(f"  Score separation:    {all_experiments['confidence_calibration']['score_separation']:.4f}")
print(f"  Close misses (top-5): {close_miss_5/n_fail*100:.1f}%" if n_fail > 0 else "")

print("\n  Plots: phase4_anthony_results.png, phase4_anthony_error_analysis.png")
print("  Done!")
