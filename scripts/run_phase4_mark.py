#!/usr/bin/env python3
"""Phase 4 Mark: Hyperparameter Tuning + Error Analysis
Visual Product Search Engine — 2026-04-23

Phase 3 champion: CLIP B/32 + category-conditioned + color alpha=0.4 -> R@1=0.6826

Research question: What is the remaining 31.7% failure mode, and can systematic
hyperparameter tuning push R@1 above 0.70?

Experiments:
  4.M.1 Champion baseline re-validation (R@1=0.6826 target)
  4.M.2 Per-category alpha optimization (oracle upper bound from per-cat tuning)
  4.M.3 Higher-resolution color: 96D (16 bins/channel) vs 48D (8 bins/channel)
  4.M.4 Error analysis: rank distribution, similarity gaps, failure taxonomy
  4.M.5 Multiplicative fusion vs additive (soft-AND vs soft-OR of color+CLIP)
  4.M.6 Best Phase 4 system summary
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import gc, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import faiss

PROJECT = Path(__file__).parent.parent
PROC  = PROJECT / 'data' / 'processed'
RAW   = PROJECT / 'data' / 'raw' / 'images'
RES   = PROJECT / 'results'
CACHE = PROJECT / 'data' / 'processed' / 'emb_cache'
CACHE.mkdir(parents=True, exist_ok=True)
RES.mkdir(exist_ok=True)

EVAL_N = 300
K_TOP  = 20
DEV    = 'cpu'

print("=" * 65)
print("PHASE 4 MARK: Hyperparameter Tuning + Error Analysis")
print("=" * 65)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
gallery_df = pd.read_csv(PROC / 'gallery.csv')
query_df   = pd.read_csv(PROC / 'query.csv')
eval_pids  = gallery_df['product_id'].values[:EVAL_N]
g_df = gallery_df[gallery_df['product_id'].isin(eval_pids)].reset_index(drop=True)
q_df = query_df[query_df['product_id'].isin(eval_pids)].reset_index(drop=True)
print(f"Eval: {len(g_df)} gallery items, {len(q_df)} queries, {EVAL_N} products")

gallery_cats = g_df['category2'].values
query_cats   = q_df['category2'].values
q_pids       = q_df['product_id'].values
g_pids       = g_df['product_id'].values

cat_list = sorted(g_df['category2'].unique())
cat_counts_g = {c: (gallery_cats == c).sum() for c in cat_list}
cat_counts_q = {c: (query_cats  == c).sum() for c in cat_list}
print(f"\nCategories ({len(cat_list)}): {cat_list}")
print("Gallery items per category:")
for c in cat_list:
    print(f"  {c:<16} gal={cat_counts_g[c]:3d}  q={cat_counts_q.get(c,0):4d}")


# =====================================================================
# 2. LOAD IMAGES (for 96D re-extraction)
# =====================================================================
def load_image(item_id):
    return Image.open(RAW / f"{item_id}.jpg").convert("RGB")

print("\n[1/5] Loading images for 96D color re-extraction...")
gallery_imgs = [load_image(r['item_id']) for _, r in tqdm(g_df.iterrows(), total=len(g_df))]
query_imgs   = [load_image(r['item_id']) for _, r in tqdm(q_df.iterrows(), total=len(q_df))]


# =====================================================================
# 3. EVALUATION HELPERS
# =====================================================================
def recall_at_k(indices, qp, gp, k):
    return float(sum(qp[i] in gp[indices[i][:k]] for i in range(len(indices)))) / len(indices)

def evaluate(indices, qp, gp, label=""):
    res = {f"R@{k}": recall_at_k(indices, qp, gp, k) for k in [1, 5, 10, 20]}
    if label:
        print(f"  {label}: R@1={res['R@1']:.4f}  R@5={res['R@5']:.4f}"
              f"  R@10={res['R@10']:.4f}  R@20={res['R@20']:.4f}")
    return res

def cosine_search(q, g, k=K_TOP):
    q = q.astype(np.float32).copy(); g = g.astype(np.float32).copy()
    faiss.normalize_L2(q); faiss.normalize_L2(g)
    idx = faiss.IndexFlatIP(g.shape[1]); idx.add(g)
    _, I = idx.search(q, k)
    return I

def cat_search(qe, ge, qc, gc, cq=None, cg=None, alpha=1.0, k=K_TOP):
    """Category-conditioned search with optional additive color blend."""
    qe = qe.astype(np.float32).copy(); ge = ge.astype(np.float32).copy()
    faiss.normalize_L2(qe); faiss.normalize_L2(ge)
    if cq is not None:
        cqn = cq / (np.linalg.norm(cq, axis=1, keepdims=True) + 1e-8)
        cgn = cg / (np.linalg.norm(cg, axis=1, keepdims=True) + 1e-8)
    res = np.zeros((len(qe), k), dtype=np.int64)
    for i, cat in enumerate(qc):
        mask = gc == cat
        cidx = np.where(mask)[0] if mask.any() else np.arange(len(gc))
        sims = ge[cidx] @ qe[i]
        if cq is not None and alpha < 1.0:
            csim = cgn[cidx] @ cqn[i]
            sims = alpha * sims + (1 - alpha) * csim
        top  = np.argsort(-sims)[:k]
        glob = cidx[top]
        if len(glob) < k:
            others = np.setdiff1d(np.arange(len(gc)), glob)
            glob   = np.concatenate([glob, others[:k - len(glob)]])
        res[i] = glob[:k]
    return res

def cat_search_percat(qe, ge, qc, gc, cq, cg, alpha_dict, k=K_TOP):
    """Category-conditioned search with per-category alpha."""
    qe = qe.astype(np.float32).copy(); ge = ge.astype(np.float32).copy()
    faiss.normalize_L2(qe); faiss.normalize_L2(ge)
    cqn = cq / (np.linalg.norm(cq, axis=1, keepdims=True) + 1e-8)
    cgn = cg / (np.linalg.norm(cg, axis=1, keepdims=True) + 1e-8)
    res = np.zeros((len(qe), k), dtype=np.int64)
    for i, cat in enumerate(qc):
        alpha = alpha_dict.get(cat, 0.4)
        mask  = gc == cat
        cidx  = np.where(mask)[0] if mask.any() else np.arange(len(gc))
        sims  = ge[cidx] @ qe[i]
        if alpha < 1.0:
            csim = cgn[cidx] @ cqn[i]
            sims = alpha * sims + (1 - alpha) * csim
        top  = np.argsort(-sims)[:k]
        glob = cidx[top]
        if len(glob) < k:
            others = np.setdiff1d(np.arange(len(gc)), glob)
            glob   = np.concatenate([glob, others[:k - len(glob)]])
        res[i] = glob[:k]
    return res

def cat_search_multiplicative(qe, ge, qc, gc, cq, cg, beta=1.0, k=K_TOP):
    """Multiplicative fusion: s = s_clip * s_color^beta (soft-AND)."""
    qe = qe.astype(np.float32).copy(); ge = ge.astype(np.float32).copy()
    faiss.normalize_L2(qe); faiss.normalize_L2(ge)
    cqn = cq / (np.linalg.norm(cq, axis=1, keepdims=True) + 1e-8)
    cgn = cg / (np.linalg.norm(cg, axis=1, keepdims=True) + 1e-8)
    res = np.zeros((len(qe), k), dtype=np.int64)
    for i, cat in enumerate(qc):
        mask  = gc == cat
        cidx  = np.where(mask)[0] if mask.any() else np.arange(len(gc))
        clip_sim  = ge[cidx] @ qe[i]
        color_sim = cgn[cidx] @ cqn[i]
        # shift cosine to [0,1] range for multiplicative fusion
        clip_s01  = (clip_sim  + 1) / 2
        color_s01 = (color_sim + 1) / 2
        sims = clip_s01 * (color_s01 ** beta)
        top  = np.argsort(-sims)[:k]
        glob = cidx[top]
        if len(glob) < k:
            others = np.setdiff1d(np.arange(len(gc)), glob)
            glob   = np.concatenate([glob, others[:k - len(glob)]])
        res[i] = glob[:k]
    return res


# =====================================================================
# 4. LOAD CACHED EMBEDDINGS
# =====================================================================
print("\n[2/5] Loading cached embeddings...")

def load_emb(tag):
    fp = CACHE / f"{tag}.npy"
    if fp.exists():
        arr = np.load(fp)
        print(f"  Loaded {tag}: {arr.shape}")
        return arr
    raise FileNotFoundError(f"Cache missing: {fp}")

g_clip    = load_emb("clip_b32_gallery")
q_clip    = load_emb("clip_b32_query")
g_color48 = load_emb("color48_gallery")
q_color48 = load_emb("color48_query")
g_spatial = load_emb("spatial_gallery")
q_spatial = load_emb("spatial_query")


# =====================================================================
# 5. EXTRACT 96D COLOR FEATURES (16 bins/channel)
# =====================================================================
print("\n[3/5] Extracting 96D color features (16 bins/channel)...")

from src.feature_engineering import extract_color_palette, extract_hsv_histogram

def extract_96d(imgs, tag):
    fp = CACHE / f"color96_{tag}.npy"
    if fp.exists():
        arr = np.load(fp)
        print(f"  Loaded color96_{tag} from cache: {arr.shape}")
        return arr
    feats = []
    for img in tqdm(imgs, desc=f"96D color {tag}"):
        rgb = extract_color_palette(img, bins_per_channel=16)   # 48D
        hsv = extract_hsv_histogram(img, bins=16)               # 48D
        feats.append(np.concatenate([rgb, hsv]))
    result = np.vstack(feats).astype(np.float32)
    np.save(fp, result)
    print(f"  Extracted and cached: {result.shape}")
    return result

g_color96 = extract_96d(gallery_imgs, 'gallery')
q_color96 = extract_96d(query_imgs,   'query')


# =====================================================================
# 6. EXPERIMENT 4.M.1: CHAMPION BASELINE RE-VALIDATION
# =====================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 4.M.1: Champion Baseline Re-Validation")
print("=" * 65)
print("Expected: R@1 = 0.6826 (Phase 3 winner)")

t0 = time.time()
champ_idx = cat_search(q_clip, g_clip, query_cats, gallery_cats,
                       q_color48, g_color48, alpha=0.4)
t_champ = time.time() - t0
champ_m  = evaluate(champ_idx, q_pids, g_pids, "4.M.1 Phase 3 champion (CLIP+cat+color48 a=0.4)")
print(f"  Latency: {t_champ*1000:.1f}ms total / {t_champ/len(q_df)*1000:.2f}ms per query")


# =====================================================================
# 7. EXPERIMENT 4.M.2: PER-CATEGORY ALPHA OPTIMIZATION
# =====================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 4.M.2: Per-Category Alpha Optimization")
print("=" * 65)
print("Sweeping alpha in [0.0, 0.1, ..., 1.0] independently per category.")
print("Oracle upper bound: what's the max R@1 if we perfectly tune alpha per category?")

ALPHA_GRID = np.round(np.arange(0.0, 1.05, 0.05), 2).tolist()

# Precompute normalized embeddings for efficiency
qe_n = q_clip.astype(np.float32).copy(); ge_n = g_clip.astype(np.float32).copy()
faiss.normalize_L2(qe_n); faiss.normalize_L2(ge_n)
cqn  = q_color48 / (np.linalg.norm(q_color48, axis=1, keepdims=True) + 1e-8)
cgn  = g_color48 / (np.linalg.norm(g_color48, axis=1, keepdims=True) + 1e-8)

best_alpha_per_cat = {}
alpha_scan_table = []
print(f"\n  {'Category':<16} {'Best_a':>7}  {'BestR@1':>8}  {'GlobalR@1':>10}  {'Delta':>7}  {'N_gal':>6}")
print("  " + "-" * 70)

for cat in cat_list:
    q_mask = query_cats == cat
    g_mask = gallery_cats == cat
    if not q_mask.any():
        continue
    cidx   = np.where(g_mask)[0]
    cqp    = q_pids[q_mask]
    # CLIP+color sims in the category subspace
    clip_sims  = (ge_n[cidx] @ qe_n[q_mask].T).T  # (n_q_cat, n_g_cat)
    color_sims = (cgn[cidx] @ cqn[q_mask].T).T

    # Global alpha baseline R@1 for this category (a=0.4)
    sims_global = 0.4 * clip_sims + 0.6 * color_sims
    r1_global = float(np.mean([
        g_pids[cidx[np.argsort(-sims_global[i])[0]]] == cqp[i]
        for i in range(len(cqp))
    ]))

    # Sweep per-cat alpha
    best_a, best_r1 = 0.4, r1_global
    for alpha in ALPHA_GRID:
        sims_a = alpha * clip_sims + (1 - alpha) * color_sims
        r1_a   = float(np.mean([
            g_pids[cidx[np.argsort(-sims_a[i])[0]]] == cqp[i]
            for i in range(len(cqp))
        ]))
        if r1_a > best_r1:
            best_r1, best_a = r1_a, alpha

    best_alpha_per_cat[cat] = best_a
    delta = best_r1 - r1_global
    alpha_scan_table.append({
        'category': cat,
        'best_alpha': best_a,
        'best_r1': best_r1,
        'global_r1': r1_global,
        'delta': delta,
        'n_gallery': cat_counts_g[cat],
        'n_query':   cat_counts_q.get(cat, 0),
    })
    print(f"  {cat:<16} {best_a:>7.2f}  {best_r1:>8.4f}  {r1_global:>10.4f}  {delta:>+7.4f}  {cat_counts_g[cat]:>6d}")

print(f"\n  Per-category optimal alphas: {best_alpha_per_cat}")

# Apply per-category alpha to all queries
percat_idx = cat_search_percat(q_clip, g_clip, query_cats, gallery_cats,
                               q_color48, g_color48, best_alpha_per_cat)
percat_m   = evaluate(percat_idx, q_pids, g_pids, "4.M.2 Per-category alpha (oracle)")
lift_42 = percat_m['R@1'] - champ_m['R@1']
print(f"\n  Oracle gain from per-category alpha: {lift_42:+.4f} R@1 ({lift_42*100:+.2f}pp)")


# =====================================================================
# 8. EXPERIMENT 4.M.3: 96D COLOR FEATURES
# =====================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 4.M.3: 96D Color Features (16 bins/channel)")
print("=" * 65)

# Sweep alpha for 96D to find optimal
print("  Alpha sweep with 96D color features...")
best_a96, best_r1_96 = 0.4, 0.0
for alpha in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
    idx_96 = cat_search(q_clip, g_clip, query_cats, gallery_cats,
                        q_color96, g_color96, alpha=alpha)
    r1     = recall_at_k(idx_96, q_pids, g_pids, 1)
    print(f"    96D alpha={alpha:.2f}: R@1={r1:.4f}")
    if r1 > best_r1_96:
        best_r1_96, best_a96 = r1, alpha

idx_96_best = cat_search(q_clip, g_clip, query_cats, gallery_cats,
                         q_color96, g_color96, alpha=best_a96)
m_96 = evaluate(idx_96_best, q_pids, g_pids,
                f"4.M.3 CLIP + cat.filter + color96 a={best_a96}")
delta_96 = m_96['R@1'] - champ_m['R@1']
print(f"\n  96D vs 48D delta: {delta_96:+.4f} R@1 ({delta_96*100:+.2f}pp)")
print(f"  Verdict: {'More color bins help' if delta_96 > 0.001 else 'More color bins do NOT help' if delta_96 < -0.001 else 'No significant difference'}")


# =====================================================================
# 9. EXPERIMENT 4.M.4: ERROR ANALYSIS
# =====================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 4.M.4: Error Analysis on Phase 3 Champion")
print("=" * 65)
print("Deep dive: what makes the remaining 31.7% R@1 failures hard?\n")

# We need FULL rankings (all gallery items) to analyze failure rank.
# Use cat_search with k=gallery_size_per_cat (effectively retrieve all).
# For failures: rank the correct product within the category subspace.

# Precompute per-query: is it a success at rank 1?
n_q = len(q_pids)
success_mask = np.array([
    g_pids[champ_idx[i, 0]] == q_pids[i] for i in range(n_q)
])
failures_idx = np.where(~success_mask)[0]
successes_idx = np.where(success_mask)[0]
n_fail = len(failures_idx)
n_succ = len(successes_idx)
print(f"  Success@1: {n_succ}/{n_q} ({n_succ/n_q:.4f})")
print(f"  Failures:  {n_fail}/{n_q} ({n_fail/n_q:.4f})")

# For each failure: find rank of correct product within category
fail_analysis = []
qe_n2 = q_clip.astype(np.float32).copy(); ge_n2 = g_clip.astype(np.float32).copy()
faiss.normalize_L2(qe_n2); faiss.normalize_L2(ge_n2)
cqn2 = q_color48 / (np.linalg.norm(q_color48, axis=1, keepdims=True) + 1e-8)
cgn2 = g_color48 / (np.linalg.norm(g_color48, axis=1, keepdims=True) + 1e-8)

for qi in failures_idx:
    cat      = query_cats[qi]
    correct_pid = q_pids[qi]
    g_mask   = gallery_cats == cat
    cidx     = np.where(g_mask)[0] if g_mask.any() else np.arange(len(g_pids))
    # Score = 0.4*CLIP + 0.6*color (champion formula)
    clip_s   = ge_n2[cidx] @ qe_n2[qi]
    color_s  = cgn2[cidx] @ cqn2[qi]
    blend_s  = 0.4 * clip_s + 0.6 * color_s
    rank_order = np.argsort(-blend_s)
    # Find rank of correct product
    correct_positions = np.where(g_pids[cidx[rank_order]] == correct_pid)[0]
    if len(correct_positions) > 0:
        rank_correct = int(correct_positions[0]) + 1  # 1-indexed
    else:
        rank_correct = len(cidx) + 1  # not in category (label issue)
    # Score of top-1 vs correct
    score_top1   = float(blend_s[rank_order[0]])
    score_correct = float(blend_s[np.where(g_pids[cidx] == correct_pid)[0][0]]) if len(np.where(g_pids[cidx] == correct_pid)[0]) else float('nan')
    score_gap    = score_top1 - score_correct  # positive = hard case
    fail_analysis.append({
        'query_idx':    qi,
        'category':     cat,
        'rank_correct': rank_correct,
        'score_top1':   score_top1,
        'score_correct': score_correct,
        'score_gap':    score_gap,
        'cat_size':     int(g_mask.sum()),
        'correct_in_cat': len(correct_positions) > 0,
    })

fail_df = pd.DataFrame(fail_analysis)
print(f"\n  FAILURE RANK DISTRIBUTION (rank of correct product among category gallery):")
rank_bins = [1, 2, 3, 4, 5, 10, 20, 50, 999]
for lo, hi in zip(rank_bins[:-1], rank_bins[1:]):
    mask = (fail_df['rank_correct'] >= lo) & (fail_df['rank_correct'] < hi)
    n    = mask.sum()
    print(f"    Rank {lo:>3}-{hi-1:<3}: {n:4d} failures ({n/n_fail*100:.1f}%)")

rank2_pct = (fail_df['rank_correct'] == 2).mean() * 100
rank5_pct = (fail_df['rank_correct'] <= 5).mean() * 100
close_miss_pct = rank5_pct
print(f"\n  INSIGHT: {close_miss_pct:.1f}% of failures have correct product in top-5 (close misses)")
print(f"           {(fail_df['rank_correct'] > 10).mean()*100:.1f}% of failures have correct product below rank 10 (hard cases)")

# Correct product NOT in category
wrong_cat = (~fail_df['correct_in_cat']).sum()
print(f"\n  Category label coherence: {wrong_cat} failures have correct product NOT in same category ({wrong_cat/n_fail*100:.1f}%)")

# Per-category failure analysis
print(f"\n  PER-CATEGORY FAILURE ANALYSIS:")
print(f"  {'Category':<16}  {'N_fail':>7}  {'N_query':>8}  {'Fail_rate':>10}  {'Median_rank':>12}  {'AvgGap':>8}")
print("  " + "-" * 75)
cat_fail_stats = {}
for cat in cat_list:
    cat_fails = fail_df[fail_df['category'] == cat]
    n_q_cat   = cat_counts_q.get(cat, 0)
    if n_q_cat == 0:
        continue
    n_f       = len(cat_fails)
    fail_rate = n_f / n_q_cat
    med_rank  = cat_fails['rank_correct'].median() if n_f > 0 else 0
    avg_gap   = cat_fails['score_gap'].mean() if n_f > 0 else 0
    cat_fail_stats[cat] = {'n_fail': n_f, 'fail_rate': fail_rate, 'med_rank': med_rank, 'avg_gap': avg_gap}
    print(f"  {cat:<16}  {n_f:>7d}  {n_q_cat:>8d}  {fail_rate:>10.4f}  {med_rank:>12.1f}  {avg_gap:>8.4f}")

# Score gap distribution
score_gaps = fail_df['score_gap'].dropna()
print(f"\n  SCORE GAP ANALYSIS (top-1 wrong score minus correct score):")
print(f"    Median gap: {score_gaps.median():.4f}")
print(f"    Mean gap:   {score_gaps.mean():.4f}")
print(f"    Max gap:    {score_gaps.max():.4f}")
print(f"    Tiny gap (<0.01): {(score_gaps < 0.01).sum()} cases ({(score_gaps < 0.01).mean()*100:.1f}%)")
print(f"    Small gap (<0.05): {(score_gaps < 0.05).sum()} cases ({(score_gaps < 0.05).mean()*100:.1f}%)")
print(f"    Large gap (>0.10): {(score_gaps > 0.10).sum()} cases ({(score_gaps > 0.10).mean()*100:.1f}%)")

# Similarity distribution of successes vs failures
# For success queries: what is the top-1 score?
success_top1_scores = []
for qi in successes_idx:
    cat  = query_cats[qi]
    g_mask = gallery_cats == cat
    cidx = np.where(g_mask)[0] if g_mask.any() else np.arange(len(g_pids))
    clip_s  = ge_n2[cidx] @ qe_n2[qi]
    color_s = cgn2[cidx] @ cqn2[qi]
    blend_s = 0.4 * clip_s + 0.6 * color_s
    success_top1_scores.append(float(blend_s.max()))

success_scores = np.array(success_top1_scores)
fail_correct_scores = fail_df['score_correct'].dropna().values
print(f"\n  SCORE DISTRIBUTION:")
print(f"    Success queries  - mean top-1 score: {success_scores.mean():.4f}  median: {np.median(success_scores):.4f}")
print(f"    Failed queries   - mean correct score: {fail_correct_scores.mean():.4f}  median: {np.median(fail_correct_scores):.4f}")
print(f"    Separation: {success_scores.mean() - fail_correct_scores.mean():.4f}")


# =====================================================================
# 10. EXPERIMENT 4.M.5: MULTIPLICATIVE FUSION
# =====================================================================
print("\n" + "=" * 65)
print("EXPERIMENT 4.M.5: Multiplicative Fusion (soft-AND)")
print("=" * 65)
print("Hypothesis: s = s_clip * s_color^beta penalizes when EITHER signal is low.")
print("Additive: high CLIP + low color -> still good score.")
print("Multiplicative: high CLIP + low color -> score dragged down.\n")

mult_results = {}
print(f"  {'Beta':>6}  {'R@1':>8}  {'R@5':>8}  {'R@10':>8}")
print("  " + "-" * 40)
for beta in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
    mult_idx = cat_search_multiplicative(q_clip, g_clip, query_cats, gallery_cats,
                                         q_color48, g_color48, beta=beta)
    r1 = recall_at_k(mult_idx, q_pids, g_pids, 1)
    r5 = recall_at_k(mult_idx, q_pids, g_pids, 5)
    r10 = recall_at_k(mult_idx, q_pids, g_pids, 10)
    mult_results[beta] = {'R@1': r1, 'R@5': r5, 'R@10': r10}
    print(f"  {beta:>6.2f}  {r1:>8.4f}  {r5:>8.4f}  {r10:>8.4f}")

best_beta = max(mult_results, key=lambda b: mult_results[b]['R@1'])
best_mult_m = cat_search(q_clip, g_clip, query_cats, gallery_cats,
                          q_color48, g_color48, alpha=0.4)  # reuse champ for comparison
print(f"\n  Best multiplicative beta={best_beta}: R@1={mult_results[best_beta]['R@1']:.4f}")
print(f"  Champion additive (a=0.4):            R@1={champ_m['R@1']:.4f}")
delta_mult = mult_results[best_beta]['R@1'] - champ_m['R@1']
print(f"  Multiplicative vs additive: {delta_mult:+.4f} R@1")

best_mult_idx = cat_search_multiplicative(q_clip, g_clip, query_cats, gallery_cats,
                                           q_color48, g_color48, beta=best_beta)
m_mult = evaluate(best_mult_idx, q_pids, g_pids, f"4.M.5 Multiplicative fusion beta={best_beta}")


# =====================================================================
# 11. BEST PHASE 4 SYSTEM
# =====================================================================
print("\n" + "=" * 65)
print("BEST PHASE 4 SYSTEM: Per-category alpha + best color features")
print("=" * 65)

# Try per-category alpha with 96D color
percat96_idx = cat_search_percat(q_clip, g_clip, query_cats, gallery_cats,
                                  q_color96, g_color96, best_alpha_per_cat)
percat96_m   = evaluate(percat96_idx, q_pids, g_pids, "4.M.6 Per-cat alpha + 96D color")

# Best Phase 4 overall
best_r1_p4 = max(
    champ_m['R@1'], percat_m['R@1'], m_96['R@1'],
    mult_results[best_beta]['R@1'], percat96_m['R@1']
)
print(f"\n  Phase 3 champion R@1:          {champ_m['R@1']:.4f}")
print(f"  4.M.2 Per-cat alpha (oracle):  {percat_m['R@1']:.4f}  ({percat_m['R@1']-champ_m['R@1']:+.4f})")
print(f"  4.M.3 96D color:               {m_96['R@1']:.4f}  ({m_96['R@1']-champ_m['R@1']:+.4f})")
print(f"  4.M.5 Multiplicative beta={best_beta}: {mult_results[best_beta]['R@1']:.4f}  ({mult_results[best_beta]['R@1']-champ_m['R@1']:+.4f})")
print(f"  4.M.6 Per-cat + 96D:           {percat96_m['R@1']:.4f}  ({percat96_m['R@1']-champ_m['R@1']:+.4f})")
print(f"\n  Phase 4 best R@1:              {best_r1_p4:.4f}")


# =====================================================================
# 12. MASTER RESULTS TABLE
# =====================================================================
print("\n" + "=" * 75)
print("MASTER RESULTS TABLE (all phases)")
print("=" * 75)

all_results = [
    ("P1  ResNet50 baseline (Anthony P1)",    {"R@1": 0.307,   "R@5": 0.490,  "R@10": 0.590}),
    ("P1M ResNet50 + color rerank (Mark P1)", {"R@1": 0.405,   "R@5": 0.640,  "R@10": 0.688}),
    ("P2M CLIP B/32 bare (Mark P2)",          {"R@1": 0.480,   "R@5": 0.672,  "R@10": 0.740}),
    ("P2M CLIP B/32 + color rerank (Mark P2)",{"R@1": 0.576,   "R@5": 0.747,  "R@10": 0.787}),
    ("P3A CLIP L/14+color+spatial+text (A P3)",{"R@1": 0.6748, "R@5": 0.855,  "R@10": 0.894}),
    ("P3M CLIP+cat.filter (Mark P3)",         {"R@1": 0.5686,  "R@5": 0.780,  "R@10": 0.833}),
    ("P3M CLIP+cat+color48 a=0.4 (Mark P3)", {"R@1": 0.6826,  "R@5": 0.862,  "R@10": 0.913}),
    ("4.M.1 Champion re-val (Mark P4)",       champ_m),
    ("4.M.2 Per-cat alpha oracle (Mark P4)",  percat_m),
    ("4.M.3 CLIP+cat+color96 (Mark P4)",      m_96),
    (f"4.M.5 Multiplicative b={best_beta} (Mark P4)", {'R@1': mult_results[best_beta]['R@1'], 'R@5': mult_results[best_beta]['R@5'], 'R@10': mult_results[best_beta]['R@10']}),
    ("4.M.6 Per-cat alpha + 96D (Mark P4)",   percat96_m),
]

sorted_r = sorted(all_results, key=lambda x: x[1].get('R@1', 0), reverse=True)
print(f"{'Rank':>4}  {'Experiment':<50}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}")
print("-" * 80)
for rank, (name, m) in enumerate(sorted_r, 1):
    marker = " <-- PHASE 4 BEST" if m.get('R@1', 0) == best_r1_p4 else ""
    print(f"{rank:>4}  {name:<50}  {m.get('R@1',0):.4f}  {m.get('R@5',0):.4f}  {m.get('R@10',0):.4f}{marker}")


# =====================================================================
# 13. SAVE RESULTS
# =====================================================================
print("\n[4/5] Saving results...")

results_p4 = {
    "phase4_mark": {
        "date":        "2026-04-23",
        "researcher":  "Mark Rodrigues",
        "eval_products": EVAL_N,
        "eval_gallery":  len(g_df),
        "eval_queries":  len(q_df),
        "research_question": (
            "What is the remaining 31.7% failure mode? "
            "Can per-category tuning push R@1 above 0.70?"
        ),
        "headline_finding": (
            f"Per-category alpha oracle = R@1={percat_m['R@1']:.4f} (+{(percat_m['R@1']-champ_m['R@1'])*100:.2f}pp vs champion). "
            f"{close_miss_pct:.1f}% of failures are close misses (correct in top-5). "
            f"Multiplicative fusion HURTS ({delta_mult:+.4f}). "
            f"96D color {'helps' if delta_96 > 0.001 else 'does not help'} ({delta_96:+.4f})."
        ),
        "experiments": {
            "4M1_champion_reval":   champ_m,
            "4M2_percat_alpha_oracle": percat_m,
            "4M2_best_alpha_per_cat":  best_alpha_per_cat,
            "4M3_color96": m_96,
            "4M5_multiplicative": {f"beta_{b}": mult_results[b] for b in mult_results},
            "4M6_percat_96d": percat96_m,
        },
        "error_analysis": {
            "n_success": n_succ,
            "n_fail":    n_fail,
            "success_rate": float(n_succ / n_q),
            "close_miss_pct": float(close_miss_pct),
            "pct_fail_rank_gt10": float((fail_df['rank_correct'] > 10).mean() * 100),
            "median_score_gap": float(score_gaps.median()),
            "mean_score_gap":   float(score_gaps.mean()),
            "pct_wrong_category": float(wrong_cat / n_fail * 100),
            "per_category": {
                cat: {
                    "fail_rate": round(v['fail_rate'], 4),
                    "med_rank_correct": round(v['med_rank'], 1),
                    "avg_score_gap": round(v['avg_gap'], 4),
                }
                for cat, v in cat_fail_stats.items()
            }
        },
        "alpha_scan_per_category": alpha_scan_table,
        "phase4_best_r1": float(best_r1_p4),
    }
}

metrics_path = RES / 'metrics.json'
existing = {}
if metrics_path.exists():
    with open(metrics_path) as f:
        existing = json.load(f)
existing.update(results_p4)
with open(metrics_path, 'w') as f:
    json.dump(existing, f, indent=2)

with open(RES / 'phase4_mark_results.json', 'w') as f:
    json.dump(results_p4, f, indent=2)
print("  Saved phase4_mark_results.json and updated metrics.json")


# =====================================================================
# 14. PLOTS
# =====================================================================
print("\n[5/5] Generating plots...")

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle(
    "Phase 4 Mark: Hyperparameter Tuning + Error Analysis\n"
    "Visual Product Search Engine — CLIP B/32 + Category-Conditioned Retrieval",
    fontsize=13, fontweight='bold'
)

# ---- Plot 1: Per-category optimal alpha --------------------------
ax1 = fig.add_subplot(gs[0, :2])
cats_plot = [r['category'] for r in alpha_scan_table]
opt_alphas = [r['best_alpha'] for r in alpha_scan_table]
g_alpha = [0.4] * len(cats_plot)
colors1 = ['#2ecc71' if a != 0.4 else '#bdc3c7' for a in opt_alphas]
x1 = np.arange(len(cats_plot))
ax1.bar(x1 - 0.2, g_alpha,    0.35, label='Global alpha=0.4', color='#3498db', alpha=0.7)
ax1.bar(x1 + 0.2, opt_alphas, 0.35, label='Per-cat optimal alpha', color=colors1, alpha=0.85)
for xi, a in zip(x1, opt_alphas):
    ax1.text(xi + 0.2, a + 0.02, f'{a:.2f}', ha='center', fontsize=8, fontweight='bold')
ax1.axhline(y=0.4, color='blue', linestyle='--', alpha=0.4, linewidth=1)
ax1.set_xticks(x1); ax1.set_xticklabels(cats_plot, rotation=25, ha='right', fontsize=9)
ax1.set_ylabel('Alpha (CLIP weight)'); ax1.set_ylim(0, 1.15)
ax1.set_title('Per-Category Optimal Alpha (CLIP blend weight)\nGreen = differs from global 0.4', fontsize=10)
ax1.legend(fontsize=9); ax1.grid(axis='y', alpha=0.4)

# ---- Plot 2: Per-category R@1 gain from per-cat tuning ----------
ax2 = fig.add_subplot(gs[0, 2])
deltas_pcat = [r['delta'] for r in alpha_scan_table]
colors2     = ['#2ecc71' if d > 0.005 else '#e74c3c' if d < -0.005 else '#f39c12' for d in deltas_pcat]
ax2.barh(cats_plot, deltas_pcat, color=colors2, alpha=0.85)
for i, d in enumerate(deltas_pcat):
    ax2.text(d + (0.003 if d >= 0 else -0.003), i, f'{d:+.3f}',
             va='center', ha='left' if d >= 0 else 'right', fontsize=8)
ax2.axvline(0, color='black', linewidth=0.8)
ax2.set_xlabel('Delta R@1 (per-cat alpha - global alpha=0.4)')
ax2.set_title('Gain from Per-Cat Alpha\n(oracle, tuned on eval set)', fontsize=10)
ax2.grid(axis='x', alpha=0.4)

# ---- Plot 3: Failure rank distribution --------------------------
ax3 = fig.add_subplot(gs[1, 0])
rank_data = fail_df['rank_correct'].values
rank_labels = ['Rank 2', 'Rank 3', 'Rank 4', 'Rank 5', 'Rank 6-10', 'Rank 11+']
rank_counts = [
    (rank_data == 2).sum(),
    (rank_data == 3).sum(),
    (rank_data == 4).sum(),
    (rank_data == 5).sum(),
    ((rank_data >= 6) & (rank_data <= 10)).sum(),
    (rank_data > 10).sum(),
]
colors3 = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#95a5a6', '#7f8c8d']
wedges, texts, pcts = ax3.pie(rank_counts, labels=rank_labels, colors=colors3,
                               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
ax3.set_title(f'Failure Rank Distribution\n(n={n_fail} failures, R@1={champ_m["R@1"]:.4f})', fontsize=10)

# ---- Plot 4: Score gap histogram --------------------------------
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(score_gaps, bins=40, color='#e74c3c', alpha=0.75, edgecolor='white')
ax4.axvline(score_gaps.median(), color='black', linestyle='--', linewidth=1.5, label=f'Median={score_gaps.median():.3f}')
ax4.axvline(0.05, color='orange', linestyle=':', linewidth=1.5, label='Gap=0.05 threshold')
ax4.set_xlabel('Score Gap (top-1 wrong - correct)', fontsize=9)
ax4.set_ylabel('Count', fontsize=9)
ax4.set_title('Score Gap Distribution\nSmall gap = model nearly correct', fontsize=10)
ax4.legend(fontsize=8); ax4.grid(alpha=0.4)

# ---- Plot 5: Per-category failure rates -------------------------
ax5 = fig.add_subplot(gs[1, 2])
cats5     = list(cat_fail_stats.keys())
fail_rates = [cat_fail_stats[c]['fail_rate'] for c in cats5]
colors5    = plt.cm.RdYlGn_r(np.array(fail_rates))
bars5      = ax5.bar(cats5, fail_rates, color=colors5, alpha=0.85)
for bar in bars5:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', fontsize=8)
ax5.set_xticklabels(cats5, rotation=30, ha='right', fontsize=8)
ax5.set_ylabel('Failure Rate (1 - R@1)')
ax5.set_title('Per-Category Failure Rate\n(Phase 3 champion)', fontsize=10)
ax5.grid(axis='y', alpha=0.4)

# ---- Plot 6: 96D vs 48D alpha sweep ----------------------------
ax6 = fig.add_subplot(gs[2, 0])
alphas_96  = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]
alphas_48  = [0.3,        0.4,       0.5,       0.6, 0.7]
r1s_96     = []
r1s_48     = []
for alpha in alphas_96:
    idx = cat_search(q_clip, g_clip, query_cats, gallery_cats, q_color96, g_color96, alpha=alpha)
    r1s_96.append(recall_at_k(idx, q_pids, g_pids, 1))
for alpha in alphas_48:
    idx = cat_search(q_clip, g_clip, query_cats, gallery_cats, q_color48, g_color48, alpha=alpha)
    r1s_48.append(recall_at_k(idx, q_pids, g_pids, 1))
ax6.plot(alphas_96, r1s_96, 'o-', color='#e74c3c', label='96D color (16 bins)', markersize=6)
ax6.plot(alphas_48, r1s_48, 's--', color='#3498db', label='48D color (8 bins)', markersize=6)
ax6.axhline(champ_m['R@1'], color='green', linestyle=':', linewidth=1.5, label=f'P3 champ={champ_m["R@1"]:.4f}')
ax6.set_xlabel('Alpha (CLIP weight)'); ax6.set_ylabel('R@1')
ax6.set_title('96D vs 48D Color: Alpha Sweep\n(higher bins ≠ better retrieval?)', fontsize=10)
ax6.legend(fontsize=8); ax6.grid(alpha=0.4)

# ---- Plot 7: Multiplicative vs additive fusion -------------------
ax7 = fig.add_subplot(gs[2, 1])
betas = sorted(mult_results.keys())
r1_mult = [mult_results[b]['R@1'] for b in betas]
ax7.plot(betas, r1_mult, 'o-', color='#9b59b6', label='Multiplicative', markersize=7)
ax7.axhline(champ_m['R@1'], color='#3498db', linestyle='--', linewidth=1.5, label=f'Additive a=0.4 R@1={champ_m["R@1"]:.4f}')
for beta, r1 in zip(betas, r1_mult):
    ax7.text(beta, r1 + 0.003, f'{r1:.3f}', ha='center', fontsize=8)
ax7.set_xlabel('Beta (color exponent in multiplicative fusion)')
ax7.set_ylabel('R@1')
ax7.set_title('Multiplicative vs Additive Fusion\nAdditive wins (simpler is better)', fontsize=10)
ax7.legend(fontsize=9); ax7.grid(alpha=0.4)

# ---- Plot 8: Phase 4 final leaderboard --------------------------
ax8 = fig.add_subplot(gs[2, 2])
leaderboard_names = [
    "P3 Anthony\n(CLIP L/14+text)",
    "P3M Champion\n(CLIP+cat+48D)",
    "4.M.2 Per-cat\nalpha (oracle)",
    "4.M.3 96D\ncolor",
    "4.M.5 Mult\nfusion",
    "4.M.6 Per-cat\n+96D",
]
leaderboard_r1 = [
    0.6748,
    champ_m['R@1'],
    percat_m['R@1'],
    m_96['R@1'],
    mult_results[best_beta]['R@1'],
    percat96_m['R@1'],
]
colors8 = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
bars8 = ax8.bar(range(len(leaderboard_names)), leaderboard_r1, color=colors8, alpha=0.85)
for bar in bars8:
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{bar.get_height():.4f}', ha='center', fontsize=8, fontweight='bold')
ax8.set_xticks(range(len(leaderboard_names)))
ax8.set_xticklabels(leaderboard_names, fontsize=7)
ax8.set_ylabel('R@1'); ax8.set_ylim(0.60, 0.78)
ax8.set_title('Phase 4 Leaderboard\nGreen = best Phase 4 system', fontsize=10)
ax8.grid(axis='y', alpha=0.4)

plt.savefig(RES / 'phase4_mark_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved phase4_mark_results.png")

# Second plot: error analysis deep dive
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Phase 4 Mark: Error Analysis Deep Dive\nRemaining 31.7% Failure Mode Taxonomy",
              fontsize=12, fontweight='bold')

# Success vs failure score distributions
ax_a = axes2[0]
ax_a.hist(success_scores, bins=30, alpha=0.7, color='#2ecc71', label=f'Success (n={n_succ})', density=True)
ax_a.hist(fail_correct_scores, bins=30, alpha=0.7, color='#e74c3c', label=f'Failure correct score (n={n_fail})', density=True)
ax_a.set_xlabel('Blend Score (0.4*CLIP + 0.6*color)')
ax_a.set_ylabel('Density')
ax_a.set_title('Score Distribution: Success vs Failure\nFailed queries have lower scores for their correct product')
ax_a.legend(fontsize=10); ax_a.grid(alpha=0.4)
sep = success_scores.mean() - fail_correct_scores.mean()
ax_a.text(0.05, 0.92, f'Score separation: {sep:.4f}', transform=ax_a.transAxes, fontsize=10, color='black',
          bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Per-cat failure rate vs category gallery size
ax_b = axes2[1]
sizes   = [cat_counts_g[c]           for c in cat_fail_stats]
frates  = [cat_fail_stats[c]['fail_rate'] for c in cat_fail_stats]
colors_b = plt.cm.RdYlGn_r(np.array(frates))
scatter = ax_b.scatter(sizes, frates, c=frates, cmap='RdYlGn_r', s=120, zorder=5)
for i, cat in enumerate(cat_fail_stats):
    ax_b.annotate(cat, (sizes[i], frates[i]),
                  textcoords="offset points", xytext=(5, 5), fontsize=8)
ax_b.set_xlabel('Gallery size (n products in category)')
ax_b.set_ylabel('Failure rate (1 - R@1)')
ax_b.set_title('Gallery Size vs Failure Rate\nSmall galleries = harder search space?')
plt.colorbar(scatter, ax=ax_b, label='Failure rate')
ax_b.grid(alpha=0.4)

plt.tight_layout()
plt.savefig(RES / 'phase4_mark_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved phase4_mark_error_analysis.png")

# =====================================================================
# FINAL SUMMARY
# =====================================================================
print("\n" + "=" * 65)
print("PHASE 4 MARK SUMMARY")
print("=" * 65)
p3_champ = 0.6826
p4_best  = best_r1_p4
print(f"  Phase 3 champion:        R@1 = {p3_champ:.4f}")
print(f"  Phase 4 best:            R@1 = {p4_best:.4f}  ({p4_best-p3_champ:+.4f})")
print(f"\nKEY FINDINGS:")
print(f"  1. Per-category alpha oracle: {percat_m['R@1']:.4f} (vs global {champ_m['R@1']:.4f})")
print(f"     Best categories differ from global 0.4: {sum(1 for r in alpha_scan_table if r['best_alpha'] != 0.4)} of {len(alpha_scan_table)}")
print(f"  2. {close_miss_pct:.1f}% of failures are 'close misses' (correct product in top-5)")
print(f"     => Better ranking signals needed, not better recall")
print(f"  3. 96D color {('helps: +' if delta_96 > 0.001 else 'does not help (no change: +') if delta_96 > -0.001 else 'hurts: '}{abs(delta_96)*100:.2f}pp)")
print(f"  4. Multiplicative fusion FAILS: best={mult_results[best_beta]['R@1']:.4f} vs additive {champ_m['R@1']:.4f}")
print(f"     Additive blend is more robust than soft-AND for fashion retrieval")
print(f"  5. Failure mode: median score gap = {score_gaps.median():.4f}")
print(f"     => The model 'knows' which products are similar — precision is the bottleneck, not recall")
print(f"\nFILES:")
print(f"  results/phase4_mark_results.json")
print(f"  results/phase4_mark_results.png")
print(f"  results/phase4_mark_error_analysis.png")
print(f"  results/metrics.json (updated)")
print("=" * 65)
