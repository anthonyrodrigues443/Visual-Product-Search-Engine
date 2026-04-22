#!/usr/bin/env python3
"""Phase 2: Multi-Model Comparison — Visual Product Search Engine.

Compares 5 model paradigms on DeepFashion In-Shop retrieval (300 products):
  1. CLIP ViT-B/32  (vision-language, 512D, 88M params)
  2. CLIP ViT-L/14  (vision-language, 768D, 304M params)
  3. DINOv2 ViT-B/14 (self-supervised, 768D, 86M params)
  4. ViT-B/16        (supervised ImageNet, 768D, 86M params)
  5. ConvNeXt-Tiny   (modern CNN, 768D, 29M params)

Plus: best model + Mark's color re-ranking at alpha={0.7, 0.5, 0.3}.

Research question: Does training paradigm (supervised / self-supervised /
vision-language) matter more than architecture (CNN vs ViT) for fashion retrieval?

Building on Phase 1:
  - Anthony: ResNet50 baseline R@1=30.7%
  - Mark: EfficientNet-B0 R@1=36.7%, color re-rank R@1=40.5%

Author: Anthony Rodrigues | Date: 2026-04-21
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
import torch
import torchvision.transforms as T
import torchvision.models as tv_models
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
print("PHASE 2: MULTI-MODEL COMPARISON")
print("Training paradigm vs architecture for fashion retrieval")
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
print(f"Downloading {len(all_idx)} images (index range 0-{max(all_idx)})...")
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
        results[f'R@{k}'] = correct / len(q_pids)
    return results


def per_cat_recall(q_df, g_pids, indices, k=1):
    cats = {}
    for cat in sorted(q_df['category2'].unique()):
        mask = (q_df['category2'] == cat).values
        qp = q_df.loc[mask, 'product_id'].values
        qi = np.where(mask)[0]
        correct = sum(1 for i, p in zip(qi, qp) if p in g_pids[indices[i, :k]])
        cats[cat] = {'R@1': correct / len(qp), 'n': int(len(qp))}
    return cats


def sim_separation(qf, gf, qp, gp):
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sims = qn @ gn.T
    corr, inc = [], []
    for qi, pid in enumerate(qp):
        cm = gp == pid
        if cm.any():
            corr.append(float(sims[qi, cm].max()))
        if (~cm).any():
            inc.append(float(sims[qi, ~cm].max()))
    c_mean = float(np.mean(corr))
    i_mean = float(np.mean(inc))
    return c_mean, i_mean, c_mean - i_mean


def faiss_search(gf, qf, k=K):
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    qn = qf / (np.linalg.norm(qf, axis=1, keepdims=True) + 1e-8)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(gn.shape[1])
    index.add(np.ascontiguousarray(gn, dtype=np.float32))
    D, I = index.search(np.ascontiguousarray(qn, dtype=np.float32), k)
    return D, I


def extract_feats(model, indices, transform, model_type='tv', bs=BS):
    feats, valid, batch, bidx = [], [], [], []
    for ix in tqdm(indices, desc='  Features', leave=False):
        ix = int(ix)
        if ix not in imgs:
            continue
        batch.append(transform(imgs[ix]))
        bidx.append(ix)
        if len(batch) >= bs:
            t = torch.stack(batch).to(DEV)
            with torch.no_grad():
                if model_type == 'clip':
                    f = model.encode_image(t)
                elif model_type == 'dino':
                    f = model(pixel_values=t).last_hidden_state[:, 0]
                else:
                    f = model(t)
            feats.append(f.cpu().float().numpy())
            valid.extend(bidx)
            batch, bidx = [], []
    if batch:
        t = torch.stack(batch).to(DEV)
        with torch.no_grad():
            if model_type == 'clip':
                f = model.encode_image(t)
            elif model_type == 'dino':
                f = model(pixel_values=t).last_hidden_state[:, 0]
            else:
                f = model(t)
        feats.append(f.cpu().float().numpy())
        valid.extend(bidx)
    return np.vstack(feats), valid


# ======================================================================
# 3. MODEL DEFINITIONS
# ======================================================================
inet_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def mk_clip_b32():
    import open_clip
    m, _, pp = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    m.eval()
    return m, pp, 'clip', 512, 88


def mk_clip_l14():
    import open_clip
    m, _, pp = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    m.eval()
    return m, pp, 'clip', 768, 304


def mk_dinov2():
    from transformers import Dinov2Model
    m = Dinov2Model.from_pretrained('facebook/dinov2-base')
    m.eval()
    return m, inet_tf, 'dino', 768, 86


def mk_vit():
    m = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads = torch.nn.Identity()
    m.eval()
    return m, inet_tf, 'tv', 768, 86


def mk_convnext():
    m = tv_models.convnext_tiny(weights=tv_models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    m.classifier[2] = torch.nn.Identity()
    m.eval()
    return m, inet_tf, 'tv', 768, 29


MODEL_SPECS = [
    ('CLIP ViT-B/32', mk_clip_b32),
    ('CLIP ViT-L/14', mk_clip_l14),
    ('DINOv2 ViT-B/14', mk_dinov2),
    ('ViT-B/16 (ImageNet)', mk_vit),
    ('ConvNeXt-Tiny', mk_convnext),
]

# ======================================================================
# 4. RUN ALL MODELS
# ======================================================================
results = {}
all_cat = {}
saved_feats = {}

for name, loader in MODEL_SPECS:
    print(f"\n{'=' * 70}")
    print(f"  MODEL: {name}")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    try:
        model, tf, mtype, dim, params = loader()
        print(f"  Loaded: {dim}D embedding, ~{params}M params")
        sys.stdout.flush()

        t0 = time.time()
        gf, gv = extract_feats(model, g_df['index'].values, tf, mtype)
        qf, qv = extract_feats(model, q_df['index'].values, tf, mtype)
        t_extract = time.time() - t0
        print(f"  Features: gallery {gf.shape}, query {qf.shape} ({t_extract:.1f}s)")

        D, I = faiss_search(gf, qf)
        rec = recall_at_k(q_pids, g_pids, I)
        cats = per_cat_recall(q_df, g_pids, I)
        corr_sim, inc_sim, gap = sim_separation(qf, gf, q_pids, g_pids)

        results[name] = {
            'paradigm': 'vision-language' if 'CLIP' in name
                        else 'self-supervised' if 'DINO' in name
                        else 'supervised-cnn' if 'ConvNeXt' in name
                        else 'supervised-vit',
            'dim': dim,
            'params_m': params,
            'extract_s': round(t_extract, 1),
            'ms_per_img': round(t_extract / (len(gv) + len(qv)) * 1000, 1),
            **{k: round(v, 4) for k, v in rec.items()},
            'separation': round(gap, 4),
            'corr_sim': round(corr_sim, 4),
            'inc_sim': round(inc_sim, 4),
        }
        all_cat[name] = cats
        saved_feats[name] = (gf.copy(), qf.copy())

        print(f"  R@1={rec['R@1']:.4f}  R@5={rec['R@5']:.4f}  "
              f"R@10={rec['R@10']:.4f}  R@20={rec['R@20']:.4f}")
        print(f"  Separation: {gap:.4f} (correct={corr_sim:.4f}, incorrect={inc_sim:.4f})")
        print(f"  Speed: {results[name]['ms_per_img']:.1f} ms/image")
        sys.stdout.flush()

        del model
        gc.collect()

    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()


# ======================================================================
# 5. COLOR RE-RANKING (best model + Mark's color features)
# ======================================================================
if results:
    best_name = max(results, key=lambda k: results[k]['R@1'])
    print(f"\n{'=' * 70}")
    print(f"  COLOR RE-RANKING: {best_name} + 48D color histogram")
    print(f"{'=' * 70}")
    sys.stdout.flush()

    from src.feature_engineering import extract_color_palette, extract_hsv_histogram

    g_colors = np.array([
        np.concatenate([
            extract_color_palette(imgs[int(ix)]),
            extract_hsv_histogram(imgs[int(ix)]),
        ])
        for ix in tqdm(g_df['index'].values, desc='  Gallery colors')
    ])
    q_colors = np.array([
        np.concatenate([
            extract_color_palette(imgs[int(ix)]),
            extract_hsv_histogram(imgs[int(ix)]),
        ])
        for ix in tqdm(q_df['index'].values, desc='  Query colors')
    ])

    gf, qf = saved_feats[best_name]
    D, I = faiss_search(gf, qf)

    gc_n = g_colors / (np.linalg.norm(g_colors, axis=1, keepdims=True) + 1e-8)
    qc_n = q_colors / (np.linalg.norm(q_colors, axis=1, keepdims=True) + 1e-8)

    for alpha in [0.7, 0.5, 0.3]:
        reranked = np.zeros_like(I)
        for qi in range(len(I)):
            cands = I[qi]
            cnn_s = D[qi]
            col_s = qc_n[qi] @ gc_n[cands].T
            blend = alpha * cnn_s + (1 - alpha) * col_s
            reranked[qi] = cands[np.argsort(-blend)]

        rr = recall_at_k(q_pids, g_pids, reranked)
        rr_cats = per_cat_recall(q_df, g_pids, reranked)
        rname = f"{best_name} + Color (a={alpha})"
        results[rname] = {
            'paradigm': 'hybrid',
            'dim': results[best_name]['dim'] + 48,
            'params_m': results[best_name]['params_m'],
            **{k: round(v, 4) for k, v in rr.items()},
        }
        all_cat[rname] = rr_cats
        print(f"  alpha={alpha}: R@1={rr['R@1']:.4f}  R@5={rr['R@5']:.4f}  "
              f"R@10={rr['R@10']:.4f}  R@20={rr['R@20']:.4f}")
    sys.stdout.flush()


# ======================================================================
# 6. FULL COMPARISON TABLE
# ======================================================================
p1_baselines = {
    'ResNet50 (P1 baseline)': {
        'paradigm': 'supervised-cnn', 'dim': 2048, 'params_m': 26,
        'R@1': 0.3067, 'R@5': 0.4927, 'R@10': 0.5901, 'R@20': 0.6913,
        'separation': 0.0485,
    },
    'EfficientNet-B0 (Mark P1)': {
        'paradigm': 'supervised-cnn', 'dim': 1280, 'params_m': 5,
        'R@1': 0.3671, 'R@5': 0.5988, 'R@10': 0.6855, 'R@20': 0.7760,
    },
    'ResNet50+Color a=0.5 (Mark)': {
        'paradigm': 'hybrid', 'dim': 2096, 'params_m': 26,
        'R@1': 0.4051, 'R@5': 0.5930, 'R@10': 0.6573, 'R@20': 0.6913,
    },
}
combined = {**p1_baselines, **results}

print(f"\n{'=' * 70}")
print("FULL COMPARISON — RANKED BY R@1")
print(f"{'=' * 70}")
header = (f"{'#':>2}  {'Model':<40} {'Para':>6} {'Dim':>5} "
          f"{'R@1':>7} {'R@5':>7} {'R@10':>7} {'R@20':>7} {'Sep':>7}")
print(header)
print("-" * len(header))
ranked = sorted(combined.items(), key=lambda x: x[1].get('R@1', 0), reverse=True)
for rank, (name, m) in enumerate(ranked, 1):
    sep_s = f"{m['separation']:.4f}" if 'separation' in m else '  —'
    para = m.get('paradigm', '?')[:6]
    print(f"{rank:>2}  {name:<40} {para:>6} {m.get('dim', ''):>5} "
          f"{m.get('R@1', 0):>7.4f} {m.get('R@5', 0):>7.4f} "
          f"{m.get('R@10', 0):>7.4f} {m.get('R@20', 0):>7.4f} {sep_s:>7}")
sys.stdout.flush()


# ======================================================================
# 7. PER-CATEGORY ANALYSIS
# ======================================================================
print(f"\n{'=' * 70}")
print("PER-CATEGORY R@1 (Phase 2 models only)")
print(f"{'=' * 70}")

p2_models = [n for n in all_cat if 'P1' not in n and 'Mark' not in n and 'Color' not in n]
cats_list = sorted(q_df['category2'].unique())

header = f"{'Category':<15} {'n':>4}" + "".join(f" {n[:18]:>18}" for n in p2_models)
print(header)
print("-" * len(header))
for cat in cats_list:
    row = f"{cat:<15} {all_cat[p2_models[0]][cat]['n']:>4}"
    for mname in p2_models:
        row += f" {all_cat[mname].get(cat, {}).get('R@1', 0):>18.4f}"
    print(row)

# Category winner analysis
print("\nCategory winners:")
for cat in cats_list:
    best_model = max(p2_models, key=lambda m: all_cat[m].get(cat, {}).get('R@1', 0))
    best_r1 = all_cat[best_model][cat]['R@1']
    worst_model = min(p2_models, key=lambda m: all_cat[m].get(cat, {}).get('R@1', 0))
    worst_r1 = all_cat[worst_model][cat]['R@1']
    spread = best_r1 - worst_r1
    print(f"  {cat:<15} BEST: {best_model} ({best_r1:.3f})  "
          f"WORST: {worst_model} ({worst_r1:.3f})  spread={spread:.3f}")
sys.stdout.flush()


# ======================================================================
# 8. PLOTS
# ======================================================================
print(f"\n{'=' * 70}")
print("GENERATING PLOTS")
print(f"{'=' * 70}")

# 8a. Main comparison bar chart + per-category heatmap
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

model_names = [n for n, _ in ranked]
r1_vals = [combined[n].get('R@1', 0) for n in model_names]

color_map = {
    'vision-language': '#e74c3c',
    'self-supervised': '#3498db',
    'supervised-vit': '#2ecc71',
    'supervised-cnn': '#9b59b6',
    'hybrid': '#f39c12',
}
bar_colors = [color_map.get(combined[n].get('paradigm', ''), '#95a5a6') for n in model_names]

bars = axes[0].barh(range(len(model_names)), r1_vals, color=bar_colors,
                     edgecolor='black', linewidth=0.5)
axes[0].set_yticks(range(len(model_names)))
axes[0].set_yticklabels([n[:35] for n in model_names], fontsize=8)
axes[0].set_xlabel('Recall@1', fontsize=12)
axes[0].set_title('All Models Ranked by R@1', fontsize=14, fontweight='bold')
axes[0].invert_yaxis()
for bar, val in zip(bars, r1_vals):
    axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=8)

# Legend for paradigms
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=c, edgecolor='black', label=p)
                for p, c in color_map.items()]
axes[0].legend(handles=legend_elems, loc='lower right', fontsize=8)

# Per-category heatmap (Phase 2 new models only)
if p2_models:
    cat_data = {}
    for mname in p2_models:
        for cat, vals in all_cat[mname].items():
            if cat not in cat_data:
                cat_data[cat] = {}
            cat_data[cat][mname] = vals['R@1']
    cat_matrix = pd.DataFrame(cat_data).T
    col_order = sorted(cat_matrix.columns, key=lambda c: cat_matrix[c].mean(), reverse=True)
    cat_matrix = cat_matrix[col_order]
    row_order = sorted(cat_matrix.index, key=lambda r: cat_matrix.loc[r].mean(), reverse=True)
    cat_matrix = cat_matrix.loc[row_order]

    sns.heatmap(cat_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[1],
                vmin=0, vmax=0.8, linewidths=0.5, cbar_kws={'shrink': 0.8})
    axes[1].set_title('Per-Category R@1 — Phase 2 Models', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Category')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=8)

plt.tight_layout()
plt.savefig(RES / 'phase2_anthony_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/phase2_anthony_comparison.png")

# 8b. Training paradigm comparison
paradigm_best = {}
for name, m in results.items():
    if 'Color' in name:
        continue
    p = m.get('paradigm', 'unknown')
    if p not in paradigm_best or m['R@1'] > paradigm_best[p]['R@1']:
        paradigm_best[p] = {'name': name, 'R@1': m['R@1'],
                            'separation': m.get('separation', 0)}

if len(paradigm_best) >= 3:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pnames = list(paradigm_best.keys())
    pr1 = [paradigm_best[p]['R@1'] for p in pnames]
    psep = [paradigm_best[p].get('separation', 0) for p in pnames]
    pcolors = [color_map.get(p, '#95a5a6') for p in pnames]

    bars = axes[0].bar(range(len(pnames)), pr1, color=pcolors, edgecolor='black')
    axes[0].set_xticks(range(len(pnames)))
    axes[0].set_xticklabels(pnames, fontsize=9, rotation=15)
    axes[0].set_ylabel('Best R@1', fontsize=12)
    axes[0].set_title('Training Paradigm vs R@1', fontsize=14, fontweight='bold')
    for i, v in enumerate(pr1):
        axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')

    bars2 = axes[1].bar(range(len(pnames)), psep, color=pcolors, edgecolor='black')
    axes[1].set_xticks(range(len(pnames)))
    axes[1].set_xticklabels(pnames, fontsize=9, rotation=15)
    axes[1].set_ylabel('Similarity Separation Gap', fontsize=12)
    axes[1].set_title('Training Paradigm vs Embedding Separation', fontsize=14,
                       fontweight='bold')
    for i, v in enumerate(psep):
        axes[1].text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(RES / 'phase2_anthony_paradigm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: results/phase2_anthony_paradigm.png")

# 8c. Dimension vs R@1 scatter (bubble = params)
fig, ax = plt.subplots(figsize=(10, 6))
for name, m in results.items():
    if 'Color' in name:
        continue
    c = color_map.get(m.get('paradigm', ''), '#95a5a6')
    ax.scatter(m['dim'], m['R@1'], s=m.get('params_m', 30) * 3, color=c,
               edgecolor='black', zorder=5, alpha=0.8)
    ax.annotate(name, (m['dim'], m['R@1']), fontsize=7,
                textcoords='offset points', xytext=(5, 5))
ax.set_xlabel('Embedding Dimension', fontsize=12)
ax.set_ylabel('Recall@1', fontsize=12)
ax.set_title('Embedding Dimension vs R@1 (bubble size = model params)',
             fontsize=13, fontweight='bold')
ax.legend(handles=legend_elems, loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(RES / 'phase2_anthony_dim_vs_r1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/phase2_anthony_dim_vs_r1.png")

# 8d. Recall@K curves for all Phase 2 models
fig, ax = plt.subplots(figsize=(10, 6))
ks_list = [1, 5, 10, 20]
for name in p2_models:
    m = results[name]
    c = color_map.get(m.get('paradigm', ''), '#95a5a6')
    vals = [m[f'R@{k}'] for k in ks_list]
    ax.plot(ks_list, vals, '-o', label=name, color=c, linewidth=2, markersize=6)

# Add Phase 1 baseline
ax.plot(ks_list, [0.3067, 0.4927, 0.5901, 0.6913], '--s', label='ResNet50 (P1)',
        color='gray', linewidth=1.5, markersize=5)
ax.set_xlabel('K', fontsize=12)
ax.set_ylabel('Recall@K', fontsize=12)
ax.set_title('Recall@K Curves — Phase 2 Models vs Baseline', fontsize=14, fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_xticks(ks_list)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RES / 'phase2_anthony_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/phase2_anthony_recall_curves.png")


# ======================================================================
# 9. SAVE RESULTS
# ======================================================================
output = {
    'phase2_anthony': {
        'date': '2026-04-21',
        'eval_products': EVAL_N,
        'eval_gallery': int(len(g_df)),
        'eval_queries': int(len(q_df)),
        'research_question': ('Does training paradigm (supervised/self-supervised/'
                              'vision-language) matter more than architecture for '
                              'fashion retrieval?'),
        'models': {k: v for k, v in results.items()},
        'per_category': {
            n: {c: round(v['R@1'], 4) for c, v in cats.items()}
            for n, cats in all_cat.items()
            if n in results
        },
    }
}

metrics_path = RES / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        existing = json.load(f)
else:
    existing = {}
existing.update(output)
with open(metrics_path, 'w') as f:
    json.dump(existing, f, indent=2)
print("\nSaved: results/metrics.json")

with open(RES / 'phase2_anthony_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("Saved: results/phase2_anthony_results.json")


# ======================================================================
# 10. SUMMARY
# ======================================================================
print(f"\n{'=' * 70}")
print("PHASE 2 COMPLETE — SUMMARY")
print(f"{'=' * 70}")

if results:
    overall_best = max(combined.items(), key=lambda x: x[1].get('R@1', 0))
    p2_best = max(results.items(), key=lambda x: x[1].get('R@1', 0))
    print(f"Overall best:    {overall_best[0]} (R@1={overall_best[1]['R@1']:.4f})")
    print(f"Phase 2 best:    {p2_best[0]} (R@1={p2_best[1]['R@1']:.4f})")
    print(f"Phase 1 ResNet50 baseline: R@1=0.3067")
    print(f"Improvement:     +{(p2_best[1]['R@1'] - 0.3067)*100:.1f}pp over baseline")

    # Paradigm ranking
    print("\nParadigm ranking (best model per paradigm):")
    for p, info in sorted(paradigm_best.items(), key=lambda x: x[1]['R@1'], reverse=True):
        print(f"  {p:>20}: {info['name']} R@1={info['R@1']:.4f}")
else:
    print("No models completed successfully.")

print(f"\nPlots saved: phase2_anthony_comparison.png, phase2_anthony_paradigm.png, "
      f"phase2_anthony_dim_vs_r1.png, phase2_anthony_recall_curves.png")

os._exit(0)
