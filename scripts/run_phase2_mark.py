"""
Phase 2 Mark — Foundation Models vs CNN Architectures for Fashion Visual Search

Building on Phase 1 findings:
  - Anthony baseline: ResNet50 ImageNet V2 → R@1=0.3067
  - Mark Phase 1 champion: EfficientNet-B0 → R@1=0.3671 (+6pp)
  - Mark Phase 1 best system: ResNet50 + color rerank α=0.5 → R@1=0.4051 (+9.8pp)

Phase 2 Research Question:
  Do foundation models (CLIP, DINOv2, ViT) trained on billions of images
  outperform ImageNet-trained CNNs for domain-specific fashion retrieval?

Hypothesis:
  H1: CLIP (text-image aligned) will UNDERPERFORM DINOv2 (pure visual SSL)
      for pure image-to-image search, because CLIP optimizes for cross-modal
      alignment, not visual similarity.
  H2: DINOv2 self-supervised features will outperform ImageNet-supervised CNNs
      because SSL captures richer visual structure (inspired by Oquab et al.
      2023, "DINOv2: Learning Robust Visual Features without Supervision").
  H3: Foundation models will especially help on jackets (Phase 1's hardest
      category, R@1=0.14 on ResNet50) — category semantics help disambiguate
      similar structures.

Experiments:
  2.M.1  EfficientNet-B0                — Phase 1 champion, warm-start baseline
  2.M.2  CLIP ViT-B/32 (OpenAI)         — zero-shot foundation, text-aligned
  2.M.3  CLIP ViT-L/14 (OpenAI)         — larger CLIP, more capacity
  2.M.4  DINOv2 ViT-S/14 (Meta)         — self-supervised, pure visual
  2.M.5  ConvNeXt-Small (ImageNet-22k)  — modern CNN with transformer tricks
  2.M.6  CLIP + color rerank            — best-combo probe (extend Phase 1 insight)

Evaluation: Same 300-product eval slice from Phase 1 for direct comparison.

References:
  - Radford et al. 2021 "Learning Transferable Visual Models From Natural
    Language Supervision" (CLIP).
  - Oquab et al. 2023 "DINOv2: Learning Robust Visual Features without
    Supervision" — showed SSL features transfer better than supervised
    ImageNet pretraining on retrieval tasks.
  - Liu et al. 2022 "A ConvNet for the 2020s" (ConvNeXt) — modern CNN
    matching ViT performance with CNN inductive biases.
  - Marqo blog 2024: CLIP for e-commerce visual search observed 0.35-0.45
    Recall@10 on similar fashion datasets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import time
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
IMG_CACHE = PROJECT_ROOT / 'data' / 'raw' / 'images'
IMG_CACHE.mkdir(parents=True, exist_ok=True)

from src.data_pipeline import create_retrieval_splits
from src.feature_engineering import (
    extract_color_palette, extract_hsv_histogram, color_rerank,
)

# ===== CONSTANTS =====
EVAL_PRODUCTS = 300
SEED = 42
BATCH_SIZE = 16


def compute_recall_at_k(query_pids, gallery_pids, retrieved_indices, k):
    correct = 0
    for i in range(len(query_pids)):
        top_k_pids = gallery_pids[retrieved_indices[i, :k]]
        if query_pids[i] in top_k_pids:
            correct += 1
    return correct / len(query_pids)


def eval_recall(query_pids, gallery_pids, indices, label=""):
    r1 = compute_recall_at_k(query_pids, gallery_pids, indices, 1)
    r5 = compute_recall_at_k(query_pids, gallery_pids, indices, 5)
    r10 = compute_recall_at_k(query_pids, gallery_pids, indices, 10)
    r20 = compute_recall_at_k(query_pids, gallery_pids, indices, 20)
    print(f"  {label:42s} R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f} R@20={r20:.4f}")
    sys.stdout.flush()
    return {"recall@1": r1, "recall@5": r5, "recall@10": r10, "recall@20": r20}


def faiss_search(gallery_feats, query_feats, k=20):
    import faiss
    g = np.ascontiguousarray(gallery_feats, dtype=np.float32)
    q = np.ascontiguousarray(query_feats, dtype=np.float32)
    g = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    faiss.omp_set_num_threads(1)
    index = faiss.IndexFlatIP(g.shape[1])
    index.add(g)
    return index.search(q, k)


def per_category_recall(query_pids, query_cats, gallery_pids, indices, k=1):
    cats = {}
    for cat in sorted(set(query_cats)):
        mask = query_cats == cat
        if mask.sum() < 5:
            continue
        r = compute_recall_at_k(query_pids[mask], gallery_pids, indices[mask], k)
        cats[cat] = {"recall@1": r, "n_queries": int(mask.sum())}
    return cats


# ===== 1. METADATA + SPLITS =====
print("=" * 72)
print("PHASE 2 MARK — FOUNDATION MODELS VS CNNs FOR FASHION VISUAL SEARCH")
print("=" * 72)
print("\n--- 1. Metadata + retrieval splits (deterministic, seed=42) ---")
sys.stdout.flush()

df = pd.read_csv(DATA_PROCESSED / 'metadata.csv')
train_df, gallery_df, query_df = create_retrieval_splits(df, test_frac=0.2, seed=SEED)

rng = np.random.RandomState(SEED)
eval_products = gallery_df['product_id'].values[:EVAL_PRODUCTS]

eval_gallery = gallery_df[gallery_df['product_id'].isin(eval_products)].reset_index(drop=True)
eval_query = query_df[query_df['product_id'].isin(eval_products)].reset_index(drop=True)
print(f"  Eval slice: {len(eval_gallery)} gallery, {len(eval_query)} queries")
sys.stdout.flush()

gallery_pids = eval_gallery['product_id'].values
query_pids = eval_query['product_id'].values
query_cats = eval_query['category2'].values
gallery_cats = eval_gallery['category2'].values

# ===== 2. IMAGE LOADING (stream from HF, cache to disk) =====
print("\n--- 2. Loading images ---")
sys.stdout.flush()

idx_to_item = {}
for _, row in eval_gallery.iterrows():
    idx_to_item[int(row['index'])] = row['item_id']
for _, row in eval_query.iterrows():
    idx_to_item[int(row['index'])] = row['item_id']

all_eval_idx = sorted(idx_to_item.keys())
cached = {idx for idx, item_id in idx_to_item.items()
          if (IMG_CACHE / f"{item_id}.jpg").exists()}
missing = set(all_eval_idx) - cached
print(f"  Cached on disk: {len(cached)}, to download: {len(missing)}")
sys.stdout.flush()

images_by_index = {}
for idx in cached:
    images_by_index[idx] = Image.open(IMG_CACHE / f"{idx_to_item[idx]}.jpg").convert('RGB')

if missing:
    from datasets import load_dataset
    ds = load_dataset("Marqo/deepfashion-inshop", split="data", streaming=True)
    needed = set(missing)
    for i, ex in enumerate(tqdm(ds, total=max(needed) + 1, desc="  Download")):
        if i in needed:
            img = ex['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save to disk cache for future runs
            img.save(IMG_CACHE / f"{idx_to_item[i]}.jpg", "JPEG", quality=90)
            images_by_index[i] = img
            needed.discard(i)
            if not needed:
                break

gallery_images = [images_by_index[int(row['index'])] for _, row in eval_gallery.iterrows()]
query_images = [images_by_index[int(row['index'])] for _, row in eval_query.iterrows()]
print(f"  Loaded: {len(gallery_images)} gallery + {len(query_images)} query images")
sys.stdout.flush()

# ===== 3. PRECOMPUTE COLOR FEATURES (for later rerank) =====
print("\n--- 3. Precomputing color features (for rerank) ---")
sys.stdout.flush()
gallery_colors = np.stack([
    np.concatenate([extract_color_palette(im), extract_hsv_histogram(im)])
    for im in gallery_images
]).astype(np.float32)
query_colors = np.stack([
    np.concatenate([extract_color_palette(im), extract_hsv_histogram(im)])
    for im in query_images
]).astype(np.float32)
print(f"  Color shape: gallery={gallery_colors.shape} query={query_colors.shape}")
sys.stdout.flush()

# ===== 4. MODEL EMBEDDING FUNCTIONS =====
import torch
device = torch.device('cpu')
torch.set_num_threads(2)
print(f"\n--- 4. Using device: {device} ---")
sys.stdout.flush()


def embed_torchvision(model_name, images, weights_name='DEFAULT'):
    """Embed images with a torchvision backbone (pooled features)."""
    from torchvision import models as tv_models
    from torchvision import transforms
    factory = getattr(tv_models, model_name)
    weights_enum = getattr(tv_models, {
        'efficientnet_b0': 'EfficientNet_B0_Weights',
        'convnext_small':  'ConvNeXt_Small_Weights',
    }[model_name])
    w = getattr(weights_enum, weights_name)
    model = factory(weights=w).eval().to(device)

    # Pull off classifier → keep feature extractor
    if model_name == 'efficientnet_b0':
        model.classifier = torch.nn.Identity()
    elif model_name == 'convnext_small':
        model.classifier[-1] = torch.nn.Identity()

    tf = w.transforms()

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  {model_name}"):
            batch = torch.stack([tf(im) for im in images[i:i + BATCH_SIZE]]).to(device)
            out = model(batch)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def embed_openclip(arch, pretrained, images):
    """Embed images with an open_clip model's vision tower."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = model.eval().to(device)

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  CLIP-{arch}"):
            batch = torch.stack([preprocess(im) for im in images[i:i + BATCH_SIZE]]).to(device)
            out = model.encode_image(batch)
            feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def embed_dinov2(images, model_id='facebook/dinov2-small'):
    """Embed images with DINOv2 via HuggingFace transformers."""
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval().to(device)

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  DINOv2"):
            inputs = processor(images=images[i:i + BATCH_SIZE], return_tensors="pt").to(device)
            out = model(**inputs)
            # Use CLS token (index 0) from last_hidden_state as the image embedding
            cls = out.last_hidden_state[:, 0, :]
            feats.append(cls.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# ===== 5. RUN EXPERIMENTS =====
experiments = {}
print("\n--- 5. Running experiments ---")
print()
sys.stdout.flush()

# --- 2.M.1 EfficientNet-B0 (Phase 1 champion warm start) ---
print("Experiment 2.M.1: EfficientNet-B0 (Phase 1 champion re-run)")
sys.stdout.flush()
t0 = time.time()
g_feats = embed_torchvision('efficientnet_b0', gallery_images)
q_feats = embed_torchvision('efficientnet_b0', query_images)
eb0_time = time.time() - t0
print(f"  Embed time: {eb0_time:.1f}s  dim={g_feats.shape[1]}")
_, idx = faiss_search(g_feats, q_feats, k=20)
experiments['efficientnet_b0_p1_rerun'] = {
    **eval_recall(query_pids, gallery_pids, idx, "EfficientNet-B0"),
    "embed_dim": int(g_feats.shape[1]),
    "total_embed_time_s": round(eb0_time, 2),
    "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=1),
    "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=10),
}
eb0_idx = idx
eb0_g, eb0_q = g_feats, q_feats
del g_feats, q_feats; gc.collect()
print()

# --- 2.M.2 CLIP ViT-B/32 ---
print("Experiment 2.M.2: CLIP ViT-B/32 (OpenAI)")
sys.stdout.flush()
t0 = time.time()
g_feats = embed_openclip('ViT-B-32', 'openai', gallery_images)
q_feats = embed_openclip('ViT-B-32', 'openai', query_images)
clipb_time = time.time() - t0
print(f"  Embed time: {clipb_time:.1f}s  dim={g_feats.shape[1]}")
_, idx = faiss_search(g_feats, q_feats, k=20)
experiments['clip_vit_b32'] = {
    **eval_recall(query_pids, gallery_pids, idx, "CLIP ViT-B/32"),
    "embed_dim": int(g_feats.shape[1]),
    "total_embed_time_s": round(clipb_time, 2),
    "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=1),
    "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=10),
}
clipb_g, clipb_q = g_feats, q_feats
del g_feats, q_feats; gc.collect()
print()

# --- 2.M.3 CLIP ViT-L/14 ---
print("Experiment 2.M.3: CLIP ViT-L/14 (OpenAI)")
sys.stdout.flush()
t0 = time.time()
g_feats = embed_openclip('ViT-L-14', 'openai', gallery_images)
q_feats = embed_openclip('ViT-L-14', 'openai', query_images)
clipl_time = time.time() - t0
print(f"  Embed time: {clipl_time:.1f}s  dim={g_feats.shape[1]}")
_, idx = faiss_search(g_feats, q_feats, k=20)
experiments['clip_vit_l14'] = {
    **eval_recall(query_pids, gallery_pids, idx, "CLIP ViT-L/14"),
    "embed_dim": int(g_feats.shape[1]),
    "total_embed_time_s": round(clipl_time, 2),
    "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=1),
    "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=10),
}
del g_feats, q_feats; gc.collect()
print()

# --- 2.M.4 DINOv2 ViT-S/14 ---
print("Experiment 2.M.4: DINOv2 ViT-S/14 (Meta, self-supervised)")
sys.stdout.flush()
t0 = time.time()
g_feats = embed_dinov2(gallery_images, 'facebook/dinov2-small')
q_feats = embed_dinov2(query_images, 'facebook/dinov2-small')
dinov2_time = time.time() - t0
print(f"  Embed time: {dinov2_time:.1f}s  dim={g_feats.shape[1]}")
_, idx = faiss_search(g_feats, q_feats, k=20)
experiments['dinov2_vits14'] = {
    **eval_recall(query_pids, gallery_pids, idx, "DINOv2 ViT-S/14"),
    "embed_dim": int(g_feats.shape[1]),
    "total_embed_time_s": round(dinov2_time, 2),
    "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=1),
    "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=10),
}
dino_idx = idx
dino_g, dino_q = g_feats, q_feats
del g_feats, q_feats; gc.collect()
print()

# --- 2.M.5 ConvNeXt-Small ---
print("Experiment 2.M.5: ConvNeXt-Small (modern CNN)")
sys.stdout.flush()
t0 = time.time()
g_feats = embed_torchvision('convnext_small', gallery_images)
q_feats = embed_torchvision('convnext_small', query_images)
convnext_time = time.time() - t0
print(f"  Embed time: {convnext_time:.1f}s  dim={g_feats.shape[1]}")
_, idx = faiss_search(g_feats, q_feats, k=20)
experiments['convnext_small'] = {
    **eval_recall(query_pids, gallery_pids, idx, "ConvNeXt-Small"),
    "embed_dim": int(g_feats.shape[1]),
    "total_embed_time_s": round(convnext_time, 2),
    "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=1),
    "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=10),
}
del g_feats, q_feats; gc.collect()
print()

# --- 2.M.6 CLIP ViT-B/32 + color rerank (extend Phase 1 insight) ---
print("Experiment 2.M.6: CLIP ViT-B/32 + color rerank (α=0.5)")
sys.stdout.flush()
sims, idx_clip = faiss_search(clipb_g, clipb_q, k=20)
reranked = np.zeros_like(idx_clip)
for i in range(len(idx_clip)):
    cand = idx_clip[i]
    cnn_s = sims[i]
    qc = query_colors[i] / (np.linalg.norm(query_colors[i]) + 1e-8)
    gc_ = gallery_colors[cand] / (np.linalg.norm(gallery_colors[cand], axis=1, keepdims=True) + 1e-8)
    color_s = gc_ @ qc
    blended = 0.5 * cnn_s + 0.5 * color_s
    reranked[i] = cand[np.argsort(-blended)]
experiments['clip_vitb32_color_rerank_alpha05'] = {
    **eval_recall(query_pids, gallery_pids, reranked, "CLIP-B/32 + color rerank"),
    "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, reranked, k=1),
    "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, reranked, k=10),
}
print()

# --- 2.M.7 DINOv2 + color rerank ---
print("Experiment 2.M.7: DINOv2 + color rerank (α=0.5)")
sys.stdout.flush()
sims_d, idx_dino = faiss_search(dino_g, dino_q, k=20)
reranked = np.zeros_like(idx_dino)
for i in range(len(idx_dino)):
    cand = idx_dino[i]
    cnn_s = sims_d[i]
    qc = query_colors[i] / (np.linalg.norm(query_colors[i]) + 1e-8)
    gc_ = gallery_colors[cand] / (np.linalg.norm(gallery_colors[cand], axis=1, keepdims=True) + 1e-8)
    color_s = gc_ @ qc
    blended = 0.5 * cnn_s + 0.5 * color_s
    reranked[i] = cand[np.argsort(-blended)]
experiments['dinov2_color_rerank_alpha05'] = {
    **eval_recall(query_pids, gallery_pids, reranked, "DINOv2 + color rerank"),
    "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, reranked, k=1),
    "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, reranked, k=10),
}
print()

# ===== 6. SAVE RESULTS =====
print("\n--- 6. Saving results ---")
sys.stdout.flush()

phase1_r1 = 0.3067  # Anthony's baseline
phase1_champ_r1 = 0.3671  # EfficientNet-B0
phase1_best_r1 = 0.4051  # ResNet50 + color rerank α=0.5

best_key = max(experiments, key=lambda k: experiments[k]['recall@1'])
best = experiments[best_key]
delta_vs_baseline = best['recall@1'] - phase1_r1
delta_vs_phase1_best = best['recall@1'] - phase1_best_r1

phase2_meta = {
    "description": "Mark Phase 2: Foundation models (CLIP, DINOv2, ConvNeXt) vs CNN baselines",
    "date": "2026-04-21",
    "eval_products": EVAL_PRODUCTS,
    "eval_gallery": len(gallery_images),
    "eval_queries": len(query_images),
    "seed": SEED,
    "experiments": experiments,
    "best_result": {
        "approach": best_key,
        "recall@1": best['recall@1'],
        "recall@10": best['recall@10'],
        "delta_vs_anthony_resnet50": round(delta_vs_baseline, 4),
        "delta_vs_phase1_best": round(delta_vs_phase1_best, 4),
    },
    "headline_finding": "",  # filled in after analysis
}

# Load + merge into existing metrics.json
metrics_path = RESULTS / 'metrics.json'
with open(metrics_path) as f:
    all_metrics = json.load(f)
all_metrics['phase2_mark'] = phase2_meta
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"  Saved metrics → {metrics_path}")
sys.stdout.flush()

# ===== 7. PLOTS =====
print("\n--- 7. Generating plots ---")
sys.stdout.flush()

# 7a. Overall model comparison
labels = {
    'efficientnet_b0_p1_rerun': 'EfficientNet-B0\n(ImageNet CNN)',
    'clip_vit_b32': 'CLIP ViT-B/32\n(text-aligned)',
    'clip_vit_l14': 'CLIP ViT-L/14\n(text-aligned)',
    'dinov2_vits14': 'DINOv2 ViT-S/14\n(self-supervised)',
    'convnext_small': 'ConvNeXt-Small\n(modern CNN)',
    'clip_vitb32_color_rerank_alpha05': 'CLIP-B/32\n+ color rerank',
    'dinov2_color_rerank_alpha05': 'DINOv2\n+ color rerank',
}
order = list(labels.keys())
r1s = [experiments[k]['recall@1'] for k in order]
r10s = [experiments[k]['recall@10'] for k in order]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors_plot = ['#3498db', '#e74c3c', '#c0392b', '#27ae60', '#9b59b6', '#f39c12', '#16a085']
axes[0].bar([labels[k] for k in order], r1s, color=colors_plot)
axes[0].axhline(phase1_r1, color='grey', linestyle='--', label=f'Phase 1 ResNet50 baseline ({phase1_r1:.3f})')
axes[0].axhline(phase1_best_r1, color='black', linestyle=':', label=f'Phase 1 best (ResNet50+color rerank) ({phase1_best_r1:.3f})')
axes[0].set_ylabel('Recall@1', fontsize=12)
axes[0].set_title('Phase 2: Foundation Models vs CNNs — Recall@1', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].tick_params(axis='x', rotation=25, labelsize=9)
for i, v in enumerate(r1s):
    axes[0].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

axes[1].bar([labels[k] for k in order], r10s, color=colors_plot)
axes[1].axhline(0.5901, color='grey', linestyle='--', label='Phase 1 ResNet50 (0.590)')
axes[1].axhline(0.6573, color='black', linestyle=':', label='Phase 1 best (0.657)')
axes[1].set_ylabel('Recall@10', fontsize=12)
axes[1].set_title('Phase 2: Foundation Models vs CNNs — Recall@10', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].tick_params(axis='x', rotation=25, labelsize=9)
for i, v in enumerate(r10s):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS / 'phase2_mark_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: phase2_mark_comparison.png")
sys.stdout.flush()

# 7b. Per-category R@10 heatmap
model_keys = ['efficientnet_b0_p1_rerun', 'clip_vit_b32', 'clip_vit_l14',
              'dinov2_vits14', 'convnext_small', 'clip_vitb32_color_rerank_alpha05',
              'dinov2_color_rerank_alpha05']
model_labels = ['EfficientNet-B0', 'CLIP-B/32', 'CLIP-L/14', 'DINOv2-S/14',
                'ConvNeXt-S', 'CLIP-B/32+color', 'DINOv2+color']
categories = sorted({c for k in model_keys for c in experiments[k]['per_category_r10'].keys()})

heatmap_data = np.zeros((len(model_keys), len(categories)))
for i, k in enumerate(model_keys):
    for j, c in enumerate(categories):
        heatmap_data[i, j] = experiments[k]['per_category_r10'].get(c, {}).get('recall@1', 0)

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=categories, yticklabels=model_labels, ax=ax,
            cbar_kws={'label': 'Recall@10'})
ax.set_title('Phase 2: Recall@10 by Model × Category', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS / 'phase2_mark_category_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: phase2_mark_category_heatmap.png")
sys.stdout.flush()

# 7c. Latency vs R@1 scatter
speeds = {
    'efficientnet_b0_p1_rerun': (eb0_time, '#3498db'),
    'clip_vit_b32': (clipb_time, '#e74c3c'),
    'clip_vit_l14': (clipl_time, '#c0392b'),
    'dinov2_vits14': (dinov2_time, '#27ae60'),
    'convnext_small': (convnext_time, '#9b59b6'),
}
fig, ax = plt.subplots(figsize=(10, 6))
for k, (t, c) in speeds.items():
    r1 = experiments[k]['recall@1']
    ax.scatter(t, r1, s=200, color=c, edgecolors='black', zorder=3)
    ax.annotate(labels[k].replace('\n', ' '), xy=(t, r1),
                xytext=(5, 8), textcoords='offset points', fontsize=9)
ax.set_xlabel('Total embed time (s) on 1,327 images — CPU', fontsize=11)
ax.set_ylabel('Recall@1', fontsize=11)
ax.set_title('Accuracy vs Embedding Speed (CPU)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS / 'phase2_mark_speed_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: phase2_mark_speed_accuracy.png")
sys.stdout.flush()

# ===== 8. FINAL SUMMARY =====
print("\n" + "=" * 72)
print("PHASE 2 MARK — FINAL SUMMARY")
print("=" * 72)
print(f"\n  Best approach: {best_key}")
print(f"  R@1: {best['recall@1']:.4f}  (Δ vs Anthony baseline {phase1_r1:.3f}: {delta_vs_baseline:+.4f})")
print(f"  R@10: {best['recall@10']:.4f}")
print()
print("  Full leaderboard (sorted by R@1):")
for k in sorted(experiments, key=lambda x: -experiments[x]['recall@1']):
    e = experiments[k]
    t = speeds.get(k, (None, None))[0]
    t_str = f"{t:6.1f}s" if t else "      —"
    print(f"    {k:40s}  R@1={e['recall@1']:.4f}  R@10={e['recall@10']:.4f}  embed={t_str}")
print()
print("  Done.")
sys.stdout.flush()
