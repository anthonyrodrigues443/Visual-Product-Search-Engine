"""
Phase 2 Mark — Resilient runner.

Same experiments as run_phase2_mark.py but:
  1. Saves metrics.json AFTER EACH experiment (not only at end).
  2. Wraps each experiment in try/except so one failure doesn't kill the session.
  3. Skips already-completed experiments on re-run (by reading phase2_mark from metrics.json).

Use this as the fallback if run_phase2_mark.py crashes mid-run.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import json
import time
import traceback
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
METRICS_PATH = RESULTS / 'metrics.json'

from src.data_pipeline import create_retrieval_splits
from src.feature_engineering import (
    extract_color_palette, extract_hsv_histogram,
)

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


def save_results(phase2_state):
    with open(METRICS_PATH) as f:
        all_metrics = json.load(f)
    all_metrics['phase2_mark'] = phase2_state
    with open(METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)


def load_or_init_phase2():
    with open(METRICS_PATH) as f:
        m = json.load(f)
    p2 = m.get('phase2_mark') or {}
    if 'experiments' not in p2:
        p2 = {
            "description": "Mark Phase 2: Foundation models (CLIP, DINOv2, ConvNeXt) vs CNN baselines",
            "date": "2026-04-21",
            "eval_products": EVAL_PRODUCTS,
            "seed": SEED,
            "experiments": {},
        }
    return p2


# ===== 1. METADATA + SPLITS =====
print("=" * 72)
print("PHASE 2 MARK (RESILIENT) — FOUNDATION MODELS VS CNNs")
print("=" * 72)
sys.stdout.flush()

df = pd.read_csv(DATA_PROCESSED / 'metadata.csv')
train_df, gallery_df, query_df = create_retrieval_splits(df, test_frac=0.2, seed=SEED)

eval_products = gallery_df['product_id'].values[:EVAL_PRODUCTS]
eval_gallery = gallery_df[gallery_df['product_id'].isin(eval_products)].reset_index(drop=True)
eval_query = query_df[query_df['product_id'].isin(eval_products)].reset_index(drop=True)

gallery_pids = eval_gallery['product_id'].values
query_pids = eval_query['product_id'].values
query_cats = eval_query['category2'].values

idx_to_item = {}
for _, row in eval_gallery.iterrows():
    idx_to_item[int(row['index'])] = row['item_id']
for _, row in eval_query.iterrows():
    idx_to_item[int(row['index'])] = row['item_id']

# ===== 2. LOAD CACHED IMAGES (skip downloading this time — assume pre-cached) =====
cached = {idx for idx, item_id in idx_to_item.items()
          if (IMG_CACHE / f"{item_id}.jpg").exists()}
missing = set(idx_to_item.keys()) - cached
print(f"\nCache: {len(cached)} cached, {len(missing)} missing")
sys.stdout.flush()

if missing and os.environ.get('SKIP_DOWNLOAD', '0') != '1':
    # Download missing ones with a hard per-run time budget
    TIME_BUDGET_S = int(os.environ.get('DOWNLOAD_BUDGET_S', '600'))  # 10 min hard cap
    print(f"  Downloading missing images (budget {TIME_BUDGET_S}s) ...")
    sys.stdout.flush()
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '30')
    start = time.time()
    try:
        from datasets import load_dataset
        ds = load_dataset("Marqo/deepfashion-inshop", split="data", streaming=True)
        needed = set(missing)
        for i, ex in enumerate(tqdm(ds, total=max(needed) + 1, desc="  download")):
            if time.time() - start > TIME_BUDGET_S:
                print(f"\n  Time budget exhausted at {len(missing) - len(needed)} of {len(missing)} — proceeding with cached")
                sys.stdout.flush()
                break
            if i in needed:
                img = ex['image']
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(IMG_CACHE / f"{idx_to_item[i]}.jpg", "JPEG", quality=90)
                needed.discard(i)
                if not needed:
                    break
    except Exception as e:
        print(f"  Download errored: {e} — continuing with what's cached")
        sys.stdout.flush()
elif missing:
    print(f"  SKIP_DOWNLOAD=1 set — proceeding with {len(cached)} cached images only")
    sys.stdout.flush()

cached = {idx for idx, item_id in idx_to_item.items()
          if (IMG_CACHE / f"{item_id}.jpg").exists()}
# Keep only items whose images we actually have
eval_gallery = eval_gallery[eval_gallery['index'].isin(cached)].reset_index(drop=True)
eval_query = eval_query[eval_query['index'].isin(cached)].reset_index(drop=True)

# Preserve product match: only keep queries whose product_id still appears in gallery
gallery_pid_set = set(eval_gallery['product_id'].values)
eval_query = eval_query[eval_query['product_id'].isin(gallery_pid_set)].reset_index(drop=True)

gallery_pids = eval_gallery['product_id'].values
query_pids = eval_query['product_id'].values
query_cats = eval_query['category2'].values

gallery_images = [Image.open(IMG_CACHE / f"{row['item_id']}.jpg").convert('RGB')
                  for _, row in eval_gallery.iterrows()]
query_images = [Image.open(IMG_CACHE / f"{row['item_id']}.jpg").convert('RGB')
                for _, row in eval_query.iterrows()]
print(f"Loaded: {len(gallery_images)} gallery + {len(query_images)} query images")
sys.stdout.flush()

# ===== 3. STATE =====
phase2 = load_or_init_phase2()
phase2['eval_gallery'] = len(gallery_images)
phase2['eval_queries'] = len(query_images)
save_results(phase2)

# ===== 4. COLOR FEATURES =====
print("\nPrecomputing color features ...")
sys.stdout.flush()
gallery_colors = np.stack([
    np.concatenate([extract_color_palette(im), extract_hsv_histogram(im)])
    for im in gallery_images
]).astype(np.float32)
query_colors = np.stack([
    np.concatenate([extract_color_palette(im), extract_hsv_histogram(im)])
    for im in query_images
]).astype(np.float32)

# ===== 5. EXPERIMENTS (each wrapped in try/except) =====
import torch
device = torch.device('cpu')
torch.set_num_threads(2)


def run_experiment(name, embed_fn):
    """Execute an experiment, save on success, skip on failure."""
    if name in phase2.get('experiments', {}):
        print(f"[SKIP] {name} — already done")
        return phase2['experiments'][name]
    print(f"\n--- Experiment: {name} ---")
    sys.stdout.flush()
    try:
        t0 = time.time()
        g_feats, q_feats = embed_fn()
        elapsed = time.time() - t0
        print(f"  Embed time: {elapsed:.1f}s  dim={g_feats.shape[1]}")
        sys.stdout.flush()
        _, idx = faiss_search(g_feats, q_feats, k=20)
        recall = eval_recall(query_pids, gallery_pids, idx, name)
        result = {
            **recall,
            "embed_dim": int(g_feats.shape[1]),
            "total_embed_time_s": round(elapsed, 2),
            "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=1),
            "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, idx, k=10),
        }
        phase2['experiments'][name] = result
        save_results(phase2)
        # Return features for potential rerank experiments
        return result, g_feats, q_feats, idx
    except Exception as e:
        print(f"  [ERROR in {name}] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return None


def embed_torchvision_fn(model_name):
    def fn():
        from torchvision import models as tv_models
        factory = getattr(tv_models, model_name)
        weights_enum = getattr(tv_models, {
            'efficientnet_b0': 'EfficientNet_B0_Weights',
            'convnext_small': 'ConvNeXt_Small_Weights',
        }[model_name])
        w = weights_enum.DEFAULT
        model = factory(weights=w).eval().to(device)
        if model_name == 'efficientnet_b0':
            model.classifier = torch.nn.Identity()
        elif model_name == 'convnext_small':
            model.classifier[-1] = torch.nn.Identity()
        tf = w.transforms()

        def encode(images):
            feats = []
            with torch.no_grad():
                for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  {model_name}"):
                    batch = torch.stack([tf(im) for im in images[i:i + BATCH_SIZE]]).to(device)
                    feats.append(model(batch).cpu().numpy())
            return np.concatenate(feats, axis=0).astype(np.float32)

        return encode(gallery_images), encode(query_images)
    return fn


def embed_openclip_fn(arch, pretrained):
    def fn():
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = model.eval().to(device)

        def encode(images):
            feats = []
            with torch.no_grad():
                for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  CLIP-{arch}"):
                    batch = torch.stack([preprocess(im) for im in images[i:i + BATCH_SIZE]]).to(device)
                    feats.append(model.encode_image(batch).cpu().numpy())
            return np.concatenate(feats, axis=0).astype(np.float32)

        return encode(gallery_images), encode(query_images)
    return fn


def embed_hf_clip_fn(model_id):
    """Embed with HuggingFace transformers CLIP — uses HF hub cache (more reliable than open_clip)."""
    def fn():
        from transformers import CLIPModel, CLIPProcessor
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id).eval().to(device)

        def encode(images):
            feats = []
            with torch.no_grad():
                for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  CLIP-hf-{model_id.split('/')[-1]}"):
                    inputs = processor(images=images[i:i + BATCH_SIZE], return_tensors="pt").to(device)
                    feats.append(model.get_image_features(**inputs).cpu().numpy())
            return np.concatenate(feats, axis=0).astype(np.float32)

        return encode(gallery_images), encode(query_images)
    return fn


def embed_dinov2_fn(model_id='facebook/dinov2-small'):
    def fn():
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).eval().to(device)

        def encode(images):
            feats = []
            with torch.no_grad():
                for i in tqdm(range(0, len(images), BATCH_SIZE), desc=f"  DINOv2"):
                    inputs = processor(images=images[i:i + BATCH_SIZE], return_tensors="pt").to(device)
                    cls = model(**inputs).last_hidden_state[:, 0, :]
                    feats.append(cls.cpu().numpy())
            return np.concatenate(feats, axis=0).astype(np.float32)

        return encode(gallery_images), encode(query_images)
    return fn


def color_rerank_fn(base_g, base_q):
    sims, idx = faiss_search(base_g, base_q, k=20)
    reranked = np.zeros_like(idx)
    for i in range(len(idx)):
        cand = idx[i]
        cnn_s = sims[i]
        qc = query_colors[i] / (np.linalg.norm(query_colors[i]) + 1e-8)
        gc_ = gallery_colors[cand] / (np.linalg.norm(gallery_colors[cand], axis=1, keepdims=True) + 1e-8)
        color_s = gc_ @ qc
        blended = 0.5 * cnn_s + 0.5 * color_s
        reranked[i] = cand[np.argsort(-blended)]
    recall = eval_recall(query_pids, gallery_pids, reranked, "color rerank")
    return {
        **recall,
        "per_category_r1": per_category_recall(query_pids, query_cats, gallery_pids, reranked, k=1),
        "per_category_r10": per_category_recall(query_pids, query_cats, gallery_pids, reranked, k=10),
    }


# Run backbones
backbone_results = {}
spec = [
    ('efficientnet_b0_p1_rerun', embed_torchvision_fn('efficientnet_b0')),
    ('clip_vit_b32', embed_hf_clip_fn('openai/clip-vit-base-patch32')),
    ('dinov2_vits14', embed_dinov2_fn('facebook/dinov2-small')),
    # ConvNeXt-Small: pytorch.org download throttled to ~36kB/s (would take 90min on this run) — deferred.
    # CLIP ViT-L/14 (890MB, 428M params): caused OOM on this CPU-only Windows machine — deferred.
    # Phase 2 headline is decisive with CLIP-B/32 (R@1=0.48) vs Phase 1 best (0.405).
]
BACKBONES_FOR_RERANK = ['clip_vit_b32', 'dinov2_vits14']
for name, fn in spec:
    r = run_experiment(name, fn)
    if r is not None and isinstance(r, tuple):
        _, g_feats, q_feats, _ = r
        backbone_results[name] = (g_feats, q_feats)
    gc.collect()

# Color rerank on CLIP-B/32 and DINOv2 — re-embed if backbones were skipped/already-cached
spec_dict = dict(spec)
for backbone in BACKBONES_FOR_RERANK:
    name = f"{backbone}_color_rerank_alpha05"
    if name in phase2.get('experiments', {}):
        print(f"[SKIP] {name} — already done")
        continue
    print(f"\n--- Experiment: {name} ---")
    sys.stdout.flush()
    try:
        if backbone in backbone_results:
            g, q = backbone_results[backbone]
        else:
            # Backbone was skipped earlier (already in metrics), but we need features for rerank.
            if backbone not in spec_dict:
                print(f"  Cannot rerank {backbone} — not in spec")
                continue
            print(f"  Re-embedding {backbone} for rerank ...")
            sys.stdout.flush()
            g, q = spec_dict[backbone]()
        phase2['experiments'][name] = color_rerank_fn(g, q)
        save_results(phase2)
        gc.collect()
    except Exception as e:
        import traceback
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        sys.stdout.flush()

# ===== 6. SUMMARY =====
print("\n" + "=" * 72)
print("PHASE 2 MARK (RESILIENT) — SUMMARY")
print("=" * 72)
exps = phase2['experiments']
if not exps:
    print("  No experiments succeeded.")
    sys.exit(1)
best_key = max(exps, key=lambda k: exps[k]['recall@1'])
best = exps[best_key]
phase1_r1 = 0.3067
phase1_best = 0.4051
phase2['best_result'] = {
    "approach": best_key,
    "recall@1": best['recall@1'],
    "recall@10": best['recall@10'],
    "delta_vs_anthony_resnet50": round(best['recall@1'] - phase1_r1, 4),
    "delta_vs_phase1_best": round(best['recall@1'] - phase1_best, 4),
}
save_results(phase2)

print(f"\n  Best: {best_key}  R@1={best['recall@1']:.4f}")
for k in sorted(exps, key=lambda x: -exps[x]['recall@1']):
    e = exps[k]
    t = e.get('total_embed_time_s', '—')
    t_str = f"{t:6.1f}s" if isinstance(t, (int, float)) else "     —"
    print(f"    {k:42s} R@1={e['recall@1']:.4f}  R@10={e['recall@10']:.4f}  embed={t_str}")
sys.stdout.flush()
print("\n  Done.")
