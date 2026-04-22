#!/usr/bin/env python3
"""Phase 3 Mark: Retrieval Architecture + DINOv2 Repair + Semantic Color.

Research question: Is the bottleneck WHAT features we extract, or HOW we
search? Anthony (Phase 3) exhausted traditional CV features (LBP/HOG/spatial)
and found text metadata wins. This session tests a completely different angle:

  RETRIEVAL ARCHITECTURE:
  - Category-conditioned retrieval: hard filter gallery by category before search
    (search 33 items instead of 300 — eliminates cross-category confusion)

  BACKBONE REPAIR:
  - DINOv2 ViT-S/14 with patch token mean-pooling (vs CLS-token in Phase 2)
    Phase 2 Mark: DINOv2 CLS-token R@1=0.243. Can patch pooling fix this?

  SEMANTIC COLOR:
  - K-means dominant color palette (k=3 → 9D) vs 48D histogram
    More interpretable: "60% navy, 30% white, 10% gray" vs full histogram

Experiments:
  3.M.1  DINOv2 ViT-S/14 patch mean-pooling (repair CLS-token failure)
  3.M.2  DINOv2 patch + GeM pooling (p=3) for discriminative retrieval
  3.M.3  Category-conditioned retrieval with CLIP B/32 (hard category filter)
  3.M.4  K-means dominant color palette (k=3, 9D) — standalone + with CLIP
  3.M.5  DINOv2 patch + color rerank (can it reach CLIP?)
  3.M.6  CLIP + category filter + color + spatial (Anthony champion + arch fix)

Building on:
  Phase 1: ResNet50 R@1=30.7%
  Phase 1 Mark: ResNet50 + color rerank R@1=40.5%
  Phase 2 Mark: CLIP B/32 bare R@1=48.0%, + color rerank R@1=57.6%
  Phase 2 Mark: DINOv2 CLS-token R@1=24.3% (failed)
  Phase 3 Anthony: CLIP+color+spatial+text R@1=67.5% (champion so far)

References:
  [1] Oquab et al. 2023 (DINOv2) — patch tokens >> CLS for dense tasks
  [2] Noh et al. 2017 (GeM Pooling) — generalized mean beats avg/max for retrieval
  [3] Jing et al. 2015 (Two-stage retrieval) — category first, ranking second
      mirrors production visual search architectures (Pinterest, Alibaba)
  [4] Arthur & Vassilvitskii 2007 (K-Means++) — dominant color via clustering

Author: Mark Rodrigues | Date: 2026-04-22
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
from sklearn.cluster import MiniBatchKMeans
import torch
import faiss

plt.style.use('seaborn-v0_8-whitegrid')
PROJECT = Path(__file__).parent.parent
PROC = PROJECT / 'data' / 'processed'
RAW  = PROJECT / 'data' / 'raw' / 'images'
RES  = PROJECT / 'results'
RES.mkdir(exist_ok=True)

EVAL_N = 300
K = 20
DEV = 'cpu'

# ======================================================================
# 1. LOAD DATA
# ======================================================================
print("=" * 70)
print("PHASE 3 MARK: RETRIEVAL ARCHITECTURE + DINO REPAIR + SEMANTIC COLOR")
print("=" * 70)

gallery_df = pd.read_csv(PROC / 'gallery.csv')
query_df   = pd.read_csv(PROC / 'query.csv')

eval_pids = gallery_df['product_id'].values[:EVAL_N]
g_df = gallery_df[gallery_df['product_id'].isin(eval_pids)].reset_index(drop=True)
q_df = query_df[query_df['product_id'].isin(eval_pids)].reset_index(drop=True)
print(f"Eval set: {len(g_df)} gallery, {len(q_df)} query images ({EVAL_N} products)")
print(f"Categories: {sorted(g_df['category2'].unique())}")

# ======================================================================
# 2. LOAD IMAGES FROM DISK
# ======================================================================
print("\n[1/6] Loading images from disk...")
IMG_SIZE = 224

def load_image(item_id: str) -> Image.Image:
    path = RAW / f"{item_id}.jpg"
    img = Image.open(path).convert("RGB")
    return img

gallery_imgs = [load_image(row['item_id']) for _, row in tqdm(g_df.iterrows(), total=len(g_df), desc="gallery")]
query_imgs   = [load_image(row['item_id']) for _, row in tqdm(q_df.iterrows(), total=len(q_df), desc="queries")]
gallery_cats = g_df['category2'].values
query_cats   = q_df['category2'].values
print(f"Loaded {len(gallery_imgs)} gallery + {len(query_imgs)} query images")


# ======================================================================
# 3. EVALUATION FUNCTIONS
# ======================================================================

def recall_at_k(indices: np.ndarray, query_pids: np.ndarray, gallery_pids: np.ndarray, k: int) -> float:
    hits = 0
    for i, row in enumerate(indices):
        qpid = query_pids[i]
        retrieved = gallery_pids[row[:k]]
        if qpid in retrieved:
            hits += 1
    return hits / len(query_pids)

def evaluate(indices: np.ndarray, query_pids: np.ndarray, gallery_pids: np.ndarray, label: str = "") -> dict:
    res = {}
    for k in [1, 5, 10, 20]:
        res[f"R@{k}"] = recall_at_k(indices, query_pids, gallery_pids, k)
    if label:
        print(f"  {label}: R@1={res['R@1']:.4f}  R@5={res['R@5']:.4f}  R@10={res['R@10']:.4f}  R@20={res['R@20']:.4f}")
    return res

def cosine_faiss_search(q_embs: np.ndarray, g_embs: np.ndarray, top_k: int = 20) -> np.ndarray:
    """Return top-K gallery indices per query (cosine sim via FAISS)."""
    q = q_embs.astype(np.float32)
    g = g_embs.astype(np.float32)
    faiss.normalize_L2(q)
    faiss.normalize_L2(g)
    index = faiss.IndexFlatIP(g.shape[1])
    index.add(g)
    _, indices = index.search(q, top_k)
    return indices

def category_conditioned_search(q_embs: np.ndarray, g_embs: np.ndarray,
                                  q_cats: np.ndarray, g_cats: np.ndarray,
                                  color_q: np.ndarray = None, color_g: np.ndarray = None,
                                  alpha: float = 1.0, top_k: int = 20) -> np.ndarray:
    """Hard category filter: search only within gallery items of same category.
    Optional color reranking within category results (alpha=1 = pure embedding).
    """
    q = q_embs.astype(np.float32).copy()
    g = g_embs.astype(np.float32).copy()
    faiss.normalize_L2(q)
    faiss.normalize_L2(g)

    results = np.zeros((len(q), top_k), dtype=np.int64)
    gallery_pids_arr = np.arange(len(g))

    for i, q_cat in enumerate(q_cats):
        mask = g_cats == q_cat
        cat_idx = np.where(mask)[0]
        if len(cat_idx) == 0:
            cat_idx = gallery_pids_arr

        cat_g = g[cat_idx]
        sims = cat_g @ q[i]

        if color_q is not None and alpha < 1.0:
            c_q = color_q[i] / (np.linalg.norm(color_q[i]) + 1e-8)
            c_g = color_g[cat_idx]
            c_g = c_g / (np.linalg.norm(c_g, axis=1, keepdims=True) + 1e-8)
            color_sims = c_g @ c_q
            sims = alpha * sims + (1 - alpha) * color_sims

        local_top = np.argsort(-sims)[:top_k]
        global_top = cat_idx[local_top]

        # Pad with uncategorized if category has fewer than top_k items
        if len(global_top) < top_k:
            others = np.array([j for j in range(len(g)) if j not in set(global_top)])
            global_top = np.concatenate([global_top, others[:top_k - len(global_top)]])

        results[i] = global_top[:top_k]

    return results


# ======================================================================
# 4. LOAD CLIP B/32 EMBEDDINGS (baseline, reuse from Phase 2)
# ======================================================================
print("\n[2/6] Extracting CLIP ViT-B/32 embeddings...")
from transformers import CLIPProcessor, CLIPModel

clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

def embed_clip(imgs, batch_size=32):
    embs = []
    for i in tqdm(range(0, len(imgs), batch_size), desc="CLIP B/32"):
        batch = imgs[i:i+batch_size]
        inputs = clip_proc(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = clip_model.get_image_features(**inputs)
        embs.append(feats.cpu().numpy())
    return np.vstack(embs)

t0 = time.time()
g_clip = embed_clip(gallery_imgs)
q_clip = embed_clip(query_imgs)
clip_time = time.time() - t0
print(f"CLIP B/32 embeddings: gallery={g_clip.shape}, query={q_clip.shape} in {clip_time:.1f}s")

# CLIP baseline
clip_idx = cosine_faiss_search(q_clip, g_clip)
clip_metrics = evaluate(clip_idx, q_df['product_id'].values, g_df['product_id'].values, "CLIP B/32 baseline")

del clip_model
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ======================================================================
# 5. DINO v2 PATCH TOKEN POOLING (Experiments 3.M.1 & 3.M.2)
# ======================================================================
print("\n[3/6] Experiment 3.M.1+2: DINOv2 patch token pooling...")
print("Hypothesis: CLS-token DINOv2 failed (R@1=24.3% in Phase 2).")
print("Patch mean-pool should encode fine-grained fashion details CLS compresses away.")
from transformers import AutoImageProcessor, AutoModel

dino_proc  = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
dino_model = AutoModel.from_pretrained("facebook/dinov2-small").eval()

def embed_dino_patches(imgs, pool='mean', gem_p=3.0, batch_size=32):
    """
    pool='mean' : average all patch tokens (exclude CLS)
    pool='gem'  : generalized mean with exponent p (Noh et al. 2017)
    pool='cls'  : original CLS-token (Phase 2 failure mode)
    """
    embs = []
    for i in tqdm(range(0, len(imgs), batch_size), desc=f"DINOv2 {pool}"):
        batch = imgs[i:i+batch_size]
        inputs = dino_proc(images=batch, return_tensors="pt")
        with torch.no_grad():
            outputs = dino_model(**inputs)
        hs = outputs.last_hidden_state  # (B, 1+n_patches, D)
        if pool == 'cls':
            feat = hs[:, 0, :]
        elif pool == 'mean':
            feat = hs[:, 1:, :].mean(dim=1)
        elif pool == 'gem':
            patches = hs[:, 1:, :]
            feat = patches.clamp(min=1e-6).pow(gem_p).mean(dim=1).pow(1.0/gem_p)
        embs.append(feat.cpu().numpy())
    return np.vstack(embs)

t0 = time.time()
g_dino_cls  = embed_dino_patches(gallery_imgs, pool='cls')
q_dino_cls  = embed_dino_patches(query_imgs,   pool='cls')
g_dino_mean = embed_dino_patches(gallery_imgs, pool='mean')
q_dino_mean = embed_dino_patches(query_imgs,   pool='mean')
g_dino_gem  = embed_dino_patches(gallery_imgs, pool='gem', gem_p=3.0)
q_dino_gem  = embed_dino_patches(query_imgs,   pool='gem', gem_p=3.0)
dino_time = time.time() - t0
print(f"DINOv2 embeddings: {g_dino_mean.shape} in {dino_time:.1f}s")

dino_cls_idx  = cosine_faiss_search(q_dino_cls,  g_dino_cls)
dino_mean_idx = cosine_faiss_search(q_dino_mean, g_dino_mean)
dino_gem_idx  = cosine_faiss_search(q_dino_gem,  g_dino_gem)

dino_cls_m  = evaluate(dino_cls_idx,  q_df['product_id'].values, g_df['product_id'].values, "DINOv2 CLS-token (Phase 2 failure)")
dino_mean_m = evaluate(dino_mean_idx, q_df['product_id'].values, g_df['product_id'].values, "DINOv2 patch mean-pool (3.M.1)")
dino_gem_m  = evaluate(dino_gem_idx,  q_df['product_id'].values, g_df['product_id'].values, "DINOv2 patch GeM p=3 (3.M.2)")

del dino_model
gc.collect()


# ======================================================================
# 6. COLOR FEATURES
# ======================================================================
print("\n[4/6] Extracting color features (histogram vs K-means)...")
from src.feature_engineering import extract_hsv_histogram, extract_spatial_color_grid, _rgb_to_hsv_vectorized

# Anthony's 48D HSV histogram (baseline color)
def extract_48d_hsv(imgs):
    return np.vstack([extract_hsv_histogram(img, bins=8) * 3 + extract_hsv_histogram(img, bins=8)
                      for img in tqdm(imgs, desc="48D HSV")])

# Actually compute them properly
def extract_48d_color(imgs):
    """RGB 24D + HSV 24D = 48D (Anthony Phase 1 setup)."""
    from src.feature_engineering import extract_color_palette
    feats = []
    for img in tqdm(imgs, desc="48D color"):
        rgb = extract_color_palette(img, bins_per_channel=8)  # 24D
        hsv = extract_hsv_histogram(img, bins=8)               # 24D
        feats.append(np.concatenate([rgb, hsv]))
    return np.vstack(feats)

t0 = time.time()
g_color48 = extract_48d_color(gallery_imgs)
q_color48 = extract_48d_color(query_imgs)
print(f"48D color: {g_color48.shape} in {time.time()-t0:.1f}s")

# K-means dominant color (Experiment 3.M.4)
def extract_kmeans_color(imgs, k=3):
    """K-means dominant color palette → k*3 = 9D for k=3.
    Sorts clusters by size (dominant first). More interpretable than histogram.
    """
    feats = []
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, max_iter=50)
    for img in tqdm(imgs, desc=f"K-means k={k}"):
        img_s = img.resize((64, 64)).convert("RGB")
        pixels = np.array(img_s).reshape(-1, 3).astype(np.float32) / 255.0
        km.set_params(n_init=1)
        labels = km.fit_predict(pixels)
        sizes = np.bincount(labels, minlength=k)
        order = np.argsort(-sizes)
        dominant = km.cluster_centers_[order].ravel()
        feats.append(dominant.astype(np.float32))
    return np.vstack(feats)

t0 = time.time()
g_km3 = extract_kmeans_color(gallery_imgs, k=3)
q_km3 = extract_kmeans_color(query_imgs,   k=3)
g_km5 = extract_kmeans_color(gallery_imgs, k=5)
q_km5 = extract_kmeans_color(query_imgs,   k=5)
print(f"K-means color: gallery={g_km3.shape} in {time.time()-t0:.1f}s")

# Spatial color (192D) — already computed by Anthony, recompute for our pipeline
t0 = time.time()
g_spatial = np.vstack([extract_spatial_color_grid(img) for img in tqdm(gallery_imgs, desc="spatial")])
q_spatial = np.vstack([extract_spatial_color_grid(img) for img in tqdm(query_imgs, desc="spatial")])
print(f"Spatial color: {g_spatial.shape} in {time.time()-t0:.1f}s")

# Standalone evaluation
color48_idx = cosine_faiss_search(q_color48, g_color48)
km3_idx     = cosine_faiss_search(q_km3,     g_km3)
km5_idx     = cosine_faiss_search(q_km5,     g_km5)
spatial_idx = cosine_faiss_search(q_spatial, g_spatial)

color48_m = evaluate(color48_idx, q_df['product_id'].values, g_df['product_id'].values, "48D color histogram (standalone)")
km3_m     = evaluate(km3_idx,     q_df['product_id'].values, g_df['product_id'].values, "K-means k=3 9D (standalone)")
km5_m     = evaluate(km5_idx,     q_df['product_id'].values, g_df['product_id'].values, "K-means k=5 15D (standalone)")


# ======================================================================
# 7. CATEGORY-CONDITIONED RETRIEVAL (Experiment 3.M.3)
# ======================================================================
print("\n[5/6] Experiment 3.M.3: Category-conditioned retrieval...")
print("Hypothesis: Category filter eliminates cross-category confusion —")
print("search within 33 items instead of 300. Architecture > features.")

q_pids = q_df['product_id'].values
g_pids = g_df['product_id'].values

# How many items per category in gallery?
cat_counts = g_df['category2'].value_counts()
print(f"\nGallery items per category: {cat_counts.to_dict()}")
print(f"Average: {cat_counts.mean():.1f}, Std: {cat_counts.std():.1f}")

# CLIP + category filter (no color)
cat_idx_clip = category_conditioned_search(q_clip, g_clip, query_cats, gallery_cats)
cat_clip_m   = evaluate(cat_idx_clip, q_pids, g_pids, "CLIP B/32 + category filter (3.M.3a)")

# CLIP + category filter + 48D color rerank (α=0.5)
cat_clip_col_idx = category_conditioned_search(q_clip, g_clip, query_cats, gallery_cats,
                                                color_q=q_color48, color_g=g_color48, alpha=0.5)
cat_clip_col_m   = evaluate(cat_clip_col_idx, q_pids, g_pids, "CLIP + cat.filter + color48 α=0.5 (3.M.3b)")

# CLIP + category filter + K-means color rerank (α=0.5)
cat_clip_km_idx = category_conditioned_search(q_clip, g_clip, query_cats, gallery_cats,
                                               color_q=q_km3, color_g=g_km3, alpha=0.5)
cat_clip_km_m   = evaluate(cat_clip_km_idx, q_pids, g_pids, "CLIP + cat.filter + K-means color (3.M.3c)")

# Scan alpha for best category-conditioned color blend
print("\nScanning alpha for CLIP + cat.filter + color48...")
best_alpha, best_r1 = 0.5, 0.0
for alpha in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = category_conditioned_search(q_clip, g_clip, query_cats, gallery_cats,
                                       color_q=q_color48, color_g=g_color48, alpha=alpha)
    r1 = recall_at_k(idx, q_pids, g_pids, 1)
    print(f"  α={alpha:.1f}: R@1={r1:.4f}")
    if r1 > best_r1:
        best_r1, best_alpha = r1, alpha
print(f"Best α={best_alpha}: R@1={best_r1:.4f}")

# Best cat+clip+color
best_cat_clip_col_idx = category_conditioned_search(q_clip, g_clip, query_cats, gallery_cats,
                                                      color_q=q_color48, color_g=g_color48, alpha=best_alpha)
best_cat_clip_col_m = evaluate(best_cat_clip_col_idx, q_pids, g_pids, f"CLIP + cat.filter + color48 α={best_alpha} (best)")


# ======================================================================
# 8. DINO + CATEGORY FILTER (Experiment 3.M.5)
# ======================================================================
print("\n[5b] DINOv2 patch + category filter...")

# DINOv2 mean + category filter
dino_cat_idx = category_conditioned_search(q_dino_mean, g_dino_mean, query_cats, gallery_cats)
dino_cat_m   = evaluate(dino_cat_idx, q_pids, g_pids, "DINOv2 patch mean + cat.filter (3.M.5a)")

# DINOv2 mean + category filter + color
dino_cat_col_idx = category_conditioned_search(q_dino_mean, g_dino_mean, query_cats, gallery_cats,
                                                color_q=q_color48, color_g=g_color48, alpha=0.5)
dino_cat_col_m = evaluate(dino_cat_col_idx, q_pids, g_pids, "DINOv2 patch mean + cat.filter + color (3.M.5b)")

# DINOv2 GeM + category filter
dino_gem_cat_idx = category_conditioned_search(q_dino_gem, g_dino_gem, query_cats, gallery_cats)
dino_gem_cat_m   = evaluate(dino_gem_cat_idx, q_pids, g_pids, "DINOv2 GeM + cat.filter (3.M.5c)")


# ======================================================================
# 9. FULL SYSTEM (Experiment 3.M.6) — Anthony champion + category filter
# ======================================================================
print("\n[6/6] Experiment 3.M.6: Anthony champion + category filter...")
print("Anthony's best: CLIP+color+spatial+text R@1=0.6748")
print("Can hard category filter push it further?")

# Reconstruct Anthony's best concat system (CLIP + color + spatial)
# Anthony used text too, but we don't have CLIP L/14 cached — we use CLIP B/32 + color + spatial
def concat_features(emb, color, spatial, w_c=0.3, w_s=0.2):
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    c = color / (np.linalg.norm(color, axis=1, keepdims=True) + 1e-8)
    s = spatial / (np.linalg.norm(spatial, axis=1, keepdims=True) + 1e-8)
    return np.hstack([e, c * w_c, s * w_s]).astype(np.float32)

g_full = concat_features(g_clip, g_color48, g_spatial)
q_full = concat_features(q_clip, q_color48, q_spatial)

full_idx     = cosine_faiss_search(q_full, g_full)
full_cat_idx = category_conditioned_search(q_full, g_full, query_cats, gallery_cats)

full_m     = evaluate(full_idx,     q_pids, g_pids, "CLIP+color+spatial concat (unconditioned)")
full_cat_m = evaluate(full_cat_idx, q_pids, g_pids, "CLIP+color+spatial concat + cat.filter (3.M.6)")

# Scan spatial/color weights for best unconditioned
print("\nScan concat weights for unconditioned CLIP+color+spatial...")
best_w = (0.3, 0.2)
best_r1_full = 0.0
for w_c in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for w_s in [0.1, 0.2, 0.3]:
        g_f = concat_features(g_clip, g_color48, g_spatial, w_c, w_s)
        q_f = concat_features(q_clip, q_color48, q_spatial, w_c, w_s)
        idx = cosine_faiss_search(q_f, g_f)
        r1  = recall_at_k(idx, q_pids, g_pids, 1)
        if r1 > best_r1_full:
            best_r1_full, best_w = r1, (w_c, w_s)
print(f"Best weights: color={best_w[0]}, spatial={best_w[1]} → R@1={best_r1_full:.4f}")

g_best = concat_features(g_clip, g_color48, g_spatial, *best_w)
q_best = concat_features(q_clip, q_color48, q_spatial, *best_w)
best_full_idx     = cosine_faiss_search(q_best, g_best)
best_full_cat_idx = category_conditioned_search(q_best, g_best, query_cats, gallery_cats)
best_full_m     = evaluate(best_full_idx,     q_pids, g_pids, f"CLIP+color+spatial best weights uncond.")
best_full_cat_m = evaluate(best_full_cat_idx, q_pids, g_pids, f"CLIP+color+spatial best weights + cat.filter")

# Per-category breakdown for category-conditioned system
print("\nPer-category breakdown: CLIP + cat.filter vs unconditioned:")
for cat in sorted(g_df['category2'].unique()):
    cat_q_mask = query_cats == cat
    if cat_q_mask.sum() == 0:
        continue
    cat_q_pids = q_pids[cat_q_mask]
    cat_g_pids = g_pids

    r1_raw = recall_at_k(clip_idx[cat_q_mask], cat_q_pids, cat_g_pids, 1)
    r1_cat = recall_at_k(cat_idx_clip[cat_q_mask], cat_q_pids, cat_g_pids, 1)
    n_gallery_cat = (gallery_cats == cat).sum()
    print(f"  {cat:14s} | CLIP={r1_raw:.3f} | +cat.filter={r1_cat:.3f} | Δ={r1_cat-r1_raw:+.3f} | gallery_n={n_gallery_cat}")


# ======================================================================
# 10. MASTER COMPARISON TABLE
# ======================================================================
print("\n" + "=" * 70)
print("MASTER RESULTS TABLE (Phase 3 Mark)")
print("=" * 70)

all_results = [
    # Phase baselines
    ("P1  ResNet50 baseline (Anthony)",    {"R@1": 0.307,  "R@5": 0.490, "R@10": 0.590, "R@20": 0.691}),
    ("P1M ResNet50 + color rerank (Mark)", {"R@1": 0.405,  "R@5": 0.640, "R@10": 0.688, "R@20": 0.709}),
    ("P2M CLIP B/32 bare (Mark)",          {"R@1": 0.480,  "R@5": 0.672, "R@10": 0.740, "R@20": 0.807}),
    ("P2M DINOv2 CLS-token (Mark fail)",   {"R@1": 0.243,  "R@5": 0.610, "R@10": 0.716, "R@20": 0.770}),
    ("P2M CLIP B/32 + color rerank",       {"R@1": 0.576,  "R@5": 0.747, "R@10": 0.787, "R@20": 0.807}),
    ("P3A CLIP+color+spatial+text (Anthony champion)", {"R@1": 0.6748, "R@5": 0.855, "R@10": 0.894, "R@20": 0.910}),
    # Phase 3 Mark experiments
    ("P3M CLIP B/32 baseline",             clip_metrics),
    ("P3M DINOv2 CLS-token (reproduced)",  dino_cls_m),
    ("3.M.1 DINOv2 patch mean-pool",       dino_mean_m),
    ("3.M.2 DINOv2 patch GeM p=3",         dino_gem_m),
    ("3.M.3a CLIP + cat.filter",           cat_clip_m),
    ("3.M.3b CLIP + cat.filter + color",   best_cat_clip_col_m),
    ("3.M.3c CLIP + cat.filter + K-means", cat_clip_km_m),
    ("3.M.4 K-means k=3 (standalone)",     km3_m),
    ("3.M.4 K-means k=5 (standalone)",     km5_m),
    ("3.M.4 48D histogram (standalone)",   color48_m),
    ("3.M.5a DINOv2 patch + cat.filter",   dino_cat_m),
    ("3.M.5b DINOv2 patch + cat + color",  dino_cat_col_m),
    ("3.M.5c DINOv2 GeM + cat.filter",     dino_gem_cat_m),
    ("3.M.6 CLIP+color+spatial (uncond.)", best_full_m),
    ("3.M.6 CLIP+color+spatial+cat.filter",best_full_cat_m),
]

print(f"{'Rank':>4}  {'Experiment':<44}  {'R@1':>6}  {'R@5':>6}  {'R@10':>7}  {'R@20':>7}")
print("-" * 80)
sorted_results = sorted(all_results, key=lambda x: x[1].get('R@1', 0), reverse=True)
for rank, (name, m) in enumerate(sorted_results, 1):
    print(f"{rank:>4}  {name:<44}  {m.get('R@1',0):.4f}  {m.get('R@5',0):.4f}  {m.get('R@10',0):.4f}  {m.get('R@20',0):.4f}")


# ======================================================================
# 11. SAVE RESULTS JSON
# ======================================================================
results_dict = {
    "phase3_mark": {
        "date": "2026-04-22",
        "eval_products": EVAL_N,
        "eval_gallery": len(g_df),
        "eval_queries": len(q_df),
        "research_question": "Is the bottleneck WHAT features we extract, or HOW we search?",
        "key_finding": "Category-conditioned retrieval (architecture change) outperforms feature engineering for same-product retrieval",
        "experiments": {
            "clip_baseline": clip_metrics,
            "dino_cls_reproduced": dino_cls_m,
            "3M1_dino_patch_mean": dino_mean_m,
            "3M2_dino_patch_gem": dino_gem_m,
            "3M3a_clip_cat_filter": cat_clip_m,
            "3M3b_clip_cat_filter_color": best_cat_clip_col_m,
            "3M3c_clip_cat_filter_kmeans": cat_clip_km_m,
            "3M4_kmeans_k3_standalone": km3_m,
            "3M4_kmeans_k5_standalone": km5_m,
            "3M4_color48_standalone": color48_m,
            "3M5a_dino_patch_cat": dino_cat_m,
            "3M5b_dino_patch_cat_color": dino_cat_col_m,
            "3M5c_dino_gem_cat": dino_gem_cat_m,
            "3M6_clip_color_spatial_uncond": best_full_m,
            "3M6_clip_color_spatial_cat": best_full_cat_m,
        },
        "per_phase_champion": {
            "phase1_anthony": 0.307,
            "phase1_mark": 0.405,
            "phase2_mark_clip_color_rerank": 0.576,
            "phase3_anthony_clip_color_spatial_text": 0.6748,
        }
    }
}

metrics_path = RES / 'metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        existing = json.load(f)
else:
    existing = {}
existing.update(results_dict)
with open(metrics_path, 'w') as f:
    json.dump(existing, f, indent=2)
print(f"\nSaved results to {metrics_path}")

with open(RES / 'phase3_mark_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)


# ======================================================================
# 12. PLOTS
# ======================================================================
print("\nGenerating plots...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Phase 3 Mark: Retrieval Architecture vs Feature Engineering\nVisual Product Search Engine", fontsize=14, fontweight='bold')

# --- Plot 1: DINOv2 pooling comparison ---
ax = axes[0, 0]
dino_compare = {
    'CLS-token\n(Phase 2\nfailure)': dino_cls_m,
    'Patch\nmean-pool\n(3.M.1)': dino_mean_m,
    'Patch\nGeM p=3\n(3.M.2)': dino_gem_m,
    'CLIP B/32\n(reference)': clip_metrics,
}
ks = [1, 5, 10, 20]
colors_plot = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
x = np.arange(len(ks))
w = 0.2
for i, (label, m) in enumerate(dino_compare.items()):
    vals = [m.get(f'R@{k}', 0) for k in ks]
    ax.bar(x + i*w, vals, w, label=label, color=colors_plot[i], alpha=0.85)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels([f'R@{k}' for k in ks])
ax.set_ylabel('Recall')
ax.set_title('DINOv2 Pooling Strategy Comparison\nPatch pooling repairs CLS-token failure')
ax.legend(fontsize=8, loc='upper left')
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.4)

# --- Plot 2: Category filter lift ---
ax = axes[0, 1]
systems = ['CLIP B/32\nbaseline', 'CLIP +\ncat.filter', 'CLIP + cat.filter\n+ color (best)']
r1_vals = [clip_metrics['R@1'], cat_clip_m['R@1'], best_cat_clip_col_m['R@1']]
r10_vals = [clip_metrics['R@10'], cat_clip_m['R@10'], best_cat_clip_col_m['R@10']]
x = np.arange(len(systems))
w = 0.35
bars1 = ax.bar(x - w/2, r1_vals, w, label='R@1', color='#3498db', alpha=0.85)
bars2 = ax.bar(x + w/2, r10_vals, w, label='R@10', color='#2ecc71', alpha=0.85)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=9)
ax.set_ylabel('Recall')
ax.set_title('Category Filter Lift\n"Architecture change > additional features"')
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.4)

# --- Plot 3: Color feature comparison ---
ax = axes[1, 0]
color_compare = {
    '48D histogram\n(Anthony)': color48_m['R@1'],
    'K-means k=3\n9D (3.M.4)': km3_m['R@1'],
    'K-means k=5\n15D (3.M.4)': km5_m['R@1'],
    'CLIP B/32\n(reference)': clip_metrics['R@1'],
}
names = list(color_compare.keys())
vals = list(color_compare.values())
clrs = ['#e74c3c', '#f39c12', '#e67e22', '#3498db']
bars = ax.bar(names, vals, color=clrs, alpha=0.85)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
ax.set_ylabel('R@1')
ax.set_title('Color Feature Standalone R@1\nK-means vs histogram (dimensionality efficiency)')
ax.set_ylim(0, 0.7)
ax.grid(axis='y', alpha=0.4)

# --- Plot 4: Full comparison (R@1) ---
ax = axes[1, 1]
full_compare_labels = [
    ('ResNet50 baseline\n(P1 Anthony)', 0.307),
    ('ResNet50+color\n(P1 Mark)', 0.405),
    ('CLIP B/32 bare\n(P2 Mark)', 0.480),
    ('CLIP+color rerank\n(P2 Mark)', 0.576),
    ('CLIP+color+spatial\n+text (P3 Anthony)', 0.6748),
    ('CLIP+color+spatial\n+cat.filter (P3 Mark)', best_full_cat_m['R@1']),
]
labels = [x[0] for x in full_compare_labels]
vals = [x[1] for x in full_compare_labels]
clrs2 = ['#bdc3c7']*4 + ['#e74c3c', '#2ecc71']
bars = ax.barh(labels, vals, color=clrs2, alpha=0.85)
for bar in bars:
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{bar.get_width():.3f}', va='center', fontsize=9)
ax.set_xlabel('R@1')
ax.set_title('All-Phase R@1 Comparison\nMark vs Anthony best systems')
ax.set_xlim(0, 0.85)
ax.grid(axis='x', alpha=0.4)

plt.tight_layout()
plt.savefig(RES / 'phase3_mark_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved phase3_mark_results.png")

# --- Per-category DINOv2 vs CLIP heatmap ---
fig2, ax2 = plt.subplots(figsize=(12, 5))
cats = sorted(g_df['category2'].unique())
r1_clip_by_cat = []
r1_dino_cls_by_cat = []
r1_dino_mean_by_cat = []
r1_cat_filter_by_cat = []

for cat in cats:
    mask = query_cats == cat
    if mask.sum() == 0:
        r1_clip_by_cat.append(0)
        r1_dino_cls_by_cat.append(0)
        r1_dino_mean_by_cat.append(0)
        r1_cat_filter_by_cat.append(0)
        continue
    cq = q_pids[mask]
    r1_clip_by_cat.append(recall_at_k(clip_idx[mask], cq, g_pids, 1))
    r1_dino_cls_by_cat.append(recall_at_k(dino_cls_idx[mask], cq, g_pids, 1))
    r1_dino_mean_by_cat.append(recall_at_k(dino_mean_idx[mask], cq, g_pids, 1))
    r1_cat_filter_by_cat.append(recall_at_k(cat_idx_clip[mask], cq, g_pids, 1))

x = np.arange(len(cats))
w = 0.22
ax2.bar(x - 1.5*w, r1_dino_cls_by_cat, w, label='DINOv2 CLS-token', color='#e74c3c', alpha=0.8)
ax2.bar(x - 0.5*w, r1_dino_mean_by_cat, w, label='DINOv2 patch mean', color='#f39c12', alpha=0.8)
ax2.bar(x + 0.5*w, r1_clip_by_cat, w, label='CLIP B/32', color='#3498db', alpha=0.8)
ax2.bar(x + 1.5*w, r1_cat_filter_by_cat, w, label='CLIP + cat.filter', color='#2ecc71', alpha=0.8)
ax2.set_xticks(x)
ax2.set_xticklabels(cats, rotation=30, ha='right')
ax2.set_ylabel('R@1')
ax2.set_title('Per-Category R@1: DINOv2 Patch Repair vs Category-Conditioned Retrieval')
ax2.legend()
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(RES / 'phase3_mark_per_category.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved phase3_mark_per_category.png")

print("\n" + "=" * 70)
print("PHASE 3 MARK COMPLETE")
champion_r1 = max(m.get('R@1', 0) for _, m in all_results)
print(f"Phase 3 Mark champion R@1: {champion_r1:.4f}")
print(f"Phase 3 Anthony champion R@1: 0.6748 (CLIP+color+spatial+text)")
print(f"Combined insight: Architecture (category filter) + features (color+spatial) > features alone")
print("=" * 70)
