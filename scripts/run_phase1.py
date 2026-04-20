"""Phase 1: EDA + Baseline — runs as a script to avoid Jupyter kernel issues."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import gc
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
RESULTS = PROJECT_ROOT / 'results'
RESULTS.mkdir(parents=True, exist_ok=True)

# ===== 1. Load metadata =====
print("=" * 60)
print("1. LOADING METADATA")
print("=" * 60)
df = pd.read_csv(DATA_PROCESSED / 'metadata.csv')
n_images = len(df)
n_products = df['product_id'].nunique()
n_categories = df['category2'].nunique()
n_colors = df['color'].nunique()
imgs_per_product = df.groupby('product_id').size()

print(f'Total images:           {n_images:,}')
print(f'Unique products:        {n_products:,}')
print(f'Product categories:     {n_categories}')
print(f'Unique colors:          {n_colors}')
print(f'Images per product:     {imgs_per_product.mean():.1f} mean, {imgs_per_product.median():.0f} median')
print(f'Missing color:          {df["color"].isna().sum()}')

# ===== 2. EDA Plots =====
print("\n" + "=" * 60)
print("2. GENERATING EDA PLOTS")
print("=" * 60)

# 2a. Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
gender_counts = df['category1'].value_counts()
axes[0].bar(gender_counts.index, gender_counts.values, color=['#e74c3c', '#3498db'])
axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Number of Images')
for i, (cat, val) in enumerate(gender_counts.items()):
    axes[0].text(i, val + 500, f'{val:,}\n({val/n_images*100:.1f}%)',
                ha='center', fontsize=11, fontweight='bold')

cat_counts = df['category2'].value_counts()
bars = axes[1].barh(cat_counts.index[::-1], cat_counts.values[::-1])
axes[1].set_title('Product Category Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Number of Images')

axes[2].hist(imgs_per_product.values, bins=range(1, imgs_per_product.max()+2),
            color='#2ecc71', edgecolor='black', alpha=0.7)
axes[2].set_title('Images per Product', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Number of Images')
axes[2].set_ylabel('Number of Products')
axes[2].axvline(imgs_per_product.mean(), color='red', linestyle='--',
               label=f'Mean={imgs_per_product.mean():.1f}')
axes[2].legend()
plt.tight_layout()
plt.savefig(RESULTS / 'eda_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/eda_distributions.png')

# 2b. Color distribution
color_counts = df['color'].value_counts().head(20)
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(color_counts.index[::-1], color_counts.values[::-1], color='#9b59b6')
ax.set_title('Top 20 Colors in Dataset', fontsize=14, fontweight='bold')
ax.set_xlabel('Number of Images')
plt.tight_layout()
plt.savefig(RESULTS / 'eda_colors.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/eda_colors.png')

# 2c. Color x Category heatmap
top_colors = df['color'].value_counts().head(10).index
top_cats = df['category2'].value_counts().head(8).index
cross = pd.crosstab(df[df['color'].isin(top_colors)]['color'],
                    df[df['category2'].isin(top_cats)]['category2'])
cross = cross.reindex(index=top_colors, columns=top_cats).fillna(0)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cross, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
ax.set_title('Color x Category Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS / 'eda_color_category_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/eda_color_category_heatmap.png')

# ===== 3. View Analysis =====
def extract_view(item_id):
    parts = item_id.split('_')
    return parts[-1] if len(parts) >= 2 else 'unknown'

df['view'] = df['item_id'].apply(extract_view)
view_counts = df['view'].value_counts()
print('\nView distribution:')
for view, count in view_counts.items():
    print(f'  {view:15s}: {count:6,} ({count/n_images*100:.1f}%)')

# ===== 4. Create Retrieval Splits =====
print("\n" + "=" * 60)
print("4. CREATING RETRIEVAL SPLITS")
print("=" * 60)
from src.data_pipeline import create_retrieval_splits
train_df, gallery_df, query_df = create_retrieval_splits(df, test_frac=0.2, seed=42)
print(f'Train: {train_df["product_id"].nunique():,} products ({len(train_df):,} images)')
print(f'Gallery: {len(gallery_df):,} (1 per test product)')
print(f'Query: {len(query_df):,} (remaining test views)')

train_df.to_csv(DATA_PROCESSED / 'train.csv', index=False)
gallery_df.to_csv(DATA_PROCESSED / 'gallery.csv', index=False)
query_df.to_csv(DATA_PROCESSED / 'query.csv', index=False)

# ===== 5. Baseline: ResNet50 + FAISS =====
print("\n" + "=" * 60)
print("5. BASELINE RETRIEVAL: ResNet50 + FAISS")
print("=" * 60)

import torch
import torchvision.transforms as T
import torchvision.models as models
import faiss

device = 'cpu'  # Use CPU to avoid MPS memory issues in background
print(f'Using device: {device}')

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device)
resnet.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Use 300 eval products to keep memory reasonable
EVAL_PRODUCTS = 300
eval_products = gallery_df['product_id'].values[:EVAL_PRODUCTS]
eval_gallery = gallery_df[gallery_df['product_id'].isin(eval_products)].reset_index(drop=True)
eval_query = query_df[query_df['product_id'].isin(eval_products)].reset_index(drop=True)

print(f'Evaluation: {len(eval_gallery)} gallery, {len(eval_query)} query images')

all_eval_indices = sorted(set(
    eval_gallery['index'].astype(int).tolist() +
    eval_query['index'].astype(int).tolist()
))
print(f'Downloading {len(all_eval_indices)} images...')

from datasets import load_dataset
ds = load_dataset('Marqo/deepfashion-inshop', split='data', streaming=True)
needed = set(all_eval_indices)
images_by_index = {}
for i, ex in enumerate(tqdm(ds, total=max(needed)+1, desc='Downloading')):
    if i in needed:
        img = ex['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images_by_index[i] = img
        needed.discard(i)
        if not needed:
            break

print(f'Downloaded {len(images_by_index)} images')

# Extract features
@torch.no_grad()
def extract_features(model, images_dict, indices, transform, device, batch_size=32):
    features = []
    valid_indices = []
    batch = []
    batch_idx = []

    for idx in tqdm(indices, desc='Extracting features'):
        idx_int = int(idx)
        if idx_int not in images_dict:
            continue
        tensor = transform(images_dict[idx_int])
        batch.append(tensor)
        batch_idx.append(idx_int)

        if len(batch) >= batch_size:
            feats = model(torch.stack(batch).to(device)).cpu().numpy()
            features.append(feats)
            valid_indices.extend(batch_idx)
            batch = []
            batch_idx = []

    if batch:
        feats = model(torch.stack(batch).to(device)).cpu().numpy()
        features.append(feats)
        valid_indices.extend(batch_idx)

    return np.vstack(features), valid_indices

gallery_features, gallery_valid_idx = extract_features(
    resnet, images_by_index, eval_gallery['index'].values, transform, device
)
query_features, query_valid_idx = extract_features(
    resnet, images_by_index, eval_query['index'].values, transform, device
)
print(f'Gallery features: {gallery_features.shape}, Query features: {query_features.shape}')

print(f'Gallery features: {gallery_features.shape}, dtype={gallery_features.dtype}')
print(f'Query features: {query_features.shape}, dtype={query_features.dtype}')
print(f'NaN check: gallery={np.isnan(gallery_features).any()}, query={np.isnan(query_features).any()}')
sys.stdout.flush()

# FAISS retrieval
print('Normalizing gallery...'); sys.stdout.flush()
gallery_norms = np.linalg.norm(gallery_features, axis=1, keepdims=True)
print(f'Gallery norms: min={gallery_norms.min():.4f}, max={gallery_norms.max():.4f}'); sys.stdout.flush()
gallery_normed = gallery_features / gallery_norms

print('Normalizing query...'); sys.stdout.flush()
query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
query_normed = query_features / query_norms

print('Creating FAISS index...'); sys.stdout.flush()
dim = gallery_normed.shape[1]
faiss.omp_set_num_threads(1)
gallery_f32 = np.ascontiguousarray(gallery_normed, dtype=np.float32)
query_f32 = np.ascontiguousarray(query_normed, dtype=np.float32)
print(f'Arrays contiguous: gallery={gallery_f32.flags["C_CONTIGUOUS"]}, query={query_f32.flags["C_CONTIGUOUS"]}'); sys.stdout.flush()
index = faiss.IndexFlatIP(dim)
print(f'Adding {len(gallery_f32)} vectors to index...'); sys.stdout.flush()
index.add(gallery_f32)

K = 20
print(f'Searching {len(query_f32)} queries...'); sys.stdout.flush()
distances, faiss_indices = index.search(query_f32, K)
print(f'Search complete: {distances.shape}'); sys.stdout.flush()

gallery_valid_df = eval_gallery[eval_gallery['index'].isin(gallery_valid_idx)].reset_index(drop=True)
query_valid_df = eval_query[eval_query['index'].isin(query_valid_idx)].reset_index(drop=True)

gallery_pids = gallery_valid_df['product_id'].values
query_pids = query_valid_df['product_id'].values

def compute_recall_at_k(query_pids, gallery_pids, retrieved_indices, k):
    correct = 0
    for i in range(len(query_pids)):
        top_k_pids = gallery_pids[retrieved_indices[i, :k]]
        if query_pids[i] in top_k_pids:
            correct += 1
    return correct / len(query_pids)

recall_1 = compute_recall_at_k(query_pids, gallery_pids, faiss_indices, 1)
recall_5 = compute_recall_at_k(query_pids, gallery_pids, faiss_indices, 5)
recall_10 = compute_recall_at_k(query_pids, gallery_pids, faiss_indices, 10)
recall_20 = compute_recall_at_k(query_pids, gallery_pids, faiss_indices, 20)

print('\n' + '=' * 50)
print('BASELINE RESULTS: ResNet50 (ImageNet V2) + FAISS')
print('=' * 50)
print(f'  Recall@1:  {recall_1:.4f} ({recall_1*100:.1f}%)')
print(f'  Recall@5:  {recall_5:.4f} ({recall_5*100:.1f}%)')
print(f'  Recall@10: {recall_10:.4f} ({recall_10*100:.1f}%)')
print(f'  Recall@20: {recall_20:.4f} ({recall_20*100:.1f}%)')

baseline_results = {
    'model': 'ResNet50_ImageNet_V2',
    'embedding_dim': dim,
    'gallery_size': int(len(gallery_valid_df)),
    'query_size': int(len(query_valid_df)),
    'recall_at_1': float(recall_1),
    'recall_at_5': float(recall_5),
    'recall_at_10': float(recall_10),
    'recall_at_20': float(recall_20),
}

# ===== 6. Per-Category Analysis =====
print("\n" + "=" * 60)
print("6. PER-CATEGORY ANALYSIS")
print("=" * 60)

query_cats = query_valid_df['category2'].values
cat_recalls = {}
for cat in sorted(set(query_cats)):
    mask = query_cats == cat
    if mask.sum() < 5:
        continue
    cat_query_pids = query_pids[mask]
    cat_idx = faiss_indices[mask]
    r1 = compute_recall_at_k(cat_query_pids, gallery_pids, cat_idx, 1)
    r10 = compute_recall_at_k(cat_query_pids, gallery_pids, cat_idx, 10)
    cat_recalls[cat] = {'recall@1': r1, 'recall@10': r10, 'n_queries': int(mask.sum())}

print(f'{"Category":15s} {"R@1":>8s} {"R@10":>8s} {"Queries":>8s}')
print('-' * 42)
for cat in sorted(cat_recalls, key=lambda x: cat_recalls[x]['recall@1'], reverse=True):
    r = cat_recalls[cat]
    print(f'{cat:15s} {r["recall@1"]:8.3f} {r["recall@10"]:8.3f} {r["n_queries"]:8d}')

# Per-category bar plot
cats_sorted = sorted(cat_recalls, key=lambda x: cat_recalls[x]['recall@1'], reverse=True)
r1_vals = [cat_recalls[c]['recall@1'] for c in cats_sorted]
r10_vals = [cat_recalls[c]['recall@10'] for c in cats_sorted]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cats_sorted))
width = 0.35
ax.bar(x - width/2, r1_vals, width, label='Recall@1', color='#3498db', alpha=0.8)
ax.bar(x + width/2, r10_vals, width, label='Recall@10', color='#e74c3c', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(cats_sorted, rotation=45, ha='right')
ax.set_ylabel('Recall')
ax.set_title('Per-Category Retrieval Performance (ResNet50 Baseline)', fontsize=14, fontweight='bold')
ax.legend()
ax.axhline(recall_1, color='#3498db', linestyle='--', alpha=0.5)
ax.axhline(recall_10, color='#e74c3c', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(RESULTS / 'baseline_per_category.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: results/baseline_per_category.png')

# ===== 7. Similarity Distribution =====
correct_sims = []
incorrect_sims = []
for i in range(len(query_pids)):
    for j in range(K):
        sim = float(distances[i, j])
        if query_pids[i] == gallery_pids[faiss_indices[i, j]]:
            correct_sims.append(sim)
        else:
            incorrect_sims.append(sim)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(correct_sims, bins=50, alpha=0.6, label=f'Correct (n={len(correct_sims)})', color='green', density=True)
ax.hist(incorrect_sims, bins=50, alpha=0.6, label=f'Incorrect (n={len(incorrect_sims)})', color='red', density=True)
ax.set_xlabel('Cosine Similarity')
ax.set_ylabel('Density')
ax.set_title('Similarity Distribution: Correct vs Incorrect', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(RESULTS / 'similarity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

sep = np.mean(correct_sims) - np.mean(incorrect_sims)
print(f'\nCorrect sims: mean={np.mean(correct_sims):.4f}, std={np.std(correct_sims):.4f}')
print(f'Incorrect sims: mean={np.mean(incorrect_sims):.4f}, std={np.std(incorrect_sims):.4f}')
print(f'Separation: {sep:.4f}')
print('Saved: results/similarity_distribution.png')

# ===== 8. Save All Metrics =====
all_metrics = {
    'phase1_baseline': baseline_results,
    'per_category': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                         for kk, vv in v.items()} for k, v in cat_recalls.items()},
    'similarity_separation': {
        'correct_mean': float(np.mean(correct_sims)),
        'incorrect_mean': float(np.mean(incorrect_sims)),
        'separation': float(sep),
    }
}
with open(RESULTS / 'metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

print('\n' + '=' * 60)
print('PHASE 1 COMPLETE')
print('=' * 60)
print(f'Recall@1={recall_1:.4f}, Recall@10={recall_10:.4f}, Recall@20={recall_20:.4f}')
print(f'Similarity separation: {sep:.4f}')
print(f'Files saved:')
for f in sorted(RESULTS.glob('*.png')) + sorted(RESULTS.glob('*.json')):
    print(f'  {f.relative_to(PROJECT_ROOT)}')

os._exit(0)
