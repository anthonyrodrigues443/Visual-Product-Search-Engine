"""Generate the missing Phase 2 Mark plots from results/metrics.json.

The resilient runner produced metrics but skipped plot generation because the
fragile run kept crashing mid-experiment. This script reads the saved data
and generates the three plots the report references:

  - phase2_mark_comparison.png  (R@1 / R@10 bar chart vs Phase 1 baselines)
  - phase2_mark_category_heatmap.png (R@10 by model x category)
  - phase2_mark_speed_accuracy.png  (CPU embed time vs R@1)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / 'results'

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

with open(RESULTS / 'metrics.json') as f:
    metrics = json.load(f)

p2 = metrics['phase2_mark']
exps = p2['experiments']

# Phase 1 anchors for the dashed comparison lines
P1_RESNET50_R1 = 0.3067   # Anthony Phase 1 baseline
P1_RESNET50_R10 = 0.5901
P1_BEST_R1 = 0.4051       # Mark Phase 1 ResNet50 + color rerank alpha=0.5
P1_BEST_R10 = 0.6573

# Display labels (newlines for axis readability)
labels = {
    'efficientnet_b0_p1_rerun': 'EfficientNet-B0\n(ImageNet CNN)',
    'clip_vit_b32': 'CLIP ViT-B/32\n(text-aligned)',
    'dinov2_vits14': 'DINOv2 ViT-S/14\n(self-supervised)',
    'clip_vit_b32_color_rerank_alpha05': 'CLIP-B/32\n+ color rerank',
    'dinov2_vits14_color_rerank_alpha05': 'DINOv2-S/14\n+ color rerank',
}
order = [k for k in labels if k in exps]
r1s = [exps[k]['recall@1'] for k in order]
r10s = [exps[k]['recall@10'] for k in order]
plot_colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#16a085']

# ===== Plot 1: overall comparison =====
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

x_labels = [labels[k] for k in order]
axes[0].bar(x_labels, r1s, color=plot_colors[:len(order)])
axes[0].axhline(P1_RESNET50_R1, color='grey', linestyle='--',
                label=f'Phase 1 ResNet50 baseline ({P1_RESNET50_R1:.3f})')
axes[0].axhline(P1_BEST_R1, color='black', linestyle=':',
                label=f'Phase 1 best (ResNet50 + color rerank, {P1_BEST_R1:.3f})')
axes[0].set_ylabel('Recall@1', fontsize=12)
axes[0].set_title('Phase 2 Mark: Foundation models vs CNNs — Recall@1',
                  fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].tick_params(axis='x', rotation=20, labelsize=9)
axes[0].set_ylim(0, max(max(r1s), P1_BEST_R1) * 1.18)
for i, v in enumerate(r1s):
    axes[0].text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

axes[1].bar(x_labels, r10s, color=plot_colors[:len(order)])
axes[1].axhline(P1_RESNET50_R10, color='grey', linestyle='--',
                label=f'Phase 1 ResNet50 ({P1_RESNET50_R10:.3f})')
axes[1].axhline(P1_BEST_R10, color='black', linestyle=':',
                label=f'Phase 1 best ({P1_BEST_R10:.3f})')
axes[1].set_ylabel('Recall@10', fontsize=12)
axes[1].set_title('Phase 2 Mark: Foundation models vs CNNs — Recall@10',
                  fontsize=13, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].tick_params(axis='x', rotation=20, labelsize=9)
axes[1].set_ylim(0, max(max(r10s), P1_BEST_R10) * 1.15)
for i, v in enumerate(r10s):
    axes[1].text(i, v + 0.008, f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
out = RESULTS / 'phase2_mark_comparison.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"saved {out}")

# ===== Plot 2: per-category heatmap (R@1) =====
short = {
    'efficientnet_b0_p1_rerun': 'EfficientNet-B0',
    'clip_vit_b32': 'CLIP-B/32',
    'dinov2_vits14': 'DINOv2-S/14',
    'clip_vit_b32_color_rerank_alpha05': 'CLIP-B/32 + color',
    'dinov2_vits14_color_rerank_alpha05': 'DINOv2 + color',
}
keys = [k for k in short if k in exps]
cats = sorted({c for k in keys for c in exps[k].get('per_category_r1', {}).keys()})
matrix = np.zeros((len(keys), len(cats)))
for i, k in enumerate(keys):
    pc = exps[k].get('per_category_r1', {})
    for j, c in enumerate(cats):
        matrix[i, j] = pc.get(c, {}).get('recall@1', np.nan)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=cats, yticklabels=[short[k] for k in keys],
            ax=ax, cbar_kws={'label': 'Recall@1'})
ax.set_title('Phase 2 Mark: Recall@1 by model x category',
             fontsize=13, fontweight='bold')
plt.tight_layout()
out = RESULTS / 'phase2_mark_category_heatmap.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"saved {out}")

# ===== Plot 3: speed vs accuracy =====
# Only backbones (not rerank ones) have embed_time metadata
backbones = ['efficientnet_b0_p1_rerun', 'clip_vit_b32', 'dinov2_vits14']
backbone_color = {
    'efficientnet_b0_p1_rerun': '#3498db',
    'clip_vit_b32': '#e74c3c',
    'dinov2_vits14': '#27ae60',
}
fig, ax = plt.subplots(figsize=(10, 6))
for k in backbones:
    if k not in exps:
        continue
    e = exps[k]
    t = e.get('total_embed_time_s')
    r1 = e['recall@1']
    if t is None:
        continue
    ax.scatter(t, r1, s=240, color=backbone_color[k], edgecolors='black', zorder=3)
    label = short[k]
    ax.annotate(label, xy=(t, r1), xytext=(8, 10),
                textcoords='offset points', fontsize=10, fontweight='bold')

ax.axhline(P1_RESNET50_R1, color='grey', linestyle='--',
           label=f'Phase 1 ResNet50 baseline ({P1_RESNET50_R1:.3f})')
ax.axhline(P1_BEST_R1, color='black', linestyle=':',
           label=f'Phase 1 best ({P1_BEST_R1:.3f})')
ax.set_xlabel('Total embed time on 1,327 images, CPU (seconds)', fontsize=11)
ax.set_ylabel('Recall@1', fontsize=11)
ax.set_title('Phase 2 Mark: accuracy vs embedding cost (CPU)',
             fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
plt.tight_layout()
out = RESULTS / 'phase2_mark_speed_accuracy.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"saved {out}")

print('\nDone.')
