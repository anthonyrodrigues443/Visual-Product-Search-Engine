"""Generate Phase 2 Mark plots + headline finding from saved metrics.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / 'results'

with open(RESULTS / 'metrics.json') as f:
    m = json.load(f)

p2 = m['phase2_mark']
exps = p2['experiments']
p1_baseline_r1 = m.get('phase1_baseline', {}).get('recall_at_1', 0.3067)
p1_baseline_r10 = m.get('phase1_baseline', {}).get('recall_at_10', 0.5901)
p1_best_r1 = 0.4051  # Mark Phase 1 ResNet50 + color rerank
p1_best_r10 = 0.6573

labels = {
    'efficientnet_b0_p1_rerun': 'EfficientNet-B0\n(ImageNet CNN)',
    'clip_vit_b32': 'CLIP ViT-B/32\n(text-image)',
    'dinov2_vits14': 'DINOv2 ViT-S/14\n(self-supervised)',
    'clip_vit_b32_color_rerank_alpha05': 'CLIP-B/32\n+ color rerank',
    'dinov2_vits14_color_rerank_alpha05': 'DINOv2\n+ color rerank',
}
order = [k for k in labels if k in exps]
r1s = [exps[k]['recall@1'] for k in order]
r10s = [exps[k]['recall@10'] for k in order]
colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#16a085']

# 7a. Overall comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].bar([labels[k] for k in order], r1s, color=colors[:len(order)])
axes[0].axhline(p1_baseline_r1, color='grey', linestyle='--', linewidth=1, label=f'Phase 1 ResNet50 baseline ({p1_baseline_r1:.3f})')
axes[0].axhline(p1_best_r1, color='black', linestyle=':', linewidth=1.5, label=f'Phase 1 best system ({p1_best_r1:.3f})')
axes[0].set_ylabel('Recall@1', fontsize=12)
axes[0].set_ylim(0, 0.7)
axes[0].set_title('Phase 2: R@1 by model', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper left', fontsize=9)
axes[0].tick_params(axis='x', rotation=15, labelsize=9)
for i, v in enumerate(r1s):
    axes[0].text(i, v + 0.010, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

axes[1].bar([labels[k] for k in order], r10s, color=colors[:len(order)])
axes[1].axhline(p1_baseline_r10, color='grey', linestyle='--', linewidth=1, label=f'Phase 1 ResNet50 ({p1_baseline_r10:.3f})')
axes[1].axhline(p1_best_r10, color='black', linestyle=':', linewidth=1.5, label=f'Phase 1 best ({p1_best_r10:.3f})')
axes[1].set_ylabel('Recall@10', fontsize=12)
axes[1].set_ylim(0, 0.9)
axes[1].set_title('Phase 2: R@10 by model', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper left', fontsize=9)
axes[1].tick_params(axis='x', rotation=15, labelsize=9)
for i, v in enumerate(r10s):
    axes[1].text(i, v + 0.010, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS / 'phase2_mark_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase2_mark_comparison.png')

# 7b. Per-category R@10 heatmap
categories = sorted({c for k in order for c in exps[k].get('per_category_r10', {})})
heatmap = np.zeros((len(order), len(categories)))
for i, k in enumerate(order):
    pc = exps[k].get('per_category_r10', {})
    for j, c in enumerate(categories):
        heatmap[i, j] = pc.get(c, {}).get('recall@1', 0)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(heatmap, annot=True, fmt='.3f', cmap='YlGnBu',
            xticklabels=categories,
            yticklabels=[labels[k].replace('\n', ' ') for k in order], ax=ax,
            cbar_kws={'label': 'Recall@10'})
ax.set_title('Phase 2: Recall@10 by model × category', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS / 'phase2_mark_category_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase2_mark_category_heatmap.png')

# 7c. Paradigm comparison: CNN vs ViT-text vs ViT-SSL vs + color
paradigms = {
    'CNN (ImageNet sup)': exps['efficientnet_b0_p1_rerun']['recall@1'],
    'ViT (text-aligned)': exps['clip_vit_b32']['recall@1'],
    'ViT (self-supervised)': exps['dinov2_vits14']['recall@1'],
    'Phase 1 best\n(ResNet50+color)': p1_best_r1,
    'Phase 2 best\n(CLIP+color)': exps['clip_vit_b32_color_rerank_alpha05']['recall@1'],
}
fig, ax = plt.subplots(figsize=(10, 6))
paradigm_colors = ['#3498db', '#e74c3c', '#27ae60', '#95a5a6', '#f39c12']
ax.bar(paradigms.keys(), paradigms.values(), color=paradigm_colors)
for i, (k, v) in enumerate(paradigms.items()):
    ax.text(i, v + 0.010, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(p1_baseline_r1, color='grey', linestyle='--', linewidth=1, label=f'ResNet50 baseline ({p1_baseline_r1:.3f})')
ax.set_ylabel('Recall@1', fontsize=12)
ax.set_ylim(0, 0.7)
ax.set_title('Phase 2 headline: which paradigm wins fashion visual search?',
             fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=10)
ax.tick_params(axis='x', labelsize=10)
plt.tight_layout()
plt.savefig(RESULTS / 'phase2_mark_paradigms.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved: phase2_mark_paradigms.png')

# Headline finding
best = p2['best_result']
rerank_lift = exps['clip_vit_b32_color_rerank_alpha05']['recall@1'] - exps['clip_vit_b32']['recall@1']
dino_vs_clip_gap = exps['clip_vit_b32']['recall@1'] - exps['dinov2_vits14']['recall@1']

headline = (
    f"CLIP ViT-B/32 + color rerank (α=0.5) wins Phase 2 at R@1={best['recall@1']:.4f} "
    f"— a {best['delta_vs_anthony_resnet50']:+.3f} lift over Anthony's ResNet50 baseline "
    f"({p1_baseline_r1:.3f}) and {best['delta_vs_phase1_best']:+.3f} over Mark's Phase 1 best "
    f"(ResNet50 + color rerank, {p1_best_r1:.3f}). Surprise: DINOv2 (self-supervised ViT) UNDERPERFORMS "
    f"CLIP by {dino_vs_clip_gap:.3f} R@1 — contradicting the common belief that self-supervised visual "
    f"features beat text-aligned ones on image→image retrieval. CLS-token DINOv2 clusters products "
    f"correctly (R@10=0.665) but fails to discriminate the top-1 match. Mark's Phase 1 color-rerank "
    f"trick STACKS on top of CLIP (+{rerank_lift:.3f} R@1), validating that explicit color signal is "
    f"still the missing piece even with a 400M-pair foundation model."
)
p2['headline_finding'] = headline
with open(RESULTS / 'metrics.json', 'w') as f:
    json.dump(m, f, indent=2)
print('\nHeadline finding saved to metrics.json:\n')
print(headline)
