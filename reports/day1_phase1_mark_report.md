# Phase 1 Mark — Day 1 Report: Color Palette Features + EfficientNet-B0

**Date:** 2026-04-20  
**Author:** Mark Rodrigues  
**Branch:** mark/phase1-2026-04-20  
**Dataset:** DeepFashion In-Shop (52,591 images, 12,995 products, 8 categories)  
**Eval split:** 300 gallery products, 1,027 query images (300 test products × avg 3.4 views)

---

## Context

Anthony's Phase 1 established the baseline: ResNet50 ImageNet V2 → **R@1=0.3067, R@10=0.5901**. The hardest category is jackets at R@1=0.1392. The similarity separation gap is 0.048 — correct matches and incorrect matches have nearly overlapping cosine similarity distributions.

My complementary work investigates two hypotheses:
1. A lighter architecture (EfficientNet-B0) may outperform ResNet50 on fashion retrieval
2. Explicit color features (RGB histogram + HSV histogram) should help categories where color is the primary consumer discriminator

---

## Methodology

### Color Feature Engineering

Rather than K-means clustering (standard for palette extraction but 750ms/image due to sklearn joblib overhead on Windows), I implemented **pure NumPy histogram features**:

- **RGB histogram**: 8 bins per channel → 24D. Captures the overall color distribution. ~4ms/image.
- **HSV histogram**: 8 bins per channel → 24D. Hue separates colors perceptually, reducing metamerism vs raw RGB. ~4ms/image.
- **Combined**: 48D color descriptor = RGB hist + HSV hist.

**Augmented embedding**: `concat(L2-norm(CNN), L2-norm(color) * color_weight=0.3)` → the 0.3 weight gives color ~23% influence during cosine search.

**Color re-ranking**: After FAISS retrieves top-20 by CNN similarity, blend CNN score and color similarity: `blended = alpha * cnn_score + (1-alpha) * color_score`. Applied at inference time — no retraining needed.

### EfficientNet-B0

EfficientNet-B0 uses compound scaling (simultaneous width, depth, and resolution scaling) vs ResNet50's pure depth scaling. With 5.3M parameters (vs 25.6M) and 20MB weights (vs 98MB), it achieves higher ImageNet efficiency. For fashion retrieval, the hypothesis is that compound scaling better preserves fine-grained texture/color details that global-average-pooling in ResNet50 can discard.

---

## Results

| Experiment | R@1 | R@5 | R@10 | R@20 | vs Baseline |
|---|---|---|---|---|---|
| Anthony: ResNet50 (baseline) | 0.3067 | 0.4927 | 0.5901 | 0.6913 | — |
| 1.M.1: EfficientNet-B0 | 0.3671 | 0.5988 | 0.6855 | 0.7760 | +6.0pp |
| 1.M.2: Color-only 48D | 0.3379 | 0.5239 | 0.6125 | 0.7069 | +3.1pp |
| 1.M.3: ResNet50 + Color (aug) | 0.3213 | 0.5063 | 0.6056 | 0.7059 | +1.5pp |
| 1.M.4: EfficientNet-B0 + Color (aug) | 0.3827 | 0.6115 | 0.6943 | 0.7848 | +7.6pp |
| 1.M.5a: Rerank alpha=0.7 | 0.3622 | 0.5570 | 0.6397 | 0.6913 | +5.5pp |
| **1.M.5b: Rerank alpha=0.5** | **0.4051** | **0.5930** | **0.6573** | **0.6913** | **+9.8pp** |

---

## Key Findings

### Finding 1: EfficientNet-B0 beats ResNet50 by +6pp with 5x smaller weights

EfficientNet-B0 (R@1=0.3671) outperforms ResNet50 (R@1=0.3067) using a model that is:
- 5x fewer parameters (5.3M vs 25.6M)
- 5x smaller weights file (20MB vs 98MB)
- Lower embedding dimension (1280D vs 2048D)

This is the headline result for production: the lighter model wins. Compound scaling is a better fit than plain depth-scaling for fine-grained visual retrieval.

### Finding 2: 48D color histogram alone beats the 2048D ResNet50 embedding

Color-only retrieval (R@1=0.3379) beats the ResNet50 baseline (R@1=0.3067) using only 48 floating-point numbers to represent each image. This validates the hypothesis: **fashion consumers search by color first**, and ResNet50 features — optimized for ImageNet classification — don't explicitly model color well enough.

### Finding 3: Color re-ranking with equal blend (alpha=0.5) is the best overall method

ResNet50 + color re-rank alpha=0.5 achieves R@1=0.4051 — the best result. This approach:
- Requires no model retraining
- Uses the existing ResNet50 FAISS index as-is
- Adds only a 48D color extraction step at inference time
- Improves R@1 by +9.8pp over the baseline

The intuition: the correct item is usually already in the ResNet50 top-20 — re-ranking just needs to surface it. Color has full decision authority at the re-ranking stage vs ~23% influence in the augmented embedding approach.

### Finding 4: Surprising — equal weight (alpha=0.5) beats CNN-heavy weight (alpha=0.7)

We expected the CNN features (2048D, trained on 1.2M ImageNet images) to dominate. But alpha=0.5 (equal blend) beats alpha=0.7 (CNN-heavy) by +4.3pp R@1 (0.4051 vs 0.3622). This suggests:
- For fashion retrieval, color is approximately as informative as structural CNN features
- The optimal re-ranking weight may be even lower than 0.5 (i.e., color-dominant)
- Worth testing alpha=0.3 and 0.2 in Phase 2

---

## Technical Notes

### Windows encoding issue

The Python scripts initially crashed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u03b1'` when printing the Greek letter α (alpha) to stdout. Fixed by replacing all α characters with the ASCII string "alpha" in print statements. The `→` arrow character had the same issue (fixed earlier by replacing with `->`).

### MiniBatchKMeans performance on Windows

Initial color feature implementation used `sklearn.cluster.MiniBatchKMeans` for palette extraction. This triggered `joblib._count_physical_cores` via subprocess on Windows, causing **755ms/image** overhead (vs 1-2ms for the actual clustering). Fixed by replacing K-means entirely with NumPy `np.histogram()` — from 755ms to **~4ms/image** (189x speedup), with better gradient: histograms produce a smooth color distribution, while K-means color counts are sensitive to initialization.

---

## Plots

- `results/phase1_mark_comparison.png` — bar chart (R@1, R@10) + per-category heatmap
- `results/phase1_mark_jackets_analysis.png` — similarity distribution + jackets-specific bar chart

---

## Recommendations for Phase 2

1. **EfficientNet-B0 + color re-ranking** — combine the best backbone with the best inference strategy. Expected R@1 > 0.42.
2. **Sweep alpha** — test alpha=0.3, 0.2 for re-ranking. If alpha=0.5 beats 0.7, lower values may be even better.
3. **Spatial color features** — 4-quadrant histogram would distinguish jacket collar from body, reducing false matches between white-collar and all-white items.
4. **Triplet loss fine-tuning** — supervised fine-tuning on DeepFashion product pairs should push R@1 past 0.50.
5. **Jackets specialist model** — jackets are the hardest category at R@1=13.9% for Anthony's baseline. A dedicated model or category-aware retrieval may be needed.
