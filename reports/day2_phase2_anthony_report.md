# Phase 2: Multi-Model Comparison — Visual Product Search Engine
**Date:** 2026-04-21
**Session:** 2 of 7
**Researcher:** Anthony Rodrigues

## Objective
Does training paradigm (supervised / self-supervised / vision-language) matter more than architecture (CNN vs ViT) for fashion visual retrieval? Which pretrained backbone gives the best retrieval features without any fine-tuning?

## Research & References
1. [Radford et al., 2021 — CLIP] — Contrastive Language-Image Pre-Training learns visual concepts from natural language supervision. CLIP ViT-L/14 is the de facto standard for production visual search.
2. [Oquab et al., 2024 — DINOv2] — Self-supervised ViT trained with self-distillation. Produces features competitive with supervised models without any labels.
3. [Liu et al., 2022 — ConvNeXt] — "A ConvNet for the 2020s" — modernized ResNet design that competes with ViTs. Tests whether the ViT architecture itself matters vs training recipe.

How research influenced experiments: These three papers represent the three dominant paradigms in computer vision. By testing all three on the same retrieval task, we isolate the effect of training paradigm from architecture.

## Dataset
| Metric | Value |
|--------|-------|
| Dataset | DeepFashion In-Shop |
| Eval gallery | 300 products (1 image each) |
| Eval queries | 1,027 images (multiple views per product) |
| Categories | 9 (tees, shorts, pants, sweatshirts, shirts, jackets, denim, sweaters, suiting) |
| Index type | FAISS FlatIP (cosine similarity) |

## Experiments

### Experiment 2.1: Five Models Head-to-Head
**Hypothesis:** CLIP should dominate because it was trained on 400M image-text pairs including fashion content, and its embedding space was specifically optimized for similarity search.

**Method:** Extract features from 5 pretrained models (no fine-tuning), build FAISS flat index per model, evaluate Recall@K on the same 300-product eval set.

**Result:**
| Rank | Model | Paradigm | Params | Dim | R@1 | R@5 | R@10 | R@20 | Sep | ms/img |
|------|-------|----------|--------|-----|-----|-----|------|------|-----|--------|
| 1 | CLIP ViT-L/14 | vision-language | 304M | 768 | **0.5531** | 0.7478 | 0.8053 | 0.8530 | -0.001 | 495.3 |
| 2 | CLIP ViT-B/32 | vision-language | 88M | 512 | 0.3934 | 0.5833 | 0.6582 | 0.7410 | -0.023 | 32.2 |
| 3 | ViT-B/16 | supervised | 86M | 768 | 0.3661 | 0.5794 | 0.6563 | 0.7605 | -0.058 | 213.6 |
| 4 | ConvNeXt-Tiny | supervised CNN | 29M | 768 | 0.3544 | 0.5774 | 0.6524 | 0.7595 | -0.050 | 148.4 |
| 5 | DINOv2 ViT-B/14 | self-supervised | 86M | 768 | 0.2921 | 0.5891 | 0.7040 | 0.7858 | -0.107 | 199.7 |

**Interpretation:** CLIP ViT-L/14 dominates at R@1=55.3%, a 24.6pp improvement over the ResNet50 baseline and 19pp over its smaller sibling. Scale within CLIP is decisive — the ViT-L/14 (304M) nearly doubles ViT-B/32 (88M) on R@1.

### Experiment 2.2: Controlled Paradigm Comparisons

**DINOv2 (86M) vs ViT-B/16 (86M) — self-supervised vs supervised, same size:**
- R@1: ViT-B/16 wins 36.6% vs 29.2% (+7.4pp)
- R@10: DINOv2 wins 70.4% vs 65.6% (+4.8pp)
- R@20: DINOv2 wins 78.6% vs 76.1% (+2.5pp)
- **Finding: Self-supervised features lose at top-1 precision but win at high-K recall. DINOv2 distributes correct products more evenly across the top-K rather than concentrating at rank 1.**

**CLIP ViT-B/32 (88M) vs ViT-B/16 (86M) — vision-language vs supervised:**
- R@1: CLIP wins 39.3% vs 36.6% (+2.7pp)
- R@10: CLIP wins 65.8% vs 65.6% (+0.2pp)
- **Finding: Vision-language training gives a modest edge at similar scale, especially at R@1.**

**ViT-B/16 (86M) vs ConvNeXt-Tiny (29M) — ViT vs CNN:**
- R@1: ViT-B/16 36.6% vs ConvNeXt 35.4% (+1.2pp)
- R@10: ViT-B/16 65.6% vs ConvNeXt 65.2% (+0.4pp)
- **Finding: Architecture barely matters. Three different architectures (ViT, EfficientNet, ConvNeXt) all land within 2pp of each other when trained on ImageNet.**

### Experiment 2.3: Per-Category Analysis

CLIP ViT-L/14 wins every category. The most dramatic improvement is on **jackets**: from R@1=13.9% (ResNet50) to 59.5% (CLIP L/14) — a 4.3x improvement. CLIP understands "jacket" as a semantic concept and can distinguish products within the category.

DINOv2's weakness is most extreme on **shorts** (R@1=18.4%) and **shirts** (21.5%). These categories have subtle visual differences that self-supervised features don't capture well without explicit product-level supervision.

### Experiment 2.4: Similarity Separation Analysis

**All Phase 2 models have negative separation** — meaning the average incorrect match has HIGHER similarity than the average correct match. Phase 1 ResNet50 had positive separation (+0.048).

| Model | Correct Sim | Incorrect Sim | Separation |
|-------|-------------|---------------|------------|
| CLIP ViT-L/14 | 0.900 | 0.901 | **-0.001** |
| CLIP ViT-B/32 | 0.881 | 0.904 | -0.023 |
| ConvNeXt-Tiny | 0.716 | 0.766 | -0.050 |
| ViT-B/16 | 0.659 | 0.716 | -0.058 |
| DINOv2 ViT-B/14 | 0.639 | 0.745 | **-0.107** |
| ResNet50 (P1) | 0.786 | 0.738 | **+0.048** |

**The paradox:** CLIP ViT-L/14 has nearly zero separation yet the best R@1. Its embedding space clusters items so tightly that the average neighbor is essentially equidistant, but the correct product is still the argmax in 55% of cases. CLIP's contrastive training creates a very tight, well-organized embedding space where the correct product is a spike above a sea of near-identical competitors.

### Experiment 2.5: CLIP ViT-L/14 + Color Re-Ranking

| Configuration | R@1 | R@5 | R@10 | Δ R@1 |
|---------------|-----|-----|------|-------|
| CLIP ViT-L/14 (base) | 0.5531 | 0.7478 | 0.8053 | — |
| + Color (alpha=0.7) | 0.6203 | 0.7936 | 0.8335 | +0.067 |
| **+ Color (alpha=0.5)** | **0.6417** | **0.7916** | **0.8306** | **+0.089** |
| + Color (alpha=0.3) | 0.6183 | 0.7741 | 0.8306 | +0.065 |

**Finding: Color re-ranking gives +8.9pp on CLIP ViT-L/14, similar to the +9.8pp on ResNet50. The boost is consistent regardless of backbone quality.** Alpha=0.5 (equal weight) again beats alpha=0.7, confirming Mark's Phase 1 finding that color deserves equal weight to deep features for fashion retrieval.

## Head-to-Head Comparison (ALL models)

| # | Model | R@1 | R@5 | R@10 | R@20 |
|---|-------|-----|-----|------|------|
| 1 | **CLIP ViT-L/14 + Color (a=0.5)** | **0.6417** | 0.7916 | 0.8306 | 0.8530 |
| 2 | CLIP ViT-L/14 + Color (a=0.7) | 0.6203 | 0.7936 | 0.8335 | 0.8530 |
| 3 | CLIP ViT-L/14 + Color (a=0.3) | 0.6183 | 0.7741 | 0.8306 | 0.8530 |
| 4 | CLIP ViT-L/14 | 0.5531 | 0.7478 | 0.8053 | 0.8530 |
| 5 | ResNet50+Color a=0.5 (Mark P1) | 0.4051 | 0.5930 | 0.6573 | 0.6913 |
| 6 | CLIP ViT-B/32 | 0.3934 | 0.5833 | 0.6582 | 0.7410 |
| 7 | EfficientNet-B0 (Mark P1) | 0.3671 | 0.5988 | 0.6855 | 0.7760 |
| 8 | ViT-B/16 (ImageNet) | 0.3661 | 0.5794 | 0.6563 | 0.7605 |
| 9 | ConvNeXt-Tiny | 0.3544 | 0.5774 | 0.6524 | 0.7595 |
| 10 | ResNet50 (P1 baseline) | 0.3067 | 0.4927 | 0.5901 | 0.6913 |
| 11 | DINOv2 ViT-B/14 | 0.2921 | 0.5891 | 0.7040 | 0.7858 |

## Key Findings

1. **Training paradigm ranking: Vision-Language (CLIP) >> Supervised ImageNet > Self-supervised (DINOv2) for R@1.** CLIP ViT-L/14 achieves R@1=55.3% zero-shot. The 400M image-text pair training data is the decisive factor, not the architecture.

2. **Scale within paradigm matters enormously.** CLIP B/32 (88M) → L/14 (304M) gains +16pp R@1. This is larger than any architecture change. For production, the bigger CLIP model is worth the 15x slower inference.

3. **Architecture doesn't matter when training is the same.** ViT-B/16 (36.6%) ≈ EfficientNet-B0 (36.7%) ≈ ConvNeXt-Tiny (35.4%) — all supervised on ImageNet, all within 2pp. The debate over CNN vs ViT is irrelevant for retrieval features.

4. **DINOv2 is the surprise: worst at R@1 (29.2%), but best non-CLIP at R@10 (70.4%).** Self-supervised features spread correct products across the top-K. This makes DINOv2 good for "fuzzy" retrieval (show similar items) but poor for exact product matching.

5. **Color re-ranking boosts all backbones by ~9pp.** CLIP L/14 + color = R@1=64.2%. The boost is additive regardless of backbone quality. Fashion consumers search by color first — deep features don't fully capture this.

6. **The separation paradox.** All semantic models (CLIP, DINOv2) have negative separation (incorrect matches are more similar than correct ones on average). Yet CLIP ViT-L/14 gets 55% R@1. The correct product is a spike above a sea of near-identical competitors.

## Next Steps
- Phase 3: Fine-tune CLIP ViT-L/14 with triplet loss on DeepFashion training pairs
- Test CLIP text-to-image retrieval ("blue cotton jacket" → matching products)
- Spatial color features (4-quadrant histogram) for structure-aware color matching
- Try combining CLIP + DINOv2 features (CLIP for R@1, DINOv2 for diversity)

## References Used Today
- [1] Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.
- [2] Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." TMLR.
- [3] Liu, Z., et al. (2022). "A ConvNet for the 2020s." CVPR.

## Code Changes
- `scripts/run_phase2_anthony.py` — 5-model comparison + color re-ranking (470 lines)
- `notebooks/phase2_anthony_model_comparison.ipynb` — executed research notebook (15 cells, 4 plots)
- `results/phase2_anthony_results.json` — all metrics
- `results/phase2_anthony_comparison.png` — bar chart + per-category heatmap
- `results/phase2_anthony_paradigm.png` — paradigm analysis
- `results/phase2_anthony_dim_vs_r1.png` — dimension vs performance scatter
- `results/phase2_anthony_recall_curves.png` — R@K curves
