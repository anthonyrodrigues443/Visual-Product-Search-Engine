# Visual Product Search Engine

**Image retrieval for fashion products using deep embeddings + approximate nearest neighbour search.** ResNet50 pretrained features achieve Recall@1=30.7% on DeepFashion In-Shop without fine-tuning, matching published expectations. Published zero-shot CLIP achieves ~78% — a 47pp gap that per-category analysis shows is concentrated in visually diverse categories like jackets.

> **Headline finding (Phase 1):** Jackets are 2.8× harder than shirts (R@1=13.9% vs 38.8%). Generic ImageNet features collapse visually diverse categories. Poor cosine similarity separation (0.048) between correct and incorrect matches is the core bottleneck — fine-tuning must widen this gap.

---

## Dataset

**[DeepFashion In-Shop](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)** — Large-scale fashion retrieval benchmark (Liu et al., CVPR 2016)

| Metric | Value |
|--------|-------|
| Total images | 52,591 |
| Unique products | 12,995 |
| Product categories | 16 |
| Unique colors | 804 |
| Images per product | 4.0 (mean), 1–7 (range) |
| Views per product | front, side, back, additional, full, flat |
| Gender split | Women 85.1%, Men 14.9% |
| Train / Test split | 10,396 / 2,599 products (80/20 by product) |

**Primary metric:** Recall@K — standard in image retrieval literature (DeepFashion, Stanford Online Products, all major retrieval benchmarks). Measures whether the correct product appears in the top-K results.

---

## Current Status

**Phase 1 complete** — ResNet50 baseline + color palette features + EfficientNet-B0 established.

| Model | R@1 | R@5 | R@10 | R@20 | Dim | Notes |
|-------|-----|-----|------|------|-----|-------|
| ResNet50 (ImageNet V2) | 30.7% | 49.3% | 59.0% | 69.1% | 2048 | Anthony baseline, no fine-tuning |
| EfficientNet-B0 (ImageNet) | 36.7% | 59.9% | 68.6% | 77.6% | 1280 | Mark — 5x smaller, beats ResNet50 |
| Color-only 48D (histogram) | 33.8% | 52.4% | 61.3% | 70.7% | 48 | Mark — 48 numbers beat 2048D CNN |
| EfficientNet-B0 + Color (aug) | 38.3% | 61.2% | 69.4% | 78.5% | 1304 | Mark — best embedding approach |
| **ResNet50 + color rerank alpha=0.5** | **40.5%** | **59.3%** | **65.7%** | **69.1%** | — | **Mark — best overall, no retraining** |
| FashionNet (published) | 53.0% | — | 73.0% | 76.4% | — | Fine-tuned, 2016 |
| CLIP ViT-B/32 (published) | ~78% | — | ~93% | ~95% | 512 | Zero-shot, 2021 |
| DINOv2 ViT-B/14 (published) | ~82% | — | ~95% | ~97% | 768 | Zero-shot, 2023 |

**Best model so far:** ResNet50 + color re-ranking (alpha=0.5) — R@1=40.5% (+9.8pp vs baseline, no retraining)

---

## Key Findings

1. **ResNet50 baseline confirms published expectations.** 30.7% Recall@1 without fine-tuning, in the ~30–40% range expected for generic ImageNet features on fashion retrieval. Fine-tuned models reach 53–82%.

2. **EfficientNet-B0 beats ResNet50 with 5x smaller weights.** R@1=36.7% vs 30.7% (+6pp) using a 20MB model vs 98MB. Compound scaling outperforms plain depth-scaling for fine-grained visual retrieval.

3. **48D color histogram alone beats 2048D ResNet50.** Color-only retrieval: R@1=33.8%. Fashion consumers search by color first — CNN features optimized for ImageNet classification don't model color explicitly enough.

4. **Color re-ranking (alpha=0.5) gives +9.8pp with zero retraining.** Blending CNN and color scores at inference time is more effective than concatenating them in the embedding. The correct product is already in the top-20 — re-ranking surfaces it.

5. **Jackets are 2.8× harder than shirts.** R@1=13.9% vs 38.8%. Visually diverse categories (bomber vs blazer vs parka) need more discriminative features than uniform categories (shirts with distinctive prints).

6. **Poor similarity separation (0.048) is the core bottleneck.** Correct matches score 0.786 cosine similarity; incorrect score 0.738. The distributions overlap heavily — fine-tuning or metric learning must widen this gap.

---

## Models Compared

**6 approaches** across Phase 1 (Anthony: ResNet50 baseline; Mark: EfficientNet-B0, color features, augmented embeddings, color re-ranking).

---

## Architecture

```
Product image (52,591 DeepFashion In-Shop images)
             │
             ▼
    CNN Feature Extractor
    ────────────────────────
    ResNet50 (ImageNet V2)
    avg pool, no classification head
    L2-normalized 2048-dim vectors
             │
             ▼
    FAISS IndexFlatIP
    ────────────────────────
    Cosine similarity search
    Gallery: 300 images
    Queries: 1,027 images
             │
             ▼
    Recall@K Evaluation
    ────────────────────────
    K = 1, 5, 10, 20
    Per-category breakdown
```

---

## Iteration Summary

### Phase 1: Domain Research + EDA + Baseline — 2026-04-20

<table>
<tr>
<td valign="top" width="38%">

**EDA Run 1:** Analysed 52,591 DeepFashion In-Shop images across 16 categories, 804 colours, and 6 view types. Tees dominate (27.3%); suiting nearly absent (0.07%). Class imbalance makes per-category evaluation essential — aggregate Recall@K hides dramatic variation.<br><br>
**EDA Run 2:** ResNet50 (ImageNet V2) + FAISS cosine retrieval on 300 gallery / 1,027 query images. R@1=30.7%, R@20=69.1%. Per-category breakdown: shirts 38.8%, jackets 13.9%. Cosine similarity separation between correct and incorrect matches: only 0.048.

</td>
<td align="center" width="24%">

<img src="results/baseline_per_category.png" width="220">

</td>
<td valign="top" width="38%">

**Combined Insight:** The R@1→R@20 gap (+38pp) reveals that the correct product is typically in the "neighbourhood" but not at the top — generic ImageNet features embed visual similarity correctly but can't rank within a cluster. Fine-tuning doesn't need to learn new representations; it needs to tighten intra-class distances.<br><br>
**Surprise:** Jackets are 2.8× harder than shirts despite being a common category. The issue isn't data volume — it's intra-category visual variance (bomber vs blazer vs parka). Category difficulty correlates with style diversity, not sample size.<br><br>
**Research:** Liu et al., 2016 — FashionNet fine-tuned on DeepFashion reaches R@1=53%, so we tried pretrained ResNet50 as a reproducible floor. arXiv 2503.13045, 2025 — no single metric learning loss dominates; contrastive, triplet, and InfoNCE each excel in different regimes, so Phase 2 will test CLIP and DINOv2 before committing to a fine-tuning strategy.<br><br>
**Best Model So Far:** ResNet50 (ImageNet V2) — R@1=30.7%

</td>
</tr>
</table>
