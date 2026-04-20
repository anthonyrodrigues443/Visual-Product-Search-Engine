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

**Phase 1 complete** — ResNet50 baseline established at R@1=30.7%.

| Model | R@1 | R@5 | R@10 | R@20 | Dim | Notes |
|-------|-----|-----|------|------|-----|-------|
| **ResNet50 (ImageNet V2)** | **30.7%** | **49.3%** | **59.0%** | **69.1%** | 2048 | Baseline, no fine-tuning |
| FashionNet (published) | 53.0% | — | 73.0% | 76.4% | — | Fine-tuned, 2016 |
| CLIP ViT-B/32 (published) | ~78% | — | ~93% | ~95% | 512 | Zero-shot, 2021 |
| DINOv2 ViT-B/14 (published) | ~82% | — | ~95% | ~97% | 768 | Zero-shot, 2023 |

**Best model so far:** ResNet50 (ImageNet V2) — R@1=30.7%

---

## Key Findings

1. **ResNet50 baseline confirms published expectations.** 30.7% Recall@1 without fine-tuning, in the ~30–40% range expected for generic ImageNet features on fashion retrieval. Fine-tuned models reach 53–82%.

2. **Jackets are 2.8× harder than shirts.** R@1=13.9% vs 38.8%. Visually diverse categories (bomber vs blazer vs parka) need more discriminative features than uniform categories (shirts with distinctive prints).

3. **Poor similarity separation (0.048) is the bottleneck.** Correct matches score 0.786 cosine similarity; incorrect score 0.738. The distributions overlap heavily — fine-tuning or metric learning must widen this gap.

4. **R@1→R@20 improvement of +38pp shows promise.** The correct product is "nearby" in embedding space; a better model should promote it to the top result.

---

## Models Compared

**1 model, 3 experiment variants** across Phase 1.

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
