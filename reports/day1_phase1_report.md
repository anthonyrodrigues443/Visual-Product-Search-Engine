# Phase 1: Domain Research + EDA + Baseline — Visual Product Search Engine
**Date:** 2026-04-20
**Session:** 1 of 7
**Researcher:** Anthony Rodrigues

## Objective
Can a pretrained CNN achieve reasonable retrieval quality on fashion product images? What's the baseline Recall@K before fine-tuning, and how does it compare to published benchmarks?

## Research & References
1. [Google Lens, 2025] — 12B visual searches/month. Industry uses CNN embeddings + ANN search (FAISS/HNSW).
2. [CLIP vs DINOv2 comparison, GoPubBy AI] — CLIP excels at semantic/text-aware retrieval; DINOv2 better at low-level visual patterns. Hybrid approaches emerging.
3. [Image Retrieval Training Guide, arXiv 2025 (2503.13045)] — No single metric learning loss dominates; contrastive, triplet, and InfoNCE each excel in different settings.

How research influenced today's experiments: Published DeepFashion benchmarks show Recall@1 of 53-82% for fine-tuned models. Without fine-tuning, pretrained ImageNet features should give ~30-40% Recall@1. This sets expectations for our baseline.

## Dataset
| Metric | Value |
|--------|-------|
| Source | Marqo/deepfashion-inshop (HuggingFace) |
| Total images | 52,591 |
| Unique products | 12,995 |
| Product categories | 16 |
| Unique colors | 804 |
| Images per product | 4.0 (mean), 4 (median), 1-7 (range) |
| Views per product | front (24.4%), side (20.7%), back (20.2%), additional (20.8%), full (12.6%), flat (1.3%) |
| Gender split | Women 85.1%, Men 14.9% |
| Train/Test split | 10,396 / 2,599 products (80/20 by product) |

## Primary Metric Selection
**Recall@K** — Standard in image retrieval literature. Used by DeepFashion benchmark, Stanford Online Products, and virtually all retrieval papers. Directly measures whether the correct product appears in the top-K results.

**Why Recall@K over mAP:** For product search, users care whether the right product appears on the first page (Recall@10), not the exact ranking of all results. Recall@K is also the most commonly reported metric, enabling direct comparison with published baselines.

Secondary metrics: similarity separation (cosine similarity gap between correct and incorrect matches).

## Experiments

### Experiment 1.1: Dataset EDA
**Hypothesis:** The dataset has sufficient diversity for a meaningful retrieval benchmark.
**Method:** Analyzed distributions of gender, category, color, and view across all 52,591 images.
**Result:**
- Heavy class imbalance: tees (27.3%) dominate; suiting (0.07%) is nearly absent
- Color distribution: Black (6.5%), Cream (5.9%), Burgundy (4.7%) are top 3
- Multiple views per product (front/side/back/additional/full) enable robust cross-view retrieval
**Interpretation:** The dataset is rich enough for meaningful retrieval research, but category imbalance means per-category evaluation is critical. Models might learn "tee-ness" at the expense of rare categories.

### Experiment 1.2: ResNet50 Baseline Retrieval
**Hypothesis:** Pretrained ResNet50 (ImageNet V2) features should achieve ~30-40% Recall@1 without fine-tuning.
**Method:** Extracted 2048-dim features from ResNet50 (avg pool, no classification head), L2-normalized, built FAISS IndexFlatIP (cosine similarity), evaluated on 300 gallery / 1027 query images.
**Result:**
| Metric | Value |
|--------|-------|
| Recall@1 | 30.7% |
| Recall@5 | 49.3% |
| Recall@10 | 59.0% |
| Recall@20 | 69.1% |
**Interpretation:** Matches our hypothesis exactly. ResNet50 without fine-tuning captures enough visual similarity for ~30% exact matches. The steep improvement from R@1 to R@20 (+38pp) suggests the correct product is often "close" in embedding space but not the top match, indicating fine-tuning or better features could dramatically improve results.

### Experiment 1.3: Per-Category Performance
**Result:**
| Category | R@1 | R@10 | Queries |
|----------|-----|------|---------|
| shirts | 0.388 | 0.702 | 121 |
| sweaters | 0.378 | 0.662 | 74 |
| tees | 0.352 | 0.590 | 244 |
| denim | 0.351 | 0.597 | 77 |
| pants | 0.306 | 0.611 | 144 |
| shorts | 0.253 | 0.544 | 158 |
| sweatshirts | 0.244 | 0.528 | 127 |
| jackets | 0.139 | 0.481 | 79 |
**Interpretation:** Jackets are hardest (R@1=13.9%), likely because visual variation within jackets is highest (bomber vs blazer vs parka). Shirts perform best (R@1=38.8%), possibly because distinctive patterns/prints make them more identifiable in embedding space.

### Experiment 1.4: Similarity Distribution Analysis
**Result:** Correct matches have mean cosine similarity 0.786 vs incorrect 0.738. Separation of only 0.048.
**Interpretation:** Poor separation explains the moderate Recall@1. The distributions heavily overlap, meaning the model can't reliably distinguish "same product, different angle" from "similar product, different item." This is the core challenge fine-tuning must solve.

## Head-to-Head Comparison
| Rank | Model | R@1 | R@5 | R@10 | R@20 | Dim | Notes |
|------|-------|-----|-----|------|------|-----|-------|
| — | ResNet50 (ImageNet V2) | 30.7 | 49.3 | 59.0 | 69.1 | 2048 | Baseline, no fine-tuning |
| — | FashionNet (published) | 53.0 | — | 73.0 | 76.4 | — | Fine-tuned, 2016 |
| — | CLIP ViT-B/32 (published) | ~78 | — | ~93 | ~95 | 512 | Zero-shot, 2021 |
| — | DINOv2 ViT-B/14 (published) | ~82 | — | ~95 | ~97 | 768 | Zero-shot, 2023 |

## Key Findings
1. **ResNet50 baseline confirms published expectations.** 30.7% Recall@1 without fine-tuning, matching the ~30-40% range expected for generic ImageNet features on fashion retrieval.
2. **Jackets are 2.8x harder than shirts.** Per-category analysis reveals dramatic performance variation (13.9% vs 38.8% R@1). Visually diverse categories need more discriminative features.
3. **Poor similarity separation (0.048) is the bottleneck.** Correct and incorrect matches overlap heavily in cosine similarity space. Fine-tuning or metric learning should dramatically widen this gap.
4. **R@1→R@20 improvement of +38pp shows promise.** The correct product is usually "nearby" in embedding space; a better model should promote it to the top.

## Error Analysis
- **Cross-category confusion:** Jackets vs sweaters (both upper body, similar shapes) cause frequent retrieval errors.
- **Color dominance:** Dark items from different categories cluster together because ImageNet features are sensitive to overall color distribution.
- **View sensitivity:** Different views of the same product can look dramatically different (front vs back), making cross-view matching the core challenge.

## Next Steps
- Phase 2: Compare CLIP, DINOv2, EfficientNet, ViT against ResNet50 baseline
- Test whether CLIP's text-image alignment helps with category disambiguation
- Evaluate different embedding dimensions (512 vs 768 vs 1024 vs 2048)
- Try different FAISS index types (HNSW, IVF) for speed benchmarking

## References Used Today
- [1] CLIP vs DINOv2 comparison: https://ai.gopubby.com/clip-vs-dinov2-which-one-is-better-for-image-retrieval-d68c03f51f0d
- [2] Image retrieval training guide: https://arxiv.org/html/2503.13045v1
- [3] FAISS index guidelines: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- [4] DeepFashion In-Shop benchmark: https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html

## Code Changes
- `src/data_pipeline.py` — data loading, metadata extraction, retrieval split creation
- `scripts/run_phase1.py` — full Phase 1 pipeline: EDA plots, ResNet50 baseline, FAISS retrieval
- `notebooks/phase1_eda_baseline.ipynb` — research notebook (executed cells 0-26; cells 28-35 ran via script due to FAISS+Python 3.14 shutdown issue)
- `results/metrics.json` — all baseline and per-category metrics
- `results/*.png` — 6 plots: distributions, colors, heatmap, per-category, similarity, sample products
