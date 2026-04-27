# Experiment Log — Visual Product Search Engine

All visual-only experiments across Phases 1–6, ranked by R@1.
Evaluation: 300 gallery products, 1,027 query images, DeepFashion In-Shop.

> Earlier phases also explored a text-augmented variant that reached
> R@1 = 0.94 by indexing the gallery's product descriptions with CLIP's
> text encoder. That path required query-side text at inference and was
> rejected as production-invalid. It has been removed from the shipping
> code; only visual-only systems are listed below.

## Master Comparison Table — Visual-Only

| Rank | Phase | System | R@1 | R@5 | R@10 | R@20 | Dim | Author |
|------|-------|--------|-----|-----|------|------|-----|--------|
| 1 | P5 | CLIP L/14 + color + spatial + cat filter (Optuna) | 0.729 | 0.882 | 0.936 | 0.974 | 1008 | Anthony |
| 2 | P4 | Per-category alpha oracle (CLIP B/32 + cat filter) | 0.695 | — | — | — | — | Mark |
| 3 | P5 | PCA-64 whitened (CLIP+color+spatial) | 0.683 | 0.859 | 0.914 | 0.949 | 64 | Anthony |
| **4** | **P3** | **CLIP B/32 + cat filter + color (α=0.4) ★ shipping** | **0.683** | **0.862** | **0.913** | **0.970** | — | **Mark** |
| 5 | P5 | PCA-128 whitened | 0.682 | 0.871 | 0.914 | 0.944 | 128 | Anthony |
| 6 | P5 | CLIP L/14 + color + spatial (Optuna, no cat filter) | 0.660 | 0.801 | 0.860 | 0.904 | 1008 | Anthony |
| 7 | P3 | CLIP L/14 + color + spatial (equal weights) | 0.646 | 0.802 | 0.860 | 0.903 | 1008 | Anthony |
| 8 | P2 | CLIP ViT-L/14 + color rerank α=0.5 | 0.642 | 0.831 | 0.853 | 0.853 | — | Anthony |
| 9 | P5 | PCA-256 whitened | 0.636 | 0.821 | 0.855 | 0.893 | 256 | Anthony |
| 10 | P2 | CLIP ViT-L/14 + color rerank α=0.3 | 0.594 | — | — | — | — | Anthony |
| 11 | P5 | CLIP L/14 + spatial only | 0.585 | 0.761 | 0.819 | 0.859 | 960 | Anthony |
| 12 | P2 | CLIP ViT-B/32 + color rerank α=0.5 | 0.576 | 0.747 | 0.787 | 0.807 | — | Mark |
| 13 | P3 | CLIP B/32 + category filter (no color) | 0.569 | — | — | — | — | Mark |
| 14 | P2 | CLIP ViT-L/14 bare | 0.553 | 0.748 | 0.805 | 0.853 | 768 | Anthony |
| 15 | P2 | CLIP ViT-B/32 bare | 0.480 | 0.672 | 0.740 | 0.807 | 512 | Mark |
| 16 | P1 | ResNet50 + color rerank α=0.5 | 0.405 | 0.593 | 0.657 | 0.691 | — | Mark |
| 17 | P2 | CLIP ViT-B/32 + color rerank α=0.3 | 0.393 | — | — | — | — | Anthony |
| 18 | P1 | EfficientNet-B0 + color (aug) | 0.383 | 0.612 | 0.694 | 0.785 | 1304 | Mark |
| 19 | P1 | EfficientNet-B0 (ImageNet) | 0.367 | 0.599 | 0.686 | 0.776 | 1280 | Mark |
| 20 | P5 | Color + spatial only (no CLIP) | 0.357 | 0.537 | 0.622 | 0.718 | 240 | Anthony |
| 21 | P5 | Spatial 192D only | 0.341 | 0.490 | 0.569 | 0.645 | 192 | Anthony |
| 22 | P5 | Color 48D only | 0.338 | 0.524 | 0.613 | 0.707 | 48 | Anthony |
| 23 | P1 | Color-only 48D (histogram) | 0.338 | 0.524 | 0.613 | 0.707 | 48 | Mark |
| 24 | P2 | DINOv2 ViT-B/14 + color rerank | 0.328 | — | — | — | — | Mark |
| 25 | P1 | ResNet50 (ImageNet V2) | 0.307 | 0.493 | 0.590 | 0.691 | 2048 | Anthony |
| 26 | P2 | DINOv2 ViT-B/14 CLS bare | 0.243 | 0.450 | 0.560 | 0.807 | 768 | Mark |
| 27 | P3 | DINOv2 patch mean-pooling | 0.150 | — | — | — | 768 | Mark |

The CLIP B/32 + cat + color α=0.4 row at rank 4 is what ships in production. The L/14 + spatial variant at rank 1 reaches a higher R@1 but adds 192D and ~80ms of CLIP-L/14 forward pass on CPU vs B/32 — the operational tradeoff isn't worth +4.6pp R@1 in most deployments.

## Key Findings Across All Phases

1. **CLIP ViT-L/14 dominates all pretrained backbones** — training paradigm (vision-language) matters more than architecture.
2. **48D color histogram beats 2048D ResNet50** — fashion retrieval is fundamentally a color-matching problem.
3. **Category filter NEVER hurts** — 0/1,027 queries degraded, +6.9pp R@1 pure upside.
4. **DINOv2 underperforms on fashion** — SSL scene-level features miss product-level discrimination.
5. **Color rerank stacks on ALL backbones** — +9.6pp on CLIP, +9.8pp on ResNet, +8.5pp on DINOv2.
6. **96D color (16 bins) catastrophically fails** — coarser 8-bin quantization is more robust to lighting variation.
7. **Spatial features are nearly redundant** — only 1.7% of queries rescued, could be removed saving 192D.

## Phase Timeline

| Phase | Date | Focus | Champion R@1 | Key Discovery |
|-------|------|-------|--------------|---------------|
| 1 | Apr 20 | Domain + EDA + Baseline | 0.307 | Jackets 2.8× harder than shirts |
| 2 | Apr 21 | Foundation models | 0.642 | CLIP ≫ DINOv2 for fashion; color rerank stacks on all backbones |
| 3 | Apr 22 | Feature engineering | 0.683 | Category filter +8.9pp; visual-only champion shipped |
| 4 | Apr 23 | Hyperparameter tuning | 0.695 | 85.3% failures are top-5 close misses; 96D color catastrophe |
| 5 | Apr 25 | Optuna + ablation | 0.729 | L/14 + spatial + cat filter reaches the visual-only ceiling |
| 6 | Apr 26 | Production pipeline | 0.683 | Visual-only stack ships with B/32 backbone for inference cost |
