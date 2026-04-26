# Experiment Log — Visual Product Search Engine

All experiments across Phases 1-6, ranked by primary metric (Recall@1).
Evaluation: 300 gallery products, 1,027 query images, DeepFashion In-Shop.

## Master Comparison Table

| Rank | Phase | System | R@1 | R@5 | R@10 | R@20 | Dim | Prod-Valid | Author |
|------|-------|--------|-----|-----|------|------|-----|------------|--------|
| 1 | P5 | Text rerank: visual top-20 + CLIP B/32 text (w_text=0.8) | 0.907 | 0.944 | 0.944 | 0.944 | — | No | Mark |
| 2 | P5 | Text+Color rerank (no CLIP visual) | 0.920 | — | — | — | — | No | Mark |
| 3 | P5 | Text-only: CLIP B/32 "a photo of {color} {category}" | 0.820 | 1.000 | 1.000 | 1.000 | — | No | Anthony |
| **4** | **P5** | **CLIP L/14 + color + spatial + cat filter (Optuna)** | **0.729** | **0.882** | **0.936** | **0.974** | **1008** | **Yes** | **Anthony** |
| 5 | P4 | Per-category alpha oracle (CLIP B/32 + cat filter) | 0.695 | — | — | — | — | Yes | Mark |
| 6 | P5 | PCA-64 whitened (CLIP+color+spatial) | 0.683 | 0.859 | 0.914 | 0.949 | 64 | Yes | Anthony |
| 7 | P5 | PCA-128 whitened | 0.682 | 0.871 | 0.914 | 0.944 | 128 | Yes | Anthony |
| 8 | P3 | CLIP B/32 + cat filter + color (alpha=0.4) | 0.683 | 0.862 | 0.913 | 0.970 | — | Yes | Mark |
| 9 | P3 | CLIP L/14 + color + spatial + text | 0.675 | 0.856 | 0.872 | 0.910 | — | No | Anthony |
| 10 | P5 | CLIP L/14 + color + spatial (Optuna, no cat filter) | 0.660 | 0.801 | 0.860 | 0.904 | 1008 | Yes | Anthony |
| 11 | P3 | CLIP L/14 + color + spatial (equal weights) | 0.646 | 0.802 | 0.860 | 0.903 | 1008 | Yes | Anthony |
| 12 | P2 | CLIP ViT-L/14 + color rerank alpha=0.5 | 0.642 | 0.831 | 0.853 | 0.853 | — | Yes | Anthony |
| 13 | P5 | PCA-256 whitened | 0.636 | 0.821 | 0.855 | 0.893 | 256 | Yes | Anthony |
| 14 | P3 | CLIP L/14 + text metadata only | 0.602 | — | — | — | — | No | Anthony |
| 15 | P2 | CLIP ViT-L/14 + color rerank alpha=0.3 | 0.594 | — | — | — | — | Yes | Anthony |
| 16 | P5 | CLIP L/14 + spatial only | 0.585 | 0.761 | 0.819 | 0.859 | 960 | Yes | Anthony |
| 17 | P2 | CLIP ViT-B/32 + color rerank alpha=0.5 | 0.576 | 0.747 | 0.787 | 0.807 | — | Yes | Mark |
| 18 | P3 | CLIP B/32 + category filter (no color) | 0.569 | — | — | — | — | Yes | Mark |
| 19 | P2 | CLIP ViT-L/14 bare | 0.553 | 0.748 | 0.805 | 0.853 | 768 | Yes | Anthony |
| 20 | P2 | CLIP ViT-B/32 bare | 0.480 | 0.672 | 0.740 | 0.807 | 512 | Yes | Mark |
| 21 | P1 | ResNet50 + color rerank alpha=0.5 | 0.405 | 0.593 | 0.657 | 0.691 | — | Yes | Mark |
| 22 | P2 | CLIP ViT-B/32 + color rerank alpha=0.3 | 0.393 | — | — | — | — | Yes | Anthony |
| 23 | P1 | EfficientNet-B0 + Color (aug) | 0.383 | 0.612 | 0.694 | 0.785 | 1304 | Yes | Mark |
| 24 | P1 | EfficientNet-B0 (ImageNet) | 0.367 | 0.599 | 0.686 | 0.776 | 1280 | Yes | Mark |
| 25 | P5 | Color + spatial only (no CLIP) | 0.357 | 0.537 | 0.622 | 0.718 | 240 | Yes | Anthony |
| 26 | P5 | Spatial 192D only | 0.341 | 0.490 | 0.569 | 0.645 | 192 | Yes | Anthony |
| 27 | P5 | Color 48D only | 0.338 | 0.524 | 0.613 | 0.707 | 48 | Yes | Anthony |
| 28 | P1 | Color-only 48D (histogram) | 0.338 | 0.524 | 0.613 | 0.707 | 48 | Yes | Mark |
| 29 | P2 | DINOv2 ViT-B/14 + color rerank | 0.328 | — | — | — | — | Yes | Mark |
| 30 | P1 | ResNet50 (ImageNet V2) | 0.307 | 0.493 | 0.590 | 0.691 | 2048 | Yes | Anthony |
| 31 | P2 | DINOv2 ViT-B/14 CLS bare | 0.243 | — | — | 0.807 | 768 | Yes | Mark |
| 32 | P3 | DINOv2 patch mean-pooling | 0.150 | — | — | — | 768 | Yes | Mark |

## Key Findings Across All Phases

1. **CLIP ViT-L/14 dominates all pretrained backbones** — training paradigm (vision-language) matters more than architecture
2. **48D color histogram beats 2048D ResNet50** — fashion retrieval is fundamentally a color-matching problem
3. **Category filter NEVER hurts** — 0/1,027 queries degraded, +6.9pp R@1 pure upside
4. **Text metadata is an evaluation trap** — R@1=60.2% text-only vs 55.3% visual, but text isn't available at inference
5. **DINOv2 underperforms on fashion** — SSL scene-level features miss product-level discrimination
6. **Color rerank stacks on ALL backbones** — +9.6pp on CLIP, +9.8pp on ResNet, +8.5pp on DINOv2
7. **96D color (16 bins) catastrophically fails** — coarser 8-bin quantization is more robust to lighting variation
8. **Spatial features are nearly redundant** — only 1.7% of queries rescued, could be removed saving 192D

## Phase Timeline

| Phase | Date | Focus | Champion R@1 | Key Discovery |
|-------|------|-------|-------------|---------------|
| 1 | Apr 20 | Domain + EDA + Baseline | 0.307 | Jackets 2.8x harder than shirts |
| 2 | Apr 21 | Foundation models | 0.642 | CLIP >> DINOv2 for fashion; color rerank stacks on all backbones |
| 3 | Apr 22 | Feature engineering | 0.683 | Category filter +8.9pp; text beats visual embeddings |
| 4 | Apr 23 | Hyperparameter tuning | 0.695 | 85.3% failures are top-5 close misses; 96D color catastrophe |
| 5 | Apr 25 | Advanced techniques | 0.729 | Visual-only ceiling; text rerank 0.907 but not prod-valid |
| 6 | Apr 25 | Explainability | 0.729 | CLIP handles 54.1% alone; category filter pure upside |
