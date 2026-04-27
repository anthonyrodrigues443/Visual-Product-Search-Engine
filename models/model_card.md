# Model Card: Visual Product Search Engine

## Model Details

This card documents two production-valid visual-only pipelines from six phases of research. Both work on raw pixels with no query-side metadata. Both use pretrained models with zero fine-tuning.

| | Research peak (Anthony, P5) | Shipping demo (Mark, P3) |
|---|---|---|
| **Backbone** | CLIP ViT-L/14 (768D image) | CLIP ViT-B/32 (512D image) |
| **Color block** | 48D RGB histogram | 48D RGB histogram |
| **Spatial block** | 192D 4×4 HSV grid | — |
| **Category filter** | yes | yes |
| **Fusion weights** | Optuna-tuned (300 trials) | α = 0.40 (CLIP : color = 0.4 : 0.6) |
| **R@1** | **0.729** | **0.683** |
| **R@5** | 0.882 | 0.862 |
| **R@10** | 0.936 | 0.913 |
| **R@20** | 0.974 | 0.970 |
| **Inference cost (per query, CPU)** | ~110ms (CLIP L/14 forward + 192D spatial extract) | ~30ms (CLIP B/32 forward + color hist) |
| **Search latency (300-product gallery)** | <1ms | <1ms |

- **Task:** Given a query fashion product image, retrieve the same product from a gallery of product images.
- **Dataset:** DeepFashion In-Shop (Liu et al., CVPR 2016) — 52,591 images, 12,995 products. Eval: 300 gallery products, 1,027 query images.
- **Primary metric:** Recall@1 (standard in image retrieval literature).
- **Framework:** PyTorch + OpenCLIP + FAISS.
- **Training paradigm:** No fine-tuning. The shipping system uses a static α; the research peak fuses four blocks with Optuna-tuned weights.

## Intended Use

Visual similarity search for fashion e-commerce: a user uploads a product photo and the system returns the closest matching products from a catalog. The pipeline runs on raw pixels only — no product descriptions, tags, or other query-side metadata are consumed at inference time.

## Architecture (both pipelines)

Both pipelines share three components and differ only in backbone choice and the optional spatial block:

1. **Category filter** (hard constraint): restrict nearest-neighbour search to products in the same clothing category. Pure upside — zero queries hurt across all 1,027 test items in either pipeline.
2. **CLIP image encoder**: pretrained ViT — L/14 (768D) for the research peak, B/32 (512D) for the shipping demo.
3. **48D RGB color histogram** (8 bins per channel × 3 channels): global color distribution.
4. **192D spatial color grid** (4×4 region HSV histograms, research-peak only): captures *where* the colors live, not just which colors appear.
5. **Score fusion**: L2-normalise each block, take cosine similarities independently, then weighted sum. Shipping system: `0.40 · clip + 0.60 · color`. Research peak: Optuna-tuned 4-way fusion across CLIP, color, spatial, and a category-conditional offset.

No text encoder is loaded in either pipeline. No product descriptions are read. No tags are consulted.

## Performance

Evaluation: 300 gallery products, 1,027 query images, 9 clothing categories. All numbers measured visual-only.

| System | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| ResNet50 baseline | 0.307 | 0.493 | 0.590 | 0.691 |
| CLIP ViT-B/32 bare | 0.480 | 0.722 | 0.807 | — |
| CLIP ViT-L/14 bare | 0.553 | 0.748 | 0.805 | 0.853 |
| CLIP B/32 + color α=0.5 | 0.576 | 0.789 | 0.858 | — |
| CLIP L/14 + color α=0.5 | 0.642 | 0.808 | 0.857 | — |
| **Shipping demo** (CLIP B/32 + cat + color α=0.4, Mark P3) | **0.683** | **0.862** | **0.913** | **0.970** |
| Per-category α oracle | 0.695 | 0.866 | 0.911 | — |
| **★ Research peak** (CLIP L/14 + color + spatial + cat, Optuna, Anthony P5) | **0.729** | **0.882** | **0.936** | **0.974** |

The shipping demo uses the lighter B/32 backbone for inference cost reasons (≈30ms image embedding on CPU vs ≈110ms for L/14, plus an extra 192D feature extract). The +4.6pp R@1 the research peak buys you is real but costs roughly 4× more compute per query.

## Component Attribution (visual-only ablation, R@1 on 1,027 queries)

Measured live via `scripts/verify_ui_numbers.py`. Each row removes one component from the production champion (cat filter + CLIP B/32 image + 48D color hist, α = 0.4):

| Component removed | R@1 | Δ vs full system |
|-------------------|-----|------------------|
| Full: cat + CLIP image + color | 0.6826 | — |
| Remove color histogram (CLIP + cat) | 0.5686 | −0.114 |
| Remove category filter (CLIP + color) | 0.5794 | −0.103 |
| Remove CLIP image (color + cat) | 0.5131 | −0.170 |
| Color-only baseline (no CLIP, no cat) | 0.3505 | −0.332 |

CLIP's image encoder is the strongest single contributor. Color is second. The category filter is pure upside on top.

## Limitations

- **Category assumption:** Category-filtered search requires knowing the query's clothing category at inference. Without it, R@1 drops from 0.683 to 0.579. In a deployment without categories, classify the query first.
- **Visual ambiguity ceiling:** 31.7% of queries fail at top-1 (1 − R@1). These are genuine cases of similar products with subtle differences in stitching, hem, or fit — visual signal alone can't always disambiguate. Resolving them would need fine-tuning on fashion-specific contrastive pairs.
- **Category bias:** Shorts sit at R@1 = 0.475 vs 0.905 for sweaters and 1.000 for suiting. The system underperforms on categories with high intra-class visual diversity (varied silhouettes, washes, lengths).
- **Evaluation scale:** Evaluated on 300 products. Performance on the full 12,995-product catalog will be lower (larger gallery increases difficulty).
- **No fine-tuning:** All features are from a pretrained model. Fine-tuning CLIP on fashion data would lift R@1 further; reproducing FashionNet (CVPR 2016) gets to 0.53 with much smaller models.

## Ethical Considerations

- The model inherits biases from CLIP's training data and DeepFashion's composition (85.1% women's clothing, 14.9% men's). Retrieval quality for underrepresented categories (e.g., gender-neutral, plus-size) has not been evaluated.
- No personal data is processed — the system operates on product catalog images only.

## Citation

```
@inproceedings{liu2016deepfashion,
  title={DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
  author={Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou},
  booktitle={CVPR},
  year={2016}
}
```
