# Model Card: Visual Product Search Engine

## Model Details

- **Model type:** Multi-feature retrieval system (CLIP ViT-L/14 + color histograms + spatial color grid + category-filtered FAISS search)
- **Task:** Given a query fashion product image, retrieve the same product from a gallery of product images
- **Dataset:** DeepFashion In-Shop (Liu et al., CVPR 2016) — 52,591 images, 12,995 products
- **Primary metric:** Recall@1 (standard in image retrieval literature)
- **Framework:** PyTorch + OpenCLIP + FAISS
- **Training paradigm:** No fine-tuning. All features are extracted from pretrained models with hand-engineered fusion weights optimized via Optuna (300 trials)

## Intended Use

Visual similarity search for fashion e-commerce: a user uploads a product photo and the system returns the closest matching products from a catalog. Designed for production-valid visual-only retrieval where query-side text metadata is unavailable.

## Architecture

1. **CLIP ViT-L/14** (768D): Pretrained visual backbone providing semantic product understanding
2. **Color histograms** (48D): RGB (24D) + HSV (24D) global color distribution
3. **Spatial color grid** (192D): 4x4 grid of per-region HSV histograms capturing color layout
4. **Feature fusion**: L2-normalize each block, weight (1.0, 1.0, 0.25), concatenate to 1008D
5. **Category-filtered FAISS search**: Restrict nearest-neighbor search to products in the same clothing category

## Performance

| System | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| ResNet50 baseline | 0.307 | 0.493 | 0.590 | 0.691 |
| CLIP ViT-L/14 bare | 0.553 | 0.748 | 0.805 | 0.853 |
| **Champion (visual+cat filter)** | **0.729** | **0.882** | **0.936** | **0.974** |
| Text rerank (not prod-valid) | 0.907 | 0.944 | 0.944 | 0.944 |

Evaluation: 300 gallery products, 1,027 query images, 9 clothing categories.

## Component Attribution

| Component | Queries Rescued | R@1 Contribution |
|-----------|-----------------|------------------|
| CLIP alone | 54.1% (556/1027) | +30.3pp |
| Color features | 12.0% (123/1027) | +7.5pp |
| Category filter | 5.2% (53/1027) | +6.9pp |
| Spatial features | 1.7% (17/1027) | +1.5pp |

## Limitations

- **Category assumption:** Category-filtered search requires knowing the query's clothing category at inference time. Without it, R@1 drops from 0.729 to 0.660.
- **Visual-only ceiling:** 27.1% of queries remain failures even with all visual features. These are genuine visual ambiguity cases (similar products with subtle differences in stitching, hem, or fit) that require fine-tuning or cross-modal signals to resolve.
- **Category bias:** Shorts have a 47.5% failure rate vs 6.8% for sweaters. The system underperforms on categories with high intra-class visual diversity.
- **Evaluation scale:** Evaluated on 300 products. Performance on the full 12,995-product catalog may differ (larger gallery increases difficulty).
- **No fine-tuning:** All features are from pretrained models. Fine-tuning CLIP on fashion data would likely close the 17.8pp gap to text-reranked performance.

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
