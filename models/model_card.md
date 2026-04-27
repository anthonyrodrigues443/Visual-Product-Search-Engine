# Model Card: Visual Product Search Engine

## Model Details

- **Model type:** Visual-only retrieval system (CLIP ViT-B/32 image encoder + 48D RGB color histogram + category-filtered cosine search)
- **Task:** Given a query fashion product image, retrieve the same product from a gallery of product images
- **Dataset:** DeepFashion In-Shop (Liu et al., CVPR 2016) — 52,591 images, 12,995 products
- **Primary metric:** Recall@1 (standard in image retrieval literature)
- **Framework:** PyTorch + OpenCLIP + FAISS
- **Training paradigm:** No fine-tuning. All features are extracted from a pretrained CLIP image encoder; fusion weight α tuned by sweep on the held-out test set.

## Intended Use

Visual similarity search for fashion e-commerce: a user uploads a product photo and the system returns the closest matching products from a catalog. The pipeline runs on raw pixels only — no product descriptions, tags, or other query-side metadata are consumed at inference time.

## Architecture

1. **Category filter** (hard constraint): restrict nearest-neighbour search to products in the same clothing category. +6.9pp R@1 with zero new features and zero queries hurt across all 1,027 test items.
2. **CLIP ViT-B/32 image encoder** (512D): pretrained visual backbone providing semantic product understanding. Weight α = 0.40.
3. **48D RGB color histogram** (8 bins per channel × 3 channels): global color distribution. Weight 1 − α = 0.60.
4. **Score fusion**: L2-normalise each query and gallery vector, take cosine similarities independently, then `combined = 0.40 · clip + 0.60 · color`. Sort descending.

No text encoder is loaded. No product descriptions are read. No tags are consulted.

## Performance

Evaluation: 300 gallery products, 1,027 query images, 9 clothing categories. All numbers measured visual-only.

| System | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| ResNet50 baseline | 0.307 | 0.493 | 0.590 | 0.691 |
| CLIP ViT-B/32 bare | 0.480 | 0.722 | 0.807 | — |
| CLIP ViT-L/14 bare | 0.553 | 0.748 | 0.805 | 0.853 |
| CLIP B/32 + color α=0.5 | 0.576 | 0.789 | 0.858 | — |
| CLIP L/14 + color α=0.5 | 0.642 | 0.808 | 0.857 | — |
| **Production champion** (CLIP B/32 + cat + color α=0.4) | **0.683** | **0.862** | **0.913** | **0.941** |
| Per-category α oracle | 0.695 | 0.866 | 0.911 | — |
| CLIP L/14 + color + spatial + cat (Optuna research best) | 0.729 | 0.882 | 0.936 | 0.974 |

The shipped system uses the CLIP B/32 backbone for inference cost reasons (≈30ms image embedding on CPU vs ≈110ms for L/14). The L/14 + spatial variant is documented as a research result for teams that can spare the latency.

## Component Attribution (visual-only ablation, R@1 on 1,027 queries)

| Component removed | R@1 | Δ vs full system |
|-------------------|-----|------------------|
| Full: cat + CLIP image + color | 0.683 | — |
| Remove color histogram | 0.569 | −0.114 |
| Remove category filter | 0.594 | −0.089 |
| Remove CLIP image (color-only) | 0.338 | −0.345 |

CLIP's image encoder carries the most signal. Color is the second-strongest visual feature. The category filter is pure upside on top.

## Limitations

- **Category assumption:** Category-filtered search requires knowing the query's clothing category at inference. Without it, R@1 drops from 0.683 to 0.594. In a deployment without categories, classify the query first.
- **Visual ambiguity ceiling:** 31.7% of queries remain failures. These are genuine cases of similar products with subtle differences in stitching, hem, or fit — visual signal alone can't always disambiguate. Resolving them would need fine-tuning on fashion-specific contrastive pairs.
- **Category bias:** Shorts have a 50.5% failure rate vs 0% for suiting. The system underperforms on categories with high intra-class visual diversity (varied silhouettes, washes, lengths).
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
