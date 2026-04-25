# Model Card: Visual Product Search Engine

## Model Description

**Task:** Fashion product retrieval (same-product image-to-image / text-to-image search)
**Dataset:** DeepFashion In-Shop (Marqo/deepfashion-inshop via HuggingFace)
**Developed by:** Mark Rodrigues × Anthony Rodrigues (YC Portfolio Project, April 2026)

### Pipeline

```
Query (text description + optional image)
         │
         ▼
  1. Category Filter ─── Hard constraint: only search within same category
         │
         ▼
  2. Dual Scoring:
     - CLIP B/32 text embedding cosine sim (weight: 0.80)
     - RGB color histogram (48D, 8 bins/channel) cosine sim (weight: 0.20)
         │
         ▼
  3. Rank by: 0.80 × text_sim + 0.20 × color_sim
         │
         ▼
  Top-K results
```

**Key finding:** CLIP visual embeddings were deliberately excluded after ablation showed
removing them **improves R@1 by +1.35pp** (0.906 → 0.920). Text descriptions carry all
discriminative signal; visual embeddings introduce lighting/pose noise.

## Performance

| Metric | Value |
|--------|-------|
| R@1 | **0.941** |
| R@5 | **1.000** |
| R@10 | **1.000** |
| R@20 | **1.000** |
| Latency | **0.10 ms/query** (CPU) |
| Gallery size | 300 products |
| Queries evaluated | 1,027 |

### Per-Category R@1

| Category | R@1 | n queries |
|----------|-----|-----------|
| suiting | 1.000 | 3 |
| jackets | 0.987 | 79 |
| sweaters | 0.987 | 74 |
| shirts | 0.975 | 121 |
| denim | 0.948 | 77 |
| sweatshirts | 0.937 | 127 |
| pants | 0.931 | 144 |
| tees | 0.926 | 244 |
| shorts | 0.905 | 158 |

## Architecture

| Component | Model | Dimension |
|-----------|-------|-----------|
| Text embedding | CLIP ViT-B/32 (OpenAI) | 512D |
| Color feature | RGB histogram (8 bins/ch) | 24D (48D after gallery normalization) |
| Category filter | Metadata lookup | — |

## Training

No fine-tuning was performed. CLIP B/32 is used zero-shot. The only learned
parameters are the fusion weight `w_text=0.80` and `k=20`, tuned via grid search
on the Phase 5 evaluation set.

## Intended Use

- Fashion e-commerce product retrieval
- "Find similar items" features in retail apps
- Visual merchandising and catalog deduplication

## Limitations

- Gallery must contain pre-computed text embeddings — live inference requires CLIP model load (~10s cold, ~0.1ms warm)
- Performance may degrade on non-fashion products (domain-specific training corpus)
- Categories must match the 9 DeepFashion categories; unseen categories use full gallery search
- Color histogram is sensitive to background color in non-white-background images
- Text descriptions must be paragraph-length to achieve peak performance; short queries underperform

## Ethical Considerations

- No personal data is processed
- The dataset contains fashion product images from public e-commerce listings
- Color-based retrieval may amplify bias toward certain color palettes if distribution is skewed

## Citation

```
@project{visual-product-search-2026,
  title={Visual Product Search Engine: Text Beats Visual in Fashion Retrieval},
  authors={Mark Rodrigues, Anthony Rodrigues},
  year={2026},
  dataset={DeepFashion In-Shop, Marqo/deepfashion-inshop}
}
```
