# Phase 6: Explainability & Model Understanding — Visual Product Search Engine
**Date:** 2026-04-25
**Session:** 6 of 7
**Researcher:** Anthony Rodrigues

## Objective
Six questions about the Phase 5 champion (CLIP L/14 + color + spatial + category filter, R@1=0.7293):
1. Per-query attribution: Which component rescues each query?
2. Similarity decomposition: What separates success from failure at the component level?
3. Failure taxonomy: What visual patterns characterize the 278 failures?
4. Category filter impact: When does it help vs hurt?
5. Embedding structure: How well do categories cluster?
6. Rank improvement anatomy: How does each component progressively improve retrieval?

## Research & References
1. **Babenko et al., 2014** — "Neural Codes for Image Retrieval" — PCA whitening on neural descriptors. Informed our t-SNE/silhouette analysis of embedding structure.
2. **Radford et al., 2021** — CLIP paper. Understanding how vision-language pretraining creates the embedding space we analyze.
3. **Liu et al., 2016** — DeepFashion. Intra-class variation (viewpoint, lighting) is the primary retrieval challenge, validated by our failure taxonomy.

How research influenced today's experiments: The failure taxonomy was designed around the insight from Liu et al. that intra-class variation dominates fashion retrieval difficulty. The silhouette analysis follows embedding quality evaluation practices from the retrieval literature.

## Dataset
| Metric | Value |
|--------|-------|
| Total products | 300 |
| Gallery size | 300 |
| Query set | 1,027 queries |
| Categories | 9 |
| Primary metric | Recall@1 (R@1) |

## Experiments

### Experiment 6.1: Per-Query Feature Attribution
**Hypothesis:** CLIP handles most queries; color and category filter rescue specific subsets.
**Method:** Run 4 system variants (CLIP-only, +color, +spatial, +cat filter) and track which component first achieves R@1=1 for each query.
**Result:**

| Rescuer | Queries | % |
|---------|---------|---|
| CLIP alone | 556 | 54.1% |
| Failed (all systems) | 278 | 27.1% |
| Color rescued | 123 | 12.0% |
| Cat filter rescued | 53 | 5.2% |
| Spatial rescued | 17 | 1.7% |

**Interpretation:** CLIP is the backbone (54.1% of queries need nothing else). Color is the most valuable supplement (12.0% of queries rescued). Category filter handles 5.2% of cross-category confusion cases. Spatial is nearly redundant (1.7%).

### Experiment 6.2: Similarity Score Decomposition
**Hypothesis:** CLIP similarity gap between success and failure will be larger than color or spatial.
**Method:** For each query, compute per-component cosine similarity to the correct gallery product, compare success vs failure distributions.
**Result:**

| Component | Success Mean | Failure Mean | Gap |
|-----------|-------------|-------------|-----|
| CLIP | 0.9189 | 0.8485 | +0.0704 |
| Color | 0.9810 | 0.9497 | +0.0313 |
| Spatial | 0.0596 | 0.0568 | +0.0029 |
| Combined | 0.9501 | 0.8994 | +0.0507 |

**HEADLINE:** CLIP is 2.25x more discriminative than color (gap 0.070 vs 0.031). When retrieval fails, the primary cause is CLIP's inability to distinguish visually similar products within the same category. Color helps but can't compensate for low CLIP similarity.

### Experiment 6.3: Failure Mode Taxonomy
**Method:** For each of the 278 failures, compare per-component similarity margins between correct and retrieved product.
**Result:**

| Failure Mode | Count | % |
|-------------|-------|---|
| Mixed signals | 135 | 48.6% |
| Both CLIP and color wrong | 52 | 18.7% |
| Ambiguous (margins < 0.02) | 51 | 18.3% |
| CLIP wrong, color right | 25 | 9.0% |
| Color wrong, CLIP right | 15 | 5.4% |

Per-category failure rates:
| Category | Fail Rate | Queries |
|----------|-----------|---------|
| shorts | 47.5% | 158 |
| denim | 31.2% | 77 |
| sweatshirts | 30.7% | 127 |
| pants | 30.6% | 144 |
| tees | 25.8% | 244 |
| jackets | 20.3% | 79 |
| shirts | 9.9% | 121 |
| sweaters | 6.8% | 74 |
| suiting | 0.0% | 3 |

**Key finding:** Nearly half of failures (48.6%) show "mixed signals" where components disagree on which product to prefer. 18.7% are total failures where both CLIP and color prefer the wrong product. Shorts are hardest (47.5% fail rate) due to extreme visual similarity within the category.

### Experiment 6.4: Category Filter Impact
**Result:**

| Impact | Queries | % |
|--------|---------|---|
| Helped | 294 | 28.6% |
| Hurt | 0 | 0.0% |
| Neutral | 733 | 71.4% |

Median rank improvement when helped: 5 positions.

**HEADLINE:** Category filter NEVER hurts. Zero queries degraded. It's pure upside: +28.6% of queries improved by median 5 rank positions, with no downside risk. This is the strongest architectural finding of the project.

### Experiment 6.5: Embedding Space Structure
**Method:** t-SNE visualization + silhouette scores on CLIP vs combined embeddings.
**Result:**

| Space | Silhouette |
|-------|-----------|
| CLIP only | 0.0035 |
| Combined | -0.0044 |
| Delta | -0.0079 |

**Interpretation:** Both silhouette scores are near zero, meaning categories barely cluster in the embedding space. This is expected: fashion items across categories share structural features (sleeves, collars, hemlines). The near-zero silhouette validates why category filtering is so impactful: the embedding space doesn't naturally separate categories, so explicit filtering is needed.

### Experiment 6.6: Rank Improvement Anatomy
**Method:** Track mean rank of correct product under each system variant.
**Result:**

| System | Mean Rank | R@1 |
|--------|-----------|-----|
| CLIP only | 8.79 | 0.5531 |
| + Color | 6.09 | 0.6456 |
| + Spatial | 6.00 | 0.6602 |
| + Cat Filter | 2.93 | 0.7293 |

**Key finding:** Each component progressively reduces mean rank. Color provides the biggest per-component lift (8.79 → 6.09, -2.70 ranks). Category filter provides the biggest R@1 lift (0.6602 → 0.7293, +6.9pp) by collapsing the search space.

## Head-to-Head Comparison (All Phases)
| Rank | System | R@1 | Production-valid? |
|------|--------|-----|-------------------|
| 1 | P5M: Text rerank (Mark) | 0.9065 | No (needs query text) |
| 2 | P5A: Text→Text oracle | 0.8199 | No (needs query text) |
| 3 | **P5A: Visual + cat filter (champion)** | **0.7293** | **Yes** |
| 4 | P4M: Per-cat alpha oracle | 0.6952 | Yes |
| 5 | P5A: Visual Optuna (no filter) | 0.6602 | Yes |
| 6 | P2: CLIP L/14+color rerank | 0.6417 | Yes |
| 7 | P2: CLIP L/14 bare | 0.5531 | Yes |
| 8 | P1: ResNet50 baseline | 0.3067 | Yes |

## Key Findings
1. **CLIP alone handles 54.1% of queries. Color rescues 12.0%. Spatial is nearly redundant (1.7%).** The system is fundamentally CLIP-first with targeted color supplements.
2. **CLIP is the most discriminative component (success-failure gap = +0.070).** Failures are primarily a CLIP limitation: it can't distinguish visually similar products within the same category.
3. **Category filter NEVER hurts (0 queries degraded).** Pure upside: 294 queries improved by median 5 rank positions. The strongest architectural finding.
4. **48.6% of failures show "mixed signals"** where CLIP and color disagree. These are genuine visual ambiguity cases that require fine-tuning to resolve.
5. **Shorts are 7x harder than sweaters** (47.5% vs 6.8% fail rate). Visual diversity within shorts is extreme.
6. **Categories don't cluster in the embedding space** (silhouette ~0). This validates why explicit category filtering adds +6.9pp R@1.

## Error Analysis
- 278 failures total (27.1% of queries)
- Failure rank median = 5 (correct product is usually close)
- 75th percentile rank = 10
- Shorts dominate failures (75/278 = 27.0% of all failures)

## Next Steps
- Phase 7: Testing + README + production pipeline + polish
- Consider CLIP fine-tuning on fashion data to address the 27.1% irreducible failures
- Vocabulary pruning experiment: remove spatial features entirely (saves 192D, loses only 1.7% of rescues)

## References Used Today
- [1] Babenko, A. et al. (2014). "Neural Codes for Image Retrieval with Aggregation." ECCV.
- [2] Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.
- [3] Liu, Z. et al. (2016). "DeepFashion: Powering Robust Clothes Recognition and Retrieval." CVPR.

## Code Changes
- `scripts/run_phase6_anthony.py` — 6 experiments: attribution, similarity decomposition, failure taxonomy, category filter impact, embedding structure, rank anatomy
- `scripts/build_phase6_notebook.py` — Notebook generator
- `notebooks/phase6_anthony_explainability.ipynb` — Executed research notebook (23 cells, 2 plots)
- `results/phase6_anthony_results.json` — All experiment results
- `results/phase6_anthony_explainability.png` — 9-panel visualization
- `results/phase6_anthony_sim_distributions.png` — 3-panel similarity distributions
- `reports/day6_phase6_anthony_report.md` — This report
