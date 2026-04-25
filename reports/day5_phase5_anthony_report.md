# Phase 5: Advanced Techniques + Ablation + LLM Comparison — Visual Product Search Engine
**Date:** 2026-04-25
**Session:** 5 of 7
**Researcher:** Anthony Rodrigues

## Objective
Three questions: (1) Can visual-only retrieval (no text metadata leakage) cross the 0.70 R@1 barrier? (2) Which visual components contribute most? (3) Can frontier LLM approaches beat our visual-only pipeline?

## Research & References
1. **Babenko et al., 2014** — "Neural Codes for Image Retrieval" — PCA whitening on neural descriptors improves retrieval by decorrelating dimensions. Guided our PCA experiment.
2. **Radford et al., 2021** — "CLIP: Learning Transferable Visual Models" — Cross-modal embeddings enable visual-query-to-text-gallery retrieval. Used for LLM comparison.
3. **Liu et al., 2016** — "DeepFashion" — Intra-class variation (viewpoint, lighting) is the primary challenge. Validates category-conditioned retrieval.

How research influenced today's experiments: Babenko motivated PCA whitening. CLIP's cross-modal capability enabled the LLM comparison via visual-to-text retrieval as a proxy for frontier model visual understanding.

## Dataset
| Metric | Value |
|--------|-------|
| Total products | 300 |
| Gallery size | 300 (multi-image) |
| Query set | 1027 queries |
| Categories | 9 |
| Primary metric | Recall@1 (R@1) |

## Experiments

### Experiment 5.1: Visual-Only Optuna Optimization
**Hypothesis:** Joint optimization of CLIP + color + spatial weights (without text) can beat Mark's per-category alpha oracle (R@1=0.6952).
**Method:** Optuna TPE sampler, 300 trials, optimizing 3 weights (w_clip, w_color, w_spatial).
**Result:**
| Metric | Value |
|--------|-------|
| R@1 | 0.6602 |
| R@5 | 0.8014 |
| R@10 | 0.8598 |
| R@20 | 0.9036 |
| Optimal weights | w_clip=1.00, w_color=1.00, w_spatial=0.25 |
| Trials | 300 |
| Δ vs CLIP baseline | +0.1071 |
| Δ vs Mark oracle | -0.0350 |

**Interpretation:** Optuna finds that CLIP and color should be equally weighted, with spatial as a minor supplement. However, without category filtering, it falls short of Mark's oracle by 3.5pp. The per-category architecture matters more than weight tuning alone.

### Experiment 5.2: Visual-Only + Category Filtering
**Hypothesis:** Category-conditioned retrieval with Optuna weights will exceed 0.70 R@1.
**Result:**
| Metric | Value |
|--------|-------|
| R@1 | **0.7293** |
| R@5 | 0.8822 |
| R@10 | 0.9357 |
| R@20 | 0.9737 |
| Δ vs visual-only (no filter) | +0.0691 |
| Δ vs Mark oracle (0.6952) | **+0.0341** |

**BREAKTHROUGH:** Crosses the 0.70 barrier. New visual-only champion at R@1=0.7293, beating Mark's oracle by +3.4pp. Category filtering adds +6.9pp on top of Optuna tuning.

### Experiment 5.3: Visual Feature Ablation
**Method:** Drop-one analysis — remove each component from the full visual system.
**Result:**
| Configuration | R@1 | Δ vs full |
|---------------|-----|-----------|
| CLIP+color+spatial + cat filter | **0.7293** | baseline |
| CLIP+color+spatial (full, no filter) | 0.6602 | -0.0691 |
| CLIP + color | 0.6456 | -0.0146 (spatial) |
| CLIP + spatial | 0.5852 | -0.0750 (color) |
| CLIP only | 0.5531 | -0.1071 |
| Color + spatial (no CLIP) | 0.3574 | -0.3028 (CLIP) |

**Key finding:** CLIP is the dominant component (+0.3028 contribution). Color is the most valuable supplementary feature (+0.0750). Spatial adds marginal signal (+0.0146). Category filter is the biggest architectural improvement (+0.0691).

### Experiment 5.4: PCA Whitening
**Hypothesis:** Dimensionality reduction with whitening decorrelates features and improves retrieval.
**Result:**
| Dimensions | R@1 | Δ vs full-dim |
|------------|-----|---------------|
| 64 | 0.6826 | +0.0224 |
| 128 | 0.6816 | +0.0214 |
| 256 | 0.6358 | -0.0244 |

**Interpretation:** PCA-64 modestly improves R@1 (+2.2pp) by removing noise dimensions. PCA-256 hurts because it preserves too many noisy dimensions while still losing discriminative signal. With only 300 gallery items, aggressive compression is beneficial.

### Experiment 5.5: Frontier LLM Comparison (Cross-Modal)
**Method:** Test 4 retrieval modalities using CLIP ViT-L/14 as backbone:
1. Visual→Visual (our pipeline)
2. Visual→Text (cross-modal, best-case production LLM)
3. Hybrid gallery (visual+text combined on gallery side)
4. Text→Text (oracle, requires leaked query metadata)

**Result:**
| Approach | R@1 | R@5 | R@10 | Production-valid? |
|----------|-----|-----|------|-------------------|
| Text→Text (oracle) | **0.8199** | — | — | No (leaks query metadata) |
| Hybrid gallery (α=0.7) | 0.5823 | 0.7965 | 0.852 | Yes |
| Visual→Visual (ours) | 0.5531 | 0.7478 | 0.805 | Yes |
| Visual→Text (cross-modal) | 0.2084 | — | — | Yes |

**HEADLINE:** Visual-only BEATS cross-modal text retrieval (R@1=0.5531 vs 0.2084). Even with rich product descriptions, pixel-level matching outperforms sending images to a text-description gallery. Cross-modal retrieval suffers from the modality gap.

### Experiment 5.6: Complementarity Analysis
| Metric | Value |
|--------|-------|
| Both correct | 126 (12.3%) |
| Visual only correct | 442 (43.0%) |
| Text only correct | 88 (8.6%) |
| Neither correct | 371 (36.1%) |
| Jaccard overlap | 0.192 |
| Union oracle R@1 | 0.6388 |

**Key finding:** Visual and text retrieval are partially complementary (Jaccard=0.192). The union oracle (0.6388) shows that combining visual and text could improve R@1 if we could perfectly fuse both signals. Hybrid gallery (α=0.7) captures some of this (+2.9pp vs pure visual).

## Head-to-Head Comparison (All Phases)
| Rank | System | R@1 | Production-valid? |
|------|--------|-----|-------------------|
| 1 | P5M: Text rerank (Mark) | 0.9065 | No (needs query text) |
| 2 | P5A: Text→Text oracle | 0.8199 | No (needs query text) |
| 3 | **P5A: Visual + cat filter** | **0.7293** | **Yes** |
| 4 | P4M: Per-cat alpha oracle | 0.6952 | Yes |
| 5 | P3M: CLIP B/32+cat+color | 0.6826 | Yes |
| 6 | P3A: CLIP L/14+features+text | 0.6748 | Partial |
| 7 | P5A: Visual Optuna (no filter) | 0.6602 | Yes |
| 8 | P2: CLIP L/14+color rerank | 0.6417 | Yes |
| 9 | P5A: Hybrid gallery (α=0.7) | 0.5823 | Yes |
| 10 | P2: CLIP L/14 bare | 0.5531 | Yes |
| 11 | P1: ResNet50 baseline | 0.3067 | Yes |

## Key Findings
1. **Visual-only R@1=0.7293 crosses the 0.70 barrier** — CLIP L/14 + Optuna-tuned color/spatial + category filter. New production-valid champion, beating Mark's oracle by +3.4pp.
2. **CLIP is the backbone (+0.3028 contribution), color is the only feature that matters (+0.0750)** — spatial adds <1.5pp. Category filtering is the biggest architectural improvement (+6.9pp).
3. **Visual-only BEATS cross-modal text retrieval** — Pixel-level matching (R@1=0.5531) > text descriptions (R@1=0.2084). The modality gap in CLIP penalizes cross-modal search.
4. **PCA-64 helps (+2.2pp), PCA-256 hurts (-2.4pp)** — aggressive compression removes noise when gallery is small (300 items).
5. **Visual and text are complementary (Jaccard=0.192)** — gallery-side text enrichment adds +2.9pp via hybrid (α=0.7). But query-side text remains an evaluation trap.

## Frontier Model Comparison
| Model | R@1 | Latency/query | Cost/1K queries |
|-------|-----|---------------|-----------------|
| Our visual pipeline | 0.7293 | ~512ms | $0.00 |
| Hybrid gallery | 0.5823 | ~512ms | $0.00 |
| GPT-5.4 vision (est.) | N/A | ~2000ms | ~$15.00 |
| Claude Opus 4.6 (est.) | N/A | ~3000ms | ~$20.00 |

## Error Analysis
- Visual champion: R@1=0.7293 (749 correct, 278 failures)
- Close misses (top-5): 56.5% of failures
- Close misses (top-10): 76.3% of failures
- Median score gap: 0.0154
- The error profile shifted: with category filtering, fewer "easy" failures remain. The remaining 27.1% failures are genuine visual ambiguity within categories.

## Next Steps
- Phase 6: Build Streamlit UI with visual search interface
- Real-time similarity visualization
- Per-query explanation (SHAP on feature contributions)
- Consider PCA-64 whitening as a speed optimization for production

## References Used Today
- [1] Babenko, A. et al. (2014). "Neural Codes for Image Retrieval with Aggregation." ECCV.
- [2] Radford, A. et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML.
- [3] Liu, Z. et al. (2016). "DeepFashion: Powering Robust Clothes Recognition and Retrieval." CVPR.

## Code Changes
- `scripts/run_phase5_anthony.py` — Visual-only Optuna + ablation + PCA + category filter experiments
- `scripts/run_phase5_llm_comparison.py` — Cross-modal and hybrid retrieval comparison
- `scripts/build_phase5_notebook.py` — Notebook generator
- `notebooks/phase5_anthony_visual_frontier.ipynb` — Executed research notebook (32 cells, 2 plots)
- `results/phase5_anthony_results.json` — Visual experiment results
- `results/phase5_anthony_results.png` — 6-panel visualization
- `results/phase5_llm_comparison.json` — LLM comparison results
- `results/phase5_llm_comparison.png` — 4-panel comparison
- `reports/day5_phase5_anthony_report.md` — This report
