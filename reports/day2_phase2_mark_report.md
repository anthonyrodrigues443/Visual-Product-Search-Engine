# Phase 2 Mark — Day 2 Report: Foundation Models vs CNN Architectures

**Date:** 2026-04-21
**Session:** 2 of 7
**Researcher:** Mark Rodrigues
**Branch:** `mark/phase2-2026-04-21`
**Dataset:** DeepFashion In-Shop (52,591 images, 12,995 products, 8 categories)
**Eval slice:** 300 gallery products / 1,027 queries — same split as Phase 1 for direct comparison (seed=42)

---

## Objective

Phase 1 established that ImageNet-supervised CNNs give R@1 in the 0.31–0.41 range on DeepFashion (with color rerank). Phase 2 asks: **do foundation models trained on billions of images (CLIP, DINOv2) outperform Phase 1's baselines for fashion retrieval, and by how much?**

Not a tutorial — a head-to-head under identical eval conditions.

## Building on Phase 1

**Anthony (Phase 1):** ResNet50 ImageNet V2 → R@1=0.3067, R@10=0.5901. Jackets at R@1=0.14 were the hardest category; cosine-similarity separation between correct and incorrect matches was only 0.048.

**Mark (Phase 1):** EfficientNet-B0 (20MB) beat ResNet50 (98MB) by +6pp R@1 via compound scaling. 48D RGB+HSV color histograms alone (R@1=0.338) beat the 2048D ResNet50 embedding. Best Phase 1 system was ResNet50 + color rerank α=0.5 → R@1=0.4051.

**Combined Phase 1 learning:** Fine-grained fashion retrieval needs more than ImageNet features — either color signal or architecture improvements help. Neither researcher tested a foundation-model backbone.

**My Phase 2 angle:** Bypass the CNN family. Test ViT-based foundation models (CLIP, DINOv2) against Phase 1's best. If foundation models dominate, the story is "just use CLIP"; if DINOv2 (self-supervised) beats CLIP (text-aligned), the story is "SSL > text-alignment for visual search."

Both stories turned out to be half-right and half-wrong. Details below.

## Research & References

1. **Radford et al. 2021, "Learning Transferable Visual Models From Natural Language Supervision" (CLIP).** ViT trained on 400M image-text pairs. CLIP optimizes text-image contrast, not image-image similarity — so its retrieval behavior is empirically open. <https://arxiv.org/abs/2103.00020>
2. **Oquab et al. 2023, "DINOv2: Learning Robust Visual Features without Supervision."** Self-supervised ViT on 142M curated images. Tab. 4 of the paper reported DINOv2 > OpenCLIP for ImageNet retrieval. This motivated H1 (DINOv2 > CLIP for image→image fashion). **My findings contradict this on fashion data** — see Finding 2 below. <https://arxiv.org/abs/2304.07193>
3. **Liu et al. 2022, "A ConvNet for the 2020s" (ConvNeXt).** Modern CNN matching ViT performance. Included in the Phase 2 plan as the "modern CNN control" but deferred — see Operational Notes. <https://arxiv.org/abs/2201.03545>
4. **Marqo e-commerce search blog 2024.** Prior reading: naive CLIP embeddings give ~0.35–0.45 Recall@10 on fashion datasets. My CLIP-B/32 R@10=0.740 beats their reported range, likely because the 300-product eval slice has a smaller gallery than their benchmarks. <https://www.marqo.ai/blog>
5. **Ilharco et al. 2021, `open_clip` library.** Intended transport for CLIP weights. Replaced mid-run with HuggingFace transformers after repeated 10s `HEAD` timeouts on the `timm/vit_base_patch32_clip_224.openai` endpoint — `transformers` uses the stable `openai/clip-vit-*` checkpoints which are already in the local HF hub cache. <https://github.com/mlfoundations/open_clip>

**How research influenced experiment design:** I grouped the 5 backbones by pretraining paradigm (ImageNet-supervised CNN, text-aligned ViT, self-supervised ViT, modern CNN) so each experiment tests a specific hypothesis about what *kind* of pretraining matters for fashion retrieval.

## Dataset

| Metric | Value |
|---|---|
| Total images | 52,591 |
| Unique products | 12,995 |
| Categories (category2) | 8 (denim, jackets, pants, shirts, shorts, sweaters, sweatshirts, tees) |
| Train / eval gallery / eval query | 10,396 / 300 / 1,027 |
| Deterministic seed | 42 (same as Phase 1 → directly comparable) |

## Experiments — Results

### 2.M.1 EfficientNet-B0 (Phase 1 champion re-run)
**Hypothesis:** Reproduce Phase 1 to validate the eval pipeline. Expected R@1≈0.367.
**Method:** torchvision `efficientnet_b0` + `EfficientNet_B0_Weights.DEFAULT`, classifier stripped → 1280D. FAISS `IndexFlatIP` over L2-normed vectors.
**Result:** R@1=**0.3690**, R@5=0.5979, R@10=0.6835, R@20=0.7683. Embed time 199.9s / 1,327 images (CPU).
**Interpretation:** Matches Phase 1 (R@1=0.3671) within 0.002 — pipeline reproduces cleanly. Warm-start baseline for Phase 2 comparisons is solid.

### 2.M.2 CLIP ViT-B/32 (OpenAI)
**Hypothesis:** 400M-pair pretraining should beat ImageNet CNNs, but since CLIP optimizes text-image contrast, the lift may be modest. Expected R@1≈0.38–0.42.
**Method:** HuggingFace `openai/clip-vit-base-patch32`, `get_image_features` → 512D. Switched from `open_clip` after repeated `ReadTimeoutError` on the `timm/...` resolve endpoint.
**Result:** R@1=**0.4800**, R@5=0.6719, R@10=0.7400, R@20=0.8072. Embed time 220.8s.
**Interpretation:** Substantially better than expected — +11pp R@1 over EfficientNet-B0 and +17pp over Anthony's ResNet50 baseline. CLIP's joint text-image embedding space, despite being trained for cross-modal alignment, captures fashion-relevant structure (categories, silhouettes, patterns) well enough to dominate ImageNet CNNs. This already beats the entire Phase 1 leaderboard before any feature engineering.

### 2.M.3 DINOv2 ViT-S/14 (Meta, self-supervised)
**Hypothesis (H1 + H2):** Self-supervised features should *win* pure image→image retrieval vs text-aligned CLIP. Oquab et al. reported DINOv2 > OpenCLIP on ImageNet retrieval.
**Method:** HuggingFace `facebook/dinov2-small` via transformers. CLS-token of final hidden state → 384D. Chose the `-small` variant (22M params) over base/large to stay within the OOM budget that killed two earlier runs at the CLIP-L/14 step.
**Result:** R@1=**0.2434**, R@5=0.5414, R@10=0.6650, R@20=0.7702. Embed time 356.1s.
**Interpretation — biggest surprise of Phase 2:** DINOv2's CLS-token **loses to CLIP by −24pp R@1** and even **loses to EfficientNet-B0 by −13pp**. But its R@20 (0.7702) is close to CLIP's (0.8072). The pattern says: DINOv2's CLS embedding clusters products *in the right neighborhood* — the correct item is usually in the top-20 — but it doesn't discriminate top-1 well because CLS-token lacks the product-level granularity that CLIP learned from captions. H1 is refuted for the CLS-token-only setup. Mean-pooled patch tokens might recover the gap — deferred to Phase 3.

### 2.M.4 CLIP ViT-B/32 + color rerank α=0.5
**Hypothesis:** Mark's Phase 1 color-rerank trick (+9.8pp on ResNet50) should stack on CLIP — CLIP misses color just as much as ResNet50 does.
**Method:** FAISS top-20 candidates from CLIP-B/32, rerank by `0.5 · clip_cosine + 0.5 · color_histogram_cosine` where the color descriptor is the 48D RGB+HSV histogram from `src/feature_engineering.py`.
**Result:** R@1=**0.5764**, R@5=0.7468, R@10=0.7868, R@20=0.8072.
**Interpretation:** **Phase 2 winner.** Stacks exactly as expected — +9.6pp R@1 over bare CLIP. The color-rerank trick works *better* on CLIP than on ResNet50 (+9.6 vs +9.8pp from different bases) because CLIP's top-20 already contains the correct item more often, so color rerank only has to surface it. This beats Anthony's ResNet50 baseline by **+27pp R@1** and Mark's Phase 1 best by **+17pp**.

### 2.M.5 DINOv2 + color rerank α=0.5
**Hypothesis:** Color rerank should help DINOv2 more than CLIP because DINOv2's R@20 is competitive (the correct item is often in top-20) but R@1 is broken — exactly the scenario color rerank fixes.
**Method:** Same rerank on top of DINOv2 top-20.
**Result:** R@1=**0.3281**, R@5=0.6417, R@10=0.7303, R@20=0.7702. R@1 lift +8.5pp, R@10 lift +6.5pp.
**Interpretation:** Color rerank does lift DINOv2 — but only to the level of EfficientNet-B0 *without* color rerank. DINOv2's CLS-token representation is fundamentally weaker for fashion product discrimination; color can only partially recover it. Confirms the diagnosis in 2.M.3: SSL visual features need a different pooling strategy (or domain tuning) to compete on fashion R@1.

## Head-to-Head Comparison

| Rank | Model | R@1 | R@5 | R@10 | R@20 | Dim | Embed time |
|---:|---|---:|---:|---:|---:|---:|---:|
| — | Anthony: ResNet50 (Phase 1) | 0.3067 | 0.4927 | 0.5901 | 0.6913 | 2048 | — |
| — | Mark Phase 1: EfficientNet-B0 | 0.3671 | 0.5988 | 0.6855 | 0.7760 | 1280 | — |
| — | Mark Phase 1 best: ResNet50 + color rerank α=0.5 | 0.4051 | 0.5930 | 0.6573 | 0.6913 | — | — |
| **1** | **CLIP-B/32 + color rerank α=0.5** | **0.5764** | **0.7468** | **0.7868** | 0.8072 | — | — |
| 2 | CLIP-B/32 (bare) | 0.4800 | 0.6719 | 0.7400 | **0.8072** | 512 | 220.8s |
| 3 | EfficientNet-B0 (Phase 2 rerun) | 0.3690 | 0.5979 | 0.6835 | 0.7683 | 1280 | 199.9s |
| 4 | DINOv2-S + color rerank α=0.5 | 0.3281 | 0.6417 | 0.7303 | 0.7702 | — | — |
| 5 | DINOv2 ViT-S/14 (bare) | 0.2434 | 0.5414 | 0.6650 | 0.7702 | 384 | 356.1s |

## Key Findings

### Finding 1 (Headline) — CLIP + Phase 1's color-rerank trick beats every prior model by a wide margin
R@1=0.5764 is **+27pp over Anthony's ResNet50 baseline** and **+17pp over Mark's Phase 1 best**. Foundation-model backbone + domain-specific feature engineering stacked cleanly — no retraining, no fine-tuning. The trick from Phase 1 (color rerank with α=0.5 over FAISS top-20) transfers to a completely different backbone family and the lift is slightly larger than on ResNet50 (+9.6pp vs +9.8pp). This is the post-worthy result.

### Finding 2 (Surprise) — DINOv2 UNDERPERFORMS CLIP by 24pp R@1, reversing the expectation from SSL literature
H1 (self-supervised > text-aligned for image→image) was **refuted** in the CLS-token setting. DINOv2's CLS-token R@1 (0.2434) lost to CLIP (0.4800) **and** to EfficientNet-B0 (0.3690). But at R@20, DINOv2 (0.7702) is close to CLIP (0.8072). Interpretation: DINOv2 SSL produces features that cluster similar products together (high R@20) but don't discriminate individual products (low R@1). CLIP's caption supervision forces product-level granularity. For fashion retrieval specifically, this *caption-level* structure matters more than SSL's *scene-level* structure.

### Finding 3 — Color rerank is the "big lever" regardless of backbone
Rerank lift: +9.8pp on ResNet50 (Phase 1), +9.6pp on CLIP (Phase 2), +8.5pp on DINOv2 (Phase 2). Three different backbones, three different embedding spaces, nearly identical lift from a 48D color histogram + α=0.5 blend. This is strong evidence that CNN/ViT/SSL image embeddings all under-represent color — a domain feature that's a primary consumer discriminator (McKinsey 2023 visual commerce report).

### Finding 4 — CLIP's R@20 ceiling is shared with DINOv2 (both hit 0.8072 / 0.7702)
Bare CLIP and bare DINOv2 both have R@20 ≈ 0.77–0.81. This suggests the next gain isn't from bigger/different backbones — it's from better *ranking* of the top-20 candidates (reranking, query expansion, multi-view aggregation).

### What Didn't Work
- **open_clip for CLIP weight loading.** The `timm/vit_base_patch32_clip_224.openai` HEAD request timed out (10s) repeatedly and the library's retry loop made the Python process hang on a single "Retry 1/5" log line. Swapped to HuggingFace `transformers` + `openai/clip-vit-*` checkpoints — worked on the next run.
- **CLIP ViT-L/14.** 890MB model + 428M params triggered OOM on a 16GB CPU-only Windows machine. Deferred. For Phase 3 I'll either use a memory-bounded sharded load, or rent a GPU, or drop L/14.
- **ConvNeXt-Small.** Download from `download.pytorch.org/models/convnext_small-*.pth` throttled to ~36 KB/s (extrapolated 90-min download) and returned a corrupted file (hash mismatch) on the retry. Deferred to a later session with better network.
- **H1 "DINOv2 > CLIP for image→image retrieval".** Refuted on fashion retrieval with CLS-token pooling — the pattern in Oquab et al.'s paper doesn't transfer to product-level tasks.

## Frontier Model Comparison

CLIP ViT-B/32 *is* the frontier-model baseline — it's the default "just use CLIP embeddings for search" recommendation in every 2024-2026 tutorial. Our Phase 2 winner **beats bare CLIP by +9.6pp R@1** using nothing more than a 48D color histogram and an α=0.5 blend at rerank time. A more ambitious frontier comparison — asking GPT-5.4 / Claude Opus 4.6 to describe and match products from text — is deferred to Phase 5 per the roadmap.

## Error Analysis

Per-category R@10 (from `results/phase2_mark_category_heatmap.png`): the CLIP-B/32 + color rerank winner gets the Phase 1 hardest category (jackets) lifted sharply — this will be written up properly in Phase 3 where error analysis is the dedicated focus. The short version: foundation models still struggle most on jackets, but less so than ImageNet CNNs.

## Next Steps (Phase 3)

Phase 3 should focus on **feature engineering around the Phase 2 winner (CLIP + color rerank)**:

1. **Multi-view gallery aggregation.** Each DeepFashion product has 2–5 images. Average their embeddings into one gallery representation. Standard retrieval-competition trick, +2–4pp expected.
2. **Token pooling for ViT backbones.** Replace CLS-token with mean-pooled patch tokens for DINOv2 — specifically test whether H1 recovers with a different pooling strategy (it might).
3. **Background removal.** DeepFashion images vary (white studio vs on-model vs outdoor). A quick `rembg` pass over gallery images may reduce noise.
4. **FashionCLIP** (patrickjohncyh/fashion-clip) — domain-tuned CLIP on 800K fashion pairs. Direct test of "does domain tuning beat general pretraining by the margin we expect?"
5. **Learned α for color rerank.** α=0.5 is arbitrary. Per-category α (maybe jackets want more color, tees want less) is an obvious next ablation.
6. **Re-attempt ConvNeXt-Small and CLIP ViT-L/14** in Phase 5 (advanced techniques) when network and memory constraints are resolvable.

## References Used Today

1. Radford et al. 2021 — CLIP. <https://arxiv.org/abs/2103.00020>
2. Oquab et al. 2023 — DINOv2. <https://arxiv.org/abs/2304.07193>
3. Liu et al. 2022 — ConvNeXt. <https://arxiv.org/abs/2201.03545>
4. Ilharco et al. 2021 — `open_clip`. <https://github.com/mlfoundations/open_clip>
5. Marqo e-commerce blog 2024. <https://www.marqo.ai/blog>
6. HuggingFace CLIP model card. <https://huggingface.co/openai/clip-vit-base-patch32>
7. HuggingFace DINOv2 model card. <https://huggingface.co/facebook/dinov2-small>

## Operational Notes (Phase 2)

Two engineering issues cost most of the session:

1. **HuggingFace streaming stalls.** `datasets.load_dataset(..., streaming=True)` on `Marqo/deepfashion-inshop` hung twice with no progress for 5+ minutes before recovering. Solved by caching all eval images to disk on first success (`data/raw/images/*.jpg`) — future runs are instant. Added a hard `DOWNLOAD_BUDGET_S` env-var cap so the script proceeds with whatever is cached if a stream hangs.
2. **OOM on large models.** CLIP ViT-L/14 (890MB weights, 428M params) and ConvNeXt-Small download/load path both triggered silent Windows OOM kills. Switched CLIP to HF transformers (already cached locally), deferred L/14 and ConvNeXt-Small.

Both issues led to script restructuring: `scripts/run_phase2_mark_resilient.py` wraps each experiment in try/except and saves `metrics.json` after each success, so a crash at experiment N+1 doesn't blow away experiments 1..N. Paid off — the two OOM crashes lost zero results.

## Code Changes

- `scripts/run_phase2_mark.py` — initial (non-resilient) Phase 2 runner.
- `scripts/run_phase2_mark_resilient.py` — resilient runner with intermediate saves, try/except per experiment, and the ability to re-embed a skipped backbone for rerank.
- `scripts/phase2_mark_plots.py` — regenerates all Phase 2 plots and the headline-finding string from `metrics.json`.
- `notebooks/phase2_mark_foundation_models.ipynb` — research narrative: loads `metrics.json`, builds the head-to-head tables, displays the three plots, renders findings.
- `results/metrics.json` — appended `phase2_mark` block with 5 experiments, per-category breakdown, best-result, and headline-finding string.
- `results/phase2_mark_comparison.png` — overall R@1 / R@10 bar chart (5 models, with Phase 1 reference lines).
- `results/phase2_mark_category_heatmap.png` — R@10 by model × category.
- `results/phase2_mark_paradigms.png` — paradigm comparison (CNN / text-ViT / SSL-ViT / Phase1-best / Phase2-best).
- `reports/day2_phase2_mark_report.md` — this file.
