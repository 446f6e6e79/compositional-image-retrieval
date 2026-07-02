# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A university research project on **compositional image retrieval** over CelebA: given a source face image and a signed textual edit (`+attr` / `-attr`), retrieve gallery images that preserve the source while applying the requested attribute changes. The **CLIP ViT-B/32 encoder is kept frozen** throughout; methods differ only in how they fuse the source embedding with attribute text vectors.

Everything lives in a single notebook, **`compositional-image-retrieval.ipynb`**. There is no Python package, no `scripts/`, no test suite, and no build/lint step. Do not port code into a package unless explicitly asked. The notebook is the deliverable.

## Running it

The notebook is written for **Google Colab**. Cell 3 mounts Google Drive and paths are Colab-absolute:

- CelebA test split (19,962 images): `/content/datasets` (passed as `root`; the `CelebA` class appends `celeba/` itself).
- Benchmark annotations: `/content/drive/MyDrive/datasets/celeba_evaluation.json`.
- Cached CLIP embeddings + per-method checkpoints: `/content/drive/MyDrive/datasets/clip_cache/` (`embeddings.pt`, `cross_attn.pt`, ...).

Image encoding and triplet generation are expensive, so results are cached to Drive and reloaded by default (`get_encoded_dataset`, `load_or_generate_triplets`, the `if "..." not in globals()` guards). When editing a method, prefer reusing the cache; only force a recompute when the embeddings/text bank actually changed.

## Architecture

**One evaluation harness, many scorers.** Every method - baseline through trained - plugs into the same driver via a scorer-factory contract:

```
make_scorer(annotation) -> scorer(source_idx) -> (N,) gallery scores
```

`evaluate(annotations, make_scorer)` (cell 50) loops queries and source images, calls `retrieve_topk` (excludes the source), and scores against ground truth. Per-query expensive work (z-scoring, building the constraint vector) happens once inside `make_scorer`; the inner `scorer` stays cheap. `evaluate_and_average` returns both raw per-source metrics (for `mean_recall_at_10`, the hyperparameter-selection scalar) and per-query averages with 95% CIs (for plotting). Metrics are Recall@K / Precision@K for K in {1, 5, 10}.

**Embeddings are L2-normalized at extraction time**, so cosine similarity is a plain dot product everywhere downstream.

**Attribute alignment:** CelebA has 40 label columns, but torchvision's `attr_names` has 41 entries (one empty string). Always use `get_attributes` (drops the empty) to keep names aligned with label columns and with learned per-attribute embedding rows.

**Benchmark JSON shape:** each annotation has `query` (e.g. `"+glasses, -smile"`) and `ground_truth`, a dict mapping `source_image_idx` (string keys) to a list of valid target indices. Access via `get_text_query` / `get_source_image_idxs` / `get_target_indices`. A target is ground truth iff it strictly satisfies the query's +/- constraints AND is within Hamming distance 2 of the source's 40-bit attribute vector.

### Methods, in narrative order

Training-free methods come first, then training-based - this order is a hard requirement (see below). Within the training-free block, each method upgrades **exactly one thing** over its predecessor:

1. **Baseline** (`baseline_scorer`) - signed latent arithmetic: `fused = img_emb + Σ(+attr) - Σ(-attr)`, score = cosine. Exposes source leakage and embedding-space negation problems.
2. **Source-Attribute Matching** (`attribute_matching_scorer`) - upgrades the *fusion mechanism*. Builds an `(N, n_attrs)` attribute-logit matrix, z-scores each column over the gallery (`zscore_columns`, since CLIP cosines have wildly different per-attribute means), then scores with `w_query * hard-constraint + w_attr * proximity-penalty + w_visual * visual-term`. Weights are grid-tuned once and then frozen.
3. **Prompt Ensembling** - upgrades only the *text bank*, keeping the same scorer and frozen weights: replaces bare-name prompts with an article-free adaptation of CLIP's ImageNet prompt-template ensemble.
4. **Cross-Attention Fusion** (`CrossAttentionFusion`, cell 86) - the trained method. The source image is a single query token attending (pre-norm Transformer-decoder layers) over sign-tagged condition vectors built from the frozen text bank. A **sign-aware FiLM layer** turns `+`/`-` into distinct per-dimension `(gamma, beta)` modulations; a **gated-residual head** (`out = v_ref + sigmoid(gate) * delta`) preserves identity by default and allows genuine subtraction. Trained with label-free triplet supervision and an InfoNCE objective; triplet labels follow `desired_target_labels` and the Hamming/constraint rule above.

CoOp and sparse-autoencoder concept editing were implemented and discarded; they may still appear as cells.

## Narrative-order constraint (important)

This is a report, and the ablation story is part of it. When adding or moving a method:

- Keep **all training-free methods before all training-based methods**.
- Each training-free step must read as a *single upgrade* over the one before it (fusion, then bank, etc.); do not reorder so a later method changes two things at once.
- Keep cross-block dependencies one-directional: training-based cells may consume training-free outputs (e.g. `Z_ENS`, `E_POS`/`E_NEG`, `get_attribute_name_embeddings()`), never the reverse.
- The author edits cells between sessions - **re-read a cell before patching it** rather than assuming its current contents.

## Conventions

- Functions carry full Google-style docstrings with Args/Returns; match that style.
- CLIP access goes through the lazily-cached `get_CLIP_model()` / `encode_text(s)` helpers - don't reinstantiate the model.
- `figures/` holds the report diagrams as `.drawio` sources plus exported `.svg`; rendered `.png`/`.pdf` are gitignored.
