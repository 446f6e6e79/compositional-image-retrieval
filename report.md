# Compositional Image Retrieval

The project studies **compositional image retrieval** on CelebA: given a *source image* and a signed textual edit (`+attr` / `−attr`), retrieve gallery images that preserve the source while applying the requested attribute changes. Throughout, the **CLIP ViT-B/32 encoder** [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020) is kept **frozen**, and every method is evaluated on a common benchmark with Recall@K and Precision@K for K ∈ {1, 5, 10}, the source image excluded. 

The task follows the CLIP-based compositional retrieval setting adopted in **CLAY** [(Lim et al., 2026)](https://arxiv.org/abs/2604.11539) and is motivated by recent studies on the compositional structure of vision-language embeddings [(Berasi et al., 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Berasi_Not_Only_Text_Exploring_Compositionality_of_Visual_Representations_in_Vision-Language_CVPR_2025_paper.html).

## Contributions

In this project, we investigate how to overcome the **static multi-condition fusion** of **CLAY** [(Lim et al., 2026)](https://arxiv.org/abs/2604.11539). CLAY performs efficient, training-free conditional retrieval by decoupling textual conditioning from visual feature extraction. For a given condition, it builds a textual subspace (via an SVD of LLM-generated prompt embeddings on the VLM's hyperspherical manifold) and retrieves by projecting the frozen visual features onto it, ensuring that database features are never re-encoded when the condition changes. Multiple conditions are merged by stacking their prompt embeddings into a single matrix and reducing it to one static projection subspace, weighting every condition uniformly. Crucially, CLAY models only *positive, focus-on* conditions: it has no notion of **signed** (additive/subtractive) attributes and no per-query mechanism to resolve multiple, let alone competing, conditions. Our work targets exactly this gap, learning a query-dependent fusion of signed `+`/`−` conditions.

As a first step, we establish a **zero-shot baseline** using the frozen CLIP ViT-B/32 encoder [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020). While straightforward, this approach exposes deep structural limitations inherent to raw embedding-space arithmetic, like the tendency of the source image to leak unwanted traits into the query, and the inability of vector subtraction to capture true semantic negation.

Motivated by these observations, we first replace direct embedding arithmetic with **Source-Attribute Matching**. Instead of editing the source representation itself, the method explicitly compares candidate images against the source on every attribute independently. Retrieval is performed in a calibrated per-attribute similarity space obtained through gallery-wise z-score normalization, combining hard enforcement of queried attributes with identity-preserving penalties on the remaining ones.

Building on this formulation, we introduce **Prompt Ensembling**, which leaves the scoring function unchanged while replacing single attribute names with an article-free adaptation of CLIP's ImageNet prompt ensemble. This improves the robustness of textual attribute representations without modifying the retrieval pipeline.

We also explored two alternative directions inspired by recent literature but ultimately discarded them after empirical evaluation: learnable prompt optimization through **CoOp** [(Zhou et al., 2022)](https://arxiv.org/abs/2109.01134), and concept editing via **Sparse Autoencoders** [(Gao et al., 2024)](https://arxiv.org/abs/2406.04093). Neither approach provided a satisfactory solution to the multi-condition fusion problem posed by the assignment.

Finally, we propose **Cross-Attention Fusion**, our main contribution. Instead of statically combining condition embeddings before retrieval, the model learns a dynamic interaction between the reference image and the sequence of signed textual conditions through cross-attention [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). 

To better distinguish positive and negative edits, we introduce a **sign-aware FiLM layer** [(Perez et al., 2018)](https://arxiv.org/abs/1709.07871), which learns independent affine modulations for additive and subtractive constraints before attention. 
Distinctively, the signed conditions then **ground themselves on the source's own visual tokens** - CLIP's global **CLS** summary of the whole image together with its 49 spatial patch tokens, all from the same frozen forward pass - and co-adapt to one another, so a localized edit can attach to the relevant region of the source rather than only shifting the pooled embedding globally.

A **gated residual head** then predicts a signed retrieval update while preserving the original source identity. Unlike CLAY's fixed conditioning strategy, our architecture learns instance-specific weighting of multiple conditions, directly addressing the dynamic multimodal fusion objective of the assignment.

---

## Data Loading and Exploration

In this section, we load and explore the dataset to understand its structure and main properties.

The dataset used is **CelebA** [(Liu et al., 2015)](https://arxiv.org/abs/1411.7766). It is a large-scale face dataset containing over 200,000 celebrity images annotated with **40 binary facial attributes**.

In this work, we use a subset of **19,962 samples**. Each sample consists of:
- a **face image** of size 178 × 218 pixels,
- a corresponding **40-dimensional attribute vector**, describing visual characteristics such as *smiling*, *eyeglasses*, *male*, *young*, etc.

CelebA is widely used for facial attribute recognition and image editing tasks due to its diversity in pose, lighting, and background conditions.

---

## Offline Feature Extraction
In this step, we use **CLIP** (specifically the ViT-B/32 variant) as a **frozen** feature extractor to convert our images into vector representations [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020).

Because these image embeddings never change, we precompute them offline. This approach offers two major benefits:

* **Zero training overhead:** We do not backpropagate through the heavy vision transformer during training.
* **Larger batch sizes:** Freeing up GPU memory allows us to use significantly larger batch sizes when training our downstream retrieval layers.

Ultimately, this step compresses every image in our dataset into a fixed vector of size 512, which acts as its static visual fingerprint.

We normalize all embeddings to unit vectors at extraction time. Because of this, calculating the similarity between a query $\mathbf{q}$ and a gallery vector $\mathbf{g}$ simplifies to a basic dot product:

$$\text{similarity}(\mathbf{q}, \mathbf{g}) = \mathbf{q} \cdot \mathbf{g}$$

Since the dot product of unit vectors is mathematically identical to cosine similarity, our evaluation metric remains unchanged. However, this allows us to scale retrieval efficiently: instead of looping through image pairs one by one, we can search the entire gallery simultaneously using a single matrix multiplication. Finally, we filter out the source image itself from the top results.

---

## Embedding Analysis: Class-Image Similarity

Before building our retrieval model, we evaluate how effectively CLIP isolates the 40 CelebA attributes. We construct a $40 \times 40$ cosine similarity heatmap between attribute text prompts and a curated set of image embeddings using the following steps:

* **Targeted Sampling:** For each attribute, we select a single "pure" image where that specific trait is active and the number of co-occurring labels is minimized. This isolates the target concept and reduces visual noise from overlapping CelebA annotations.
* **Matrix Construction:** We compute the cosine similarities between all 40 text prompts and the 40 isolated image embeddings.

If CLIP is properly aligned with these facial concepts, the diagonal of the heatmap should dominate each row. Strong off-diagonal values immediately expose semantic overlap or confusion between related attributes, such as `Wavy_Hair` versus `Straight_Hair`.

#### Cosine Heatmap Analysis

The resulting heatmap reveals that the diagonal does not dominate. Instead, the matrix is driven by strong row and column biases: certain text prompts score highly against almost all images, while specific image columns light up across completely unrelated prompts.

Furthermore, all similarity values sit in a narrow, compressed band ($\sim 0.13$ to $0.27$), meaning raw cosine scores carry very little discriminative signal. These findings indicate that raw CLIP embeddings capture only coarse facial semantics and fail to isolate fine-grained attributes.

---

## Metrics

We evaluate every method with two standard top-$K$ retrieval metrics, computed per *(query, source image)* pair at $K \in \{1, 5, 10\}$. Let $\mathcal{R}_K$ be the ordered set of top-$K$ retrieved gallery images (the source image itself excluded) and $\mathcal{G}$ the ground-truth set of valid retrievals for that query.

- **Recall@K (hit rate).** Whether *at least one* valid image appears in the top $K$:
$$\text{Recall@}K = \mathbb{1}\big[\,|\mathcal{R}_K \cap \mathcal{G}| > 0\,\big] \in \{0, 1\}.$$

- **Precision@K.** The fraction of the top $K$ retrievals that are valid:
$$\text{Precision@}K = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{K}.$$

Both metrics are first averaged over the source images of each query (reported with a 95% confidence interval), and the **mean Recall@10** across all *(query, source)* pairs is the single headline scalar we use to compare methods and tune hyperparameters.

---

## Evaluation Protocol
To assess the performance of our retrieval system, we utilize a standardized benchmark of queries stored in a JSON file. Each entry in the dataset follows this structure:

* **`query`**: A string representing the textual modification (e.g., `"+glasses, -smiling"`).
* **`ground_truth`**: A dictionary where:
    * **Keys** are the indices of the **source images** used as the starting point.
    * **Values** are lists of indices for images considered valid retrievals for that specific source.

### Example Structure
```json
{
    "query": "+glasses, -smiling",
    "ground_truth": {
        "0": [1, 2, 3],
        "4": [5, 6, 7]
    }
}
```
In this example, image 0 serves as a source image (e.g., a smiling person without glasses). The system is expected to retrieve images 1, 2, or 3, which represent the "target" state (a non-smiling person with glasses), which should be visually similar to the source image but with the specified modifications applied.

### Formal validity rule

Fix a source image $s$ with binary attribute vector $\mathbf{b}_s \in \{0,1\}^{40}$ and a signed query $q=(q^+,q^-)$, where $q^+$ / $q^-$ are the attributes to add / remove. The **ideal target attribute vector** $\mathbf{b}^\star$ is $\mathbf{b}_s$ with the queried bits set to their requested values:
$$\mathbf{b}^\star_j=\begin{cases}1 & j\in q^+\\ 0 & j\in q^-\\ (\mathbf{b}_s)_j & \text{otherwise.}\end{cases}$$
An image $x$ is a **valid retrieval** for $(s,q)$ iff it satisfies every signed constraint **and** stays within a Hamming budget of the ideal attribute vector:
$$\mathcal{G}(s,q)=\Big\{\,x:\ (\mathbf{b}_x)_j=1\ \forall j\in q^+,\ \ (\mathbf{b}_x)_j=0\ \forall j\in q^-,\ \ \lVert\mathbf{b}_x-\mathbf{b}^\star\rVert_1\le 2\,\Big\}.$$
This set $\mathcal{G}$ is exactly the ground-truth set used by $\text{Recall@}K$ and $\text{Precision@}K$ above. The Hamming-$\le 2$ budget keeps a valid target visually close to the source: only the requested edits, plus at most two incidental attribute changes.

### Evaluation Function

We evaluate the retrieval performance of each fusion mechanism on the benchmark dataset, comparing it against the baseline method.

We compute the recall and precision metrics for each source image in the query for `"K = {1, 5, 10}"`.
Then we average the result across all source images and keep track on each query separately.

---

## Baseline Method
To establish a baseline for our retrieval system, we evaluate a **zero-shot, training-free approach** that relies exclusively on CLIP embeddings and cosine similarity.

The baseline uses simple latent space arithmetic by combining the attribute and image embeddings, without any learning or explicit alignment.
The query is decomposed into signed attribute terms: starting from the source image embedding, each `+` attribute embedding is added and each `−` attribute embedding is subtracted, and the resulting vector is used to find the nearest neighbours in the dataset.

### Scorer

Let $\mathbf{v}_s \in \mathbb{R}^D$ represent the raw CLIP visual embedding of the source image, and let $\hat{\mathbf{e}}_s = \frac{\mathbf{v}_s}{\|\mathbf{v}_s\|_2}$ be its corresponding unit-norm vector. For each text attribute $j$, $\mathbf{t}_j \in \mathbb{R}^D$ denotes the raw CLIP text embedding generated from the bare-name attribute prompt.

Let $q^+$ be the set of attributes to be added, and $q^-$ be the set of attributes to be removed. The unnormalized composite query vector $\mathbf{f} \in \mathbb{R}^D$ is constructed by shifting the source embedding along the text vector directions:

$$\mathbf{f} = \mathbf{v}_s + \sum_{j \in q^+} \mathbf{t}_j - \sum_{j \in q^-} \mathbf{t}_j$$

To evaluate and rank candidate images from the gallery, we compute the cosine similarity between the composite query and each gallery embedding. Let $\hat{\mathbf{e}}_x$ represent the pre-normalized, unit-norm CLIP embedding of a gallery image $x$ (where $\|\hat{\mathbf{e}}_x\|_2 = 1$). The final retrieval score for a given candidate $x$ is defined as the inner product of $\hat{\mathbf{e}}_x$ and the unit-normalized query vector:

$$\text{score}(x) = \hat{\mathbf{e}}_x^{\top} \left( \frac{\mathbf{f}}{\|\mathbf{f}\|_2} \right)$$

Gallery images are then sorted in descending order based on this score, directly optimizing the retrieval ranking in the shared latent space.

## Source-Attribute Matching (Training-Free)

The baseline model composes raw embeddings by summing text and image vectors directly in CLIP space. This approach introduces two critical structural problems:

1. **Source Leakage:** The raw image embedding implicitly injects *every* attribute of the source person—including those the query explicitly requests to change. This biases the retrieval heavily toward exact look-alikes of the source.
2. **Embedding-Space Negation:** Subtracting an attribute embedding (i.e., $- \mathbf{t}_j$) does not project into a region that represents the true linguistic complement of that concept in CLIP space.

To overcome these limitations, **Source-Attribute Matching (SAM)** utilizes the same core components (text and image cosine similarities) but fundamentally recasts the composition inside a **per-attribute similarity space**. This design is mathematically aligned with the ground-truth evaluation rules of the CelebA benchmark, where a valid target must satisfy the query's signed constraints while remaining within a **Hamming distance of $\le 2$** from the source image's 40-bit attribute vector.

Instead of manipulating an opaque, high-dimensional visual vector, SAM explicitly breaks down image composition into localized attribute agreements:

* **Per-Attribute Logits:** We map every gallery image against an explicit text bank consisting of one naturalized prompt per attribute. This yields a raw similarity matrix where each entry represents the cosine similarity between a specific image and a specific facial trait.
* **Statistical Calibration (Z-Scoring):** Raw CLIP cosine similarities suffer from severe alignment discrepancies; certain prominent attributes exhibit uniformly high means across the dataset, while subtle ones remain low. To prevent high-variance or high-mean attributes from dominating the retrieval process, we apply **z-score normalization** independently to each attribute column. This statistical calibration centers each attribute's distribution around a mean of 0 and scales it to a standard deviation of 1, placing all 40 semantic concepts onto an identical, comparable scale.
* **Explicit Decomposed Scoring:** The source image's own row within this calibrated matrix acts as its semantic profile. Candidates are evaluated using a multi-term objective balancing query adherence, background attribute preservation, and global visual identity.

By shifting the operations to a calibrated similarity space, negation is handled by simply inverting a normalized score rather than subtracting vectors in a latent space, completely resolving both baseline limitations.

### Scorer

Let $L \in \mathbb{R}^{N \times 40}$ be the raw attribute logit matrix, where $N$ is the total number of gallery images. The element $L_{xj}$ represents the single-prompt cosine similarity between the unit-norm embedding of gallery image $x$ ($\hat{\mathbf{e}}_x$) and the text embedding of attribute $j$ ($\mathbf{t}_j$):

$$L_{xj} = \hat{\mathbf{e}}_x^{\top}\mathbf{t}_j$$

We apply column-wise z-scoring across the entire gallery to construct the calibrated similarity matrix $Z \in \mathbb{R}^{N \times 40}$. For an image $x$ and attribute $j$, the calibrated score $Z_{xj}$ is defined as:

$$Z_{xj} = \frac{L_{xj} - \mu_j}{\sigma_j}$$

where the dataset-wide mean $\mu_j$ and standard deviation $\sigma_j$ for attribute $j$ are computed as:

$$\mu_j = \frac{1}{N}\sum_{x=1}^{N} L_{xj}, \qquad \sigma_j = \sqrt{\frac{1}{N}\sum_{x=1}^{N} (L_{xj} - \mu_j)^2}$$

Given a source image $s$ and a signed query $q$, the final retrieval score for any candidate gallery image $x$ is formalized as:

$$\text{score}(x) = \underbrace{w_q \sum_{j \in q} s_j Z_{xj}}_{\text{Queried Constraints}} \;-\; \underbrace{w_r \sum_{j \notin q} \big(Z_{xj} - Z_{sj}\big)^2}_{\text{Attribute Proximity}} \;+\; \underbrace{w_v \hat{\mathbf{e}}_x^{\top}\hat{\mathbf{e}}_s}_{\text{Visual Similarity}}$$

The operational mechanics of these three components are defined as follows:

* **Queried Constraints:** Evaluates candidate compliance for attributes targeted by the edit. The sign modifier $s_j$ is set to $+1$ if the attribute is requested ($j \in q^+$) and $-1$ if it is to be removed ($j \in q^-$).
* **Attribute Proximity:** Acts as a continuous surrogate for the benchmark's strict Hamming-$\le 2$ budget. It penalizes any semantic drift on the unqueried background attributes, forcing the retriever to preserve the source identity's secondary traits.
* **Visual Similarity:** Computes the raw visual cosine similarity between the candidate ($\hat{\mathbf{e}}_x$) and source ($\hat{\mathbf{e}}_s$) embeddings to retain fine-grained spatial characteristics like pose, lighting, and ambient context.

The structural hyperparameters $(w_q, w_r, w_v)$ are static scalar weights optimized via grid search on a validation split to balance the three retrieval priorities.

The three weights `w_query`, `w_attr`, `w_visual` are training-free hyperparameters tuned with a
deliberately small grid so sensitivity stays visible. They are tuned **once**, using the simple
bare-name attribute bank (Source-Attribute Matching); Prompt Ensembling reuses the same `SAM_WEIGHTS` with its
improved bank, making the two methods directly comparable.

---

## Prompt Ensembling (training-free)

Source-Attribute Matching fixed the **fusion mechanism**; its remaining weakness is the **text embeddings themselves**: a single bare prompt per attribute is a noisy estimate of the concept. Prompt Ensembling keeps the scoring layer **and the very same weights** frozen, and upgrades only the per-attribute bank:

- **3a. Expanded template bank.** Each attribute's positive embedding `e⁺` is the ensemble of several person-referring phrases run through an article-free adaptation of CLIP's ImageNet prompt set (55 unique templates) plus a handful of portrait-specific templates.
- **3b. Linguistic negatives.** A separate `e⁻` is built from real negative descriptions ("a person without {attr}", "a clean-shaven person", ...). The attribute logit becomes the **pos-minus-neg margin** `cos(x, e⁺) − cos(x, e⁻)`, a stronger, noise-cancelled signal than a single positive cosine, and a real linguistic complement instead of a sign flip.

Because the fusion mechanism and its weights are inherited unchanged from Source-Attribute Matching, any improvement here is attributable to the embedding bank alone.


**Formal definition.** For attribute $j$, let $P^+_j$ / $P^-_j$ be its positive / negative phrase banks and $\mathcal{T}$ the template set. Each ensembled embedding L2-normalises every phrase$\times$template encoding before mean-pooling, then re-normalises:
$$\mathbf{e}^{\pm}_j=\operatorname{normalise}\!\Big(\tfrac{1}{|P^{\pm}_j|\,|\mathcal{T}|}\!\!\sum_{p\in P^{\pm}_j}\sum_{\rho\in\mathcal{T}}\operatorname{normalise}\big(\text{CLIP}_{\text{text}}(\rho(p))\big)\Big).$$
The per-attribute logit fed to the **unchanged** attribute-matching scorer becomes a pos-minus-neg cosine margin:
$$L_{xj}=\hat{\mathbf{e}}_x^{\top}\mathbf{e}^{+}_j-\hat{\mathbf{e}}_x^{\top}\mathbf{e}^{-}_j=\cos(\hat{\mathbf{e}}_x,\mathbf{e}^{+}_j)-\cos(\hat{\mathbf{e}}_x,\mathbf{e}^{-}_j),$$
a noise-cancelled signal whose negative pole is a real linguistic complement rather than a sign flip. Everything downstream ($Z$, the three-term score, the weights) is identical to Source-Attribute Matching, so any gain is attributable to the bank alone.

### Scorer

Only the bank changes: Prompt Ensembling builds an **ensembled** pos/neg bank and feeds it to **Source-Attribute Matching's `attribute_matching_scorer`, unchanged**. Any gain over Source-Attribute Matching is therefore attributable to the embedding bank alone.

#### CLIP ImageNet prompt templates

The per-attribute banks are ensembled over [CLIP's official ImageNet prompt templates](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb) (the canonical 80-template zero-shot set), plus a few portrait-specific templates for CelebA faces.

We adapt the templates to **full noun phrases**: each `{phrase}` already carries its own article (e.g. *"a person with glasses"*), so we drop the template's leading article to avoid *"a **a** person with glasses"*. Removing that article makes the `a {}` and `the {}` variants identical, collapsing the official 80 templates to **55 unique** ones, the set we actually ensemble over.

---

## Other experiments

Before settling on cross-attention, two other *learned* methods were built and then dropped. Both keep CLIP frozen and learn a different piece of the pipeline, and both reuse the existing evaluation harness, so each slots in as a drop-in replacement for either the embedding bank or the edit rule.

**CoOp (Context Optimization, [Zhou et al. 2022](https://arxiv.org/abs/2109.01134)).** Instead of writing the prompt prefix by hand, CoOp learns it: the handcrafted words in front of each attribute are replaced by $M=16$ continuous context vectors that live in CLIP's word-embedding space and are shared across all 40 attributes. We frame CelebA as multi-label classification, building a positive prompt ("a person with {attribute}") and a negative prompt ("a person without {attribute}") for every attribute, and train the shared context (about $M\times 512\approx 8\text{k}$ parameters) with binary cross-entropy on the per-attribute margin $\cos(\text{img},e_+)-\cos(\text{img},e_-)$. The learned positive/negative text bank then drops into the *same* profile-matching scorer the training-free methods use, with the same weights. It trained cleanly and slightly edged the hand-written banks, but gave no decisive gain: the bottleneck is the fixed additive fusion, not the prompt wording.

**TopK-SAE concept editing ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)).** This route learns the *representation* rather than the prompt. A TopK sparse autoencoder is trained, with no labels and no text, to reconstruct the cached CLIP image embeddings through an overcomplete dictionary of $H=4096$ unit-norm atoms, each a direction in CLIP space; sparsity pushes those atoms toward near-monosemantic concepts. A textual condition is grounded zero-shot onto its top few atoms (cosine affinity, mean-centred to drop the shared text direction), and the source is edited along only those atoms, $\mathbf{v}_{\text{target}}=\mathbf{v}_s+\sum_c\sigma_c\,\gamma_c\,\hat{\mathbf{u}}_c$, leaving every other atom untouched so identity is preserved for free. Reconstruction was faithful enough for retrieval (the residual is inert), but the edit direction was the failure point: a query attribute seldom grounds onto a single clean, monosemantic atom, so the added term behaved as noise and could not reliably realise the attribute.

Both methods leave the image-condition *interaction* hand-designed. Cross-attention learns that interaction instead.

---

## Training-Based Method: Cross-Attention Fusion

Every method so far combines the visual reference and its textual conditions with a *fixed* rule (latent arithmetic, profile matching, or a learned prompt). 

**Cross-Attention Fusion** instead *learns* the combination: a small trained module reads the reference image together with its `±attribute` conditions and produces a single composite query that keeps the reference's identity while applying the requested edits. The reference image attends over its conditions and decides, per image, how strongly each one should count.

CLIP stays frozen; only the lightweight fusion module is trained, with a contrastive objective. The conditions reuse the frozen bare-name attribute text bank shared with the training-free methods.

End to end, for the query `+Eyeglasses & -Smiling`:

1. **Encode inputs.** Frozen CLIP encodes the reference image and each condition phrase into 512-d vectors, each condition tagged with its sign (`+` should be present, `-` should be absent).
2. **Sign-aware modulation.** Each sign reshapes its attribute vector so that `+attr` and `-attr` become genuinely distinct directions, giving the model a handle on negation and composed queries.
3. **Build the condition sequence.** The modulated conditions are stacked into a variable-length sequence (1 to 3 per query), with padding masked so attention never reads empty slots.
4. **Cross-attention.** The image is the query and the conditions are the keys and values: a stack of Transformer-decoder layers refines how this particular image weighs and combines its conditions.
5. **Gated-residual fusion.** The module returns the reference plus a *gated, signed* edit, so identity is anchored by default while the edit can either add or remove an attribute.
6. **Retrieve.** The fused query is L2-normalised and the frozen gallery is ranked by cosine similarity, returning the top-K nearest images.

The *Architecture* below frames why a Transformer fits this fusion, and the detailed cell that follows derives the four learned stages.

### Architecture

**How the Transformer maps to our problem.** The sequence the attention runs over is neither image patches nor text sub-words: it is the query's own short list of `±attribute` edits (one to three of them), so the sequence length is simply *how many things you asked to change*. The **source image is the single query token** ($Q$), while the **sign-modulated condition vectors are the keys and values** ($K=V$). Read semantically, the image asks *"given who I am, how strongly should I weigh each requested edit?"* This is **cross-attention** - the image reads the conditions - and because there is exactly one query token, the output is just a content-based weighted average of those conditions, with the weights computed from the image itself. That per-image, per-condition weighting is exactly what the fixed latent arithmetic of the training-free methods cannot express. Before that read, the conditions first attend over a *second* sequence - the source's own visual tokens, the CLS token plus its 49 spatial patches - so each edit can ground on the relevant region of this particular source before the image weighs it.

At a high level (diagram below), frozen CLIP encodes the source image into a unit-norm embedding $\mathbf{v}_{\text{ref}}$ **and** its sequence of visual tokens $[\text{CLS};\,49\text{ patches}]$, and each condition into the bare-name text vector $\mathbf{t}_a$ of its attribute; the trained module $\Phi_\theta$ fuses them into a single query $\mathbf{q}$, which ranks the frozen gallery by cosine similarity. It does so in four learned stages - **sign-aware FiLM**, **patch grounding**, **stacked cross-attention**, and a **gated residual** - derived in turn below.

![High-level architecture of the cross-attention fusion module](figures/architecture.svg)

The diagram below expands the module layer by layer. Frozen CLIP encodes the reference into a unit-norm vector $\mathbf{v}_{\text{ref}}\in\mathbb{R}^{D}$ ($D=512$) and exposes its visual-token sequence $V_{\text{raw}}=[\text{CLS};\,\text{patch}_1,\dots,\text{patch}_{49}]\in\mathbb{R}^{50\times d_{\text{clip}}}$ ($d_{\text{clip}}=768$ for ViT-B/32). A query carries up to $T=3$ conditions, each a pair $(a_k,s_k)$ with $a_k$ a CelebA attribute index and $s_k\in\{+1,-1,0\}$ its sign ($0$ marks a padding slot). Each $a_k$ selects its frozen bare-name text vector $\mathbf{t}_{a_k}\in\mathbb{R}^{D}$, so the conditions enter as a $(T,D)$ block that the four trained stages reshape, ground, read, and fuse onto $\mathbf{v}_{\text{ref}}$.

**1. Sign-aware FiLM conditioning** [(Perez et al., 2018)](https://arxiv.org/abs/1709.07871). Each sign selects one of two learned embeddings in a table $E_{\text{sign}}\in\mathbb{R}^{2\times D}$, which a single affine layer turns into a per-dimension scale and shift applied to the attribute's text vector:
$$\mathbf{z}_{s_k}=E_{\text{sign}}\big[\mathbb{1}[s_k<0]\big],\qquad (\boldsymbol{\gamma}_k,\boldsymbol{\beta}_k)=W_{\text{FiLM}}\,\mathbf{z}_{s_k}+\mathbf{b}_{\text{FiLM}},\qquad \mathbf{c}_k=(\mathbf{1}+\boldsymbol{\gamma}_k)\odot\mathbf{t}_{a_k}+\boldsymbol{\beta}_k.$$
$W_{\text{FiLM}}$ and $\mathbf{b}_{\text{FiLM}}$ are zero-initialised, so training starts from the raw CLIP semantics ($\mathbf{c}_k=\mathbf{t}_{a_k}$) and only gradually learns how a sign should bend each coordinate. Because the modulation is multiplicative and attribute-specific, the $+$ and $-$ versions of one attribute become distinct per-dimension directions rather than mirror images - a plain additive offset would translate every attribute identically. The modulated conditions form the memory $C=[\mathbf{c}_1,\dots,\mathbf{c}_T]\in\mathbb{R}^{T\times D}$, with padding slots recorded in a boolean mask.

**2. Patch grounding.** Before the image weighs the conditions, the conditions read the source. The visual tokens are projected into the fusion space, $V=V_{\text{raw}}W_{\text{vis}}^{\top}\in\mathbb{R}^{50\times D}$, plus a two-row learned *type* embedding distinguishing the CLS token from the 49 patches (CLIP's positional embeddings already encode patch location). The conditions $C$ are then the target of a pre-norm Transformer-decoder layer with memory $V$:
$$C\leftarrow C+\operatorname{SelfAttn}(\operatorname{LN}C),\quad C\leftarrow C+\operatorname{CrossAttn}(\operatorname{LN}C,\,V),\quad C\leftarrow C+\operatorname{FFN}(\operatorname{LN}C),$$
so in one block the conditions co-adapt to one another and ground spatially on the source - `+Eyeglasses` can read the eye patches of *this* face rather than a generic direction. The grounded $C$ replaces the raw conditions below.

**3. Stacked cross-attention** [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). The reference enters as a single query token $\mathbf{x}_0=\mathbf{v}_{\text{ref}}$ and is refined by $L=2$ pre-norm decoder layers reading from $C$, each with self-attention, cross-attention, and a feed-forward sublayer (dropout $p=0.1$):
$$\mathbf{x}\leftarrow\mathbf{x}+\operatorname{SelfAttn}(\operatorname{LN}\mathbf{x}),\quad \mathbf{x}\leftarrow\mathbf{x}+\operatorname{CrossAttn}(\operatorname{LN}\mathbf{x},\,C),\quad \mathbf{x}\leftarrow\mathbf{x}+\operatorname{FFN}(\operatorname{LN}\mathbf{x}).$$
The cross-attention sublayer, with $h=4$ heads of dimension $d_h=D/h=128$, is where the image reads the conditions:
$$\operatorname{Attn}(\mathbf{q},C)=\operatorname{softmax}\!\Big(\frac{(\mathbf{q}W_Q)(CW_K)^{\top}}{\sqrt{d_h}}+M\Big)(CW_V),$$
where the additive mask $M$ sends padded positions to $-\infty$ so they receive zero weight. With exactly one query token the output is a content-based weighted average of the unmasked conditions, weighted by the image itself - the per-image, per-condition weighting fixed arithmetic cannot express. The feed-forward sublayer ($\mathbb{R}^{D}\to\mathbb{R}^{2D}\to\mathbb{R}^{D}$ with GELU) adds per-token capacity. Stacking $L=2$ layers lets the image read, update, and read again, yielding the attended vector $\mathbf{a}=\operatorname{Decoder}(\mathbf{v}_{\text{ref}};C)\in\mathbb{R}^{D}$.

**4. Gated-residual fusion.** The attended vector is concatenated with the reference, $\mathbf{u}=[\mathbf{v}_{\text{ref}};\mathbf{a}]\in\mathbb{R}^{2D}$, and consumed by two heads: an edit head (a GELU MLP) producing a signed displacement $\boldsymbol{\delta}$, and a gate head (affine + sigmoid) producing a per-dimension gate $\mathbf{g}\in(0,1)^{D}$:
$$\boldsymbol{\delta}=W_2^{\delta}\,\operatorname{GELU}(W_1^{\delta}\mathbf{u}),\quad \mathbf{g}=\sigma(W^{g}\mathbf{u}),\quad \mathbf{q}=\Phi_\theta\big(\mathbf{v}_{\text{ref}},\{(a_k,s_k)\}\big)=\frac{\mathbf{v}_{\text{ref}}+\mathbf{g}\odot\boldsymbol{\delta}}{\lVert\mathbf{v}_{\text{ref}}+\mathbf{g}\odot\boldsymbol{\delta}\rVert_2}.$$
The gate's bias is initialised to $2$ ($\mathbf{g}\approx\sigma(2)\approx0.88$), so it starts mostly open and gradient flows into the edit head from the first step; once trained, the per-dimension gate localises an edit to the few coordinates an attribute should move and leaves the rest of $\mathbf{v}_{\text{ref}}$ intact. Two properties matter for the task: the edit is *additive* onto $\mathbf{v}_{\text{ref}}$, so the module learns a correction rather than reconstructing the embedding and identity stays anchored; and $\boldsymbol{\delta}$ is *signed*, free to point against an attribute direction so the model can genuinely subtract a feature - a combiner that mixes image and text through a softmax forms only convex combinations of its inputs and cannot remove an attribute this way [(Baldrati et al., 2022)](https://dblp.org/rec/conf/cvpr/BaldratiBUB22a.html). The final L2-normalisation returns $\mathbf{q}$ to the unit sphere so retrieval by cosine similarity reduces to a dot product against the unit-norm gallery (see the Scorer below).

The module is deliberately lightweight: on top of an otherwise frozen CLIP it adds only the two-row sign table, one FiLM layer, the visual-token projection and type table, the grounding decoder layer, $L=2$ decoder layers, and the two fusion heads - a small fraction of the backbone, so training is fast.

![Detailed layer-by-layer architecture of the cross-attention fusion module](figures/architecture_details.svg)

---

### Training

The fusion module is trained with synthetic, label-free supervision built from CelebA's attribute annotations, so no manual labelling is needed. At each step we take a reference image, flip a few of its attributes to form a signed query, and fetch from the training split a real image that matches the edited attribute profile to serve as the **positive target**; optionally we also mine a **hard negative** that looks right but violates one requested sign. A contrastive objective then pulls the fused query toward its target and pushes it away from the other images in the batch, including that hard negative. Only the fusion module is updated, while CLIP and the attribute text bank stay frozen. The diagram below traces one training triplet from the reference to the loss.

![Label-free triplet supervision and the InfoNCE objective](figures/training.svg)

The diagram above shows one triplet; this section spells out how the triplets are synthesised and how the objective is optimised.

**Triplet synthesis.** Each triplet starts from a random reference image $s$ with its 40-bit CelebA attribute vector $\mathbf{b}_s$. We sample 1 to 3 of its attributes and flip them into a signed query $q$ (a flipped-on attribute becomes a `+` term, a flipped-off attribute a `-` term), and build the ideal target attribute vector $\mathbf{b}^\star$ by copying $\mathbf{b}_s$ and applying exactly those flips. The **positive target** $t$ is drawn from the training split among real images that satisfy the query and lie within the benchmark's Hamming budget of the ideal profile, $\lVert \mathbf{b}_t - \mathbf{b}^\star \rVert_1 \le 2$. This is the same rule the benchmark uses to judge a correct retrieval, so the model is trained against the exact target definition it is later evaluated on. Since $\mathbf{b}_s$, the chosen flips, and the candidate labels are all that is required, the triplets $(s,q,t)$ are produced with no human annotation, purely from the attribute table. A large pool, on the order of $10^5$ training triplets plus a few thousand held out for validation, is generated once and cached, keyed to the synthesis settings, so that re-training under different model or optimiser hyperparameters reuses the same pool instead of regenerating it.

**Hard negatives.** When enabled, one **constraint-violating** distractor $h$ is mined per query: a real image that keeps the reference's other attributes but breaks exactly one requested sign (for the query $-\text{Smiling}$, a face that is otherwise valid yet still smiling). Such an image is close to the target in every respect except the single attribute the query cares about, so using it as a negative forces the model to key on the edited attribute rather than on overall resemblance to the source. This is the main defence against the failure mode where a combiner simply returns look-alikes of the reference. Queries for which no such image exists in the split fall back to using only the in-batch negatives for that row.

**InfoNCE objective** [(van den Oord et al., 2018)](https://arxiv.org/abs/1807.03748). For a batch of $B$ triplets, let $\mathbf{q}_i=\Phi_\theta(\mathbf{v}_{s_i},q_i)$ be the fused query, $\mathbf{t}_i$ the embedding of its positive target, $\mathbf{h}_i$ its optional hard negative, and $\tau$ CLIP's own (frozen) temperature. Every other target in the batch acts as an in-batch negative, and the per-row hard negative is appended as one extra negative, giving the cross-entropy
$$\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{t}_i)}{\displaystyle\sum_{j=1}^{B}\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{t}_j)\;+\;\mathbb{1}[h_i\ \text{exists}]\,\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{h}_i)}.$$
Minimising $\mathcal{L}$ raises the cosine similarity between each fused query and its true target while lowering it against the $B-1$ other targets and the hard negative. Because every embedding is unit-norm and the temperature matches CLIP's, the geometry the loss optimises is exactly the one the retrieval scorer uses at test time, so improvements on the objective translate directly into retrieval gains.

**Optimisation and model selection.** Only the fusion module $\Phi_\theta$ receives gradients; CLIP's image and text encoders and the attribute text bank are frozen, and the image features of the training split are pre-extracted once so each step runs only the small module rather than the backbone. We optimise with AdamW (learning rate $2\times10^{-4}$, weight decay $10^{-2}$) under a cosine-annealed learning-rate schedule over 20 epochs with batch size 512, relying on dropout inside the decoder and the fusion heads together with weight decay for regularisation. After every epoch we measure Recall@10 on the held-out triplets: each fused validation query is ranked against the whole training gallery with the source excluded, and a hit is counted when a returned image both satisfies the query and lies within the Hamming budget of the ideal target, exactly the benchmark rule. The checkpoint with the best validation Recall@10 is kept, and on later runs that cached checkpoint is reloaded so evaluation never requires re-training.

### Scorer

At evaluation the scorer builds **one** composite query embedding per source image and ranks the frozen gallery against it by cosine similarity.

1. Parse the query string (`+A & -B & …`) into attribute indices and signs.
2. Fuse the source embedding with its conditions through the trained module $\Phi_\theta$ (its output is already L2-normalised):

$$\mathbf{q} \;=\; \Phi_\theta\!\big(\mathbf{v}_{\text{ref}},\, \{(\mathbf{t}_a, s_a)\}\big), \qquad \lVert \mathbf{q} \rVert_2 = 1,$$

where $\mathbf{t}_a$ is the frozen CLIP text vector of attribute $a$ and $s_a \in \{+1, -1\}$ its sign.

3. Score every gallery image $\mathbf{g}_i$ (already unit-norm). Since both vectors are normalised, cosine similarity is a dot product:

$$\operatorname{score}(i) \;=\; \cos(\mathbf{g}_i, \mathbf{q}) \;=\; \mathbf{g}_i^{\top}\mathbf{q}.$$

4. Retrieve the top-$K$ most similar images, excluding the source itself:

$$\mathcal{R}_K \;=\; \operatorname{top\text{-}}K \,\{\, \mathbf{g}_i^{\top}\mathbf{q} \;:\; i \neq \text{ref} \,\}.$$

Scoring the whole gallery is a single matrix-vector product, giving the $(N,)$ similarity scores at once.

### Cross-Attention: Qualitative Inspection

To see *what the trained model does* and where it breaks, we inspect a **SUCCESS** and a **FAILURE** case for two query types the benchmark stresses: a single-attribute **negation** (e.g. `-Heavy Makeup`) and a **composed** multi-attribute query (e.g. `+Eyeglasses, -Smiling`). For each, we automatically pick, from that query's own benchmark sources, one source the model gets right (a ground-truth target in its top-k) and one it gets wrong (none in top-k); nothing is hardcoded.

For each `(source, query)` we read out:

- **Top-k retrieval under the edit**: the images the *fused* query pulls to the top (source excluded), each marked ✓/✗ for satisfying the requested attributes and tagged `GT` when it is a benchmark target. This shows directly whether the edit moved retrieval toward the request rather than toward look-alikes of the source.
- **Residual gate** $\sigma(g)\in[0,1]$ from the gated-residual head: its mean summarises overall edit strength, while a low mean with a few high dimensions signals a localised edit and a flat $\approx 0.5$ means the head barely moved off its initialisation.

The trained weights are reused exactly; nothing is re-trained.

### Limitations

Read against the task, the trained module has clear ceilings worth stating plainly.

- **The attention does relatively light work.** A query carries at most $T=3$ conditions, so the cross-attention arbitrates among a handful of vectors rather than modelling long-range structure over a long sequence. Most of what makes the method work is the **sign-aware FiLM** (which makes $+$ and $-$ genuinely distinct) and the **gated residual** (which anchors identity and permits subtraction); the attention mainly reweights. The Transformer is the right *frame*, but it is not where the heavy lifting happens.
- **The remaining capacity ceiling is the pooled gallery target, not the source.** The source side now enters as CLIP's full visual-token sequence (the CLS token plus 49 patches), so the conditions can ground on the region an edit should touch - "remove the glasses" can read the eye patches - rather than only shifting a pooled vector. The gallery *target* side, however, is still scored as a single pooled 512-d embedding (retrieval ranks `gallery_embeddings @ q`), so a localised edit must ultimately be matched against a holistic image vector. Lifting this would mean a patch-level gallery index, at a real cost in storage and retrieval time.
- **The text bank is frozen and non-compositional.** Conditions are bare-name CLIP text vectors, and CLIP text behaves like a bag of concepts. Interacting attributes (e.g. *Smiling* and *Mouth Slightly Open*) enter as independent conditions the module can only reweight and ground; it cannot learn their joint semantics.
- **Negation still rides on the CLIP geometry.** FiLM and the signed $\boldsymbol{\delta}$ approximate "absence of an attribute" as a direction in an embedding space that was never trained for negation - a learned workaround, not a true representation of *not*.
- **Identity is preserved at the cost of under-editing.** Because the output defaults to $\mathbf{v}_{\text{ref}}$, the path of least resistance is to leave the embedding nearly unchanged, so strongly requested edits can be damped: keeping the source is always the safe option for the contrastive loss.

---

## Final Comparison: all methods

All methods evaluated on the same benchmark JSON and the same precomputed image embeddings: the training-free series (baseline, source-attribute matching, prompt ensembling) followed by the training-based Cross-Attention Fusion.

---

## References

1. **CLIP** — A. Radford, J. W. Kim, C. Hallacy, et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
2. **Transformer** — A. Vaswani, N. Shazeer, N. Parmar, et al. *Attention Is All You Need.* NeurIPS 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
3. **FiLM** — E. Perez, F. Strub, H. de Vries, V. Dumoulin, A. Courville. *FiLM: Visual Reasoning with a General Conditioning Layer.* AAAI 2018. [arXiv:1709.07871](https://arxiv.org/abs/1709.07871)
4. **InfoNCE / CPC** — A. van den Oord, Y. Li, O. Vinyals. *Representation Learning with Contrastive Predictive Coding.* 2018. [arXiv:1807.03748](https://arxiv.org/abs/1807.03748)
5. **CelebA** — Z. Liu, P. Luo, X. Wang, X. Tang. *Deep Learning Face Attributes in the Wild.* ICCV 2015. [arXiv:1411.7766](https://arxiv.org/abs/1411.7766)
6. **TIRG** — N. Vo, L. Jiang, C. Sun, et al. *Composing Text and Image for Image Retrieval — An Empirical Odyssey.* CVPR 2019. [arXiv:1812.07119](https://arxiv.org/abs/1812.07119)
7. **Combiner (CLIP4Cir)** — A. Baldrati, M. Bertini, T. Uricchio, A. Del Bimbo. *Effective Conditioned and Composed Image Retrieval Combining CLIP-Based Features.* CVPR 2022. [dblp](https://dblp.org/rec/conf/cvpr/BaldratiBUB22a.html)
8. **Pic2Word** — K. Saito, K. Sohn, X. Zhang, et al. *Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval.* CVPR 2023. [arXiv:2302.03084](https://arxiv.org/abs/2302.03084)
9. **CoOp** — K. Zhou, J. Yang, C. C. Loy, Z. Liu. *Learning to Prompt for Vision-Language Models.* IJCV 2022. [arXiv:2109.01134](https://arxiv.org/abs/2109.01134)
10. **TopK-SAE** — L. Gao, T. Dupré la Tour, H. Tillman, et al. *Scaling and Evaluating Sparse Autoencoders.* 2024. [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
11. **CLIP prompt templates** — OpenAI. *Prompt Engineering for ImageNet* (notebook). [GitHub](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb)
