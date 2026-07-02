# Compositional Image Retrieval

The project studies **compositional image retrieval** on CelebA: given a *source image* and a signed textual edit (`+attr` / `−attr`), the objective is to retrieve gallery images that preserve the source while applying the requested attribute changes.
This work adopts the CLIP-based compositional retrieval setting introduced in **CLAY** [(Lim et al., 2026)](https://arxiv.org/abs/2604.11539), utilizing a **frozen CLIP encoder** [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020).

## Contributions

In this project, we investigate how to overcome the **static multi-condition fusion** of **CLAY** [(Lim et al., 2026)](https://arxiv.org/abs/2604.11539). CLAY enables efficient, training-free retrieval by separating text conditioning from visual features. It achieves this by projecting frozen visual features onto a textual subspace constructed for each condition. Multiple conditions are then merged into a single static subspace that weights every condition uniformly. Crucially, CLAY only supports positive, additive conditions; it lacks a mechanism for signed (additive or subtractive) attributes and cannot resolve multiple or competing conditions for a single query.

We propose the following training-free methods for compositional image retrieval:

- **Baseline**: a zero-shot, training-free starting point using raw embedding-space arithmetic on the frozen CLIP ViT-B/32 encoder [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020).
- **Source-Attribute Matching**: upgrades the *fusion mechanism*, replacing embedding arithmetic with explicit per-attribute comparison against the source in a calibrated similarity space.
- **Prompt Ensembling**: upgrades only the *text bank*, leaving the scorer untouched, with an article-free adaptation of CLIP's ImageNet prompt ensemble.

Before settling on a trained model, we also built and discarded two *learned* alternatives: prompt optimization via **CoOp** [(Zhou et al., 2022)](https://arxiv.org/abs/2109.01134) and concept editing via **Sparse Autoencoders** [(Gao et al., 2024)](https://arxiv.org/abs/2406.04093) - neither of which showed improvement over the multi-condition fusion problem.

Our main contribution, **Cross-Attention Fusion**, addresses this by dynamically modeling the interaction between the reference image and its sequence of signed textual conditions via cross-attention [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). Rather than statically combining condition embeddings prior to retrieval, this approach computes image-conditioned weights for each attribute, moving past the uniform, static weighting used in CLAY.

---

## Data Loading and Exploration

The dataset is **CelebA** [(Liu et al., 2015)](https://arxiv.org/abs/1411.7766), a large-scale face dataset of over 200,000 celebrity images annotated with **40 binary facial attributes**. We use a subset of **19,962 samples**, each consisting of:
- a **face image** of size 178 × 218 pixels,
- a corresponding **40-dimensional attribute vector** describing traits such as *smiling*, *eyeglasses*, *male*, *young*.

---

## Offline Feature Extraction

We use **CLIP** (ViT-B/32) as a **frozen** feature extractor, encoding every image into a fixed 512-d vector [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020). Since these embeddings never change, we precompute them offline once.

All embeddings are L2-normalized to unit vectors at extraction time. Similarity between a query $\mathbf{q}$ and a gallery vector $\mathbf{g}$ is then a plain dot product:

$$\text{similarity}(\mathbf{q}, \mathbf{g}) = \mathbf{q} \cdot \mathbf{g},$$

Retrieval over the whole gallery is therefore a single matrix multiplication rather than a pairwise loop.

---

## Embedding Analysis: Class-Image Similarity

Before constructing the retrieval model, we first assess how well CLIP distinguishes the 40 CelebA attributes. For each attribute, we select a "pure" image: one that has the attribute active while minimizing co-occurring traits, breaking ties at random. We then compute a \(40 \times 40\) cosine similarity matrix between the image embeddings and the 40 attribute text prompts.

The diagonal is not dominant. Instead, the matrix is driven by row and column biases, with some prompts and images scoring consistently highly across unrelated pairs. Moreover, all cosine similarities fall within a narrow range (approximately \(0.13\)–\(0.27\)), providing little discriminative signal. Raw CLIP cosine similarity therefore captures broad facial semantics rather than the fine-grained attribute information required for reliable retrieval.

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
* **`ground_truth`**: A dictionary mapping source image indices to lists of valid target image indices.

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

In this example, image `0` is the source (e.g., a smiling person without glasses). The system must retrieve images `1`, `2`, or `3`, which represent the modified target state (not smiling, with glasses).

### Ground Truth Validity Criteria

An image is considered a valid retrieval for a given source and query if it meets two conditions:

1. **Strict Attribute Matching:** It must perfectly satisfy the explicit edits requested in the query (e.g., it *must* have glasses and *must not* be smiling).
2. **Visual Consistency:** To ensure the target image remains visually close to the source, we enforce a strict threshold allowing a maximum of **two incidental attribute changes** outside of the requested query.

---

## Baseline Method

To establish a baseline, we evaluate a **zero-shot, training-free approach** that relies on simple latent space arithmetic using pre-computed CLIP embeddings.

Starting from the source image embedding, we add the embeddings of the requested attributes (`+`) and subtract the embeddings of the removed attributes (`−`). The resulting vector is then used to find the nearest neighbors in the dataset via cosine similarity.

Let $\mathbf{v}_s$ be the raw CLIP visual embedding of the source image, and $\mathbf{t}_j$ be the raw CLIP text embedding for attribute $j$. Given the sets of attributes to add ($q^+$) and remove ($q^-$), the unnormalized composite query vector $\mathbf{f}$ is:

$$\mathbf{f} = \mathbf{v}_s + \sum_{j \in q^+} \mathbf{t}_j - \sum_{j \in q^-} \mathbf{t}_j$$

### Scorer

To rank candidate images from the gallery, we normalize the composite query vector and compute its cosine similarity (inner product) with each unit-normalized gallery embedding $\hat{\mathbf{e}}_x$:

$$\text{score}(x) = \hat{\mathbf{e}}_x^{\top} \left( \frac{\mathbf{f}}{\|\mathbf{f}\|_2} \right)$$

## Source-Attribute Matching (Training-Free)

The baseline approach introduces two critical structural problems:

1. **Source Leakage:** The raw image embedding implicitly contains every visual attribute of the original person, including those the query explicitly requests to change. As a result, the system is heavily biased toward retrieving exact look-alikes of the source image rather than successfully applying the modifications.
2. **Embedding-Space Negation:** Subtracting an attribute embedding does not project into a region that represents the true linguistic complement of that concept in CLIP space.

To overcome these limitations, **Source-Attribute Matching (SAM)** utilizes the same core components (text and image cosine similarities) but fundamentally recasts the composition inside a **per-attribute similarity space**:

* **Per-Attribute Logits:** Every gallery image is compared against a text bank of all attributes. This creates a matrix where each entry represents the direct cosine similarity between a specific image and a specific trait.
* **Statistical Calibration (Z-Scoring):** To prevent dominant traits from overpowering the representation, we **z-score normalize** each attribute column independently. This centers each trait's distribution around a mean of 0 and a standard deviation of 1, placing all concepts on a comparable scale.
* **Explicit Decomposed Scoring:** The source image's row in the calibrated matrix serves as its semantic profile. Candidates are then evaluated using a scoring objective that balances three goals: satisfying the explicit query, preserving unmentioned background traits, and maintaining global visual identity.

By shifting operations to this calibrated similarity space, negation is handled by simply inverting a normalized score rather than subtracting vectors in latent space, resolving both baseline limitations.

### Scorer

Given a source image $s$ and a signed query $q$, the retrieval score for any candidate gallery image $x$ is defined by a multi-part objective that balances query compliance, background preservation, and global visual identity:

$$\text{score}(x) = \underbrace{w_q \sum_{j \in q} s_j Z_{xj}}_{\text{Queried Constraints}} \;-\; \underbrace{w_r \sum_{j \notin q} \big(Z_{xj} - Z_{sj}\big)^2}_{\text{Attribute Proximity}} \;+\; \underbrace{w_v \hat{\mathbf{e}}_x^{\top}\hat{\mathbf{e}}_s}_{\text{Visual Similarity}}$$

The operational mechanics of these three components are defined as follows:

* **Queried Constraints ($w_q$):** This drives the requested edits. It rewards candidate images that match the new query; either by looking for high positive scores when adding a trait, or high negative scores when removing one. Because we are using calibrated trait scores, removing a feature is as simple as flipping a mathematical sign.
* **Attribute Proximity ($w_r$):** This acts as a protective shield for the rest of the person's face. It penalizes any changes to the background traits that the query didn't mention, ensuring that unedited features (like hair color, age, or expression) stay as close to the original source image as possible.
* **Visual Similarity ($w_v$):** This preserves the overall look and feel of the original image using raw CLIP embeddings. It captures fine-grained details that text descriptions miss entirely.

---

## Prompt Ensembling (Training-free)

While Source-Attribute Matching improves the fusion mechanism, a single text prompt per attribute remains a noisy estimate of a concept. **Prompt Ensembling** addresses this by upgrading the text bank.

Instead of relying on a single word, each trait is defined by combining a bank of descriptive phrases with dozens of prompt templates. We average these combinations to create two distinct, well-rounded CLIP profiles for every trait: a **positive profile** capturing its presence, and a **negative profile** capturing its explicit absence.
The benefits of this approach are twofold:

* **Noise Cancellation:** Averaging across dozens of prompt templates smooths out the linguistic quirks and inherent noise of individual prompts, establishing a stable baseline.
* **True Semantic Negation:** Rather than relying on simple arithmetic inversion, this approach uses a dedicated linguistic negative embedding.

### Scorer

To evaluate this method, only the embedding bank is modified, which it's passed directly to the Source-Attribute Matching framework.

#### CLIP ImageNet prompt templates

The per-attribute banks are ensembled over [CLIP's official ImageNet prompt templates](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb) (the canonical 80-template zero-shot set), plus a few portrait-specific templates for CelebA faces.


---

## Other experiments

Prior to adopting cross-attention, two alternative trained-based approaches were evaluated:

**CoOp (Context Optimization, [Zhou et al. 2022](https://arxiv.org/abs/2109.01134)):** Instead of utilizing handcrafted prompt prefixes, we employ CoOp to learn a set of $M=16$ continuous context vectors within CLIP’s word-embedding space, which are shared across all attributes. Once trained, this learned positive/negative text bank integrates directly into the same profile-matching scorer, utilizing identical weights.

While this optimization showed marginal improvements over prompt ensembling, it was ultimately excluded because the primary bottleneck resides in the fusion mechanism rather than prompt enrichment. Consequently, we chose to dedicate our investigation entirely to advancing architectural fusion strategies rather than tuning representation inputs.

**TopK-SAE Concept Editing ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)):** This approach explores a representation-level modification aimed at extending CLIP while strictly preserving its zero-shot capabilities. To achieve this, a **TopK sparse autoencoder** is trained in an unsupervised manner to reconstruct cached CLIP image embeddings using an overcomplete dictionary of $H=4096$ unit-norm atoms, each isolating a specific directional feature.

At retrieval time, textual conditions are grounded zero-shot onto their highest-affinity atoms via mean-centered cosine similarity. The source image embedding is subsequently modified by shifting it exclusively along these selected atomic vectors, while the source identity is preserved via the unmodified residual embedding:

$$\mathbf{v}_{\text{target}} = \mathbf{v}_s + \sum_c \sigma_c \, \gamma_c \, \hat{\mathbf{u}}_c$$

Where $\mathbf{v}_{\text{target}}$ is the edited query embedding, $\mathbf{v}_s$ is the source embedding, $\hat{\mathbf{u}}_c$ is the unit-norm dictionary atom for condition $c$, $\sigma_c \in \{+1, -1\}$ is the condition sign, and $\gamma_c \ge 0$ is the retrieval-time edit magnitude.

Despite providing high-fidelity reconstructions, this approach failed to deliver decisive gains over the baseline. Query attributes rarely align cleanly with isolated monosemantic atoms. As a result, editing the grounded atoms injects unexpected semantic noise rather than precisely altering the intended attribute.

---

## Cross-Attention Fusion (Training-based)

Every method so far combines the visual reference and its textual conditions with a fixed, query-agnostic rule. These hard-coded combinations suffer from three specific architectural limitations:

* **Embedding-space negation:** Subtracting a text vector in CLIP space merely creates a geometric mirror point rather than a true semantic opposite.
* **Ungrounded edit directions:** A generic attribute vector remains identical across all inputs, failing to adapt to the specific visual context of the reference image.
* **Source leakage:** The reference embedding entangles every attribute of the person, including the one being edited, so a fixed combination is pulled toward look-alikes of the source and dilutes the requested change instead of separating what to keep from what to modify.

**Cross-Attention Fusion** instead *learns* the combination with a small module $\Phi_\theta$ trained on top of frozen CLIP: the reference image attends over its signed conditions to weigh each edit *per image* rather than by a fixed rule.

### Architecture

To achieve this adaptive combination, the source image is first processed at two distinct granularity simultaneously. CLIP processes the input into a regular grid of non-overlapping **spatial patch tokens** $V_{\text{raw}}$, which act as localized descriptors of specific regions like an eye or the hairline, enabling targeted, position-aware editing. Alongside these localized patches, the encoder extracts a specialized **CLS token**, a single global vector $\mathbf{v}_{\text{ref}}$ that pools and summarizes the entire face. In parallel, each textual condition (without the sign) is encoded into its own attribute vector $\mathbf{t}_a$.

The fusion process begins by passing each attribute vector $\mathbf{t}_a$ through **sign-aware FiLM conditioning** [(Perez et al., 2018)](https://arxiv.org/abs/1709.07871), a lightweight affine layer that rewrites each condition based on its positive or negative sign so that adding and removing a feature become distinct learned directions $\mathbf{c}_k$ rather than simple geometric mirror images.

Next, these conditions $\mathbf{c}_k$ undergo **patch grounding**: acting as queries, they read the projected patch tokens $V$ through the cross-attention of a Transformer-decoder layer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), while self-attention lets the conditions co-adapt with one another. This anchors each requested edit directly onto the specific region of the reference face it should physically alter.

The core of the system relies on **stacked cross-attention**. The global image summary $\mathbf{v}_{\text{ref}}$ acts as the single query token, while the grounded, signed conditions $C$ serve as keys and values. In this setup, the image dynamically determines how strongly weight each requested edit based on its own identity. Stacking $L$ such decoder layers allows the image to iteratively read the conditions and update its representation, ultimately producing the attended vector $\mathbf{a}$.

Finally, **gated-residual fusion** [(Vo et al., 2019)](https://arxiv.org/abs/1812.07119) applies this attended vector $\mathbf{a}$ as a signed correction $\boldsymbol{\delta}$, gated per-dimension by $\mathbf{g}$, directly onto the global identity vector $\mathbf{v}_{\text{ref}}$. This mechanism preserves the source's original identity by default while allowing targeted features to be genuinely removed, ultimately producing a single unit-norm query $\mathbf{q}$ ready to rank the gallery by cosine similarity.

![High-level architecture of the cross-attention fusion module](figures/architecture.svg)

The following section provides a detailed breakdown of each individual component and the overall system architecture.

**Sign-aware FiLM conditioning** [(Perez et al., 2018)](https://arxiv.org/abs/1709.07871): This component processes an attribute vector $\mathbf{t}_{a_k}$ alongside its sign $s_k$. The sign indexes one of two learned vectors in a embedding table $E_{\text{sign}}$. A single affine layer then transforms this vector into a per-dimension scale $\boldsymbol{\gamma}_k$ and shift $\boldsymbol{\beta}_k$ to modulate the original attribute:

$$\mathbf{c}_k=(\mathbf{1}+\boldsymbol{\gamma}_k)\odot\mathbf{t}_{a_k}+\boldsymbol{\beta}_k$$

The underlying weights and biases ($W_{\text{FiLM}}$, $\mathbf{b}_{\text{FiLM}}$) are zero-initialized, meaning training safely starts from the raw CLIP vector ($\mathbf{c}_k=\mathbf{t}_{a_k}$) before gradually learning the modulation. Because this transformation is multiplicative and sign-specific, the positive and negative forms of an attribute become distinct, independent directions in the embedding space rather than mere mirror images. Stacked together, these modulated conditions form the system memory $C=[\mathbf{c}_1,\dots,\mathbf{c}_T]$. A boolean mask handles the padding slots, allowing downstream operations to seamlessly process varying numbers of attributes.

**Patch grounding** [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): This stage anchors the modulated conditions $C$ directly to the source image's visual tokens $V_{\text{raw}}$. The visual tokens are projected into the fusion space and tagged with a learned type embedding to distinguish the CLS token from the patch tokens. A pre-norm Transformer-decoder layer then refines the conditions in three steps, each with a distinct purpose. **Self-attention** lets the edits co-adapt, so multiple or conflicting requests (like `+Bald` and `+Bangs`) settle into a mutually consistent set before they touch the image. **Cross-attention** then has each condition read the image patches $V$, turning a generic attribute direction into one localized on the region of this specific face it should change, so `-Eyeglasses` grounds on the eye patches rather than acting everywhere. Finally, the **feed-forward sublayer** transforms each condition non-linearly, giving the block the capacity to blend these co-adapted and grounded signals into a single, richer edit representation.
The resulting grounded conditions $C$ are now tied to the region of the face each edit should act on, and they replace the raw conditions in every downstream stage.

**Stacked cross-attention** [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762): This final stage fuses the reference token $\mathbf{v}_{\text{ref}}$ with the grounded conditions $C$ by passing the reference through $L$ identical pre-norm decoder layers. Within each layer, three sequential steps refine the token. First, **self-attention** operates on the query side; because the reference is the sole token here, it transforms this single vector individually rather than mixing information across a sequence. Next, **cross-attention** allows the reference token to read the grounded conditions. It is run as multi-head attention, letting the image weigh the conditions in several representation subspaces in parallel, and because there is only one query token, the operation collapses into a content-based weighted average of the edits, dynamically weighted by the image itself. Finally, a **feed-forward sublayer** applies a non-linear transformation through a GELU activation, a smooth non-linearity that, unlike ReLU, keeps a gradient for negative inputs, providing the capacity to map these weighted edits into a coherent update. Each sublayer is wrapped in a pre-norm residual connection, ensuring that any single layer only incrementally updates the reference. Stacking $L$ such layers enables the image to iteratively read the conditions, update its representation, and read them again. This process yields the final attended vector $\mathbf{a}$, which represents a tailored, per-image blend of the requested edits.

**Gated-residual fusion** [(Vo et al., 2019)](https://arxiv.org/abs/1812.07119): This final stage applies the attended vector $\mathbf{a}$ to the reference $\mathbf{v}_{\text{ref}}$ as a targeted correction rather than a replacement. The two vectors are concatenated into $\mathbf{u}=[\mathbf{v}_{\text{ref}};\mathbf{a}]$ and processed by two parallel heads. The **edit head** (a **GELU MLP**) proposes *what* to change as a signed displacement $\boldsymbol{\delta}$, while the **gate head** (an affine layer with a sigmoid activation) determines *how much* to change via a per-dimension gate $\mathbf{g}\in(0,1)^{D}$. This gated displacement is added to the reference, and the result is projected back onto the unit sphere:

$$\mathbf{q}=\Phi_\theta\big(\mathbf{v}_{\text{ref}},\{(a_k,s_k)\}\big)=\frac{\mathbf{v}_{\text{ref}}+\mathbf{g}\odot\boldsymbol{\delta}}{\lVert\mathbf{v}_{\text{ref}}+\mathbf{g}\odot\boldsymbol{\delta}\rVert_2}$$

Once trained, the gate localizes edits to specific coordinates while leaving the rest of the reference intact. Ultimately, this additive correction produces a unit-norm query $\mathbf{q}$ on the unit sphere, allowing downstream image retrieval to reduce to a simple dot product.

![Detailed layer-by-layer architecture of the cross-attention fusion module](figures/architecture_details.svg)

---

### Training

![Label-free triplet supervision and the InfoNCE objective](figures/training.svg)

**Triplet synthesis:** Each triplet originates from a random reference image $s$ with its corresponding binary attribute vector $\mathbf{b}_s$. A few attributes are randomly sampled and flipped to construct a signed edit query $q$. Applying these exact flips to the original vector yields the ideal target profile, $\mathbf{b}^\star$. The **positive target** $t$ is then drawn from real images that satisfy the query constraints while remaining within a strict Hamming distance budget of the ideal profile.

**Hard negatives:** When enabled, one **constraint-violating** distractor $h$ is mined per query: a real image that satisfies every requested edit but one, which it violates. For the query `-Smiling`, this is a face that is otherwise valid yet still smiling. Because it resembles the target in every respect except the single edited attribute, using it as a negative forces the model to key on the requested change rather than on overall similarity to the source. It is the primary defense against the failure mode where the network simply returns look-alikes of the reference. When no such image exists for a query, that row falls back to its in-batch negatives alone.

**InfoNCE objective** [(van den Oord et al., 2018)](https://arxiv.org/abs/1807.03748): To optimize the model, we employ the InfoNCE loss over a batch of $B$ triplets:

$$\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{t}_i)}{\displaystyle\sum_{j=1}^{B}\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{t}_j)\;+\;\mathbb{1}[h_i\ \text{exists}]\,\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{h}_i)}$$

Here, the fused query $\mathbf{q}_i$ is evaluated against its positive target $\mathbf{t}_i$, $B-1$ in-batch negative targets $\mathbf{t}_j$, and an optional hard negative $\mathbf{h}_i$ modulated by the indicator function $\mathbb{1}$. The frozen parameter $\tau$ scales the dot products according to CLIP's temperature.
Minimizing $\mathcal{L}$ maximizes the similarity between the query and its true target while minimizing it against all negative distractors. 

**Optimisation:** We optimize the model using **AdamW** which is well-suited for this architecture. Because its per-parameter adaptive step sizes accommodate the diverse gradient scales of our heterogeneous modules, while decoupled weight decay provides effective regularization. The learning rate follows a **cosine-annealed schedule** over the epochs, decaying smoothly to zero to promote stable convergence. We use a large batch size to enrich the InfoNCE objective with a higher volume of in-batch contrastive negatives. To prevent overfitting on the synthetic triplets, we combine weight decay with a dropout rate in the decoder and fusion heads.

**Early Stopping:** A held-out set of validation triplets is scored once per epoch with the same InfoNCE loss used for training, with a baseline value also recorded for the untrained model before the first epoch. Whenever the validation loss reaches a new minimum, we snapshot the module's weights as the current best and immediately write that checkpoint to disk, so an interrupted run loses at most the epochs since the last improvement.

### Scorer

At evaluation the scorer builds **one** composite query embedding per source image and ranks the frozen gallery against it.

1. Parse the query string (`+A & -B & …`) into attribute indices and signs.
2. Fuse the source embedding with its conditions through the trained module $\Phi_\theta$ (its output is already L2-normalised)

### Cross-Attention: Qualitative Inspection

To see *what the trained model does* and where it breaks, we inspect a **SUCCESS** and a **FAILURE** case for two query types the benchmark stresses: a single-attribute **negation** (e.g. `-Heavy Makeup`) and a **composed** multi-attribute query (e.g. `+Eyeglasses, -Smiling`). For each, we automatically pick, from that query's own benchmark sources, one source the model gets right (a ground-truth target in its top-k) and one it gets wrong (none in top-k); nothing is hardcoded.

For each `(source, query)` we read out:

- **Top-k retrieval under the edit**: the images the *fused* query pulls to the top (source excluded), each marked ✓/✗ for satisfying the requested attributes and tagged `GT` when it is a benchmark target. This shows directly whether the edit moved retrieval toward the request rather than toward look-alikes of the source.
- **Residual gate** $\sigma(g)\in[0,1]$ from the gated-residual head: its mean summarises overall edit strength, while a low mean with a few high dimensions signals a localised edit and a flat $\approx 0.5$ means the head barely moved off its initialisation.

The trained weights are reused exactly; nothing is re-trained.

### Limitations

Our proposed method is a step forward, but the following limitations remain:

- **The attention does relatively light work.** A query carries at most $T=3$ conditions, so the cross-attention only arbitrates among a handful of vectors. Most of the lift comes from the sign-aware FiLM and the gated residual; the attention mainly reweights. The Transformer is the right *frame*, but not where the heavy lifting happens.

- **The capacity ceiling is now the pooled gallery target, not the source.** The source enters as CLIP's full visual-token sequence, so edits can ground on the region they should touch, but each gallery image is still indexed as a single pooled 512-d embedding. A localised edit must therefore be matched against a holistic vector; lifting this would require a patch-level gallery index, at a real cost in storage and retrieval time.

- **The text bank is frozen and non-compositional.** Conditions are bare-name CLIP text vectors, and CLIP text behaves like a bag of concepts. Interacting attributes (e.g. *Smiling* and *Mouth Slightly Open*) enter as independent conditions that can only be reweighted, not jointly understood.

- **Negation still rides on the CLIP geometry.** Approximating "absence of an attribute" as a direction in an embedding space never trained for negation is a learned workaround, not a true representation of *not*.

- **Identity is preserved at the cost of under-editing.** Because the output defaults to $\mathbf{v}_{\text{ref}}$, leaving the embedding nearly unchanged is always the safe option for the contrastive loss, so strongly requested edits can be damped.

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
