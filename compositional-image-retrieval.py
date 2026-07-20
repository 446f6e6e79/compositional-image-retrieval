#==============================================================================
# Cell   8 [code] - Global paths (dataset root, benchmark annotations, embedding cache)
#==============================================================================

# Do *not* append `celeba/` to CELEBA_ROOT, the dataset class does that itself
CELEBA_ROOT = Path("/content/datasets")
BENCHMARK_ANNOTATIONS_PATH = Path("/content/drive/MyDrive/datasets/celeba_evaluation.json")
EVALUATION_CACHE_DIR = Path("/content/drive/MyDrive/datasets/clip_cache")
GALLERY_FEATS_PATH = EVALUATION_CACHE_DIR / "embeddings.pt"   # test-split (gallery) CLIP features


#==============================================================================
# Cell  10 [code] - CLIP model & encoding helpers (get_CLIP_model, encode_texts)
#==============================================================================

MODEL_NAME = "openai/clip-vit-base-patch32"

_model = None
_processor = None


def get_CLIP_model():
    """Lazily load and cache the CLIP model and processor.

    Returns:
        The cached (model, processor) pair, with the model on `device` and in eval mode.
    """
    global _model, _processor

    if _model is None:
        print("Loading CLIP model...")
        _model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
        _model.eval()

    if _processor is None:
        _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def _as_feature_tensor(out) -> torch.Tensor:
    """Normalize a CLIP feature output into a plain Tensor.

    Different transformers versions return either a plain Tensor or a
    BaseModelOutputWithPooling-style object exposing .text_embeds / .image_embeds /
    .pooler_output.

    Args:
        out: Output of CLIPModel.get_text_features / get_image_features.

    Returns:
        The embedding Tensor extracted from `out`.
    """
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    if isinstance(out, tuple) and len(out) > 0:
        return out[0]
    raise TypeError(f"Unexpected feature output type: {type(out)}")


@torch.no_grad()
def encode_texts(prompts: list[str], device) -> torch.Tensor:
    """Encode a batch of text prompts with CLIP in one call.

    Args:
        prompts: Text prompts to encode.
        device: Device to place the embeddings on.

    Returns:
        A (P, D) tensor, L2-normalized per row, on `device`.
    """
    model, processor = get_CLIP_model()
    inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    embs = _as_feature_tensor(model.get_text_features(**inputs))
    return F.normalize(embs, p=2, dim=-1)


def _collate_keep_pil(batch):
    """Collate (image, label) samples, keeping the images as PIL objects.

    Unlike PyTorch's default ``collate_fn``, the images are returned as a plain
    Python list (so the CLIP processor can do its own preprocessing) while the
    labels are stacked into a single batched tensor.

    Args:
        batch: List of (image, label) samples produced by the dataset.

    Returns:
        An (images, labels) pair: a list of PIL images in batch order and the
        stacked (B, ...) label tensor.
    """
    imgs = [item[0] for item in batch]
    lbls = torch.stack([item[1] for item in batch], dim=0)
    return imgs, lbls


def _encode_dataset(
    dataset,
    device,
    encode_batch: Callable,
    out_shape: tuple[int, ...],
    out_dtype: torch.dtype,
    indices: list[int] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    keep_labels: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode a dataset with CLIP in one DataLoader pass, with progress logging.

    Args:
        dataset: Dataset yielding (image, label) pairs.
        device: Device the CLIP forward runs on.
        encode_batch: Callable (model, inputs) -> (B, ...) CPU tensor, where `inputs` is the
            CLIP processor output for the batch, already moved to `device`. The per-sample
            shape/dtype of this output must match `out_shape`/`out_dtype`.
        out_shape: Per-sample output shape (excluding the batch dim), e.g. (D,) for pooled
            embeddings or (50, 768) for patch tokens. Must match what `encode_batch` returns.
        out_dtype: Output dtype, e.g. torch.float32 or torch.float16. Must match what
            `encode_batch` returns.
        indices: Optional subset of dataset indices to encode, in order. None encodes all.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        keep_labels: Whether to also accumulate and return labels.

    Returns:
        An (encoded, labels) pair: encoded is (N, *out_shape) on CPU; labels is the (N, ...)
        concatenation of the dataset labels if `keep_labels`, else None.
    """
    model, processor = get_CLIP_model()
    source = dataset if indices is None else torch.utils.data.Subset(dataset, list(indices))
    # Stream batches in dataset order; the collate keeps PIL images for the CLIP processor
    loader = torch.utils.data.DataLoader(
        source,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=_collate_keep_pil,
    )

    n_total = len(source)
    pad = len(str(n_total))
    pos = 0

    encoded = torch.empty((n_total, *out_shape), dtype=out_dtype)
    labels_list: list[torch.Tensor] = []
    # Iterate over the DataLoader, encoding each batch and accumulating results
    for imgs_batch, lbls_batch in loader:
        inputs = processor(images=imgs_batch, return_tensors="pt").to(device)
        batch = encode_batch(model, inputs)
        encoded[pos:pos + batch.shape[0]] = batch
        if keep_labels:
            labels_list.append(lbls_batch)
        pos += batch.shape[0]
        print(f"Encoded {pos:>{pad}}/{n_total} images ({100 * pos / n_total:.1f}%)")

    labels = torch.cat(labels_list, dim=0) if keep_labels else None
    return encoded, labels


def _load_or_encode(
    cache_path: str | Path,
    encode_pass: Callable,
    *,
    from_cache: Callable = lambda blob: blob,
):
    """Load cached data from `cache_path` if present, else encode, cache, and return it.

    Shared skeleton for get_encoded_dataset and get_encoded_patches: ensures the cache directory
    exists, and either loads an existing cache or runs `encode_pass`, saves its result, and
    returns it.

    Args:
        cache_path: Path to load data from / save data to.
        encode_pass: Callable () -> (blob, result); `blob` is the exact dict to `torch.save`,
            `result` is what to return to the caller on a cache miss.
        from_cache: Callable (blob) -> result, extracting the return value from a loaded blob.

    Returns:
        Whatever `from_cache(blob)` (cache hit) or `encode_pass()`'s `result` (cache miss)
        produces.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"Loading cached data from {cache_path}.")
        return from_cache(torch.load(cache_path, map_location="cpu"))

    print("Cache not found. Encoding...")
    blob, result = encode_pass()
    torch.save(blob, cache_path)
    print(f"Saved to {cache_path}.")
    return result


@torch.no_grad()
def get_encoded_dataset(
    dataset,
    device,
    cache_path: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all images in a dataset, with on-disk caching.

    Args:
        dataset: Dataset yielding (image, label) pairs.
        device: Device to place the features on.
        cache_path: Path to load features from / save features to.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        A (features, labels) pair: features (N, D) on `device`, L2-normalized per
        row; labels (N, ...) on CPU, as produced by the dataset.
    """
    model, _ = get_CLIP_model()
    embed_dim = model.config.projection_dim

    def encode_batch(model, inputs) -> torch.Tensor:
        e = _as_feature_tensor(model.get_image_features(**inputs))
        return F.normalize(e, p=2, dim=-1).cpu()

    def from_cache(blob):
        feats, labels = blob["features"].to(device), blob["labels"]
        print(f"features: {tuple(feats.shape)}, labels: {tuple(labels.shape)}")
        return feats, labels

    def encode_pass():
        feats, labels = _encode_dataset(dataset, device, encode_batch,
                                         out_shape=(embed_dim,), out_dtype=torch.float32,
                                         batch_size=batch_size, num_workers=num_workers)
        print(f"features: {tuple(feats.shape)}, labels: {tuple(labels.shape)}")
        return {"features": feats, "labels": labels}, (feats.to(device), labels)

    return _load_or_encode(cache_path, encode_pass, from_cache=from_cache)


@torch.no_grad()
def get_encoded_patches(
    dataset,
    device,
    cache_path: str | Path,
    indices: list[int] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    """Encode images into CLIP's per-token visual sequence, with caching.

    Each image's visual sequence is exactly two things concatenated:
    - one CLS token: CLIP's learned global summary of the whole image, not a patch;
    - 49 spatial patch tokens.

    Both come from the vision tower's ``last_hidden_state``, giving the fusion
    module spatially grounded conditions.

    Args:
        dataset: Dataset yielding (image, label) pairs.
        device: Device the CLIP forward runs on.
        cache_path: Path to load tokens from / save tokens to.
        indices: Optional subset of dataset indices to encode (in this order). None encodes all.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.

    Returns:
        A (len(indices) or N, 50, 768) fp16 tensor of visual tokens on CPU, row-aligned to
        `indices` (or to the dataset order when `indices` is None).
    """
    model, _ = get_CLIP_model()
    vision_config = model.config.vision_config
    hidden_size = vision_config.hidden_size
    num_patches = (vision_config.image_size // vision_config.patch_size) ** 2
    seq_len = num_patches + 1  # +1 for the CLS token

    def encode_batch(model, inputs) -> torch.Tensor:
        return model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state.half().cpu()

    def from_cache(blob):
        patches = blob["patches"]
        print(f"patches: {tuple(patches.shape)}")
        return patches

    def encode_pass():
        patches, _ = _encode_dataset(dataset, device, encode_batch, indices=indices,
                                      out_shape=(seq_len, hidden_size), out_dtype=torch.float16,
                                      batch_size=batch_size, num_workers=num_workers,
                                      keep_labels=False)
        print(f"patches: {tuple(patches.shape)}")
        return {"patches": patches}, patches

    return _load_or_encode(cache_path, encode_pass, from_cache=from_cache)


def build_patch_bank(dataset, device, cache_path, source_indices: list[int], total_size: int) -> Callable:
    """Load visual tokens for a dataset subset and return an index-based getter.

    Args:
        dataset: Dataset yielding (image, label) pairs.
        device: Device the CLIP encoding forward runs on.
        cache_path: Path to load the token bank from / save it to.
        source_indices: Original dataset indices to include in the bank.
        total_size: Full dataset length, sizing the lookup table.

    Returns:
        ``patches_for(idx)``: maps (B,) original dataset indices (any device, must be in
        `source_indices`) to a (B, 50, 768) fp16 token tensor on `idx`'s device.
    """
    # Load the cached visual tokens for the source indices, and build a lookup table
    patches = get_encoded_patches(dataset, device, cache_path, indices=source_indices)
    lookup = torch.full((total_size,), -1, dtype=torch.long)
    lookup[torch.tensor(source_indices)] = torch.arange(len(source_indices))

    def patches_for(idx: torch.Tensor) -> torch.Tensor:
        """Gather cached CLIP visual tokens for original dataset indices."""
        rows = lookup[idx.cpu()]
        return patches[rows].to(idx.device)

    return patches_for

#==============================================================================
# Cell  11 [code] - Shared attribute/query helpers (names, index maps, signed-query parser)
#==============================================================================

_attribute_cache: list[str] | None = None


def get_attributes(dataset=None) -> list[str]:
    """Return the dataset's attribute names, filtering out any empty strings.

    Memoized after the first call: CelebA's `attr_names` has 41 entries (one empty
    string), so this filters and caches the 40 non-empty names once, instead of
    recomputing the same list comprehension at every call site.

    Args:
        dataset: A CelebA dataset object exposing `attr_names`. Defaults to the global
            `celeba` test-split dataset (loaded in the "Load CelebA test split" cell).

    Returns:
        The 40 non-empty attribute names, aligned with the label columns.
    """
    global _attribute_cache
    if _attribute_cache is None:
        if dataset is None:
            dataset = celeba
        _attribute_cache = [name for name in dataset.attr_names if name]
    return _attribute_cache


_attribute_index_cache: dict[str, int] | None = None


def attribute_to_index(name: str) -> int:
    """Return the column index of a CelebA attribute name.

    Args:
        name: Attribute name, e.g. "Bald".

    Returns:
        The attribute's position in the 40-column label vector.
    """
    global _attribute_index_cache
    if _attribute_index_cache is None:
        _attribute_index_cache = {n: i for i, n in enumerate(get_attributes())}
    return _attribute_index_cache[name]


def index_to_attribute(index: int) -> str:
    """Return the CelebA attribute name at a given column index.

    Args:
        index: Position in the 40-column label vector.

    Returns:
        The attribute name at that position.
    """
    return get_attributes()[index]


def query_to_signed_indices(text_query: str) -> tuple[list[int], list[int]]:
    """Parse a signed text query into "+" and "-" attribute-index lists.

    The shared query parser: splits each comma-separated term into its sign and
    attribute name, and maps the name to its label-column index.

    Args:
        text_query: Comma-separated signed query, e.g. "+Bald, -Eyeglasses".

    Returns:
        A (pos_idx, neg_idx) pair of attribute-index lists for the "+" and "-" terms.
    """
    pos_idx, neg_idx = [], []
    for component in text_query.split(","):
        component = component.strip()
        if not component:
            continue
        sign_char, attr_name = component[0], component[1:].strip()
        j = attribute_to_index(attr_name)
        (pos_idx if sign_char == "+" else neg_idx).append(j)
    return pos_idx, neg_idx


_attribute_name_embs_cache: torch.Tensor | None = None


def get_attribute_name_embeddings(device) -> torch.Tensor:
    """Lazily encode and cache the cleaned attribute-name CLIP text bank for the 40 attributes.

    Every attribute name becomes a cleaned lowercase name (e.g. "Wearing_Hat" -> "wearing
    hat"); training-free and training-based methods alike reuse this same bank, so it's
    encoded once and cached rather than re-run per call site.

    Args:
        device: Device to place the cached embeddings on.

    Returns:
        A (40, D) L2-normalized CLIP text embedding, one row per `get_attributes()` entry.
    """
    global _attribute_name_embs_cache
    if _attribute_name_embs_cache is None:
        cleaned_attribute_names = [name.replace("_", " ").lower() for name in get_attributes()]
        _attribute_name_embs_cache = encode_texts(cleaned_attribute_names, device)
    return _attribute_name_embs_cache


#==============================================================================
# Cell  12 [code] - def plot_images(celeba_dataset: object, indices: list[int], n_cols: int, n_ro…
#==============================================================================

def plot_images(celeba_dataset: object, indices: list[int], n_cols: int, n_rows: int, figsize: tuple[int, int]=(20, 10)):
    """Plot a grid of CelebA images given their dataset indices.

    Args:
        celeba_dataset: The CelebA dataset object.
        indices: Indices of the images to plot.
        n_cols: Number of columns in the grid.
        n_rows: Number of rows in the grid.
        figsize: Figure size as (width, height).
    """
    if len(indices) > n_cols * n_rows:
        raise ValueError("Number of indices exceeds the grid capacity")

    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # Blank every cell up front so a partially filled last row shows no empty frames
    for ax in axes.flat:
        ax.axis("off")
    for counter, img_idx in enumerate(indices):
        img, _ = celeba_dataset[img_idx]
        axes[counter // n_cols, counter % n_cols].imshow(img)

    plt.tight_layout()
    plt.show()


def plot_image_row(images, titles=None, title_colors=None, figsize=None):
    """Plot a single row of images with optional per-image titles and colors.

    Companion to plot_images (which lays out an untitled grid by dataset index): this
    takes already-loaded images plus optional titles, used wherever a row of results
    needs per-image captions.

    Args:
        images: Already-loaded images to plot, left to right.
        titles: Optional per-image title strings.
        title_colors: Optional per-image title colors, aligned with `titles`.
        figsize: Optional figure size as (width, height); defaults to a size scaled by image count.
    """
    n = len(images)
    _, axes = plt.subplots(1, n, figsize=figsize or (3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis("off")
        if titles is not None:
            ax.set_title(
                titles[i],
                color=(title_colors[i] if title_colors else "black"),
                fontsize=9,
            )
    plt.tight_layout()
    plt.show()


def plot_image_with_attributes(idx: int, figsize: tuple[int, int]=(10, 5)):
    """Plot a single image with its active attributes listed as text alongside it.

    Args:
        idx: Dataset index of the image to plot.
        figsize: Figure size as (width, height).
    """
    img, labels = celeba[idx]
    active_attrs = [index_to_attribute(j) for j, value in enumerate(labels) if value == 1]

    _, (ax_img, ax_text) = plt.subplots(1, 2, figsize=figsize)
    ax_img.imshow(img)
    ax_img.axis("off")

    ax_text.axis("off")
    ax_text.text(0.5, 0.5, "\n".join(active_attrs), fontsize=10, ha="center", va="center")

    plt.tight_layout()
    plt.show()

#==============================================================================
# Cell  14 [code] - def plot_metrics_across_k(average_results_per_query: list[dict], title: str =…
#==============================================================================

def plot_metrics_across_k(average_results_per_query: list[dict], title: str = "Retrieval Metrics across K"):
    """Plot Recall@K and Precision@K as grouped bar charts with 95% confidence intervals.

    One bar per query, grouped by K, in side-by-side recall and precision panels.

    Args:
        average_results_per_query: List of per-query average dicts, as produced by compute_query_average_results().
        title: Title for the overall figure.
    """
    k_values = [1, 5, 10]
    n_queries = len(average_results_per_query)
    x = np.arange(n_queries)
    width = 0.25
    offsets = [-width, 0.0, width]
    colors = [plt.cm.tab10(i) for i in range(len(k_values))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    for k, offset, color in zip(k_values, offsets, colors):
        recall_means    = [q[f"Recall@{k}"]       for q in average_results_per_query]
        recall_cis      = [q[f"Recall@{k}_CI"]    for q in average_results_per_query]
        precision_means = [q[f"Precision@{k}"]    for q in average_results_per_query]
        precision_cis   = [q[f"Precision@{k}_CI"] for q in average_results_per_query]

        ax1.bar(x + offset, recall_means,    width, yerr=recall_cis,    capsize=4, ecolor="black", color=color, label=f"K={k}")
        ax2.bar(x + offset, precision_means, width, yerr=precision_cis, capsize=4, ecolor="black", color=color, label=f"K={k}")

    for ax, metric in [(ax1, "Recall"), (ax2, "Precision")]:
        ax.set_xlabel("Query")
        ax.set_ylabel(f"{metric}@K")
        ax.set_title(f"{metric}@K per query")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in range(n_queries)])
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(title="K")

    plt.tight_layout()
    plt.show()


def plot_methods_comparison(method_results: dict[str, list[dict]], title: str = "Method Comparison across queries"):
    """Plot a per-query line comparison of N retrieval methods.

    Layout is a 2 x 3 grid: rows are Recall (top) and Precision (bottom); columns
    are K in {1, 5, 10}. In each subplot, one segmented line per method connects the
    metric values over the query axis, so per-query differences between methods are
    visible.

    Args:
        method_results: Dict mapping method name to its average_results_per_query (same shape consumed by plot_metrics_across_k).
        title: Title for the overall figure.
    """
    k_values = [1, 5, 10]
    method_names = list(method_results.keys())
    n_methods = len(method_names)
    if n_methods == 0:
        raise ValueError("method_results must contain at least one method.")

    n_queries = len(next(iter(method_results.values())))
    x = np.arange(n_queries)
    colors = [plt.cm.tab10(i % 10) for i in range(n_methods)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    fig.suptitle(title)

    for row_idx, metric in enumerate(["Recall", "Precision"]):
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            for method, color in zip(method_names, colors):
                ys = [q[f"{metric}@{k}"] for q in method_results[method]]
                ax.plot(x, ys, marker="o", color=color, label=method)
            ax.set_title(f"{metric}@{k}")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Q{i+1}" for i in range(n_queries)], rotation=45, ha="right")

    for ax in axes[:, 0]:
        ax.set_ylabel("Score")
    for ax in axes[-1, :]:
        ax.set_xlabel("Query")

    axes[0, 0].legend(title="Method", loc="best")

    plt.tight_layout()
    plt.show()


def plot_results_table(
    method_results: dict[str, list[dict]],
    title: str = "Method Comparison — summary table",
    metrics: tuple[str, ...] = ("Recall", "Precision"),
    k_values: tuple[int, ...] = (1, 5, 10),
):
    """Render a summary table of mean Recall@K / Precision@K per method.

    Rows are methods and columns are mean Recall@K / Precision@K. Each cell is the
    mean over queries of the per-query average - the same aggregation used by
    plot_methods_comparison. The best method per column is highlighted (bold + shaded).

    Args:
        method_results: Dict mapping method name to its average_results_per_query (same shape consumed by plot_methods_comparison).
        title: Title for the figure.
        metrics: Metric families to include as columns.
        k_values: K cutoffs to include per metric.
    """
    method_names = list(method_results.keys())
    if not method_names:
        raise ValueError("method_results must contain at least one method.")
    col_labels = [f"{m}@{k}" for m in metrics for k in k_values]

    # (n_methods, n_cols) matrix of mean scores across queries
    matrix = np.array([
        [float(np.mean([q[f"{m}@{k}"] for q in method_results[name]]))
         for m in metrics for k in k_values]
        for name in method_names
    ])

    fig, ax = plt.subplots(figsize=(1.3 * len(col_labels) + 2.5, 0.55 * len(method_names) + 1.2))
    ax.axis("off")
    ax.set_title(title, pad=12)

    table = ax.table(
        cellText=[[f"{v:.3f}" for v in row] for row in matrix],
        rowLabels=method_names,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Bold + shade the best (max) method in each column; row 0 is the header
    for col, best_row in enumerate(matrix.argmax(axis=0)):
        cell = table[best_row + 1, col]
        cell.set_text_props(weight="bold")
        cell.set_facecolor("#d4edda")

    plt.tight_layout()
    plt.show()

#==============================================================================
# Cell  16 [code] - Load CelebA test split
#==============================================================================

celeba = CelebA(root=CELEBA_ROOT, split="test", download=False)

# The test split should contain 19,962 samples
print("Number of samples:", len(celeba))

# Show element size
sample_img, sample_attrs = celeba[0]
print(f"Sample image size: {sample_img.size}")
print(f"Number of attributes: {len(sample_attrs)}")

#==============================================================================
# Cell  18 [code] - Visualize 50 random samples
#==============================================================================

# Get 50 random images and visualize them
indices = np.random.choice(len(celeba), size=50, replace=False)
plot_images(celeba, indices=indices, n_cols=10, n_rows=5)

#==============================================================================
# Cell  20 [code] - Attribute frequency table
#==============================================================================

all_labels = celeba.attr.numpy()

attr_counts = all_labels.sum(axis=0)
attr_freq = all_labels.mean(axis=0)

print(f"{'Attribute':<20} {'Count':>10} {'Frequency':>10}")
print("-" * 45)

for attr, count, freq in zip(get_attributes(celeba), attr_counts, attr_freq):
    print(f"{attr:<20} {count:>10} {freq:>10.3f}")

#==============================================================================
# Cell  22 [code] - Attribute name/index maps & retrieve_by_attributes
#==============================================================================

def retrieve_by_attributes(query: dict) -> list[int]:
    """Retrieve all dataset images that satisfy the given attribute conditions.

    Args:
        query: Dict mapping attribute name to "+" (must have the attribute) or "-" (must not have the attribute).

    Returns:
        Indices of images that satisfy every specified condition.
    """
    # Boolean mask over the precomputed label matrix
    mask = np.ones(len(all_labels), dtype=bool)

    # For each attribute condition, update the mask to keep only images that satisfy it
    for attr_name, sign in query.items():
        attr_idx = attribute_to_index(attr_name)
        if sign == "+":
            mask &= all_labels[:, attr_idx] == 1
        elif sign == "-":
            mask &= all_labels[:, attr_idx] == 0
        else:
            raise ValueError(f"Invalid sign for attribute condition: {sign}. Use '+' or '-'.")

    return np.nonzero(mask)[0].tolist()


#==============================================================================
# Cell  24 [code] - Inspect a single image's attributes
#==============================================================================

IMAGE_INDEX = 99
plot_image_with_attributes(IMAGE_INDEX)


#==============================================================================
# Cell  26 [code] - Example signed attribute query
#==============================================================================

# YOU CAN CHANGE THIS QUERY TO TEST DIFFERENT ATTRIBUTE COMBINATIONS.
# RERUN THE CELL TO SEE THE RESULTS.
test_query = {
    "Bald": "+",
    "Smiling": "+",
    "Eyeglasses": "-",
}

retrieved_images = retrieve_by_attributes(test_query)
print(f"Number of retrieved images: {len(retrieved_images)}")

# Plot up to 10 random retrieved images
n_samples = min(10, len(retrieved_images))
if n_samples == 0:
    print("No images match this query.")
else:
    sampled_indices = np.random.choice(retrieved_images, size=n_samples, replace=False)
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))
    plot_images(celeba, indices=sampled_indices, n_cols=n_cols, n_rows=n_rows)


#==============================================================================
# Cell  28 [code] - Encode (or load cached) gallery embeddings
#==============================================================================

# Get the encoded dataset, using cached features if available
gallery_embeddings, gallery_labels = get_encoded_dataset(celeba, device, GALLERY_FEATS_PATH, batch_size=128)


#==============================================================================
# Cell  30 [code] - Nearest neighbors by cosine similarity (sanity check)
#==============================================================================

if "gallery_embeddings" not in globals():
    raise RuntimeError(
        "Embeddings not found. Run the offline feature extraction cell above first."
    )

SANITY_SOURCE_IDX = 10006

# Dot product == cosine similarity for unit-norm embeddings
similarities = gallery_embeddings @ gallery_embeddings[SANITY_SOURCE_IDX]

# Get the 6 highest-similarity matches and drop the source itself.
top_vals, top_idx = torch.topk(similarities, k=6)
nearest_indices = top_idx[1:].tolist()
nearest_similarities = top_vals[1:].tolist()

images = [celeba[SANITY_SOURCE_IDX][0]] + [celeba[idx][0] for idx in nearest_indices]
titles = ["Source"] + [f"Cosine sim: {sim:.4f}" for sim in nearest_similarities]
title_colors = ["tab:blue"] + ["black"] * len(nearest_indices)
plot_image_row(images, titles=titles, title_colors=title_colors, figsize=(25, 5))


#==============================================================================
# Cell  34 [code] - def _select_pure_image_idxs(all_labels: np.ndarray, rng: np.random.Generator)…
#==============================================================================

def _select_pure_image_idxs(all_labels: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Return a list of image indices, one per attribute, that are "pure" positives for that attribute.

    A "pure" positive is defined as an image that has the attribute in question and has the least
    number of other attributes. If multiple images have the same minimal number of other
    attributes, one is chosen at random.

    Args:
        all_labels: (N, n_attrs) binary attribute-label matrix.
        rng: Random generator used to break ties and pick fallbacks.

    Returns:
        One selected image index per attribute, in attribute order.
    """
    selected = []
    for attr_idx in range(all_labels.shape[1]):
        candidates = np.where(all_labels[:, attr_idx] == 1)[0]
        if len(candidates) == 0:
            selected.append(int(rng.integers(0, all_labels.shape[0])))
            continue
        other_counts = all_labels[candidates].sum(axis=1) - 1
        purest = candidates[other_counts == other_counts.min()]
        selected.append(int(rng.choice(purest)))
    return selected


def plot_cosine_heatmap(cos_mat: np.ndarray, attr_names: list[str]) -> None:
    """Plot a cosine-similarity heatmap of text prompts versus sampled images.

    Args:
        cos_mat: (n_attrs, n_attrs) cosine-similarity matrix (prompts x images).
        attr_names: Attribute names used as axis tick labels.
    """
    n = len(attr_names)
    fig, ax = plt.subplots(figsize=(14, 14))
    im = ax.imshow(cos_mat, cmap="viridis", aspect="equal")
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(attr_names, rotation=90, fontsize=8)
    ax.set_yticklabels(attr_names, fontsize=8)
    ax.set_xlabel("Sampled image (chosen as 'pure' positive for this attribute)")
    ax.set_ylabel("Text prompt: 'A picture of a person with {attr}'")
    ax.set_title(f"CLIP cosine similarity: {n} attribute prompts × {n} sampled images")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    plt.tight_layout()
    plt.show()


def _print_cosine_diagnostics(cos_mat: np.ndarray) -> None:
    """Print diagonal versus off-diagonal cosine-similarity statistics.

    Args:
        cos_mat: (n_attrs, n_attrs) cosine-similarity matrix whose diagonal pairs each prompt with its matching image.
    """
    n = cos_mat.shape[0]
    diag = np.diag(cos_mat)
    off_diag_mean = (cos_mat.sum() - diag.sum()) / (cos_mat.size - n)
    diag_argmax_rate = float((cos_mat.argmax(axis=1) == np.arange(n)).mean())
    print(f"Mean diagonal cosine:      {diag.mean():.4f}")
    print(f"Mean off-diagonal cosine:  {off_diag_mean:.4f}")
    print(f"Diagonal-argmax rate:      {diag_argmax_rate:.2%} (attributes where the matching image is the row's argmax)")


selected_idxs = _select_pure_image_idxs(all_labels, np.random.default_rng(seed=0))
selected_img_embs = gallery_embeddings[selected_idxs].to(device)

cos_mat = (get_attribute_name_embeddings(device) @ selected_img_embs.T).detach().cpu().numpy()
_print_cosine_diagnostics(cos_mat)


#==============================================================================
# Cell  36 [code] - Plot attribute cosine-similarity heatmap
#==============================================================================

plot_cosine_heatmap(cos_mat, get_attributes())


#==============================================================================
# Cell  38 [code] - def evaluate_retrieval(
#==============================================================================

def evaluate_retrieval(
    retrieved_indices: list[int],
    ground_truth_indices: list[int],
    k: int
) -> dict:
    """Evaluate retrieval performance for a single source image.

    Args:
        retrieved_indices: Image IDs predicted by the model, ordered by similarity (descending).
        ground_truth_indices: Valid target IDs from the benchmark JSON.
        k: Cutoff for top-K evaluation (e.g. 1, 5, 10).

    Returns:
        A dict with Recall@K and Precision@K for the given `k`.
    """
    # Get the top K retrieved indices
    top_k_retrieved = retrieved_indices[:k]

    # Calculate the intersection between predictions and ground truth
    hits = set(top_k_retrieved).intersection(set(ground_truth_indices))
    num_hits = len(hits)

    # Metrics calculations
    # Recall@K (Hit Rate): 1 if at least one match is found, 0 otherwise
    recall_at_k = 1 if num_hits > 0 else 0

    # Precision@K: Fraction of top K predictions that are correct
    precision_at_k = num_hits / k

    return {
        f"Recall@{k}": recall_at_k,
        f"Precision@{k}": precision_at_k
    }


def mean_recall_at_10(evaluation_results: list[dict]) -> float:
    """Compute mean Recall@10 over every (query, source image) pair.

    This is the scalar used to compare hyperparameter settings across the
    training-free methods.

    Args:
        evaluation_results: Per-query result dicts, each mapping source image to its per-K metrics.

    Returns:
        The mean Recall@10 across all (query, source image) pairs.
    """
    vals = [
        metrics["Recall@10"]
        for query_results in evaluation_results
        for metrics in query_results.values()
    ]
    return float(np.mean(vals))


#==============================================================================
# Cell  42 [code] - Load benchmark annotations JSON
#==============================================================================

# Open the JSON file containing the benchmark annotations
with open(BENCHMARK_ANNOTATIONS_PATH, "r") as f:
    annotations = json.load(f)

# Print the number of annotations loaded
print(f"Loaded {len(annotations)} queries!")

# Display a sample annotation to understand the structure of the data
print("Annotation keys:", list(annotations[0].keys()))

# Extract and print first text query
print("Text-Query example:", annotations[0].get("query", ""))

# Extract and print the source image ID for the first annotation
print("Source-Image example:", list(annotations[0].get("ground_truth", {}).keys())[:5], "...")

# Extract and print the list of ground truth indices for the first annotation
print("List of ground truth indices for the first annotation:", annotations[0].get("ground_truth", {}).get("13", [])[:5], "...")

#==============================================================================
# Cell  44 [code] - Helper functions to extract query and ground-truth info from benchmark annotations
#==============================================================================

def get_text_query(annotation: dict) -> str:
    """Extract the text query from a benchmark annotation.

    Args:
        annotation: Benchmark annotation dict for a single query.

    Returns:
        The text query string (e.g. "+glasses, -smile").
    """
    return annotation.get("query", "")


def get_source_image_idxs(annotation: dict) -> list[int]:
    """Extract the source image IDs from a benchmark annotation.

    Args:
        annotation: Benchmark annotation dict for a single query.

    Returns:
        The source image IDs as integers.
    """
    # The "ground_truth" keys must be converted to int since JSON keys are always strings
    return [int(key) for key in annotation.get("ground_truth", {}).keys()]


def get_target_indices(annotation: dict, source_image_idx: int) -> list[int]:
    """Extract the target image IDs for a given source image from a benchmark annotation.

    Args:
        annotation: Benchmark annotation dict for a single query.
        source_image_idx: Index of the source image whose ground-truth targets to retrieve.

    Returns:
        The valid target IDs (integers) considered correct matches for the given query.
    """
    return annotation.get("ground_truth", {}).get(str(source_image_idx), [])


#==============================================================================
# Cell  46 [code] - Sanity-check annotation helper functions
#==============================================================================

# Let's test these utility functions on the first annotation in the dataset
annotation = annotations[1]

text_query = get_text_query(annotation)
print("Text query:", text_query )

source_image_idx = get_source_image_idxs(annotation)[0]
print("Source image index:", source_image_idx)
plot_image_with_attributes(source_image_idx, figsize=(4, 4))

# Get the first 5 ground truth indices for this annotation and source image
ground_truth_indices = get_target_indices(annotation, source_image_idx)[:5]
print("Ground truth indices for this query:", ground_truth_indices)

plot_images(celeba, indices=ground_truth_indices, n_cols=5, n_rows=1, figsize=(10, 2))


#==============================================================================
# Cell  48 [code] - def retrieve_topk(scores: torch.Tensor, exclude_idx: int, k: int = 10) -> lis…
#==============================================================================

from scipy.stats import binomtest


def retrieve_topk(scores: torch.Tensor, exclude_idx: int, k: int = 10) -> list[int]:
    """Return the top-k gallery indices by score, excluding the source image.

    Args:
        scores: (N,) similarity or composite score for every gallery image.
        exclude_idx: Index of the source image to remove from results.
        k: Number of results to return.

    Returns:
        Up to k gallery indices, ranked by descending score.
    """
    # Retrieve the top k+1 indices (including the source image)
    _, topk = torch.topk(scores, k=k + 1)
    # Filter out the source image index and return the top k results
    return topk[topk != exclude_idx][:k].tolist()


def evaluate(
    annotations: list[dict],
    make_scorer: Callable,
    verbose: bool = False
) -> list[dict]:
    """Evaluate retrieval performance on the benchmark.

    Single driver for all methods. Each method supplies a *scorer factory*:
        make_scorer(annotation) -> scorer(source_idx) -> (N,) gallery scores.

    Per-query work (z-scoring, prompt tokenisation, constraint vectors) happens
    once inside ``make_scorer``; the inner ``scorer`` loop is then fast.

    Args:
        annotations: Benchmark annotations (loaded from the JSON file).
        make_scorer: Given an annotation, returns ``scorer(source_idx) -> (N,) score tensor``.
        verbose: Whether to print per-query progress.

    Returns:
        Per-query results as ``list[dict[source_idx -> metrics_dict]]``, where each
        metrics_dict maps "Recall@K" / "Precision@K" to its value for K in {1, 5, 10}.
    """
    results = []
    for i, annotation in enumerate(annotations):
        if verbose:
            print(f"Evaluating query Q{i+1}: {get_text_query(annotation)}")
        # Setup the scorer for this query
        scorer = make_scorer(annotation)

        per_source = {}
        for src in get_source_image_idxs(annotation):
            retrieved = retrieve_topk(scorer(src), exclude_idx=src, k=10)
            per_source[src] = {
                name: value
                for k in (1, 5, 10)
                for name, value in evaluate_retrieval(retrieved, get_target_indices(annotation, src), k).items()
            }
        results.append(per_source)
    return results


def _mean_and_ci(values: list[float]) -> tuple[float, float]:
    """Compute a mean and its 95% Wald confidence interval from the empirical standard deviation.

    Recall@K is 0/1, so its Bernoulli variance p(1-p)/n happens to coincide with this; Precision@K
    is a fraction in {0, 1/k, ..., 1} for which the Bernoulli formula is wrong, so both metrics use
    this shared empirical-std estimator instead.

    Args:
        values: Per-source metric values for one query.

    Returns:
        A (mean, 95%_CI_half_width) pair; the CI is 0 when fewer than 2 values are given.
    """
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    ci = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return mean, float(ci)


def compute_query_average_results(query_evaluation_results: dict) -> dict:
    """Average Recall@K and Precision@K across a query's source images, for K in {1, 5, 10}.

    Args:
        query_evaluation_results: Dict mapping source image index to its flat metrics dict
            (Recall@K / Precision@K for K in {1, 5, 10}).

    Returns:
        A dict of average Recall@K / Precision@K plus their 95% confidence intervals for the query.
    """
    average_results = {}

    for k in [1, 5, 10]:
        # Collect the per-source Recall@K and Precision@K values for this query
        recall_vals = [m[f"Recall@{k}"] for m in query_evaluation_results.values()]
        precision_vals = [m[f"Precision@{k}"] for m in query_evaluation_results.values()]

        # Average each metric and attach its 95% confidence interval (empirical-std based; see
        # _mean_and_ci - the naive Bernoulli formula only applies to Recall, not Precision)
        average_results[f"Recall@{k}"], average_results[f"Recall@{k}_CI"] = _mean_and_ci(recall_vals)
        average_results[f"Precision@{k}"], average_results[f"Precision@{k}_CI"] = _mean_and_ci(precision_vals)

    return average_results


def evaluate_and_average(annotations: list[dict], make_scorer: Callable, verbose: bool = False):
    """Run evaluate() then per-query averaging in one call.

    Args:
        annotations: Benchmark annotations (loaded from the JSON file).
        make_scorer: Scorer factory; given an annotation returns ``scorer(source_idx) -> (N,) score tensor``.
        verbose: Whether to print per-query progress.

    Returns:
        A (evaluation_results, average_results_per_query) pair: the raw per-source metrics
        (for mean_recall_at_10) and the per-query averages (for plotting) - the pair every
        method downstream needs.
    """
    results = evaluate(annotations, make_scorer, verbose=verbose)
    return results, [compute_query_average_results(q) for q in results]


def recall10_hits(evaluation_results: list[dict]) -> np.ndarray:
    """Flatten a method's per-(query, source) Recall@10 outcomes into one aligned 0/1 vector.

    evaluate() iterates annotations and sources in a fixed order, so vectors extracted this
    way are paired across methods: position i is the same (query, source) for every method.

    Args:
        evaluation_results: Raw per-source metrics, as returned by evaluate().

    Returns:
        A 1D 0/1 int array, one entry per (query, source) pair.
    """
    return np.array([
        metrics["Recall@10"]
        for query_results in evaluation_results
        for metrics in query_results.values()
    ], dtype=int)


def mcnemar_pvalue(hits_a: np.ndarray, hits_b: np.ndarray) -> tuple[float, int, int]:
    """Exact McNemar test on paired binary outcomes.

    Only discordant pairs carry information: b counts sources where A hits and B misses,
    c the reverse. Under H0 (no difference) each discordant pair is a fair coin, so the
    p-value is an exact two-sided binomial test.

    Args:
        hits_a: 0/1 outcome vector of method A.
        hits_b: 0/1 outcome vector of method B, paired with `hits_a`.

    Returns:
        A (p_value, b, c) tuple; p_value is 1.0 when there are no discordant pairs.
    """
    b = int(((hits_a == 1) & (hits_b == 0)).sum())
    c = int(((hits_a == 0) & (hits_b == 1)).sum())
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return p, b, c


#==============================================================================
# Cell  51 [code] - def baseline_scorer(gallery_embeddings: torch.Tensor) -> Callable
#==============================================================================

def baseline_scorer(gallery_embeddings: torch.Tensor) -> Callable:
    """Build the scorer factory for the signed-arithmetic baseline.
    Decomposes the query into signed attribute terms and fuses them with the source
    image embedding by simple latent arithmetic.

    Args:
        gallery_embeddings: (N, D) gallery image embeddings, L2-normalized per row.

    Returns:
        A ``make_scorer(annotation)`` factory consumed by evaluate().
    """
    attr_name_embs = get_attribute_name_embeddings(gallery_embeddings.device)

    def make_scorer(annotation: dict) -> Callable:
        """Build a per-query scorer from the query's signed attribute delta."""
        pos_idx, neg_idx = query_to_signed_indices(get_text_query(annotation))
        # Initialize delta vector to zero, then add/subtract attribute embeddings based on the query
        delta = torch.zeros(gallery_embeddings.shape[1], device=gallery_embeddings.device)
        if pos_idx:
            delta = delta + attr_name_embs[pos_idx].sum(dim=0)
        if neg_idx:
            delta = delta - attr_name_embs[neg_idx].sum(dim=0)

        def scorer(source_idx: int) -> torch.Tensor:
            """Score every gallery image against the fused source query embedding."""
            fused = gallery_embeddings[source_idx] + delta
            return gallery_embeddings @ F.normalize(fused, dim=0)

        return scorer
    return make_scorer


#==============================================================================
# Cell  53 [code] - Evaluate & plot baseline
#==============================================================================

evaluation_results_baseline, average_results_per_query_baseline = evaluate_and_average(
    annotations,
    baseline_scorer(gallery_embeddings),
    verbose=False,
)

plot_metrics_across_k(average_results_per_query_baseline, title="Baseline Fusion Performance across K")


#==============================================================================
# Cell  55 [code] - Benchmark ground-truth rule (constants & label helpers)
#==============================================================================

# Both constants encode the benchmark's ground-truth rule, so they are shared by the
# validation-query builder (training-free tuning) and the triplet synthesis (training-based).
MAX_TERMS      = 3    # max attribute conditions per query (benchmark-dictated)
HAMMING_BUDGET = 2    # max Hamming distance for a valid target (matches benchmark)


def desired_target_labels(source_labels: torch.Tensor, pos_idx: list[int], neg_idx: list[int]) -> torch.Tensor:
    """Compute the ideal target label vector for a source under a signed query.

    Args:
        source_labels: Boolean attribute-label vector of the source image.
        pos_idx: Attribute indices the target must have.
        neg_idx: Attribute indices the target must not have.

    Returns:
        The source labels with the queried attributes forced on/off.
    """
    target = source_labels.clone()
    if pos_idx:
        target[pos_idx] = True
    if neg_idx:
        target[neg_idx] = False
    return target


def query_satisfied(labels_bool: torch.Tensor, pos_idx: list[int], neg_idx: list[int]) -> torch.Tensor:
    """Return a boolean mask of candidates satisfying a signed query.

    A candidate satisfies the query if it has all the positive attributes and none of
    the negative attributes, regardless of other attributes.

    Args:
        labels_bool: (N, n_attrs) boolean candidate label matrix.
        pos_idx: Attribute indices a candidate must have.
        neg_idx: Attribute indices a candidate must not have.

    Returns:
        An (N,) boolean mask, True where the candidate satisfies the query.
    """
    ok = torch.ones(labels_bool.shape[0], dtype=torch.bool, device=labels_bool.device)
    if pos_idx:
        ok &= labels_bool[:, pos_idx].all(dim=1)
    if neg_idx:
        ok &= (~labels_bool[:, neg_idx]).all(dim=1)
    return ok


def find_valid_targets(labels_bool: torch.Tensor, source_labels: torch.Tensor,
                       pos_idx: list[int], neg_idx: list[int]) -> torch.Tensor:
    """Return indices of valid target images for a source and query.

    Valid targets satisfy the query and lie within HAMMING_BUDGET of the ideal target -
    the exact rule the benchmark JSON was built with.

    Args:
        labels_bool: (N, n_attrs) boolean candidate label matrix.
        source_labels: Boolean attribute-label vector of the source image.
        pos_idx: Attribute indices the target must have.
        neg_idx: Attribute indices the target must not have.

    Returns:
        Indices into `labels_bool` of the valid target images.
    """
    # Compute the ideal target attribute vector for this source and query
    target = desired_target_labels(source_labels, pos_idx, neg_idx)
    # First filter to candidates that satisfy the query and whose labels are within HAMMING_BUDGET
    ok = query_satisfied(labels_bool, pos_idx, neg_idx)
    hamming = (labels_bool != target.unsqueeze(0)).sum(dim=1)
    return (ok & (hamming <= HAMMING_BUDGET)).nonzero(as_tuple=True)[0]


#==============================================================================
# Cell  56 [code] - Grid-search weight candidates
#==============================================================================

GRID_W_ATTR = [0.05, 0.1, 0.2, 0.4]   # attribute-proximity penalty weight candidates
GRID_W_VISUAL  = [0.0, 0.5, 1.0]          # visual identity weight candidates


#==============================================================================
# Cell  57 [code] - Load CelebA train split & pre-extract features
#==============================================================================

# The train split serves two roles: it is the pool the tuning validation queries are built
# from here (so the benchmark is never used for hyperparameter selection), and it is later
# reused as the training corpus for the training-based method.
TRAIN_FEATS_PATH = EVALUATION_CACHE_DIR / "train_embeddings.pt"

# Load the full CelebA train split
print(f"Loading CelebA train split from {CELEBA_ROOT} ...")
celeba_train = CelebA(root=CELEBA_ROOT, split="train", download=False)
print(f"CelebA train split size: {len(celeba_train)}")

# Pre-extract image features once and cache
train_embeddings, train_labels = get_encoded_dataset(
    celeba_train, device, TRAIN_FEATS_PATH, batch_size=128
)
print(f"train_embeddings dtype: {train_embeddings.dtype}, device: {train_embeddings.device}")

train_labels_bool = (train_labels.to(device) > 0)        # (M, 40) on GPU, for candidate filtering
train_labels_bool_np = train_labels_bool.cpu().numpy()   # CPU copy, for cheap per-sample query sampling
TRAIN_N = train_labels_bool.shape[0]
n_attrs = train_labels_bool.shape[1]


#==============================================================================
# Cell  58 [code] - Synthetic validation queries for hyperparameter tuning
#==============================================================================

VAL_QUERY_COUNT   = 36   # 3x the benchmark's query count, for a more robust hyperparameter estimate
VAL_MAX_SOURCES   = 50   # cap sources per query to bound grid-search cost
VAL_MIN_TARGETS   = 5    # benchmark rule: a source needs >= 5 valid targets
VAL_SOURCE_TRIES  = 2000 # sampled source candidates per query before giving up
VAL_SEEDS         = [12, 34, 56]  # independent query draws averaged in the grid search below, so
                                  # the chosen weights aren't just the ones that got lucky on one draw


def build_validation_annotations(
    labels_bool: torch.Tensor,
    n_queries: int,
    seed: int,
    max_sources: int = VAL_MAX_SOURCES,
    min_targets: int = VAL_MIN_TARGETS,
    source_tries: int = VAL_SOURCE_TRIES,
) -> list[dict]:
    """Build benchmark-shaped annotations from random signed queries over a label pool.

    Mirrors the benchmark construction: each annotation pairs a signed query with sources
    that have at least `min_targets` valid targets under the Hamming rule
    (find_valid_targets). Emitting the exact benchmark JSON shape means evaluate() and
    every scorer factory can be reused unchanged for hyperparameter tuning.

    Args:
        labels_bool: (N, n_attrs) boolean label matrix of the pool (e.g. the train split).
        n_queries: Number of annotations to build; term counts cycle 1..MAX_TERMS so the
            complexity mix mirrors the benchmark's simple and composed queries.
        seed: Seed for the random generator.
        max_sources: Max sources kept per query.
        min_targets: Min valid targets a source needs to be kept.
        source_tries: Random source candidates examined per query.

    Returns:
        A list of `{"query": ..., "ground_truth": {str(src): [targets...]}}` dicts.
    """
    rng = np.random.default_rng(seed)
    names = get_attributes()
    annotations_out = []
    while len(annotations_out) < n_queries:
        # Cycle 1..MAX_TERMS terms and draw a random sign per attribute
        n_terms = 1 + len(annotations_out) % MAX_TERMS
        attrs = [int(j) for j in rng.choice(labels_bool.shape[1], size=n_terms, replace=False)]
        term_signs = rng.integers(0, 2, size=n_terms)
        pos_idx = [a for a, s in zip(attrs, term_signs) if s == 1]
        neg_idx = [a for a, s in zip(attrs, term_signs) if s == 0]

        # Keep sampled sources that have enough valid targets under the benchmark rule
        ground_truth = {}
        for s in rng.choice(labels_bool.shape[0], size=source_tries, replace=False):
            targets = find_valid_targets(labels_bool, labels_bool[int(s)], pos_idx, neg_idx)
            targets = targets[targets != int(s)]
            if targets.numel() >= min_targets:
                ground_truth[str(int(s))] = targets.tolist()
            if len(ground_truth) >= max_sources:
                break
        if not ground_truth:
            continue   # degenerate query (e.g. contradictory attributes) - resample

        query = ", ".join([f"+{names[j]}" for j in pos_idx] + [f"-{names[j]}" for j in neg_idx])
        annotations_out.append({"query": query, "ground_truth": ground_truth})
    return annotations_out


val_annotation_sets = [
    build_validation_annotations(train_labels_bool, VAL_QUERY_COUNT, seed=seed)
    for seed in VAL_SEEDS
]
val_annotations = val_annotation_sets[0]  # kept around for the inspection printout below
print(f"Built {len(VAL_SEEDS)} validation query sets of {VAL_QUERY_COUNT} queries each (seeds {VAL_SEEDS}):")
for ann in val_annotations:
    print(f"  {get_text_query(ann):<45} sources: {len(ann['ground_truth'])}")


#==============================================================================
# Cell  59 [code] - compute_attribute_logits
#==============================================================================

def compute_attribute_logits(
    gallery_embeddings: torch.Tensor,
    E_pos: torch.Tensor,
    E_neg: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the (N, n_attrs) raw attribute logit matrix.

    With a pos/neg bank the logit is the pos-minus-neg cosine margin; with a single
    bank it is the plain cosine.

    Args:
        gallery_embeddings: (N, D) gallery image embeddings, L2-normalized per row.
        E_pos: (n_attrs, D) positive attribute text embeddings.
        E_neg: (n_attrs, D) negative attribute text embeddings, or None.

    Returns:
        An (N, n_attrs) attribute logit matrix.
    """
    logits = gallery_embeddings @ E_pos.T
    if E_neg is not None:
        logits = logits - gallery_embeddings @ E_neg.T
    return logits


def zscore_columns(logits: torch.Tensor) -> torch.Tensor:
    """Standardize each attribute column over the gallery.

    CLIP cosines have very different per-attribute means/spreads; without this,
    high-mean attributes dominate any sum across columns.

    Args:
        logits: (N, n_attrs) attribute logit matrix.

    Returns:
        The column-wise z-scored logits, same shape as `logits`.
    """
    mean = logits.mean(dim=0, keepdim=True)
    std = logits.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (logits - mean) / std


def attribute_matching_scorer(
    gallery_embeddings: torch.Tensor,
    E_pos: torch.Tensor,
    E_neg: torch.Tensor | None = None,
    w_query: float = 1.0,
    w_attr: float = 0.1,
    w_visual: float = 0.0,
) -> Callable:
    """Scorer factory for attribute-attribute vector matching.

    Pre-computes the z-scored attribute logit matrix once, then returns a
    ``make_scorer`` closure that pre-computes the per-query constraint vector once;
    the inner per-source ``scorer`` only slices rows and sums.

    Used by Source-Attribute Matching and Prompt Ensembling - they differ only in E_pos/E_neg.

    Args:
        gallery_embeddings: (N, D) gallery image embeddings, L2-normalized per row.
        E_pos: (n_attrs, D) positive attribute text embeddings.
        E_neg: (n_attrs, D) negative attribute text embeddings, or None.
        w_query: Weight for the constraint (queried-attribute) term.
        w_attr: Weight for the identity-preservation (attribute-proximity) term.
        w_visual: Weight for direct visual cosine similarity.

    Returns:
        A ``make_scorer(annotation)`` factory consumed by evaluate().
    """
    Z = zscore_columns(compute_attribute_logits(gallery_embeddings, E_pos, E_neg))  # (N, n_attrs)

    def make_scorer(annotation: dict) -> Callable:
        """Build a per-query scorer from the query's constraint vector."""
        pos_idx, neg_idx = query_to_signed_indices(get_text_query(annotation))
        queried   = set(pos_idx + neg_idx)
        unqueried = [j for j in range(Z.shape[1]) if j not in queried]
        constraint = Z[:, pos_idx].sum(dim=1) - Z[:, neg_idx].sum(dim=1)  # (N,)
        Z_unq = Z[:, unqueried]  # gathered once per query; the per-source loop only slices rows

        def scorer(source_idx: int) -> torch.Tensor:
            """Score every gallery image by constraint, attribute proximity, and visual similarity."""
            attr_proximity = ((Z_unq - Z_unq[source_idx]) ** 2).sum(dim=1)
            scores  = w_query * constraint - w_attr * attr_proximity
            if w_visual > 0:
                scores = scores + w_visual * (gallery_embeddings @ gallery_embeddings[source_idx])
            return scores

        return scorer
    return make_scorer


#==============================================================================
# Cell  62 [code] - Grid search over fusion weights (on the validation queries)
#==============================================================================

# The grid is scored on the synthetic train-split validation queries, never on the
# benchmark: the selected weights are frozen here and only then reported on the benchmark.
# Each cell is averaged over VAL_SEEDS independent query draws (same scorer, evaluated against
# each draw) so the pick isn't just the weights that got lucky on a single sample of queries.
grid_rows = []
for w_p in GRID_W_ATTR:
    for w_v in GRID_W_VISUAL:
        scorer_factory = attribute_matching_scorer(
            train_embeddings, get_attribute_name_embeddings(device), w_query=1.0, w_attr=w_p, w_visual=w_v
        )
        r10_per_seed = [
            mean_recall_at_10(evaluate(val_ann, scorer_factory, verbose=False))
            for val_ann in val_annotation_sets
        ]
        r10 = float(np.mean(r10_per_seed))
        grid_rows.append((w_p, w_v, r10))
        seeds_str = ", ".join(f"{x:.4f}" for x in r10_per_seed)
        print(f"w_attr={w_p:<5} w_visual={w_v:<4} mean val Recall@10={r10:.4f}  (per-seed: {seeds_str})")

best_w_attr, best_w_visual, best_r10 = max(grid_rows, key=lambda row: row[2])
print(f"\nBest: w_attr={best_w_attr}, w_visual={best_w_visual} (mean val Recall@10={best_r10:.4f})")
SAM_WEIGHTS = dict(w_query=1.0, w_attr=best_w_attr, w_visual=best_w_visual)


#==============================================================================
# Cell  64 [code] - Evaluate & plot Source-Attribute Matching
#==============================================================================

evaluation_results_sam, average_results_per_query_sam = evaluate_and_average(
    annotations,
    attribute_matching_scorer(gallery_embeddings, get_attribute_name_embeddings(device), **SAM_WEIGHTS),
)
print(f"Source-Attribute Matching: mean Recall@10 = {mean_recall_at_10(evaluation_results_sam):.4f}")

plot_metrics_across_k(
    average_results_per_query_sam,
    title="Source-Attribute Matching — Performance across K",
)


#==============================================================================
# Cell  69 [code] - Attribute description bank (positive/negative)
#==============================================================================

# Person-referring positive AND negative descriptions for each CelebA attribute.
attribute_descriptions_pos = {
    "5_o_Clock_Shadow":     ["a person with a 5 o'clock shadow", "a person with light facial stubble", "a person with short beard stubble", "a face with a 5 o'clock shadow", "a man with a 5 o'clock shadow", "a person with visible beard stubble"],
    "Arched_Eyebrows":      ["a person with arched eyebrows", "a person with curved eyebrows", "a face with high arched eyebrows", "a portrait with strongly arched eyebrows", "a person whose eyebrows are clearly arched"],
    "Attractive":           ["an attractive person", "a good-looking person", "a visually appealing person", "a beautiful person", "an attractive face", "a strikingly attractive person"],
    "Bags_Under_Eyes":      ["a person with bags under the eyes", "a person with eye bags", "a tired-looking person with under-eye bags", "a face with visible under-eye bags", "a portrait of a person with bags under the eyes"],
    "Bald":                 ["a bald person", "a person with no hair", "a person with a shaved head", "a person with a completely bald head", "a man who is bald", "a portrait of a bald person"],
    "Bangs":                ["a person with bangs", "a person with fringe hair", "a face with bangs across the forehead", "a portrait of someone with bangs", "a person whose hair has bangs"],
    "Big_Lips":             ["a person with full lips", "a person with big lips", "a face with prominent lips", "a portrait of a person with very full lips"],
    "Big_Nose":             ["a person with a big nose", "a person with a large nose", "a face with a prominent nose", "a portrait of a person with a noticeably big nose"],
    "Black_Hair":           ["a person with black hair", "a person with dark black hair", "a portrait of a black-haired person", "a person whose hair is black"],
    "Blond_Hair":           ["a person with blond hair", "a person with blonde hair", "a person with light blonde hair", "a portrait of a blond person", "a person whose hair is blonde"],
    "Blurry":               ["a blurry photo of a person", "an out-of-focus image of a person", "a blurred image of a person", "a low-quality blurry portrait", "a defocused photograph of a face"],
    "Brown_Hair":           ["a person with brown hair", "a person with dark brown hair", "a portrait of a brown-haired person", "a person whose hair is brown"],
    "Bushy_Eyebrows":       ["a person with bushy eyebrows", "a person with thick eyebrows", "a face with very thick eyebrows", "a portrait of a person with bushy eyebrows"],
    "Chubby":               ["a chubby person", "a person with a round face", "a person with a chubby face", "a portrait of a chubby person"],
    "Double_Chin":          ["a person with a double chin", "a person with a noticeable double chin", "a face with a clear double chin", "a portrait of a person with a double chin"],
    "Eyeglasses":           ["a person wearing eyeglasses", "a person wearing glasses", "a person with glasses", "a face with eyeglasses", "a portrait of a person wearing glasses", "a person who wears glasses"],
    "Goatee":               ["a person with a goatee", "a person with a goatee beard", "a man with a goatee", "a portrait of a person with a goatee"],
    "Gray_Hair":            ["a person with gray hair", "a person with grey hair", "a person with silver hair", "a portrait of a gray-haired person", "an older person with gray hair"],
    "Heavy_Makeup":         ["a person wearing heavy makeup", "a person with noticeable makeup", "a person with strong makeup", "a face with heavy makeup", "a portrait of a person wearing heavy makeup"],
    "High_Cheekbones":      ["a person with high cheekbones", "a person with prominent cheekbones", "a face with sharply defined high cheekbones"],
    "Male":                 ["a man", "a male person", "a portrait of a man", "a photograph of a man", "a male face"],
    "Mouth_Slightly_Open":  ["a person with their mouth slightly open", "a person with slightly open lips", "a face with parted lips", "a portrait of a person whose mouth is slightly open"],
    "Mustache":             ["a person with a mustache", "a person with facial hair and a mustache", "a man with a mustache", "a portrait of a person with a mustache"],
    "Narrow_Eyes":          ["a person with narrow eyes", "a person with small eyes", "a face with narrow eyes", "a portrait of a person with narrow eyes"],
    "No_Beard":             ["a clean-shaven person", "a person without a beard", "a person with no facial hair", "a portrait of a clean-shaven person", "a face without any beard"],
    "Oval_Face":            ["a person with an oval face", "a person with an oval-shaped face", "a portrait of a person with an oval face"],
    "Pale_Skin":            ["a person with pale skin", "a person with light skin tone", "a portrait of a person with pale skin", "a face with very pale skin"],
    "Pointy_Nose":          ["a person with a pointy nose", "a person with a sharp nose", "a face with a pointy nose"],
    "Receding_Hairline":    ["a person with a receding hairline", "a person with thinning hairline", "a portrait of a person whose hairline is receding"],
    "Rosy_Cheeks":          ["a person with rosy cheeks", "a person with flushed cheeks", "a face with rosy cheeks"],
    "Sideburns":            ["a person with sideburns", "a person with long sideburns", "a face with sideburns"],
    "Smiling":              ["a smiling person", "a person who is smiling", "a person with a happy expression", "a person with a big smile", "a portrait of a smiling person", "a face with a smile"],
    "Straight_Hair":        ["a person with straight hair", "a person with smooth straight hair", "a portrait of a person with straight hair"],
    "Wavy_Hair":            ["a person with wavy hair", "a person with curly wavy hair", "a portrait of a person with wavy hair"],
    "Wearing_Earrings":     ["a person wearing earrings", "a person with earrings", "a portrait of a person wearing earrings"],
    "Wearing_Hat":          ["a person wearing a hat", "a person with a hat", "a portrait of a person wearing a hat"],
    "Wearing_Lipstick":     ["a person wearing lipstick", "a person with lipstick", "a portrait of a person wearing lipstick"],
    "Wearing_Necklace":     ["a person wearing a necklace", "a person with a necklace", "a portrait of a person wearing a necklace"],
    "Wearing_Necktie":      ["a person wearing a necktie", "a person with a tie", "a portrait of a person wearing a necktie"],
    "Young":                ["a young person", "a youthful person", "a person who looks young", "a portrait of a young person", "a young-looking face"],
}

# Linguistic negatives. We avoid the "not X" construction wherever possible because
# CLIP's text encoder attends to the object token regardless of the "not" - phrasing
# matters. Where a clean linguistic opposite exists (e.g. clean-shaven vs bearded)
# we use it; otherwise we lean on "without {attr}" / "no {attr}" framings.
attribute_descriptions_neg = {
    "5_o_Clock_Shadow":     ["a clean-shaven person", "a person with no facial stubble", "a person without a 5 o'clock shadow", "a smoothly shaven face"],
    "Arched_Eyebrows":      ["a person with flat eyebrows", "a person whose eyebrows are not arched", "a face with straight eyebrows"],
    "Attractive":           ["an unattractive person", "a plain-looking person", "an ordinary-looking person"],
    "Bags_Under_Eyes":      ["a person without bags under the eyes", "a person with no eye bags", "a fresh-looking face without under-eye bags"],
    "Bald":                 ["a person with hair", "a person with a full head of hair", "a person who is not bald"],
    "Bangs":                ["a person without bangs", "a person with no fringe", "a face without bangs"],
    "Big_Lips":             ["a person with thin lips", "a person with small lips", "a person without big lips"],
    "Big_Nose":             ["a person with a small nose", "a person without a big nose", "a face with a small nose"],
    "Black_Hair":           ["a person without black hair", "a person whose hair is not black"],
    "Blond_Hair":           ["a person without blond hair", "a person whose hair is not blonde"],
    "Blurry":               ["a sharp clear photo of a person", "a high quality in-focus portrait", "a crisp clear image of a face"],
    "Brown_Hair":           ["a person without brown hair", "a person whose hair is not brown"],
    "Bushy_Eyebrows":       ["a person with thin eyebrows", "a person without bushy eyebrows"],
    "Chubby":               ["a thin person", "a person with a slim face", "a person who is not chubby"],
    "Double_Chin":          ["a person without a double chin", "a person with a defined jawline"],
    "Eyeglasses":           ["a person without eyeglasses", "a person not wearing glasses", "a face without glasses", "a person with bare eyes"],
    "Goatee":               ["a person without a goatee", "a clean-shaven person", "a person with no goatee"],
    "Gray_Hair":            ["a person without gray hair", "a person whose hair is not gray"],
    "Heavy_Makeup":         ["a person with no makeup", "a person without makeup", "a face without heavy makeup", "a person with a bare natural face"],
    "High_Cheekbones":      ["a person without high cheekbones", "a person with flat cheeks"],
    "Male":                 ["a woman", "a female person", "a portrait of a woman", "a female face"],
    "Mouth_Slightly_Open":  ["a person with a closed mouth", "a person with closed lips", "a person whose mouth is shut"],
    "Mustache":             ["a clean-shaven person", "a person without a mustache", "a person with no mustache"],
    "Narrow_Eyes":          ["a person with wide eyes", "a person with big eyes", "a person without narrow eyes"],
    "No_Beard":             ["a person with a beard", "a bearded person", "a person with facial hair"],
    "Oval_Face":            ["a person without an oval face", "a person with a round face", "a person with a square face"],
    "Pale_Skin":            ["a person with dark skin", "a person with a tanned complexion", "a person without pale skin"],
    "Pointy_Nose":          ["a person with a rounded nose", "a person without a pointy nose"],
    "Receding_Hairline":    ["a person with a full hairline", "a person without a receding hairline"],
    "Rosy_Cheeks":          ["a person without rosy cheeks", "a person with pale cheeks"],
    "Sideburns":            ["a clean-shaven person", "a person without sideburns"],
    "Smiling":              ["a person with a neutral expression", "a person who is not smiling", "a serious-looking person", "a person with a straight face"],
    "Straight_Hair":        ["a person with curly hair", "a person without straight hair"],
    "Wavy_Hair":            ["a person with straight hair", "a person without wavy hair"],
    "Wearing_Earrings":     ["a person without earrings", "a person not wearing any earrings"],
    "Wearing_Hat":          ["a person without a hat", "a person not wearing a hat", "a bare-headed person"],
    "Wearing_Lipstick":     ["a person without lipstick", "a person with bare lips"],
    "Wearing_Necklace":     ["a person without a necklace", "a person with a bare neck"],
    "Wearing_Necktie":      ["a person without a necktie", "a person with an open collar"],
    "Young":                ["an old person", "an elderly person", "an older person", "a senior person"],
}


#==============================================================================
# Cell  71 [code] - CLIP ImageNet-style prompt templates
#==============================================================================

# CLIP's official ImageNet templates, article-stripped for the {description} slot
clip_imagenet_templates = [
    "a bad photo of {description}.", "a photo of many {description}.", "a sculpture of {description}.",
    "a photo of the hard to see {description}.", "a low resolution photo of {description}.", "a rendering of {description}.",
    "graffiti of {description}.", "a cropped photo of {description}.", "a tattoo of {description}.",
    "the embroidered {description}.", "a photo of a hard to see {description}.", "a bright photo of {description}.",
    "a photo of a clean {description}.", "a photo of a dirty {description}.", "a dark photo of {description}.",
    "a drawing of {description}.", "a photo of my {description}.", "the plastic {description}.",
    "a photo of the cool {description}.", "a close-up photo of {description}.", "a black and white photo of {description}.",
    "a painting of {description}.", "a pixelated photo of {description}.", "a plastic {description}.",
    "a photo of the dirty {description}.", "a jpeg corrupted photo of {description}.", "a blurry photo of {description}.",
    "a photo of {description}.", "a good photo of {description}.", "a {description} in a video game.",
    "a photo of one {description}.", "a doodle of {description}.", "the origami {description}.",
    "a sketch of {description}.", "a origami {description}.", "the toy {description}.",
    "a rendition of {description}.", "a photo of the clean {description}.", "a photo of a large {description}.",
    "a photo of a nice {description}.", "a photo of a weird {description}.", "a cartoon {description}.",
    "art of {description}.", "a embroidered {description}.", "itap of {description}.",
    "a plushie {description}.", "a photo of the nice {description}.", "a photo of the small {description}.",
    "a photo of the weird {description}.", "the cartoon {description}.", "a photo of the large {description}.",
    "the plushie {description}.", "a toy {description}.", "a photo of a cool {description}.",
    "a photo of a small {description}.",
]
portrait_templates = [
    "a portrait of {description}.",
    "a portrait photograph of {description}.",
    "a closeup headshot of {description}.",
    "a candid photo of {description}.",
    "a studio portrait of {description}.",
    "a high-resolution headshot of {description}.",
    "a face photo of {description}.",
    "a photo showing the face of {description}.",
    "a frontal photo of {description}.",
    "a clear photo of {description}.",
]
prompt_templates = clip_imagenet_templates + portrait_templates


@torch.no_grad()
def _encode_descriptions_through_templates(descriptions: list[str], templates: list[str]) -> torch.Tensor:
    """Encode every (description x template) pair, mean-pool the embeddings, and re-normalize.

    Each pair is L2-normalized before pooling; batched in a single processor/model
    call for speed.

    Args:
        descriptions: Attribute descriptions to expand across templates.
        templates: Prompt templates with a `{description}` placeholder.

    Returns:
        A single (D,) L2-normalized ensemble embedding.
    """
    # Build the full prompt list as description x template
    prompts = [template.format(description=description) for description in descriptions for template in templates]
    # Encode all prompts in one batch
    embs = encode_texts(prompts, device)   # (P, D), per-row normalized
    mean_emb = embs.mean(dim=0)
    return mean_emb / mean_emb.norm()


@torch.no_grad()
def precompute_attribute_description_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-attribute positive and negative text-embedding banks.

    E_pos[i] = ensemble over (positive descriptions for attribute i) x (templates)
    E_neg[i] = ensemble over (negative descriptions for attribute i) x (templates)

    Returns:
        An (E_pos, E_neg) pair, each (40, 512) and L2-normalized.
    """
    pos_embs, neg_embs = [], []
    for attribute_name in get_attributes():
        # Retrieve the positive and negative natural-language descriptions for this attribute
        pos_descriptions = attribute_descriptions_pos[attribute_name]
        neg_descriptions = attribute_descriptions_neg[attribute_name]
        # Encode each description through the prompt templates, mean-pool, and L2-normalize
        pos_embs.append(_encode_descriptions_through_templates(pos_descriptions, prompt_templates))
        neg_embs.append(_encode_descriptions_through_templates(neg_descriptions, prompt_templates))
    E_pos = torch.stack(pos_embs, dim=0)
    E_neg = torch.stack(neg_embs, dim=0)
    return E_pos, E_neg


print("Precomputing pos/neg attribute embeddings with the expanded template bank (this may take a minute)...")
E_POS, E_NEG = precompute_attribute_description_embeddings()
E_POS = E_POS.to(gallery_embeddings.device)
E_NEG = E_NEG.to(gallery_embeddings.device)
print(f"E_POS: {tuple(E_POS.shape)},  E_NEG: {tuple(E_NEG.shape)}")


#==============================================================================
# Cell  73 [code] - Evaluate & plot Prompt Ensembling
#==============================================================================

# Same scoring layer and weights as Source-Attribute Matching - only the embedding bank changes.
evaluation_results_promptens, average_results_per_query_promptens = evaluate_and_average(
    annotations,
    attribute_matching_scorer(gallery_embeddings, E_POS, E_NEG, **SAM_WEIGHTS),
    verbose=False,
)
print(f"Prompt Ensembling: mean Recall@10 = {mean_recall_at_10(evaluation_results_promptens):.4f}")


plot_methods_comparison(
    {
        "Baseline":                  average_results_per_query_baseline,
        "Source-Attribute Matching": average_results_per_query_sam,
        "Prompt Ensembling":         average_results_per_query_promptens,
    },
    title="Training-Free Method Comparison — per-query Recall@K and Precision@K",
)


#==============================================================================
# Cell  77 [code] - Cross-Attention Fusion hyperparameters
#==============================================================================

CA_HEADS          = 4          # cross-attention heads
CA_LAYERS         = 2          # stacked cross-attention (transformer decoder) layers
CA_FFN_MULT       = 2          # transformer FFN hidden size = CA_FFN_MULT * dim
CA_DROPOUT        = 0.1        # dropout inside the transformer layers
CA_GROUND_LAYERS  = 1          # patch-grounding decoder layers (conditions read the visual tokens)
CA_GROUND_HEADS   = 4          # attention heads in the grounding decoder
CLIP_VIS_DIM      = 768        # CLIP ViT-B/32 hidden width of the visual tokens (CLS + 49 patches)
CA_TRAIN_TRIPLETS = 100_000    # synthetic training triplets (own pool)
CA_VAL_TRIPLETS   = 2_000      # synthetic validation triplets
CA_EPOCHS         = 20         # training epochs
CA_BATCH          = 512        # mini-batch size
CA_LR             = 2e-4       # AdamW learning rate
CA_WD             = 1e-2       # AdamW weight decay

CA_FILM_SIGN_STD  = 0.02       # FiLM sign-embedding init std
CA_GATE_BIAS_INIT = 2.0        # gated-residual gate-open bias; sigmoid(2.0)≈0.88 keeps gate open
CA_SEED           = 0          # init seed, re-applied before every build so the full model and
                               # the ablation variants below differ only in architecture

# Ablation variants: each disables exactly one component of the fusion module. Consumed by the
# ablation study cells further down; the full model passes no flags at all
ABLATION_VARIANTS = {
    "− sign-aware FiLM":   {"use_film":   False},
    "− patch grounding":   {"use_ground": False},
    "− cross-attention":   {"use_attn":   False},
    "− residual gate":     {"use_gate":   False},
}


#==============================================================================
# Cell  82 [code] - class CrossAttentionFusion(nn.Module)
#==============================================================================

class CrossAttnPoolLayer(nn.Module):
    """Pre-norm layer: cross-attend `tgt` over `memory`, then apply a feed-forward block."""

    def __init__(self, dim: int, n_heads: int, dim_feedforward: int, dropout: float = 0.1):
        """Build one cross-attention-only pooling layer.

        Args:
            dim: Embedding dimension D.
            n_heads: Number of cross-attention heads.
            dim_feedforward: Hidden size of the feed-forward sub-block.
            dropout: Dropout probability used after attention and inside the FFN.
        """
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Cross-attend `tgt` over `memory`, then apply the feed-forward block.

        Args:
            tgt: (B, Tq, D) query tokens.
            memory: (B, Tm, D) keys/values to attend over.
            memory_key_padding_mask: (B, Tm) bool mask, True marks a padded key to ignore.

        Returns:
            A (B, Tq, D) tensor, same shape as `tgt`.
        """
        x = tgt
        attended, _ = self.cross_attn(
            self.norm1(x), memory, memory,
            key_padding_mask=memory_key_padding_mask, need_weights=False,
        )
        x = x + self.dropout1(attended)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class CrossAttnPoolDecoder(nn.Module):
    """Stack of `CrossAttnPoolLayer`s, each attending over the same `memory`."""

    def __init__(self, dim: int, n_heads: int, n_layers: int, dim_feedforward: int, dropout: float = 0.1):
        """Build the layer stack.

        Args:
            dim: Embedding dimension D.
            n_heads: Number of cross-attention heads per layer.
            n_layers: Number of stacked `CrossAttnPoolLayer`s.
            dim_feedforward: Hidden size of each layer's feed-forward sub-block.
            dropout: Dropout probability passed to every layer.
        """
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnPoolLayer(dim, n_heads, dim_feedforward, dropout) for _ in range(n_layers)
        ])

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Run `tgt` through every stacked layer, each attending over the same `memory`.

        Args:
            tgt: (B, Tq, D) query tokens.
            memory: (B, Tm, D) keys/values to attend over.
            memory_key_padding_mask: (B, Tm) bool mask, True marks a padded key to ignore.

        Returns:
            A (B, Tq, D) tensor, same shape as `tgt`.
        """
        x = tgt
        for layer in self.layers:
            x = layer(x, memory, memory_key_padding_mask=memory_key_padding_mask)
        return x


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: the source image queries its sign-tagged conditions.

    The attended result is fused back onto the image embedding. See the
    "Training-Based Method: Cross-Attention Fusion" section of the report for the
    full architecture rationale.
    """

    def __init__(self, attr_name_embs: torch.Tensor, dim: int, n_heads: int = 4,
                 n_layers: int = 2, ffn_mult: int = 2, dropout: float = 0.1,
                 film_sign_std: float = 0.02, gate_bias_init: float = 2.0,
                 clip_dim: int = 768, ground_layers: int = 1, ground_heads: int = 4,
                 *, use_film: bool = True, use_ground: bool = True,
                 use_attn: bool = True, use_gate: bool = True):
        """Build the cross-attention fusion module.

        Args:
            attr_name_embs: (n_attrs, D) frozen cleaned attribute-name CLIP text bank.
            dim: Embedding dimension D.
            n_heads: Number of attention heads per decoder layer.
            n_layers: Number of Transformer-decoder layers.
            ffn_mult: Feed-forward hidden-size multiplier (hidden = ffn_mult * dim).
            dropout: Dropout probability used in attention and the FFN/heads.
            film_sign_std: Std of the normal init for the FiLM sign-embedding table.
            gate_bias_init: Initial gated-residual gate bias (sigmoid(bias) is the gate at step 0).
            clip_dim: Hidden width of CLIP's raw visual tokens (768 for ViT-B/32).
            ground_layers: Number of patch-grounding Transformer-decoder layers.
            ground_heads: Number of attention heads in the grounding decoder.
            use_film: Keep sign-aware FiLM. When False, conditions are the frozen attribute
                vectors multiplied by their raw sign (plain arithmetic negation).
            use_ground: Keep patch grounding. When False, conditions never read the visual tokens.
            use_attn: Keep the stacked cross-attention. When False, the conditions are pooled by
                an unweighted masked mean, i.e. the static query-agnostic fusion this project
                argues against.
            use_gate: Keep the per-dimension residual gate. When False the gate is fixed to 1,
                leaving a plain residual (the zero-init identity property is preserved either way).

        The four ``use_*`` flags exist for the ablation study; all default to True, so the full
        model is the plain constructor call.
        """
        super().__init__()
        self.use_film, self.use_ground = use_film, use_ground
        self.use_attn, self.use_gate = use_attn, use_gate

        # Frozen cleaned attribute-name CLIP text bank, indexed per condition in forward()
        self.register_buffer("attr_name", attr_name_embs)  # (n_attrs, D)

        if use_film:
            # Sign embedding: each sign (+/-) gets a learned vector for FiLM modulation
            self.sign_embed = nn.Embedding(2, dim)              # weight: (2, D)   [0: +, 1: -]
            nn.init.normal_(self.sign_embed.weight, std=film_sign_std)

            # Sign-conditioned FiLM: maps a sign embedding (D) to a per-dim scale + shift (2D),
            # letting "+" and "-" modulate the same frozen attribute vector in opposite, learned ways
            self.film = nn.Linear(dim, 2 * dim)                 # (B, T, D) -> (B, T, 2D)
            nn.init.zeros_(self.film.weight)    # gamma
            nn.init.zeros_(self.film.bias)      # beta

        if use_ground:
            # Patch grounding: project frozen CLIP visual tokens back into the fusion dimension
            self.vis_proj = nn.Linear(clip_dim, dim)             # (B, 50, clip_dim) -> (B, 50, D)
            self.vis_type = nn.Embedding(2, dim)                 # weight: (2, D)   [0: CLS, 1: patch]
            nn.init.normal_(self.vis_type.weight, std=film_sign_std)   # same small init std as the sign table
            ground_layer = nn.TransformerDecoderLayer(
                dim, ground_heads, dim_feedforward=ffn_mult * dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
            self.ground = nn.TransformerDecoder(ground_layer, num_layers=ground_layers)
            # tgt: (B, T, D) conditions, memory: (B, 50, D) grounded visual tokens -> (B, T, D)

        if use_attn:
            # Stacked cross-attention: image (1 query token) attends over the grounded conditions
            self.decoder = CrossAttnPoolDecoder(dim, n_heads, n_layers, ffn_mult * dim, dropout)
            # tgt: (B, 1, D) image query, memory: (B, T, D) conditions -> (B, 1, D)

        # Gated residual head: a sigmoid gate weighs a non-linear delta added back onto v_ref
        self.delta = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )                                                   # (B, 2D) -> (B, D)
        if use_gate:
            self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())
                                                                # (B, 2D) -> (B, D), in (0, 1)

        # Zero-init the delta head's last layer so the module is exactly the identity map at
        # step 0 (out = v_ref, matching the FiLM identity init); its weights still receive
        # gradient because their inputs are non-zero. This holds for every ablation variant:
        # a zero delta is the identity whether or not it is gated
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

        if use_gate:
            # Gate starts open (sigmoid(2) ~ 0.88) so the delta head is not doubly suppressed at init
            nn.init.constant_(self.gate[0].bias, gate_bias_init)

    def forward(self, img_emb: torch.Tensor, vis_tokens: torch.Tensor,
                cond_attr: torch.Tensor, cond_sign: torch.Tensor) -> torch.Tensor:
        """Fuse the source image with its sign-tagged conditions via cross-attention.

        Args:
            img_emb: (B, D) L2-normalized source image embeddings.
            vis_tokens: (B, 50, clip_dim) raw CLIP visual tokens (CLS at position 0, then
                49 patches) for the source.
            cond_attr: (B, T) attribute indices for each condition.
            cond_sign: (B, T) signs in {+1, -1, 0}; 0 marks padding.

        Returns:
            A (B, D) L2-normalized fused embedding.
        """
        pad_mask = cond_sign == 0             # (B, T) bool, True = ignore
        sign_id  = (cond_sign < 0).long()     # (B, T) in {0, 1}; padding -> 0 (masked anyway)
        attr     = self.attr_name[cond_attr]  # (B, T, D) frozen text

        # 1. Sign-aware FiLM: look up each condition's sign embedding, then run it through a
        # single Linear(D, 2D) whose output is split into a per-dimension scale (gamma) and
        # shift (beta). This lets "+" and "-" modulate the same frozen attribute text vector
        # in opposite, learned ways instead of just negating it.
        if self.use_film:
            sign_emb    = self.sign_embed(sign_id)                    # (B, T, D)
            gamma, beta = self.film(sign_emb).chunk(2, dim=-1)        # (B, T, D) each
            conds       = (1.0 + gamma) * attr + beta                 # (B, T, D)
        else:
            # Ablation: plain arithmetic negation, the geometric mirror point FiLM replaces
            conds = cond_sign.unsqueeze(-1).to(attr.dtype) * attr     # (B, T, D)

        # 2. Patch grounding: project the source's visual tokens, tag CLS vs patch, and let the
        # conditions self-attend (co-adapt, tempering contradictory edits) and cross-attend
        # (ground spatially) over them.
        if self.use_ground:
            V = self.vis_proj(vis_tokens.to(self.vis_proj.weight.dtype))         # (B, 50, D)
            type_id = torch.ones(V.shape[1], dtype=torch.long, device=V.device)  # (50,)
            type_id[0] = 0                                                       # position 0 is the CLS token
            V = V + self.vis_type(type_id)                                       # (B, 50, D), broadcast over batch
            conds = self.ground(conds, V, tgt_key_padding_mask=pad_mask)         # (B, T, D) grounded

        # 3. Stacked cross-attention: the image (1 query token) reads the grounded conditions
        if self.use_attn:
            q        = img_emb.unsqueeze(1)                                      # (B, 1, D)
            attended = self.decoder(q, conds, memory_key_padding_mask=pad_mask)  # (B, 1, D)
            attended = attended.squeeze(1)                                       # (B, D)
        else:
            # Ablation: unweighted masked mean. Every condition gets the same weight and the
            # image has no say in the pooling, which is exactly the static fusion of CLAY
            keep    = (~pad_mask).unsqueeze(-1).to(conds.dtype)                  # (B, T, 1)
            n_keep  = keep.sum(dim=1).clamp(min=1.0)                             # (B, 1)
            attended = (conds * keep).sum(dim=1) / n_keep                        # (B, D)

        # 4. Gated-residual fusion: v_ref preserved by default, delta can add or subtract
        fused = torch.cat([img_emb, attended], dim=-1)   # (B, 2D)
        delta = self.delta(fused)                        # (B, D)
        # Ablation (use_gate=False): gate fixed to 1, leaving a plain residual
        gate  = self.gate(fused) if self.use_gate else 1.0
        out   = img_emb + gate * delta                   # (B, D)
        return F.normalize(out, dim=-1)                  # (B, D), L2-normalized


def build_ca_model(**flags) -> CrossAttentionFusion:
    """Build a cross-attention fusion model from the module-level hyperparameters.

    Reseeds immediately before construction so the full model and every ablation variant start
    from the same initialisation draw and differ only in the component being ablated.

    Args:
        **flags: Optional ``use_film`` / ``use_ground`` / ``use_attn`` / ``use_gate`` overrides.
            Passing none builds the full model.

    Returns:
        The constructed module, moved onto `device`.
    """
    torch.manual_seed(CA_SEED)
    return CrossAttentionFusion(
        get_attribute_name_embeddings(device), gallery_embeddings.shape[1],
        n_heads=CA_HEADS, n_layers=CA_LAYERS, ffn_mult=CA_FFN_MULT, dropout=CA_DROPOUT,
        film_sign_std=CA_FILM_SIGN_STD, gate_bias_init=CA_GATE_BIAS_INIT,
        clip_dim=CLIP_VIS_DIM, ground_layers=CA_GROUND_LAYERS, ground_heads=CA_GROUND_HEADS,
        **flags,
    ).to(device)


# Instantiate the cross-attention fusion model
ca_model = build_ca_model()


#==============================================================================
# Cell  84 [code] - Parameter count & forward self-check  (NEW - add to notebook)
#==============================================================================

n_params = sum(p.numel() for p in ca_model.parameters() if p.requires_grad)
print(f"Cross-Attention trainable parameters: {n_params:,}")

# Forward self-check: shapes and unit-norm output (cheap correctness gate). Run over the full
# model and every ablation variant, so a variant that silently breaks the contract (wrong shape,
# un-normalized output, or a broken identity init that would hand it a different starting point
# from the full model) is caught here rather than showing up as a bogus ablation number
_b    = 4
_img  = F.normalize(torch.randn(_b, gallery_embeddings.shape[1], device=device), dim=-1)
_vis  = torch.randn(_b, 50, CLIP_VIS_DIM, device=device)
_attr = torch.randint(0, len(get_attributes()), (_b, 3), device=device)
_sign = torch.tensor([[1, -1, 0], [1, 0, 0], [-1, -1, 1], [1, 1, -1]], device=device)

for _name, _flags in [("full model", {}), *ABLATION_VARIANTS.items()]:
    _m = ca_model if not _flags else build_ca_model(**_flags)
    _m.eval()
    with torch.no_grad():
        _out = _m(_img, _vis, _attr, _sign)

    assert _out.shape == (_b, gallery_embeddings.shape[1]), (_name, _out.shape)
    assert torch.allclose(_out.norm(dim=-1), torch.ones(_b, device=device), atol=1e-5), _name
    # Zero-init delta head -> the untrained module must be exactly the identity map
    assert torch.allclose(_out, _img, atol=1e-5), _name

    _n = sum(p.numel() for p in _m.parameters() if p.requires_grad)
    print(f"Forward self-check passed: {_name:<20} {_n:>10,} params")

ca_model.train()


#==============================================================================
# Cell  89 [code] - def find_hard_negative(labels_bool, source_labels, pos_idx, neg_idx, ...)
#==============================================================================

def find_hard_negative(labels_bool: torch.Tensor, source_labels: torch.Tensor,
                       pos_idx: list[int], neg_idx: list[int],
                       source_idx: int, rng) -> int:
    """Return the index of a hard negative for a source and query.

    A hard negative breaks the query on exactly one sampled attribute while staying
    within HAMMING_BUDGET of that violated ideal profile, so it resembles a valid
    target in every other respect. Returns -1 when the query is empty or no
    candidate qualifies.

    Args:
        labels_bool: (N, n_attrs) boolean candidate label matrix.
        source_labels: Boolean attribute-label vector of the source image.
        pos_idx: Attribute indices the (satisfied) query would require present.
        neg_idx: Attribute indices the (satisfied) query would require absent.
        source_idx: Index of the source image, excluded from candidates.
        rng: Random generator used to pick the violated attribute and the negative.

    Returns:
        The index of a sampled hard negative, or -1 if none exists.
    """
    queried = pos_idx + neg_idx
    if not queried:
        return -1

    # Compute the ideal target attribute vector for this source and query
    target = desired_target_labels(source_labels, pos_idx, neg_idx)
    # Randomly select one of the queried attributes to violate
    j = int(rng.choice(queried))
    violated = target.clone()
    violated[j] = ~violated[j] # Break the query on attribute j

    # Filter to candidates that violate the query on attribute j and are within HAMMING_BUDGET of the source
    ok = (labels_bool[:, queried] == violated[queried].unsqueeze(0)).all(dim=1)
    ok &= (labels_bool != violated.unsqueeze(0)).sum(dim=1) <= HAMMING_BUDGET
    # Exclude the source image itself from the candidates
    cand = ok.nonzero(as_tuple=True)[0]
    cand = cand[cand != source_idx]
    if cand.numel() == 0:
        return -1
    return int(cand[int(rng.integers(0, cand.numel()))])


#==============================================================================
# Cell  91 [code] - def build_condition_row(pos_idx: list[int], neg_idx: list[int], width: int) -…
#==============================================================================

def build_condition_row(pos_idx: list[int], neg_idx: list[int], width: int) -> tuple[list[int], list[int]]:
    """Build fixed-width attribute and sign condition rows for a signed query.

    Rows are padded with 0 to `width` so all triplets fit in fixed-width tensors for
    efficient batch processing.

    Args:
        pos_idx: Attribute indices with a "+" sign.
        neg_idx: Attribute indices with a "-" sign.
        width: Fixed row width to pad to.

    Returns:
        An (attrs_row, signs_row) pair, each a length-`width` list padded with 0.
    """
    attrs = pos_idx + neg_idx
    signs = [1] * len(pos_idx) + [-1] * len(neg_idx)
    pad = width - len(attrs)
    return attrs + [0] * pad, signs + [0] * pad


def generate_triplet_pool(n_triplets: int, seed: int, log_every: int = 5000):
    """Sample a pool of (source, target, cond_attr, cond_sign, hard) triplet rows.

    Args:
        n_triplets: Number of triplet rows to sample.
        seed: Seed for the random generator.
        log_every: Print progress every this many sampled rows.

    Returns:
        A tuple of tensors (src, tgt, cond_attr, cond_sign, hard), each with n_triplets rows.
    """
    rng = np.random.default_rng(seed)
    src, tgt, cond_attr, cond_sign, hard = [], [], [], [], []
    while len(src) < n_triplets:
        # Sample a source and a query that actually *changes* it: queried positives are
        # attributes the source lacks, queried negatives are attributes it already has
        s = int(rng.integers(0, TRAIN_N))
        a_np = train_labels_bool_np[s]
        n_terms = int(rng.integers(1, MAX_TERMS + 1))
        attrs = [int(j) for j in rng.choice(n_attrs, size=n_terms, replace=False)]
        pos_idx = [j for j in attrs if not a_np[j]]
        neg_idx = [j for j in attrs if a_np[j]]

        # Find a valid target satisfying the query within the Hamming budget (resample if none)
        candidates = find_valid_targets(train_labels_bool, train_labels_bool[s], pos_idx, neg_idx)
        candidates = candidates[candidates != s]
        if candidates.numel() == 0:
            continue
        t = int(candidates[int(rng.integers(0, candidates.numel()))])

        # Mine one constraint-violating hard negative for this query (-1 if none exists)
        h = find_hard_negative(train_labels_bool, train_labels_bool[s], pos_idx, neg_idx, s, rng)

        # Pad the condition to fixed width and record the triplet row
        attrs_row, signs_row = build_condition_row(pos_idx, neg_idx, MAX_TERMS)
        src.append(s)
        tgt.append(t)
        cond_attr.append(attrs_row)
        cond_sign.append(signs_row)
        hard.append(h)
        if len(src) % log_every == 0:
            print(f"  {len(src)}/{n_triplets} triplets")
    return (
        torch.tensor(src), torch.tensor(tgt),
        torch.tensor(cond_attr), torch.tensor(cond_sign),
        torch.tensor(hard),
    )


#==============================================================================
# Cell  93 [code] - def load_or_generate_triplets(n_triplets: int, seed: int, cache_path: str)
#==============================================================================

def load_or_generate_triplets(n_triplets: int, seed: int, cache_path: str | Path):
    """Load a cached triplet pool if its generation key matches, else generate and cache it.

    The pool is fully determined by (seed, n_triplets, MAX_TERMS, HAMMING_BUDGET) and the train
    labels; that key is stored alongside the tensors so a stale pool can never be silently reused
    after those constants change. Mirrors get_encoded_dataset's feature-cache pattern, and lets a
    re-train (e.g. tweaking model/optimizer hyperparameters) skip the triplet synthesis entirely.

    Args:
        n_triplets: Number of triplet rows the pool should contain.
        seed: Seed used to generate the pool.
        cache_path: Path to load the pool from / save it to.

    Returns:
        A tuple of triplet tensors (src, tgt, cond_attr, cond_sign, hard).
    """
    key = {"seed": seed, "n_triplets": n_triplets,
           "MAX_TERMS": MAX_TERMS, "HAMMING_BUDGET": HAMMING_BUDGET}
    cache_path = Path(cache_path)
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu")
        if blob.get("key") == key:
            print(f"Loaded {n_triplets} cached triplets (seed={seed}) from {cache_path}.")
            return tuple(blob["tensors"])
        print(f"Triplet cache {cache_path} key {blob.get('key')} != {key}; regenerating.")
    pool = generate_triplet_pool(n_triplets, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"key": key, "tensors": [t.cpu() for t in pool]}, cache_path)
    print(f"Saved {n_triplets} triplets (seed={seed}) to {cache_path}.")
    return pool


# --- Materialise the train/val triplet pools here, in the synthesis cell (cached and keyed on the
#     generation parameters). The training cell below just consumes these tensors. ---
CA_TRIPLETS_TRAIN_PATH = EVALUATION_CACHE_DIR / "cross_attn_triplets_train.pt"
CA_TRIPLETS_VAL_PATH   = EVALUATION_CACHE_DIR / "cross_attn_triplets_val.pt"
ca_trip_src, ca_trip_tgt, ca_trip_attr, ca_trip_sign, ca_trip_hard = load_or_generate_triplets(
    CA_TRAIN_TRIPLETS, 10, CA_TRIPLETS_TRAIN_PATH)
ca_val_src,  ca_val_tgt,  ca_val_attr,  ca_val_sign,  ca_val_hard  = load_or_generate_triplets(
    CA_VAL_TRIPLETS, 11, CA_TRIPLETS_VAL_PATH)
print(f"train triplets: {ca_trip_src.shape[0]}, val triplets: {ca_val_src.shape[0]}")
print(f"hard negatives found for {(ca_trip_hard >= 0).float().mean().item():.1%} of training triplets")


# --- Patch tokens for the unique train sources the triplets reference. Extracting only these
#     (rather than the whole train split) bounds storage (see build_patch_bank). ---
CA_TRAIN_PATCHES_PATH = EVALUATION_CACHE_DIR / "patches_train.pt"
ca_patch_src = torch.unique(torch.cat([ca_trip_src, ca_val_src])).tolist()
train_patches_for = build_patch_bank(celeba_train, device, CA_TRAIN_PATCHES_PATH, ca_patch_src, TRAIN_N)
print(f"train patch bank: {len(ca_patch_src)} unique sources")


#==============================================================================
# Cell  95 [code] - Cross-Attention Fusion training setup
#==============================================================================

model, _ = get_CLIP_model()
logit_scale_value = model.logit_scale.exp().detach()   # Frozen CLIP temperature for InfoNCE

# Move the synthesised train/val triplet tensors onto the device once. Cheap and harmless even
# when the cell below loads a cached checkpoint and skips training
ca_trip_src_dev, ca_trip_tgt_dev = ca_trip_src.to(device), ca_trip_tgt.to(device)
ca_trip_attr_dev, ca_trip_sign_dev = ca_trip_attr.to(device), ca_trip_sign.to(device)
ca_trip_hard_dev = ca_trip_hard.to(device)
ca_val_src_dev, ca_val_tgt_dev = ca_val_src.to(device), ca_val_tgt.to(device)
ca_val_attr_dev, ca_val_sign_dev = ca_val_attr.to(device), ca_val_sign.to(device)
ca_val_hard_dev = ca_val_hard.to(device)


def in_batch_valid_target_mask(src_idx: torch.Tensor, tgt_idx: torch.Tensor,
                               cond_attr: torch.Tensor, cond_sign: torch.Tensor,
                               labels_bool: torch.Tensor) -> torch.Tensor:
    """Mark which in-batch targets are valid targets for each row's (source, query) pair.

    Applies find_valid_targets' ground-truth rule (query satisfied AND within
    HAMMING_BUDGET of the ideal target vector) to every (row, batch target) pair at once.
    ca_infonce_loss uses this to drop *false negatives* from its denominator: with
    Hamming-close CelebA faces and large batches, another row's target regularly happens
    to satisfy row i's query too, and pushing the fused query away from a genuinely
    correct candidate would directly fight the retrieval objective.

    Args:
        src_idx: (B,) source image indices per row, into `labels_bool`.
        tgt_idx: (B,) positive-target image indices per row, into `labels_bool`.
        cond_attr: (B, T) attribute indices for each condition.
        cond_sign: (B, T) signs in {+1, -1, 0}; 0 marks padding.
        labels_bool: (N, n_attrs) boolean label matrix the indices refer to.

    Returns:
        A (B, B) boolean mask, True at (i, j) iff batch target j is a valid target for
        row i. The diagonal (each row's own positive) is True by construction.
    """
    active = cond_sign != 0                                            # (B, T)
    want   = cond_sign > 0                                             # (B, T)

    # Ideal target vector per row: source labels with the queried attributes forced on/off
    # (vectorized desired_target_labels)
    desired = labels_bool[src_idx].clone()                             # (B, A)
    rows = torch.arange(desired.shape[0], device=desired.device).unsqueeze(1).expand_as(cond_attr)
    desired[rows[active], cond_attr[active]] = want[active]

    # Query satisfaction: target j must match row i's sign on every non-padded condition
    tgt_labels = labels_bool[tgt_idx]                                  # (B, A)
    at_queried = tgt_labels[:, cond_attr]                              # (B_tgt, B_row, T)
    satisfied  = ((at_queried == want.unsqueeze(0)) | ~active.unsqueeze(0)).all(dim=-1).T

    # Pairwise Hamming distance to the ideal target, XOR written as a float matmul
    d, l = desired.float(), tgt_labels.float()
    hamming = d @ (1.0 - l).T + (1.0 - d) @ l.T                        # (B_row, B_tgt)
    return satisfied & (hamming <= HAMMING_BUDGET)


def ca_infonce_loss(q: torch.Tensor, tgt_idx: torch.Tensor, hard_idx: torch.Tensor,
                    embeddings: torch.Tensor, logit_scale: torch.Tensor,
                    valid_target_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Compute the InfoNCE loss over in-batch targets plus one mined hard negative per row.

    Args:
        q: (B, D) fused query embeddings.
        tgt_idx: (B,) indices of the positive target per row, into `embeddings`.
        hard_idx: (B,) mined hard-negative indices per row; negative marks "none".
        embeddings: (N, D) L2-normalized embedding bank the indices refer to.
        logit_scale: Scalar temperature multiplier (frozen CLIP logit scale).
        valid_target_mask: Optional (B, B) mask from in_batch_valid_target_mask; True
            off-diagonal entries are false negatives, excluded from the denominator
            (each row's own positive on the diagonal always stays).

    Returns:
        The scalar InfoNCE loss.
    """
    t = embeddings[tgt_idx]
    no_hard = hard_idx < 0
    hf = embeddings[hard_idx.clamp(min=0)]                # (B, D); invalid rows masked below
    hard_sim = (q * hf).sum(-1, keepdim=True)             # (B, 1) per-row hard-negative score
    logits = logit_scale * torch.cat([q @ t.T, hard_sim], dim=1)   # (B, B+1)
    logits[:, -1] = logits[:, -1].masked_fill(no_hard, -1e9)
    if valid_target_mask is not None:
        false_neg = valid_target_mask.clone()
        false_neg.fill_diagonal_(False)
        logits[:, :-1] = logits[:, :-1].masked_fill(false_neg, -1e9)
    labels_ce = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels_ce)


@torch.no_grad()
def ca_val_loss(model: nn.Module) -> float:
    """Mean validation InfoNCE loss over the held-out triplets.

    Reuses the training objective (in-batch plus mined hard negatives, via ca_infonce_loss)
    so the train and validation loss curves are directly comparable. This is the metric used
    for checkpoint selection, and it is comparable across the ablation variants because they
    all share the same cached validation triplets.

    Args:
        model: The fusion module to score.

    Returns:
        The mean validation loss.
    """
    model.eval()
    loss_sum = 0.0
    n_val = ca_val_src_dev.shape[0]
    for start in range(0, n_val, CA_BATCH):
        sl = slice(start, min(start + CA_BATCH, n_val))
        q = model(train_embeddings[ca_val_src_dev[sl]], train_patches_for(ca_val_src_dev[sl]),
                  ca_val_attr_dev[sl], ca_val_sign_dev[sl])
        vt_mask = in_batch_valid_target_mask(ca_val_src_dev[sl], ca_val_tgt_dev[sl],
                                             ca_val_attr_dev[sl], ca_val_sign_dev[sl], train_labels_bool)
        loss_sum += float(ca_infonce_loss(q, ca_val_tgt_dev[sl], ca_val_hard_dev[sl],
                                          train_embeddings, logit_scale_value, vt_mask)) * q.shape[0]
    return loss_sum / n_val


#==============================================================================
# Cell  97 [code] - def plot_training_curve(history: dict, title: str = "Cross-Attention Training…
#==============================================================================

def plot_training_curve(history: dict, title: str = "Cross-Attention Training Curve"):
    """Plot the cross-attention learning curve as a single-axis loss figure.

    Shows the raw per-step training loss alongside the per-epoch validation loss
    (starting from a step-0 baseline). The best epoch (minimum validation loss) is starred.

    Args:
        history: Dict with per-step keys "step", "loss" and per-epoch keys
            "epoch_step", "val_loss", plus "best_epoch".
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(history["step"], history["loss"], color="tab:blue", linewidth=1, label="Train loss (per step)")
    ax.plot(history["epoch_step"], history["val_loss"], color="tab:orange", marker="o", linewidth=2, label="Val loss (per epoch)")

    best = history.get("best_epoch", -1)
    if 0 <= best < len(history["epoch_step"]):
        ax.scatter(history["epoch_step"][best], history["val_loss"][best], color="tab:orange",
                   marker="*", s=320, edgecolor="black", zorder=5,
                   label=f"Best (val loss={history['val_loss'][best]:.3f})")

    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("InfoNCE loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title)

    plt.tight_layout()
    plt.show()


#==============================================================================
# Cell  99 [code] - Train (or load cached) Cross-Attention model
#==============================================================================

CA_CKPT = EVALUATION_CACHE_DIR / "cross_attn_patch.pt"


def train_cross_attention(model: nn.Module, ckpt_path: Path, *, epochs: int = CA_EPOCHS,
                          plot: bool = True, label: str = "cross-attention") -> dict:
    """Train a fusion module with InfoNCE, or load it from cache if the checkpoint exists.

    Selects the minimum-validation-loss weights, writing the checkpoint on every improvement so
    an interrupted Colab run loses at most the epochs since the last improvement. Taking the
    module and checkpoint path as arguments lets the ablation variants below reuse this exact
    training procedure, which is what makes their scores comparable to the full model's.

    Args:
        model: The fusion module to train, modified in place; ends holding the best weights.
        ckpt_path: Where to read/write the checkpoint. An existing file short-circuits training.
        epochs: Number of training epochs.
        plot: Whether to draw the learning curve (off for the ablation runs, which would
            otherwise emit one figure per variant).
        label: Name used in progress messages.

    Returns:
        The training history dict, with an added "val_loss_best" key.
    """
    cached = torch.load(ckpt_path, map_location=device) if ckpt_path.exists() else None
    if cached is not None:
        model.load_state_dict(cached["state_dict"])
        model.eval()
        print(f"Loaded cached {label} from {ckpt_path} "
              f"(val loss={cached.get('val_loss', float('nan')):.4f}) — skipping training.")
        history = cached.get("history")
        if history is None:
            print("Cached checkpoint predates training history — re-train to regenerate the learning curve.")
            return {"val_loss_best": cached.get("val_loss", float("nan"))}
        if plot:
            plot_training_curve(history)
        return {**history, "val_loss_best": cached.get("val_loss", float("nan"))}

    # Reseed so every variant sees the same batch order and dropout draws as the full model:
    # without this the ablation deltas would carry run-to-run noise on top of the component effect
    torch.manual_seed(CA_SEED)

    # Optimizer / LR schedule.
    optimizer = torch.optim.AdamW(model.parameters(), lr=CA_LR, weight_decay=CA_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss, best_ca_state, best_epoch = float("inf"), None, -1
    n_train_trip = ca_trip_src_dev.shape[0]

    # Learning-curve history: per-step training loss and per-epoch validation loss.
    history = {"step": [], "loss": [], "epoch_step": [], "val_loss": [], "best_epoch": -1}
    step = 0

    def _save_ca_checkpoint():
        """Persist the current best checkpoint to `ckpt_path`."""
        torch.save(
            {"state_dict": {k: v.cpu() for k, v in best_ca_state.items()},
             "val_loss": best_val_loss, "history": history},
            ckpt_path,
        )

    # Baseline validation loss at step 0 (untrained model): plotted, but not a checkpoint candidate.
    history["epoch_step"].append(0)
    history["val_loss"].append(ca_val_loss(model))

    # Training loop: InfoNCE on the fused query vs target, keeping the min-val-loss weights.
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train_trip, device=device)
        for start in range(0, n_train_trip, CA_BATCH):
            idx = perm[start:start + CA_BATCH]
            q = model(train_embeddings[ca_trip_src_dev[idx]], train_patches_for(ca_trip_src_dev[idx]),
                      ca_trip_attr_dev[idx], ca_trip_sign_dev[idx])
            vt_mask = in_batch_valid_target_mask(ca_trip_src_dev[idx], ca_trip_tgt_dev[idx],
                                                 ca_trip_attr_dev[idx], ca_trip_sign_dev[idx], train_labels_bool)
            loss = ca_infonce_loss(q, ca_trip_tgt_dev[idx], ca_trip_hard_dev[idx],
                                   train_embeddings, logit_scale_value, vt_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            history["step"].append(step)
            history["loss"].append(float(loss.detach()))
        scheduler.step()

        # Per-epoch validation loss and best-checkpoint tracking (minimum val loss).
        val_loss = ca_val_loss(model)
        history["epoch_step"].append(step)
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, len(history["val_loss"]) - 1
            best_ca_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            _save_ca_checkpoint()
        history["best_epoch"] = best_epoch

        # Live, in-place redraw of the learning curve so training state is visible as it runs.
        if plot:
            clear_output(wait=True)
            plot_training_curve(history)
        print(f"[{label}] epoch {epoch+1:3d}/{epochs}  train loss={history['loss'][-1]:.4f}  "
              f"val loss={val_loss:.4f}  (best epoch {best_epoch})")

    # Restore best weights, draw the final figure, and cache weights + history.
    model.load_state_dict(best_ca_state)
    if plot:
        clear_output(wait=True)
        plot_training_curve(history)
    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    _save_ca_checkpoint()  # final history (post-loop) may differ from the last improvement's save
    print(f"Saved {label} to {ckpt_path}")
    return {**history, "val_loss_best": best_val_loss}


ca_history = train_cross_attention(ca_model, CA_CKPT)


#==============================================================================
# Cell 101 [code] - Cross-Attention gallery scorer
#==============================================================================

# Gallery visual tokens (CLS + 49 patches) for the source side of retrieval. The gallery *target*
# side stays the pooled, frozen ``gallery_embeddings``, so retrieval cost is unchanged; only the
# query computation reads patches. Only the benchmark's *source* images are ever queried this way
# (scorer/inspection both index by source_idx), so - mirroring the train-split bank above - we
# encode just that subset rather than all 19,962 test images (see build_patch_bank).
GALLERY_PATCHES_PATH = EVALUATION_CACHE_DIR / "patches_test.pt"
gallery_patch_src = sorted({src for ann in annotations for src in get_source_image_idxs(ann)})
gallery_patches_for = build_patch_bank(celeba, device, GALLERY_PATCHES_PATH,
                                       gallery_patch_src, gallery_embeddings.shape[0])


def query_to_condition_rows(text_query: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a benchmark query string into condition tensors for the model.

    Args:
        text_query: Comma-separated signed query, e.g. "+Bald, -Eyeglasses".

    Returns:
        A (cond_attr, cond_sign) pair of (1, T) tensors, padded as needed.
    """
    pos_idx, neg_idx = query_to_signed_indices(text_query)
    width = max(MAX_TERMS, len(pos_idx) + len(neg_idx))
    attrs, signs = build_condition_row(pos_idx, neg_idx, width)
    cond_attr = torch.tensor([attrs], device=gallery_embeddings.device)
    cond_sign = torch.tensor([signs], device=gallery_embeddings.device)
    return cond_attr, cond_sign


@torch.no_grad()
def fuse_source_query(model: nn.Module, source_idx: int,
                      cond_attr: torch.Tensor, cond_sign: torch.Tensor) -> torch.Tensor:
    """Run a trained fusion module on one (source image, query) pair.

    Shared by the evaluation scorer and the qualitative inspection (which additionally hooks the
    gate), so the fused-query computation for a single source is defined in exactly one place.
    The module is passed in rather than read from the global so the ablation variants below can
    be scored through this same path.

    Args:
        model: Trained CrossAttentionFusion module to fuse with.
        source_idx: Gallery index of the source image; must be a benchmark source (see
            `gallery_patches_for`).
        cond_attr: (1, T) attribute indices for each condition.
        cond_sign: (1, T) signs in {+1, -1, 0}; 0 marks padding.

    Returns:
        A (D,) L2-normalized fused query embedding.
    """
    idx = torch.tensor([source_idx], device=gallery_embeddings.device)
    return model(
        gallery_embeddings[idx], gallery_patches_for(idx), cond_attr, cond_sign
    ).squeeze(0)


def cross_attn_scorer(gallery_embeddings: torch.Tensor, model: nn.Module) -> Callable:
    """Build the scorer factory for Cross-Attention Fusion.

    Builds the fused query embedding once per annotation and returns gallery
    cosine similarities against it.

    Args:
        gallery_embeddings: (N, D) gallery image embeddings, L2-normalized per row.
        model: Trained CrossAttentionFusion model. Also used for the ablation variants, which
            are scored through this same factory.

    Returns:
        A ``make_scorer(annotation)`` factory consumed by evaluate().
    """
    model.eval()

    def make_scorer(annotation: dict) -> Callable:
        """Build a per-query scorer from the query's condition rows."""
        cond_attr, cond_sign = query_to_condition_rows(get_text_query(annotation))

        @torch.no_grad()
        def scorer(source_idx: int) -> torch.Tensor:
            """Score every gallery image against the fused source query embedding."""
            q = fuse_source_query(model, source_idx, cond_attr, cond_sign)
            return gallery_embeddings @ q

        return scorer
    return make_scorer


#==============================================================================
# Cell 103 [code] - Evaluate & plot Cross-Attention Fusion
#==============================================================================

evaluation_results_ca, average_results_per_query_ca = evaluate_and_average(
    annotations,
    cross_attn_scorer(gallery_embeddings, ca_model),
    verbose=True,
)
plot_metrics_across_k(
    average_results_per_query_ca,
    title="Cross-Attention Fusion — Performance across K",
)


#==============================================================================
# Cell 105 [code] - Qualitative attention inspection
#==============================================================================

@torch.no_grad()
def fuse_and_gate(source_idx: int, text_query: str) -> tuple[torch.Tensor, np.ndarray]:
    """Run the trained model on one (source image, query) and capture its residual gate.

    The gate is read with a forward hook so the trained weights are reused exactly.

    Args:
        source_idx: Gallery index of the source image.
        text_query: Comma-separated signed query, e.g. "+Bald, -Eyeglasses".

    Returns:
        A (q, gate) pair: the fused query embedding (D,) and the per-dimension residual
        gate (D,) in [0, 1].
    """
    cond_attr, cond_sign = query_to_condition_rows(text_query)
    captured = {}

    def grab_gate(module, inputs, output):
        """Forward hook that stashes the gate module's output."""
        captured["gate"] = output.detach()                   # (B, D)

    handle = ca_model.gate.register_forward_hook(grab_gate)
    ca_model.eval()
    try:
        q = fuse_source_query(ca_model, source_idx, cond_attr, cond_sign)
    finally:
        handle.remove()
    return q, captured["gate"][0].cpu().numpy()


def plot_cross_attention_inspection(
    source_idx: int,
    text_query: str,
    k: int = 5,
    label: str = "",
    ground_truth: set[int] | None = None,
) -> None:
    """Plot the source image and the top-k images the fused query retrieves, with a gate summary.

    Each retrieved image is annotated ✓/✗ for whether it satisfies the query attributes and, when
    ``ground_truth`` is given, tagged "GT" if it is an actual benchmark target - so a SUCCESS panel
    visibly contains a ground-truth hit and a FAILURE panel does not. ``label`` (e.g.
    "SUCCESS (negation)") is prepended to the source caption so the figure is self-describing in the
    report. This replaces the per-condition attention bar chart: for a single-term query that bar is a
    trivial 1.0 (one softmax key) and carries no signal; the retrieved set, in contrast, shows whether
    the edit actually pulled retrieval toward the requested attributes.

    Args:
        source_idx: Gallery index of the source image.
        text_query: Comma-separated signed query, e.g. "+Bald, -Eyeglasses".
        k: Number of top retrieved images to show.
        label: Optional caption prefix on the source image (e.g. "SUCCESS (composed)").
        ground_truth: Optional benchmark target indices for this (query, source); marks GT hits.
    """
    q, gate = fuse_and_gate(source_idx, text_query)

    sims = gallery_embeddings @ q
    sims[source_idx] = -1.0                                   # Exclude the source itself
    topk = sims.topk(k).indices.tolist()

    pos_idx, neg_idx = query_to_signed_indices(text_query)
    gallery_bool = gallery_labels > 0                       # (N, 40) bool
    gt = set(ground_truth or [])

    head = f"{label}\n" if label else ""
    images = [celeba[source_idx][0]]
    titles = [f"{head}source #{source_idx}\nquery: {text_query}"]
    colors = ["black"]
    for idx in topk:
        ok = bool(query_satisfied(gallery_bool[idx].unsqueeze(0), pos_idx, neg_idx).item())
        tag = "✓ satisfies" if ok else "✗ violates"
        if idx in gt:
            tag += "  ·  GT"
        images.append(celeba[idx][0])
        titles.append(f"#{idx}  cos={sims[idx].item():.2f}\n{tag}")
        colors.append("green" if ok else "crimson")
    plot_image_row(images, titles=titles, title_colors=colors, figsize=(3 * (k + 1), 3.2))

    print(f"Residual gate — mean {gate.mean():.3f}, median {np.median(gate):.3f}, "
          f"max {gate.max():.3f}  (0 = keep source dim, 1 = full edit)")


def find_success_failure_sources(annotation: dict, k: int = 5) -> tuple[int | None, int | None]:
    """Find one source whose top-k retrieval hits a ground-truth target (success) and one whose
    top-k misses every target (failure), under the trained Cross-Attention model.

    Success/failure is judged by the benchmark hit rule (Recall@k), so it matches how the method is
    scored rather than attribute satisfaction alone.

    Args:
        annotation: Benchmark annotation dict for a single query.
        k: Retrieval cutoff used to decide a hit (and shown in the panel).

    Returns:
        A (success_idx, failure_idx) pair; either is None if the query has no such source.
    """
    scorer = cross_attn_scorer(gallery_embeddings, ca_model)(annotation)
    success = failure = None
    for src in get_source_image_idxs(annotation):
        retrieved = retrieve_topk(scorer(src), exclude_idx=src, k=k)
        hit = len(set(retrieved) & set(get_target_indices(annotation, src))) > 0
        if hit and success is None:
            success = src
        if not hit and failure is None:
            failure = src
        if success is not None and failure is not None:
            break
    return success, failure


def first_query(annotations: list[dict], predicate: Callable) -> dict | None:
    """Return the first annotation whose (pos_idx, neg_idx) satisfy ``predicate``, else None.

    Args:
        annotations: Benchmark annotations.
        predicate: Called as ``predicate(pos_idx, neg_idx) -> bool``.

    Returns:
        The first matching annotation, or None.
    """
    for ann in annotations:
        pos_idx, neg_idx = query_to_signed_indices(get_text_query(ann))
        if predicate(pos_idx, neg_idx):
            return ann
    return None


# Inspect a SUCCESS and a FAILURE case for two query types the spec calls out: a
# single-attribute negation and a composed multi-attribute query.
INSPECT_K = 5
inspection_queries = [
    ("negation", first_query(annotations, lambda p, n: len(n) >= 1 and len(p) + len(n) == 1)),
    ("composed", first_query(annotations, lambda p, n: len(p) + len(n) >= 2)),
]

for kind, ann in inspection_queries:
    if ann is None:
        print(f"No {kind} query in the benchmark — skipping.")
        continue
    query = get_text_query(ann)
    success_idx, failure_idx = find_success_failure_sources(ann, k=INSPECT_K)
    print(f"\n===== {kind.upper()} query: '{query}' =====")
    for case, src in [("SUCCESS", success_idx), ("FAILURE", failure_idx)]:
        if src is None:
            print(f"  No {case.lower()} case among this query's sources.")
            continue
        plot_cross_attention_inspection(
            src, query, k=INSPECT_K,
            label=f"{case} ({kind})",
            ground_truth=set(get_target_indices(ann, src)),
        )


#==============================================================================
# Cell 106 [code] - Ablation study: train one variant per removed component
#==============================================================================

# Each variant disables exactly one component of the fusion module (see ABLATION_VARIANTS) and is
# retrained from scratch on the same cached triplet pool, from the same init seed, with the same
# optimizer and schedule as the full model. The only difference between two rows is therefore the
# component itself. Checkpoints are cached per variant, so re-running this cell is free
ablation_models, ablation_results, ablation_val_loss = {}, {}, {}

for variant_name, flags in ABLATION_VARIANTS.items():
    slug = "_".join(k.removeprefix("use_") for k in flags)      # e.g. "film" -> cross_attn_ablate_film.pt
    print(f"\n===== Ablation: {variant_name} =====")

    variant_model = build_ca_model(**flags)
    variant_history = train_cross_attention(
        variant_model, EVALUATION_CACHE_DIR / f"cross_attn_ablate_{slug}.pt",
        plot=False, label=variant_name,
    )

    evaluation_results_v, average_results_per_query_v = evaluate_and_average(
        annotations, cross_attn_scorer(gallery_embeddings, variant_model),
    )
    ablation_models[variant_name] = variant_model
    ablation_results[variant_name] = (evaluation_results_v, average_results_per_query_v)
    ablation_val_loss[variant_name] = variant_history["val_loss_best"]
    print(f"{variant_name}: mean Recall@10 = {mean_recall_at_10(evaluation_results_v):.3f}")


#==============================================================================
# Cell 107 [code] - Ablation study: results table and significance vs the full model
#==============================================================================

# Retrieval table: the full model on top, each variant below it
ablation_table = {"Cross-Attention (full)": average_results_per_query_ca}
ablation_table.update({name: avg for name, (_, avg) in ablation_results.items()})

plot_results_table(
    ablation_table,
    title="Ablation: mean Recall@K / Precision@K per removed component",
)

# Effect size and significance of each removal, measured against the full model. The benchmark is
# small, so a raw Recall@10 gap is easy to over-read; the McNemar test on paired per-source
# outcomes says whether a gap is larger than what resampling alone would produce
full_hits = recall10_hits(evaluation_results_ca)
full_val  = ca_history.get("val_loss_best", float("nan"))

print(f"Ablation vs the full model ({len(full_hits)} paired (query, source) outcomes)\n")
print(f"{'Removed component':<24} {'R@10':>6} {'ΔR@10':>7} {'val loss':>9} {'Δval':>7} "
      f"{'full>abl':>9} {'abl>full':>9} {'p-value':>9}")
print(f"{'(none: full model)':<24} {full_hits.mean():>6.3f} {'':>7} {full_val:>9.4f} "
      f"{'':>7} {'':>9} {'':>9} {'':>9}")

for variant_name, (evaluation_results_v, _) in ablation_results.items():
    hits_v = recall10_hits(evaluation_results_v)
    p, b, c = mcnemar_pvalue(full_hits, hits_v)
    val_v = ablation_val_loss[variant_name]
    print(f"{variant_name:<24} {hits_v.mean():>6.3f} {hits_v.mean() - full_hits.mean():>+7.3f} "
          f"{val_v:>9.4f} {val_v - full_val:>+7.4f} {b:>9} {c:>9} {p:>9.2g}")

print("\nΔ columns are variant minus full model: negative ΔR@10 and positive Δval loss both mean "
      "\nthe removed component was contributing. Parameter counts differ for the grounding and "
      "\ncross-attention rows, so those deltas mix mechanism with capacity.")


#==============================================================================
# Cell 108 [code] - Aggregate results across all methods
#==============================================================================

all_methods_results = {
    "Baseline":                  average_results_per_query_baseline,
    "Source-Attribute Matching": average_results_per_query_sam,
    "Prompt Ensembling":         average_results_per_query_promptens,
    "Cross-Attention":           average_results_per_query_ca,
}

plot_methods_comparison(
    all_methods_results,
    title="Final Method Comparison — per-query Recall@K and Precision@K",
)

plot_results_table(
    all_methods_results,
    title="Final Method Comparison — mean Recall@K / Precision@K",
)


#==============================================================================
# Cell 110 [code] - Paired significance tests (McNemar on per-source Recall@10)
#==============================================================================

method_hits = {
    "Baseline":                  recall10_hits(evaluation_results_baseline),
    "Source-Attribute Matching": recall10_hits(evaluation_results_sam),
    "Prompt Ensembling":         recall10_hits(evaluation_results_promptens),
    "Cross-Attention":           recall10_hits(evaluation_results_ca),
}

# Consecutive narrative pairs, plus the overall baseline-to-best comparison
method_pairs = list(zip(list(method_hits), list(method_hits)[1:]))
method_pairs.append(("Baseline", "Cross-Attention"))

n_pairs = len(next(iter(method_hits.values())))
print(f"McNemar test on paired per-source Recall@10 hits ({n_pairs} (query, source) pairs)\n")
print(f"{'Method A':<28} {'Method B':<28} {'R@10 A':>7} {'R@10 B':>7} {'A>B':>5} {'B>A':>5} {'p-value':>9}")
for name_a, name_b in method_pairs:
    hits_a, hits_b = method_hits[name_a], method_hits[name_b]
    p, b, c = mcnemar_pvalue(hits_a, hits_b)
    print(f"{name_a:<28} {name_b:<28} {hits_a.mean():>7.3f} {hits_b.mean():>7.3f} "
          f"{b:>5} {c:>5} {p:>9.2g}")
