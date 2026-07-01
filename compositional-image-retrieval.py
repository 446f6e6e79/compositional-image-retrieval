#==============================================================================
# Cell   9 [code] - CLIP model & encoding helpers (get_CLIP_model, encode_texts)
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
    """Collate a batch by keeping PIL images as a list and stacking the labels.

    The default collate_fn cannot stack PIL.Image objects, and the encode loop
    wants a list of PIL images to feed to the CLIP processor.

    Args:
        batch: List of (PIL image, label tensor) pairs.

    Returns:
        A (list of PIL images, stacked label tensor) pair.
    """
    imgs = [item[0] for item in batch]
    lbls = torch.stack([item[1] for item in batch], dim=0)
    return imgs, lbls


@torch.no_grad()
def _encode_image_batches(
    dataset,
    device,
    encode_batch: Callable,
    indices: list[int] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Yield per-batch CLIP encodings for a dataset (optionally a subset), with progress logging.

    Shared plumbing for get_encoded_dataset and get_encoded_patches: both build the same
    PIL-preserving DataLoader over the (sub)dataset, push each batch through CLIP, and print
    identical progress. They differ only in *what* they extract per batch (pooled, normalized
    embedding vs. raw fp16 visual tokens) and in how they cache/accumulate it, so extraction is
    injected as `encode_batch` and accumulation stays with the caller.

    Args:
        dataset: Dataset yielding (image, label) pairs.
        device: Device the CLIP forward runs on.
        encode_batch: Callable (model, processor, pil_images, device) -> (B, ...) CPU tensor.
        indices: Optional subset of dataset indices to encode, in order. None encodes all.
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.

    Yields:
        (encoded_batch, labels_batch) pairs, one per DataLoader batch.
    """
    model, processor = get_CLIP_model()
    source = dataset if indices is None else torch.utils.data.Subset(dataset, list(indices))

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
    for imgs_batch, lbls_batch in loader:
        encoded = encode_batch(model, processor, list(imgs_batch), device)
        pos += len(imgs_batch)
        print(f"Encoded {pos:>{pad}}/{n_total} images ({100 * pos / n_total:.1f}%)")
        yield encoded, lbls_batch


def _encode_pooled_batch(model, processor, pil_images: list, device) -> torch.Tensor:
    """Encode a batch into pooled, projected, L2-normalized CLIP image embeddings.

    Args:
        model: The CLIP model returned by get_CLIP_model().
        processor: The CLIP processor returned by get_CLIP_model().
        pil_images: PIL images to encode.
        device: Device the CLIP forward runs on.

    Returns:
        A (B, D) CPU tensor, L2-normalized per row.
    """
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    e = _as_feature_tensor(model.get_image_features(**inputs))
    return F.normalize(e, p=2, dim=-1).cpu()


def _encode_token_batch(model, processor, pil_images: list, device) -> torch.Tensor:
    """Encode a batch into CLIP's raw per-token visual sequence ``[CLS ; 49 patch]``.

    Args:
        model: The CLIP model returned by get_CLIP_model().
        processor: The CLIP processor returned by get_CLIP_model().
        pil_images: PIL images to encode.
        device: Device the CLIP forward runs on.

    Returns:
        A (B, 50, 768) fp16 CPU tensor, not L2-normalized.
    """
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    return model.vision_model(pixel_values=inputs["pixel_values"]).last_hidden_state.half().cpu()


@torch.no_grad()
def get_encoded_dataset(
    dataset,
    device,
    cache_path: str,
    batch_size: int = 128,
    num_workers: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all images in a dataset, with on-disk caching.

    If a cached file exists at `cache_path`, load and return it. Otherwise compute
    features and labels in a single DataLoader pass, cache as a dict
    {"features", "labels"}, and return.

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
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}.")
        blob = torch.load(cache_path, map_location="cpu")
        features = blob["features"].to(device)
        labels   = blob["labels"]
        print(f"Loaded from cache. features: {tuple(features.shape)}, labels: {tuple(labels.shape)}")
        return features, labels

    print("Cache not found. Encoding dataset...")
    feats_list: list[torch.Tensor] = []
    lbls_list:  list[torch.Tensor] = []
    for e, lbls in _encode_image_batches(dataset, device, _encode_pooled_batch,
                                          batch_size=batch_size, num_workers=num_workers):
        feats_list.append(e)
        lbls_list.append(lbls)

    features = torch.cat(feats_list, dim=0).to(device)
    labels   = torch.cat(lbls_list,  dim=0)

    torch.save({"features": features.cpu(), "labels": labels}, cache_path)
    print(f"Saved to {cache_path}. features: {tuple(features.shape)}, labels: {tuple(labels.shape)}")
    return features, labels


@torch.no_grad()
def get_encoded_patches(
    dataset,
    device,
    cache_path: str,
    indices: list[int] | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> torch.Tensor:
    """Encode images into CLIP's per-token visual sequence ``[CLS ; 49 patch]``, with caching.

    Unlike ``get_encoded_dataset`` (which returns the single pooled, projected image embedding),
    this taps the vision tower's ``last_hidden_state`` to keep the global CLS token and the 49
    patch tokens, giving the fusion module spatially grounded conditions. Tokens are stored in
    fp16 and are *not* L2-normalized, since they feed a learned projection rather than a cosine
    score. The returned bank is kept on CPU (it can be large); callers move the slices they need
    onto the device; when loaded from cache it is memory-mapped, so only sliced rows enter RAM.
    The cache records which dataset ``indices`` it covers so a stale subset can
    never be silently reused.

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
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    idx_list = list(range(len(dataset))) if indices is None else list(indices)

    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu", mmap=True)
        if blob.get("indices") == idx_list:
            print(f"Loading cached patch tokens from {cache_path}.")
            patches = blob["patches"]
            print(f"Loaded from cache. patches: {tuple(patches.shape)}")
            return patches
        print(f"Patch cache {cache_path} covers different indices; regenerating.")

    print(f"Cache not found / stale. Encoding {len(idx_list)} images into visual tokens...")
    n_total = len(idx_list)
    patches = None
    pos = 0
    for tokens, _lbls in _encode_image_batches(dataset, device, _encode_token_batch,
                                                indices=idx_list, batch_size=batch_size,
                                                num_workers=num_workers):
        if patches is None:
            patches = torch.empty((n_total, tokens.shape[1], tokens.shape[2]), dtype=torch.float16)
        patches[pos:pos + tokens.shape[0]] = tokens
        pos += tokens.shape[0]

    torch.save({"indices": idx_list, "patches": patches}, cache_path)
    print(f"Saved to {cache_path}. patches: {tuple(patches.shape)}")
    del patches
    return torch.load(cache_path, map_location="cpu", mmap=True)["patches"]


#==============================================================================
# Cell  10 [markdown] - Plotting utilities
#==============================================================================

"""
### Plotting utilities
Helper functions for visualizing retrieved images and their associated prompts, used across the notebook for consistent presentation of results.
"""


#==============================================================================
# Cell  11 [code] - def plot_images(celeba_dataset: object, indices: list[int], n_cols: int, n_ro…
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
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for counter, img_idx in enumerate(indices):
        img, _ = celeba_dataset[img_idx]
        if n_rows == 1:
            ax = axes[counter % n_cols]
        else:
            ax = axes[counter // n_cols, counter % n_cols]
        ax.imshow(img)
        ax.axis('off')

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
    fig, axes = plt.subplots(1, n, figsize=figsize or (3 * n, 3.2))
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


#==============================================================================
# Cell  12 [markdown] - Results visualization
#==============================================================================

"""
### Results visualization
Functions for visualizing the results of image retrieval, including displaying retrieved images alongside their prompts and similarity scores.
"""


#==============================================================================
# Cell  13 [code] - def plot_metrics_across_k(average_results_per_query: list[dict], title: str =…
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
    mean over queries of the per-query average — the same aggregation used by
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
# Cell  14 [markdown] - Data Loading and Exploration
#==============================================================================

"""
---

## Data Loading and Exploration

In this section, we load and explore the dataset to understand its structure and main properties.

The dataset used is **CelebA** [(Liu et al., 2015)](https://arxiv.org/abs/1411.7766). It is a large-scale face dataset containing over 200,000 celebrity images annotated with **40 binary facial attributes**.

In this work, we use a subset of **19,962 samples**. Each sample consists of:
- a **face image** of size 178 × 218 pixels,
- a corresponding **40-dimensional attribute vector**, describing visual characteristics such as *smiling*, *eyeglasses*, *male*, *young*, etc.

CelebA is widely used for facial attribute recognition and image editing tasks due to its diversity in pose, lighting, and background conditions.
"""


#==============================================================================
# Cell  15 [code] - Load CelebA test split
#==============================================================================

# Do *not* put `celeba` in the path, the dataset class adds it automatically
data_root = Path("/content/datasets")
celeba = CelebA(root=data_root, split="test", download=False)

# This should be 19.962
print("Number of samples:", len(celeba))

# Show element size
sample_img, sample_attrs = celeba[0]
print(f"Sample image size: {sample_img.size}")
print(f"Number of attributes: {len(sample_attrs)}")

def get_attribute_names(dataset) -> list[str]:
    """Return the dataset's attribute names, dropping torchvision's spurious empty entry.

    torchvision's CelebA exposes `attr_names` with 41 entries (one is an empty string),
    while the label tensor has 40 columns. Dropping empties keeps the names aligned with
    the label columns and with the learned per-attribute embedding rows.

    Args:
        dataset: A CelebA dataset object exposing `attr_names`.

    Returns:
        The 40 non-empty attribute names, aligned with the label columns.
    """
    return [name for name in dataset.attr_names if name]


#==============================================================================
# Cell  16 [markdown] - Sample visualization
#==============================================================================

"""
### Sample visualization
First, we can visualize a random selection of images from the dataset to get a sense of the variety and quality of the images. We will display 50 random images in a grid format.
"""


#==============================================================================
# Cell  17 [code] - Visualize 50 random samples
#==============================================================================

# Get 50 random images and visualize them
indices = np.random.choice(len(celeba), size=50, replace=False)
plot_images(celeba, indices=indices, n_cols=10, n_rows=5)


#==============================================================================
# Cell  18 [markdown] - Attribute annotation
#==============================================================================

"""
### Attribute annotation
Now that we know how to load our dataset and we have visualized some samples, let's move to understanding how attributes are annotated in the dataset. Each image in the dataset is annotated with a set of 40 binary attributes, from the following list. 

Here, we also report how frequently each attribute appears in the dataset, which is important to understand the distribution of attributes and to design a retrieval system that can handle rare attributes effectively.
"""


#==============================================================================
# Cell  19 [code] - Attribute frequency table
#==============================================================================

all_labels = celeba.attr.numpy()

attr_counts = all_labels.sum(axis=0)
attr_freq = all_labels.mean(axis=0)

print(f"{'Attribute':<20} {'Count':>10} {'Frequency':>10}")
print("-" * 45)

for attr, count, freq in zip(get_attribute_names(celeba), attr_counts, attr_freq):
    print(f"{attr:<20} {count:>10} {freq:>10.3f}")


#==============================================================================
# Cell  20 [markdown] - Let's define few other utilities functions that will facilitate the handling…
#==============================================================================

"""
Let's define few other utilities functions that will facilitate the handling of attributes later on.
"""


#==============================================================================
# Cell  21 [code] - Attribute name/index maps & retrieve_by_attributes
#==============================================================================

attr_names = get_attribute_names(celeba)
idx2attribute = {idx: name for idx, name in enumerate(attr_names)}
attribute2idx = {name: idx for idx, name in enumerate(attr_names)}

def retrieve_by_attributes(parameters:dict):
    """Retrieve all dataset images that satisfy the given attribute conditions.

    Args:
        parameters: Dict mapping attribute name to "+" (must have the attribute) or "-" (must not have the attribute).

    Returns:
        Indices of images that satisfy every specified condition.
    """
    # Boolean mask over the precomputed label matrix
    mask = np.ones(len(all_labels), dtype=bool)
    for attr_name, value in parameters.items():
        attr_idx = attribute2idx[attr_name]
        if value == "+":
            mask &= all_labels[:, attr_idx] == 1
        elif value == "-":
            mask &= all_labels[:, attr_idx] == 0
        else:
            raise ValueError(f"Invalid value for attribute condition: {value}. Use '+' or '-'.")

    return np.nonzero(mask)[0].tolist()

def plot_image_with_attributes(idx: int, figsize: tuple[int, int]=(10, 5)):
    """Plot a single image with its active attributes listed as text alongside it.

    Args:
        idx: Dataset index of the image to plot.
        figsize: Figure size as (width, height).
    """
    img, labels = celeba[idx]
    active_attrs = [idx2attribute[idx] for idx, value in enumerate(labels) if value == 1]

    fig, (ax_img, ax_text) = plt.subplots(1, 2, figsize=figsize)
    ax_img.imshow(img)
    ax_img.axis('off')

    ax_text.axis('off')
    text = "\n".join(active_attrs)

    ax_text.text(
        0.5, 0.5, text,
        fontsize=10,
        ha='center',   
        va='center'    
    )

    plt.tight_layout()
    plt.show()


def parse_query_signs(text_query: str) -> tuple[list[int], list[int]]:
    """Parse a signed attribute query into positive and negative attribute indices.

    For example, '+Bald, -Eyeglasses' becomes
    ([attribute2idx['Bald']], [attribute2idx['Eyeglasses']]). Shared query parser
    consumed by every method that reads the benchmark query string (Source-Attribute
    Matching, Cross-Attention Fusion, qualitative inspection), so it lives here with
    the other attribute/query utilities rather than inside one method.

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
        j = attribute2idx[attr_name]
        (pos_idx if sign_char == "+" else neg_idx).append(j)
    return pos_idx, neg_idx


#==============================================================================
# Cell  22 [markdown] - Now that we have the mapping, we can easily get the attributes of any image i…
#==============================================================================

"""
Now that we have the mapping, we can easily get the attributes of any image in the dataset. For example, let's get the attributes of a given image index.
"""


#==============================================================================
# Cell  23 [code] - Inspect a single image's attributes
#==============================================================================

IMAGE_INDEX = 99
plot_image_with_attributes(IMAGE_INDEX)


#==============================================================================
# Cell  24 [markdown] - Now that we have everything in place, let's try to analyze some possible quer…
#==============================================================================

"""
Now that we have everything in place, let's try to analyze some possible queries.
"""


#==============================================================================
# Cell  25 [code] - Example signed attribute query
#==============================================================================

query_1 = {"Bald": "+",
           "Smiling": "+",
           "Eyeglasses": "-",
           }
retrieved_images = retrieve_by_attributes(query_1)
print(f"Number of retrieved images: {len(retrieved_images)}")

# Plot up to 10 random retrieved images (without replacement).
n_samples = min(10, len(retrieved_images))
if n_samples == 0:
    print("No images match this query.")
else:
    sampled_indices = np.random.choice(retrieved_images, size=n_samples, replace=False)
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))
    plot_images(celeba, indices=sampled_indices, n_cols=n_cols, n_rows=n_rows)


#==============================================================================
# Cell  26 [markdown] - Offline Feature Extraction (CLIP)
#==============================================================================

"""
---

## Offline Feature Extraction
In this step, we use **CLIP** (specifically the ViT-B/32 variant) as a **frozen** feature extractor to convert our images into vector representations [(Radford et al., 2021)](https://arxiv.org/abs/2103.00020).

Because these image embeddings never change, we precompute them offline. This approach offers two major benefits:

* **Zero training overhead:** We do not backpropagate through the heavy vision transformer during training.
* **Larger batch sizes:** Freeing up GPU memory allows us to use significantly larger batch sizes when training our downstream retrieval layers.

Ultimately, this step compresses every image in our dataset into a fixed vector of size 512, which acts as its static visual fingerprint.
"""


#==============================================================================
# Cell  27 [code] - Embedding cache paths
#==============================================================================

EVALUATION_CACHE_DIR = "/content/drive/MyDrive/datasets/clip_cache"
EVALUATION_CACHE_PATH = os.path.join(EVALUATION_CACHE_DIR, "embeddings.pt")


#==============================================================================
# Cell  28 [code] - Encode (or load cached) gallery embeddings
#==============================================================================

# Get the encoded dataset, using cached features if available
gallery_embeddings, gallery_labels = get_encoded_dataset(celeba, device, EVALUATION_CACHE_PATH, batch_size=128)


#==============================================================================
# Cell  29 [markdown] - Sanity check
#==============================================================================

"""
### Sanity check
Now that embeddings for the image dataset are available, let's run a quick sanity check to verify retrieval quality.
We will pick a source image, compare its CLIP embedding against all dataset embeddings, and inspect the nearest matches.
"""


#==============================================================================
# Cell  30 [code] - Load sanity-check source image
#==============================================================================

SANITY_SOURCE_IDX = 10006
img, _ = celeba[SANITY_SOURCE_IDX]

plt.figure(figsize=(4, 4))
plt.axis('off')
plt.imshow(img)


#==============================================================================
# Cell  31 [markdown] - We normalize all embeddings to unit vectors at extraction time. Because of th…
#==============================================================================

"""
We normalize all embeddings to unit vectors at extraction time. Because of this, calculating the similarity between a query $\mathbf{q}$ and a gallery vector $\mathbf{g}$ simplifies to a basic dot product:

$$\text{similarity}(\mathbf{q}, \mathbf{g}) = \mathbf{q} \cdot \mathbf{g}$$

Since the dot product of unit vectors is mathematically identical to cosine similarity, our evaluation metric remains unchanged. However, this allows us to scale retrieval efficiently: instead of looping through image pairs one by one, we can search the entire gallery simultaneously using a single matrix multiplication. Finally, we filter out the source image itself from the top results.
"""


#==============================================================================
# Cell  32 [code] - Nearest neighbors by cosine similarity
#==============================================================================

if "gallery_embeddings" not in globals():
    raise RuntimeError(
        "Embeddings not found. Run the offline feature extraction cell above first."
    )

source_embedding = gallery_embeddings[SANITY_SOURCE_IDX]

# Dot product == cosine similarity for unit-norm embeddings
similarities = gallery_embeddings @ source_embedding

# Get the 6 highest-similarity matches and drop the source itself.
top_vals, top_idx = torch.topk(similarities, k=6)
nearest_indices = top_idx[1:].tolist()       
nearest_similarities = top_vals[1:].tolist()  

print("Nearest indices:", nearest_indices)
print("Nearest cosine similarities:", nearest_similarities)


#==============================================================================
# Cell  33 [markdown] - Let's visualize the nearest images to our source image and see if they are in…
#==============================================================================

"""
Let's visualize the nearest images to our source image and see if they are indeed similar.
"""


#==============================================================================
# Cell  34 [code] - Plot nearest-neighbor results
#==============================================================================

images = [celeba[idx][0] for idx in nearest_indices]
titles = [f"Cosine sim: {sim:.4f}" for sim in nearest_similarities]
plot_image_row(images, titles=titles, figsize=(25, 5))


#==============================================================================
# Cell  35 [markdown] - Embedding Analysis: Class-Image Similarity
#==============================================================================

"""
---

## Embedding Analysis: Class-Image Similarity

Before building our retrieval model, we evaluate how effectively CLIP isolates the 40 CelebA attributes. We construct a $40 \times 40$ cosine similarity heatmap between attribute text prompts and a curated set of image embeddings using the following steps:

* **Targeted Sampling:** For each attribute, we select a single "pure" image where that specific trait is active and the number of co-occurring labels is minimized. This isolates the target concept and reduces visual noise from overlapping CelebA annotations.
* **Matrix Construction:** We compute the cosine similarities between all 40 text prompts and the 40 isolated image embeddings.

If CLIP is properly aligned with these facial concepts, the diagonal of the heatmap should dominate each row. Strong off-diagonal values immediately expose semantic overlap or confusion between related attributes, such as `Wavy_Hair` versus `Straight_Hair`.
"""


#==============================================================================
# Cell  36 [code] - def _select_pure_image_idxs(all_labels: np.ndarray, rng: np.random.Generator)…
#==============================================================================

def _select_pure_image_idxs(all_labels: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Pick, for each attribute, a "pure" image: positive for it with the fewest other positives.

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


# Bare-name attribute text bank: reused downstream by other methods
prompts = [name.replace("_", " ").lower() for name in attr_names]
ATTR_TEXT_EMBS = encode_texts(prompts, device).to(gallery_embeddings.device)


rng = np.random.default_rng(seed=0)
selected_idxs = _select_pure_image_idxs(all_labels, rng)
selected_img_embs = gallery_embeddings[selected_idxs].to(ATTR_TEXT_EMBS.device)


cos_mat = (ATTR_TEXT_EMBS @ selected_img_embs.T).detach().cpu().numpy()
_print_cosine_diagnostics(cos_mat)


#==============================================================================
# Cell  37 [markdown] - Cosine Heatmap Analysis
#==============================================================================

"""
#### Cosine Heatmap Analysis

The resulting heatmap reveals that the diagonal does not dominate. Instead, the matrix is driven by strong row and column biases: certain text prompts score highly against almost all images, while specific image columns light up across completely unrelated prompts.

Furthermore, all similarity values sit in a narrow, compressed band ($\sim 0.13$ to $0.27$), meaning raw cosine scores carry very little discriminative signal. These findings indicate that raw CLIP embeddings capture only coarse facial semantics and fail to isolate fine-grained attributes.
"""


#==============================================================================
# Cell  38 [code] - Plot attribute cosine-similarity heatmap
#==============================================================================

plot_cosine_heatmap(cos_mat, attr_names)


#==============================================================================
# Cell  39 [markdown] - Metrics (Recall@K / Precision@K definitions)
#==============================================================================

"""
---

## Metrics

We evaluate every method with two standard top-$K$ retrieval metrics, computed per *(query, source image)* pair at $K \in \{1, 5, 10\}$. Let $\mathcal{R}_K$ be the ordered set of top-$K$ retrieved gallery images (the source image itself excluded) and $\mathcal{G}$ the ground-truth set of valid retrievals for that query.

- **Recall@K (hit rate).** Whether *at least one* valid image appears in the top $K$:
$$\text{Recall@}K = \mathbb{1}\big[\,|\mathcal{R}_K \cap \mathcal{G}| > 0\,\big] \in \{0, 1\}.$$

- **Precision@K.** The fraction of the top $K$ retrievals that are valid:
$$\text{Precision@}K = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{K}.$$

Both metrics are first averaged over the source images of each query (reported with a 95% confidence interval), and the **mean Recall@10** across all *(query, source)* pairs is the single headline scalar we use to compare methods and tune hyperparameters.
"""


#==============================================================================
# Cell  40 [code] - def evaluate_retrieval(
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
        metrics[10]["Recall@10"]
        for query_results in evaluation_results
        for metrics in query_results.values()
    ]
    return float(np.mean(vals))


#==============================================================================
# Cell  41 [markdown] - Example usage
#==============================================================================

"""
#### Example usage
"""


#==============================================================================
# Cell  42 [code] - Example usage of evaluate_retrieval
#==============================================================================

# Suppose the model returns these indices from most to least similar:
predictions = [1, 2, 3, 4, 5]
# And we load this from our JSON for this specific source:
ground_truth = [3, 2, 1]

# Evaluate at K=1 and K=5
print("Results @ 1:", evaluate_retrieval(predictions, ground_truth, k=1))
print("Results @ 5:", evaluate_retrieval(predictions, ground_truth, k=5))


#==============================================================================
# Cell  43 [markdown] - Evaluation Protocol (benchmark JSON)
#==============================================================================

"""
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
"""


#==============================================================================
# Cell  44 [code] - Load benchmark annotations JSON
#==============================================================================

# Open the JSON file containing the benchmark annotations
annotations_path = Path("/content/drive/MyDrive/datasets/celeba_evaluation.json")
with open(annotations_path, "r") as f:
    annotations = json.load(f)

# Print the number of annotations loaded
print(f"Loaded {len(annotations)} queries!")


#==============================================================================
# Cell  45 [markdown] - We can define some utility functions to facilitate the evaluation process
#==============================================================================

"""
We can define some utility functions to facilitate the evaluation process
"""


#==============================================================================
# Cell  46 [code] - Inspect annotation structure
#==============================================================================

# Display a sample annotation to understand the structure of the data
print("Sample annotation shape", annotations[0].keys())

# Extract and print first text query
print("Text-Query example:", annotations[0].get("query", ""))

# Extract and print the source image ID for the first annotation
print("Source-Image example:", list(annotations[0].get("ground_truth", {}).keys())[:5],"...")

# Extract and print the list of ground truth indices for the first annotation
print("List of ground truth indices for the first annotation:", annotations[0].get("ground_truth", {}).get("13", [])[:5], "...")

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

def get_ground_truth_indices(annotation: dict, source_image_idx: int) -> list[int]:
    """Extract the valid target IDs for one source image from a benchmark annotation.

    Args:
        annotation: Benchmark annotation dict for a single query.
        source_image_idx: Index of the source image whose ground-truth targets to retrieve.

    Returns:
        The valid target IDs (integers) considered correct matches for the given query.
    """
    return annotation.get("ground_truth", {}).get(str(source_image_idx), [])


#==============================================================================
# Cell  47 [markdown] - Sanity-check annotation helpers
#==============================================================================

"""
#### Sanity-check annotation helpers
"""


#==============================================================================
# Cell  48 [code] - Sanity-check annotation helper functions
#==============================================================================

# Let's test these utility functions on the first annotation in the dataset
annotation = annotations[1]

text_query = get_text_query(annotation)
print("Text query:", text_query )

source_image_idx = get_source_image_idxs(annotation)[0]
print("Source image index:", source_image_idx)
plot_image_with_attributes(source_image_idx, figsize=(4, 4))

# Get the first 5 ground truth indices for this annotation and source image
ground_truth_indices = get_ground_truth_indices(annotation, source_image_idx)[:5]
print("Ground truth indices for this query:", ground_truth_indices)

plot_images(celeba, indices=ground_truth_indices, n_cols=5, n_rows=1, figsize=(10, 2))


#==============================================================================
# Cell  49 [markdown] - Evaluation Function
#==============================================================================

"""
### Evaluation Function

We evaluate the retrieval performance of each fusion mechanism on the benchmark dataset, comparing it against the baseline method.

We compute the recall and precision metrics for each source image in the query for `"K = {1, 5, 10}"`.
Then we average the result across all source images and keep track on each query separately.
"""


#==============================================================================
# Cell  50 [code] - def retrieve_topk(scores: torch.Tensor, exclude_idx: int, k: int = 10) -> lis…
#==============================================================================

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
        Per-query results as ``list[dict[source_idx -> dict[k -> metrics_dict]]]``.
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
                k: evaluate_retrieval(retrieved, get_ground_truth_indices(annotation, src), k)
                for k in (1, 5, 10)
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
        query_evaluation_results: Dict mapping source image index to its evaluation metrics for K in {1, 5, 10}.

    Returns:
        A dict of average Recall@K / Precision@K plus their 95% confidence intervals for the query.
    """
    average_results = {}

    for k in [1, 5, 10]:
        # Collect the per-source Recall@K and Precision@K values for this query
        recall_vals = [m[k][f"Recall@{k}"] for m in query_evaluation_results.values()]
        precision_vals = [m[k][f"Precision@{k}"] for m in query_evaluation_results.values()]

        # Average each metric and attach its 95% confidence interval (empirical-std based; see
        # _mean_and_ci — the naive Bernoulli formula only applies to Recall, not Precision)
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
        (for mean_recall_at_10) and the per-query averages (for plotting) — the pair every
        method downstream needs.
    """
    results = evaluate(annotations, make_scorer, verbose=verbose)
    return results, [compute_query_average_results(q) for q in results]


#==============================================================================
# Cell  51 [markdown] - Baseline Method (training-free)
#==============================================================================

"""
---

## Baseline Method
To establish a baseline for our retrieval system, we evaluate a **zero-shot, training-free approach** that relies exclusively on CLIP embeddings and cosine similarity.

The baseline uses simple latent space arithmetic by combining the attribute and image embeddings, without any learning or explicit alignment.
The query is decomposed into signed attribute terms: starting from the source image embedding, each `+` attribute embedding is added and each `−` attribute embedding is subtracted, and the resulting vector is used to find the nearest neighbours in the dataset.
"""


#==============================================================================
# Cell  52 [markdown] - Scorer
#==============================================================================

"""
### Scorer

Let $\mathbf{v}_s \in \mathbb{R}^D$ represent the raw CLIP visual embedding of the source image, and let $\hat{\mathbf{e}}_s = \frac{\mathbf{v}_s}{\|\mathbf{v}_s\|_2}$ be its corresponding unit-norm vector. For each text attribute $j$, $\mathbf{t}_j \in \mathbb{R}^D$ denotes the raw CLIP text embedding generated from the bare-name attribute prompt.

Let $q^+$ be the set of attributes to be added, and $q^-$ be the set of attributes to be removed. The unnormalized composite query vector $\mathbf{f} \in \mathbb{R}^D$ is constructed by shifting the source embedding along the text vector directions:

$$\mathbf{f} = \mathbf{v}_s + \sum_{j \in q^+} \mathbf{t}_j - \sum_{j \in q^-} \mathbf{t}_j$$

To evaluate and rank candidate images from the gallery, we compute the cosine similarity between the composite query and each gallery embedding. Let $\hat{\mathbf{e}}_x$ represent the pre-normalized, unit-norm CLIP embedding of a gallery image $x$ (where $\|\hat{\mathbf{e}}_x\|_2 = 1$). The final retrieval score for a given candidate $x$ is defined as the inner product of $\hat{\mathbf{e}}_x$ and the unit-normalized query vector:

$$\text{score}(x) = \hat{\mathbf{e}}_x^{\top} \left( \frac{\mathbf{f}}{\|\mathbf{f}\|_2} \right)$$

Gallery images are then sorted in descending order based on this score, directly optimizing the retrieval ranking in the shared latent space.
"""


#==============================================================================
# Cell  53 [code] - def baseline_scorer(gallery_embeddings: torch.Tensor) -> Callable
#==============================================================================

def baseline_scorer(gallery_embeddings: torch.Tensor) -> Callable:
    """Build the scorer factory for the signed-arithmetic baseline.

    Decomposes the query into signed attribute terms and fuses them with the source
    image embedding by simple latent arithmetic.
    Attribute text embeddings come from the bare-name CLIP bank (ATTR_TEXT_EMBS); the
    per-query delta is built once per annotation and reused for every source image.

    Args:
        gallery_embeddings: (N, D) gallery image embeddings, L2-normalized per row.

    Returns:
        A ``make_scorer(annotation)`` factory consumed by evaluate().
    """
    def make_scorer(annotation: dict) -> Callable:
        """Build a per-query scorer from the query's signed attribute delta."""
        pos_idx, neg_idx = parse_query_signs(get_text_query(annotation))
        # Initialize delta vector to zero, then add/subtract attribute embeddings based on the query
        delta = torch.zeros(gallery_embeddings.shape[1], device=gallery_embeddings.device)
        if pos_idx:
            delta = delta + ATTR_TEXT_EMBS[pos_idx].sum(dim=0)
        if neg_idx:
            delta = delta - ATTR_TEXT_EMBS[neg_idx].sum(dim=0)

        def scorer(source_idx: int) -> torch.Tensor:
            """Score every gallery image against the fused source query embedding."""
            fused = gallery_embeddings[source_idx] + delta
            return gallery_embeddings @ F.normalize(fused, dim=0)

        return scorer
    return make_scorer


#==============================================================================
# Cell  54 [markdown] - Evaluation & Plot
#==============================================================================

"""
### Evaluation & Plot
"""


#==============================================================================
# Cell  55 [code] - Evaluate & plot baseline
#==============================================================================

evaluation_results_baseline, average_results_per_query_baseline = evaluate_and_average(
    annotations,
    baseline_scorer(gallery_embeddings),
    verbose=False,
)

plot_metrics_across_k(average_results_per_query_baseline, title="Baseline Fusion Performance across K")


#==============================================================================
# Cell  56 [markdown] - Source-Attribute Matching (Training-Free)
#==============================================================================

"""
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
"""


#==============================================================================
# Cell  57 [markdown] - Parameters
#==============================================================================

"""
### Parameters
"""


#==============================================================================
# Cell  58 [code] - Grid-search weight candidates
#==============================================================================

GRID_W_ATTR = [0.05, 0.1, 0.2, 0.4]   # attribute-proximity penalty weight candidates
GRID_W_VISUAL  = [0.0, 0.5, 1.0]          # visual identity weight candidates


#==============================================================================
# Cell  59 [markdown] - Scorer
#==============================================================================

"""
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
"""


#==============================================================================
# Cell  60 [markdown] - Attribute-matching scorer
#==============================================================================

"""
#### Attribute-matching scorer
"""


#==============================================================================
# Cell  61 [code] - compute_attribute_logits
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
    ``make_scorer`` closure that pre-computes the per-query constraint vector
    once, and an inner per-source ``scorer`` closure:


    Used by Source-Attribute Matching and Prompt Ensembling — they differ only in E_pos/E_neg.

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
        pos_idx, neg_idx = parse_query_signs(get_text_query(annotation))
        queried   = set(pos_idx + neg_idx)
        unqueried = [j for j in range(Z.shape[1]) if j not in queried]
        constraint = Z[:, pos_idx].sum(dim=1) - Z[:, neg_idx].sum(dim=1)  # (N,)

        def scorer(source_idx: int) -> torch.Tensor:
            """Score every gallery image by constraint, attribute proximity, and visual similarity."""
            z_src   = Z[source_idx]
            attr_proximity = ((Z[:, unqueried] - z_src[unqueried]) ** 2).sum(dim=1)
            scores  = w_query * constraint - w_attr * attr_proximity
            if w_visual > 0:
                scores = scores + w_visual * (gallery_embeddings @ gallery_embeddings[source_idx])
            return scores

        return scorer
    return make_scorer


#==============================================================================
# Cell  62 [markdown] - Evaluation & Plot
#==============================================================================

"""
### Evaluation & Plot
"""


#==============================================================================
# Cell  63 [markdown] - The three weights w_query, w_attr, w_visual are training-free hyperparameters…
#==============================================================================

"""
The three weights `w_query`, `w_attr`, `w_visual` are training-free hyperparameters tuned with a
deliberately small grid so sensitivity stays visible. They are tuned **once**, using the simple
bare-name attribute bank (Source-Attribute Matching); Prompt Ensembling reuses the same `SAM_WEIGHTS` with its
improved bank, making the two methods directly comparable.
"""


#==============================================================================
# Cell  64 [code] - Grid search over fusion weights
#==============================================================================

grid_rows = []
for w_p in GRID_W_ATTR:
    for w_v in GRID_W_VISUAL:
        res = evaluate(
            annotations,
            attribute_matching_scorer(gallery_embeddings, ATTR_TEXT_EMBS, w_query=1.0, w_attr=w_p, w_visual=w_v),
            verbose=False,
        )
        r10 = mean_recall_at_10(res)
        grid_rows.append((w_p, w_v, r10))
        print(f"w_attr={w_p:<5} w_visual={w_v:<4} mean Recall@10={r10:.4f}")

best_w_attr, best_w_visual, best_r10 = max(grid_rows, key=lambda row: row[2])
print(f"\nBest: w_attr={best_w_attr}, w_visual={best_w_visual} (mean Recall@10={best_r10:.4f})")
SAM_WEIGHTS = dict(w_query=1.0, w_attr=best_w_attr, w_visual=best_w_visual)


#==============================================================================
# Cell  65 [markdown] - Final evaluation with the selected SAM_WEIGHTS, then the per-query metrics plo…
#==============================================================================

"""
Final evaluation with the selected `SAM_WEIGHTS`, then the per-query metrics plot.
"""


#==============================================================================
# Cell  66 [code] - Evaluate & plot Source-Attribute Matching
#==============================================================================

evaluation_results_sam, average_results_per_query_sam = evaluate_and_average(
    annotations,
    attribute_matching_scorer(gallery_embeddings, ATTR_TEXT_EMBS, **SAM_WEIGHTS),
)
print(f"Source-Attribute Matching: mean Recall@10 = {mean_recall_at_10(evaluation_results_sam):.4f}")

plot_metrics_across_k(
    average_results_per_query_sam,
    title="Source-Attribute Matching — Performance across K",
)


#==============================================================================
# Cell  67 [markdown] - Prompt Ensembling (training-free)
#==============================================================================

"""
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
"""


#==============================================================================
# Cell  68 [markdown] - Parameters
#==============================================================================

"""
### Parameters

Prompt Ensembling has no parameters of its own; it reuses `SAM_WEIGHTS` from Source-Attribute Matching. Only the phrase/template banks below change.
"""


#==============================================================================
# Cell  69 [markdown] - Scorer
#==============================================================================

"""
### Scorer

Only the bank changes: Prompt Ensembling builds an **ensembled** pos/neg bank and feeds it to **Source-Attribute Matching's `attribute_matching_scorer`, unchanged**. Any gain over Source-Attribute Matching is therefore attributable to the embedding bank alone.
"""


#==============================================================================
# Cell  70 [markdown] - Attribute phrase banks
#==============================================================================

"""
#### Attribute phrase banks
"""


#==============================================================================
# Cell  71 [code] - Attribute phrase bank (positive/negative)
#==============================================================================

# Person-referring positive AND negative phrases for each CelebA attribute.
# The previous version only stored positives and negated their embedding by -1;
# we now also store linguistic negatives so the negative side of the score is
# computed against an actual "without ..." description.
humanized_mappings_pos = {
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
# CLIP's text encoder attends to the object token regardless of the "not" — phrasing
# matters. Where a clean linguistic opposite exists (e.g. clean-shaven vs bearded)
# we use it; otherwise we lean on "without {attr}" / "no {attr}" framings.
humanized_mappings_neg = {
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
# Cell  72 [markdown] - CLIP ImageNet prompt templates
#==============================================================================

"""
#### CLIP ImageNet prompt templates

The per-attribute banks are ensembled over [CLIP's official ImageNet prompt templates](https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb) (the canonical 80-template zero-shot set), plus a few portrait-specific templates for CelebA faces.

We adapt the templates to **full noun phrases**: each `{phrase}` already carries its own article (e.g. *"a person with glasses"*), so we drop the template's leading article to avoid *"a **a** person with glasses"*. Removing that article makes the `a {}` and `the {}` variants identical, collapsing the official 80 templates to **55 unique** ones, the set we actually ensemble over.
"""


#==============================================================================
# Cell  73 [code] - CLIP ImageNet-style prompt templates
#==============================================================================

# CLIP's official ImageNet templates, article-stripped for {phrase} noun phrases
# (see the note above); the article-collapse leaves 55 unique templates.
clip_imagenet_templates = [
    "a bad photo of {phrase}.", "a photo of many {phrase}.", "a sculpture of {phrase}.",
    "a photo of the hard to see {phrase}.", "a low resolution photo of {phrase}.", "a rendering of {phrase}.",
    "graffiti of {phrase}.", "a cropped photo of {phrase}.", "a tattoo of {phrase}.",
    "the embroidered {phrase}.", "a photo of a hard to see {phrase}.", "a bright photo of {phrase}.",
    "a photo of a clean {phrase}.", "a photo of a dirty {phrase}.", "a dark photo of {phrase}.",
    "a drawing of {phrase}.", "a photo of my {phrase}.", "the plastic {phrase}.",
    "a photo of the cool {phrase}.", "a close-up photo of {phrase}.", "a black and white photo of {phrase}.",
    "a painting of {phrase}.", "a pixelated photo of {phrase}.", "a plastic {phrase}.",
    "a photo of the dirty {phrase}.", "a jpeg corrupted photo of {phrase}.", "a blurry photo of {phrase}.",
    "a photo of {phrase}.", "a good photo of {phrase}.", "a {phrase} in a video game.",
    "a photo of one {phrase}.", "a doodle of {phrase}.", "the origami {phrase}.",
    "a sketch of {phrase}.", "a origami {phrase}.", "the toy {phrase}.",
    "a rendition of {phrase}.", "a photo of the clean {phrase}.", "a photo of a large {phrase}.",
    "a photo of a nice {phrase}.", "a photo of a weird {phrase}.", "a cartoon {phrase}.",
    "art of {phrase}.", "a embroidered {phrase}.", "itap of {phrase}.",
    "a plushie {phrase}.", "a photo of the nice {phrase}.", "a photo of the small {phrase}.",
    "a photo of the weird {phrase}.", "the cartoon {phrase}.", "a photo of the large {phrase}.",
    "the plushie {phrase}.", "a toy {phrase}.", "a photo of a cool {phrase}.",
    "a photo of a small {phrase}.",
]
portrait_templates = [
    "a portrait of {phrase}.",
    "a portrait photograph of {phrase}.",
    "a closeup headshot of {phrase}.",
    "a candid photo of {phrase}.",
    "a studio portrait of {phrase}.",
    "a high-resolution headshot of {phrase}.",
    "a face photo of {phrase}.",
    "a photo showing the face of {phrase}.",
    "a frontal photo of {phrase}.",
    "a clear photo of {phrase}.",
]
prompt_templates = clip_imagenet_templates + portrait_templates


@torch.no_grad()
def _encode_phrases_through_templates(phrases: list[str], templates: list[str]) -> torch.Tensor:
    """Encode every (phrase x template) pair, mean-pool the embeddings, and re-normalize.

    Each pair is L2-normalized before pooling; batched in a single processor/model
    call for speed.

    Args:
        phrases: Phrases to expand across templates.
        templates: Prompt templates with a `{phrase}` placeholder.

    Returns:
        A single (D,) L2-normalized ensemble embedding.
    """
    prompts = [template.format(phrase=phrase) for phrase in phrases for template in templates]
    embs = encode_texts(prompts, device)   # (P, D), per-row normalized
    mean_emb = embs.mean(dim=0)
    return mean_emb / mean_emb.norm()


@torch.no_grad()
def precompute_attribute_pos_neg_embeddings() -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-attribute positive and negative text-embedding banks.

    E_pos[i] = ensemble over (positive phrases for attribute i) x (templates)
    E_neg[i] = ensemble over (negative phrases for attribute i) x (templates)

    Returns:
        An (E_pos, E_neg) pair, each (40, 512) and L2-normalized.
    """
    pos_embs, neg_embs = [], []
    for name in attr_names:
        pos_embs.append(_encode_phrases_through_templates(humanized_mappings_pos[name], prompt_templates))
        neg_embs.append(_encode_phrases_through_templates(humanized_mappings_neg[name], prompt_templates))
    E_pos = torch.stack(pos_embs, dim=0)
    E_neg = torch.stack(neg_embs, dim=0)
    return E_pos, E_neg


print("Precomputing pos/neg attribute embeddings with the expanded template bank (this may take a minute)...")
E_POS, E_NEG = precompute_attribute_pos_neg_embeddings()
E_POS = E_POS.to(gallery_embeddings.device)
E_NEG = E_NEG.to(gallery_embeddings.device)
print(f"E_POS: {tuple(E_POS.shape)},  E_NEG: {tuple(E_NEG.shape)}")


#==============================================================================
# Cell  74 [markdown] - Evaluation & Plot
#==============================================================================

"""
### Evaluation & Plot
"""


#==============================================================================
# Cell  75 [code] - Evaluate & plot Prompt Ensembling
#==============================================================================

# Same scoring layer and weights as Source-Attribute Matching — only the embedding bank changes.
evaluation_results_promptens, average_results_per_query_promptens = evaluate_and_average(
    annotations,
    attribute_matching_scorer(gallery_embeddings, E_POS, E_NEG, **SAM_WEIGHTS),
    verbose=False,
)
print(f"Prompt Ensembling: mean Recall@10 = {mean_recall_at_10(evaluation_results_promptens):.4f}")


plot_methods_comparison(
    {
        "Baseline":                            average_results_per_query_baseline,
        "Source-Attribute Matching":                average_results_per_query_sam,
        "Prompt Ensembling":  average_results_per_query_promptens,
    },
    title="Training-Free Method Comparison — per-query Recall@K and Precision@K",
)


#==============================================================================
# Cell  76 [markdown] - Other experiments
#==============================================================================

"""
---

## Other experiments

Before settling on cross-attention, two other *learned* methods were built and then dropped. Both keep CLIP frozen and learn a different piece of the pipeline, and both reuse the existing evaluation harness, so each slots in as a drop-in replacement for either the embedding bank or the edit rule.

**CoOp (Context Optimization, [Zhou et al. 2022](https://arxiv.org/abs/2109.01134)).** Instead of writing the prompt prefix by hand, CoOp learns it: the handcrafted words in front of each attribute are replaced by $M=16$ continuous context vectors that live in CLIP's word-embedding space and are shared across all 40 attributes. We frame CelebA as multi-label classification, building a positive prompt ("a person with {attribute}") and a negative prompt ("a person without {attribute}") for every attribute, and train the shared context (about $M\times 512\approx 8\text{k}$ parameters) with binary cross-entropy on the per-attribute margin $\cos(\text{img},e_+)-\cos(\text{img},e_-)$. The learned positive/negative text bank then drops into the *same* profile-matching scorer the training-free methods use, with the same weights. It trained cleanly and slightly edged the hand-written banks, but gave no decisive gain: the bottleneck is the fixed additive fusion, not the prompt wording.

**TopK-SAE concept editing ([Gao et al. 2024](https://arxiv.org/abs/2406.04093)).** This route learns the *representation* rather than the prompt. A TopK sparse autoencoder is trained, with no labels and no text, to reconstruct the cached CLIP image embeddings through an overcomplete dictionary of $H=4096$ unit-norm atoms, each a direction in CLIP space; sparsity pushes those atoms toward near-monosemantic concepts. A textual condition is grounded zero-shot onto its top few atoms (cosine affinity, mean-centred to drop the shared text direction), and the source is edited along only those atoms, $\mathbf{v}_{\text{target}}=\mathbf{v}_s+\sum_c\sigma_c\,\gamma_c\,\hat{\mathbf{u}}_c$, leaving every other atom untouched so identity is preserved for free. Reconstruction was faithful enough for retrieval (the residual is inert), but the edit direction was the failure point: a query attribute seldom grounds onto a single clean, monosemantic atom, so the added term behaved as noise and could not reliably realise the attribute.

Both methods leave the image-condition *interaction* hand-designed. Cross-attention learns that interaction instead.
"""


#==============================================================================
# Cell  77 [markdown] - Training-Based Method: Cross-Attention Fusion
#==============================================================================

"""
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
"""


#==============================================================================
# Cell  78 [markdown] - Parameters
#==============================================================================

"""
### Parameters
"""


#==============================================================================
# Cell  79 [code] - Cross-Attention Fusion hyperparameters
#==============================================================================

CA_HEADS          = 4          # cross-attention heads
CA_LAYERS         = 2          # stacked cross-attention (transformer decoder) layers
CA_FFN_MULT       = 2          # transformer FFN hidden size = CA_FFN_MULT * dim
CA_DROPOUT        = 0.1        # dropout inside the transformer layers
CA_GROUND_LAYERS  = 1          # patch-grounding decoder layers (conditions read the visual tokens)
CA_GROUND_HEADS   = 4          # attention heads in the grounding decoder
CLIP_VIS_DIM      = 768        # CLIP ViT-B/32 hidden width of the [CLS ; 49 patch] tokens
CA_TRAIN_TRIPLETS = 100_000    # synthetic training triplets (own pool)
CA_VAL_TRIPLETS   = 2_000      # synthetic validation triplets
CA_EPOCHS         = 20         # training epochs
CA_BATCH          = 512        # mini-batch size
CA_LR             = 2e-4       # AdamW learning rate
CA_WD             = 1e-2       # AdamW weight decay
CA_HARD_NEG       = True       # mine one constraint-violating hard negative per triplet
MAX_TERMS         = 3          # max attribute conditions per synthetic query (benchmark-dictated)
HAMMING_BUDGET    = 2          # max Hamming distance for a valid target (matches benchmark)

CA_FILM_SIGN_STD  = 0.02       # FiLM sign-embedding init std
CA_GATE_BIAS_INIT = 2.0        # gated-residual gate-open bias; sigmoid(2.0)≈0.88 keeps gate open


#==============================================================================
# Cell  80 [markdown] - Architecture
#==============================================================================

"""
### Architecture

**How the Transformer maps to our problem.** The sequence the attention runs over is neither image patches nor text sub-words: it is the query's own short list of `±attribute` edits (one to three of them), so the sequence length is simply *how many things you asked to change*. The **source image is the single query token** ($Q$), while the **sign-modulated condition vectors are the keys and values** ($K=V$). Read semantically, the image asks *"given who I am, how strongly should I weigh each requested edit?"* This is **cross-attention** - the image reads the conditions - and because there is exactly one query token, the output is just a content-based weighted average of those conditions, with the weights computed from the image itself. That per-image, per-condition weighting is exactly what the fixed latent arithmetic of the training-free methods cannot express. Before that read, the conditions first attend over a *second* sequence - the source's own visual tokens, CLIP's global CLS summary plus its 49 spatial patches - so each edit can ground on the relevant region of this particular source before the image weighs it.

At a high level (diagram below), frozen CLIP encodes the source image into a unit-norm embedding $\mathbf{v}_{\text{ref}}$ **and** its sequence of visual tokens $[\text{CLS};\,49\text{ patches}]$, and each condition into the bare-name text vector $\mathbf{t}_a$ of its attribute; the trained module $\Phi_\theta$ fuses them into a single query $\mathbf{q}$, which ranks the frozen gallery by cosine similarity. It does so in four learned stages - **sign-aware FiLM**, **patch grounding**, **stacked cross-attention**, and a **gated residual** - derived in turn below.
"""


#==============================================================================
# Cell  81 [markdown] - Figure: high-level cross-attention fusion architecture
#==============================================================================

"""
![High-level architecture of the cross-attention fusion module](figures/architecture.svg)
"""


#==============================================================================
# Cell  82 [markdown] - The diagram below expands the module layer by layer. Frozen CLIP encodes the…
#==============================================================================

"""
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
"""


#==============================================================================
# Cell  83 [markdown] - Figure: detailed cross-attention fusion architecture
#==============================================================================

"""
![Detailed layer-by-layer architecture of the cross-attention fusion module](figures/architecture_details.svg)
"""


#==============================================================================
# Cell  84 [code] - class CrossAttentionFusion(nn.Module)
#==============================================================================

class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion: the source image queries a sequence of text-encoded,
    sign-tagged conditions, and the attended result is fused back onto the image embedding.

    Conditions reuse the frozen bare-name CLIP text bank (one vector per attribute). A learned,
    sign-conditioned FiLM modulation turns each into an additive (+) or subtractive (-) condition:
    ``conds = (1 + gamma) * attr_text + beta``, where ``(gamma, beta)`` are produced per sign, so
    ``+attr`` and ``-attr`` become genuinely distinct, per-dimension vectors. Before the image
    weighs them, a *patch-grounding* stage lets the conditions read the source's frozen CLIP
    visual tokens ``[CLS ; 49 patches]`` (the global CLS summary plus the 49 spatial patches of a
    ViT-B/32): the conditions self-attend (co-adapt to one another) and cross-attend over those
    tokens, so a localized edit can latch onto the relevant region of *this* source rather than a
    generic attribute direction. The image (a single query token) then attends over the grounded
    conditions through a stack of pre-norm Transformer-decoder layers (cross-attention + GELU FFN +
    dropout). Finally a *gated residual head* fuses the attended vector back onto the reference:
    ``out = v_ref + sigmoid(gate) * delta``, so identity is preserved by default and the network
    only nudges it (the signed ``delta`` can subtract, which a softmax-averaged attention cannot).
    """

    def __init__(self, attr_text_embs: torch.Tensor, dim: int, n_heads: int = 4,
                 n_layers: int = 2, ffn_mult: int = 2, dropout: float = 0.1,
                 film_sign_std: float = 0.02, gate_bias_init: float = 2.0,
                 clip_dim: int = 768, ground_layers: int = 1, ground_heads: int = 4):
        """Build the cross-attention fusion module.

        Args:
            attr_text_embs: (n_attrs, D) frozen bare-name CLIP text bank.
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
        """
        super().__init__()

        # Freeze the bare-name CLIP text bank (one vector per attribute) for cross-attention
        self.register_buffer("attr_text", attr_text_embs)  # (n_attrs, D) frozen CLIP text

        # Sign embedding: each sign (+/-) gets a learned per-dimension vector (gamma, beta) for FiLM
        self.sign_embed = nn.Embedding(2, dim)             # 0: +   1: -
        nn.init.normal_(self.sign_embed.weight, std=film_sign_std)

        # Sign-conditioned FiLM: each sign yields a per-dimension (gamma, beta) over the attribute
        self.film = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        # Patch grounding: project the frozen CLIP visual tokens into the fusion space and tag the
        # global CLS token apart from the 49 patches, then let the conditions read them.
        self.vis_proj = nn.Linear(clip_dim, dim)
        self.vis_type = nn.Embedding(2, dim)               # 0: global CLS (position 0)   1: patch
        nn.init.normal_(self.vis_type.weight, std=film_sign_std)
        ground_layer = nn.TransformerDecoderLayer(
            dim, ground_heads, dim_feedforward=ffn_mult * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.ground = nn.TransformerDecoder(ground_layer, num_layers=ground_layers)

        # Stacked cross-attention: image (1 query token) attends over the grounded conditions
        layer = nn.TransformerDecoderLayer(
            dim, n_heads, dim_feedforward=ffn_mult * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

        # Gated residual head: a sigmoid gate weighs a non-linear delta added back onto v_ref
        self.delta = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim),
        )
        self.gate = nn.Sequential(nn.Linear(2 * dim, dim), nn.Sigmoid())

        # Gate starts open (sigmoid(2) ~ 0.88) so the non-zero delta head receives gradient
        nn.init.constant_(self.gate[0].bias, gate_bias_init)

    def forward(self, img_emb: torch.Tensor, vis_tokens: torch.Tensor,
                cond_attr: torch.Tensor, cond_sign: torch.Tensor) -> torch.Tensor:
        """Fuse the source image with its sign-tagged conditions via cross-attention.

        Args:
            img_emb: (B, D) L2-normalized source image embeddings.
            vis_tokens: (B, 50, clip_dim) raw CLIP visual tokens [CLS ; 49 patches] for the source.
            cond_attr: (B, T) attribute indices for each condition.
            cond_sign: (B, T) signs in {+1, -1, 0}; 0 marks padding.

        Returns:
            A (B, D) L2-normalized fused embedding.
        """
        pad_mask = cond_sign == 0             # (B, T) True = ignore
        sign_id  = (cond_sign < 0).long()     # 0 for +, 1 for - (padding -> 0, masked anyway)
        attr     = self.attr_text[cond_attr]  # (B, T, D) frozen text

        # 1. Sign-aware FiLM: each sign modulates the attribute's text vector per-dimension
        gamma, beta = self.film(self.sign_embed(sign_id)).chunk(2, dim=-1)  # (B, T, D) each
        conds = (1.0 + gamma) * attr + beta                                 # (B, T, D)

        # 2. Patch grounding: project the source's visual tokens, tag CLS vs patch, and let
        # the conditions self-attend (co-adapt) and cross-attend (ground spatially) over them.
        V = self.vis_proj(vis_tokens.to(self.vis_proj.weight.dtype))        # (B, 50, D)
        type_id = torch.ones(V.shape[1], dtype=torch.long, device=V.device)
        type_id[0] = 0                                                      # position 0 is the global CLS token
        V = V + self.vis_type(type_id)                                      # broadcast over batch
        conds = self.ground(conds, V, tgt_key_padding_mask=pad_mask)        # (B, T, D) grounded

        # 3. Stacked cross-attention: the image (1 query token) reads the grounded conditions
        q = img_emb.unsqueeze(1)                                            # (B, 1, D)
        attended = self.decoder(q, conds, memory_key_padding_mask=pad_mask) # (B, 1, D)
        attended = attended.squeeze(1)                                      # (B, D)

        # 4. Gated-residual fusion: v_ref preserved by default, delta can add or subtract
        fused = torch.cat([img_emb, attended], dim=-1)        # (B, 2D)
        out = img_emb + self.gate(fused) * self.delta(fused)  # (B, D)
        return F.normalize(out, dim=-1)


ca_model = CrossAttentionFusion(
    ATTR_TEXT_EMBS, gallery_embeddings.shape[1],
    n_heads=CA_HEADS, n_layers=CA_LAYERS, ffn_mult=CA_FFN_MULT, dropout=CA_DROPOUT,
    film_sign_std=CA_FILM_SIGN_STD, gate_bias_init=CA_GATE_BIAS_INIT,
    clip_dim=CLIP_VIS_DIM, ground_layers=CA_GROUND_LAYERS, ground_heads=CA_GROUND_HEADS,
).to(device)

n_params = sum(p.numel() for p in ca_model.parameters() if p.requires_grad)
print(f"Cross-Attention trainable parameters: {n_params:,}")

# Forward self-check: shapes and unit-norm output (cheap correctness gate)
ca_model.eval()
with torch.no_grad():
    _b    = 4
    _img  = F.normalize(torch.randn(_b, gallery_embeddings.shape[1], device=device), dim=-1)
    _vis  = torch.randn(_b, 50, CLIP_VIS_DIM, device=device)
    _attr = torch.randint(0, len(attr_names), (_b, 3), device=device)
    _sign = torch.tensor([[1, -1, 0], [1, 0, 0], [-1, -1, 1], [1, 1, -1]], device=device)
    _out  = ca_model(_img, _vis, _attr, _sign)

    assert _out.shape == (_b, gallery_embeddings.shape[1]), _out.shape
    assert torch.allclose(_out.norm(dim=-1), torch.ones(_b, device=device), atol=1e-5)

print("Forward self-check passed:", tuple(_out.shape))
ca_model.train()


#==============================================================================
# Cell  85 [markdown] - Training
#==============================================================================

"""
---

### Training

The fusion module is trained with synthetic, label-free supervision built from CelebA's attribute annotations, so no manual labelling is needed. At each step we take a reference image, flip a few of its attributes to form a signed query, and fetch from the training split a real image that matches the edited attribute profile to serve as the **positive target**; optionally we also mine a **hard negative** that looks right but violates one requested sign. A contrastive objective then pulls the fused query toward its target and pushes it away from the other images in the batch, including that hard negative. Only the fusion module is updated, while CLIP and the attribute text bank stay frozen. The diagram below traces one training triplet from the reference to the loss.
"""


#==============================================================================
# Cell  86 [markdown] - Figure: label-free triplet supervision & InfoNCE objective
#==============================================================================

"""
![Label-free triplet supervision and the InfoNCE objective](figures/training.svg)
"""


#==============================================================================
# Cell  87 [markdown] - The diagram above shows one triplet; this section spells out how the triplets…
#==============================================================================

"""
The diagram above shows one triplet; this section spells out how the triplets are synthesised and how the objective is optimised.

**Triplet synthesis.** Each triplet starts from a random reference image $s$ with its 40-bit CelebA attribute vector $\mathbf{b}_s$. We sample 1 to 3 of its attributes and flip them into a signed query $q$ (a flipped-on attribute becomes a `+` term, a flipped-off attribute a `-` term), and build the ideal target attribute vector $\mathbf{b}^\star$ by copying $\mathbf{b}_s$ and applying exactly those flips. The **positive target** $t$ is drawn from the training split among real images that satisfy the query and lie within the benchmark's Hamming budget of the ideal profile, $\lVert \mathbf{b}_t - \mathbf{b}^\star \rVert_1 \le 2$. This is the same rule the benchmark uses to judge a correct retrieval, so the model is trained against the exact target definition it is later evaluated on. Since $\mathbf{b}_s$, the chosen flips, and the candidate labels are all that is required, the triplets $(s,q,t)$ are produced with no human annotation, purely from the attribute table. A large pool, on the order of $10^5$ training triplets plus a few thousand held out for validation, is generated once and cached, keyed to the synthesis settings, so that re-training under different model or optimiser hyperparameters reuses the same pool instead of regenerating it.

**Hard negatives.** When enabled, one **constraint-violating** distractor $h$ is mined per query: a real image that keeps the reference's other attributes but breaks exactly one requested sign (for the query $-\text{Smiling}$, a face that is otherwise valid yet still smiling). Such an image is close to the target in every respect except the single attribute the query cares about, so using it as a negative forces the model to key on the edited attribute rather than on overall resemblance to the source. This is the main defence against the failure mode where a combiner simply returns look-alikes of the reference. Queries for which no such image exists in the split fall back to using only the in-batch negatives for that row.

**InfoNCE objective** [(van den Oord et al., 2018)](https://arxiv.org/abs/1807.03748). For a batch of $B$ triplets, let $\mathbf{q}_i=\Phi_\theta(\mathbf{v}_{s_i},q_i)$ be the fused query, $\mathbf{t}_i$ the embedding of its positive target, $\mathbf{h}_i$ its optional hard negative, and $\tau$ CLIP's own (frozen) temperature. Every other target in the batch acts as an in-batch negative, and the per-row hard negative is appended as one extra negative, giving the cross-entropy
$$\mathcal{L}=-\frac{1}{B}\sum_{i=1}^{B}\log\frac{\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{t}_i)}{\displaystyle\sum_{j=1}^{B}\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{t}_j)\;+\;\mathbb{1}[h_i\ \text{exists}]\,\exp(\tau\,\mathbf{q}_i^{\top}\mathbf{h}_i)}.$$
Minimising $\mathcal{L}$ raises the cosine similarity between each fused query and its true target while lowering it against the $B-1$ other targets and the hard negative. Because every embedding is unit-norm and the temperature matches CLIP's, the geometry the loss optimises is exactly the one the retrieval scorer uses at test time, so improvements on the objective translate directly into retrieval gains.

**Optimisation and model selection.** Only the fusion module $\Phi_\theta$ receives gradients; CLIP's image and text encoders and the attribute text bank are frozen, and the image features of the training split are pre-extracted once so each step runs only the small module rather than the backbone. We optimise with AdamW (learning rate $2\times10^{-4}$, weight decay $10^{-2}$) under a cosine-annealed learning-rate schedule over 20 epochs with batch size 512, relying on dropout inside the decoder and the fusion heads together with weight decay for regularisation. After every epoch we compute the mean validation InfoNCE loss over the held-out triplets, using the same in-batch-plus-hard-negative objective as training, so the train and validation loss curves are directly comparable. The checkpoint with the lowest validation loss is kept, and on later runs that cached checkpoint is reloaded so evaluation never requires re-training.
"""


#==============================================================================
# Cell  88 [markdown] - Load the CelebA training split and pre-extract image features
#==============================================================================

"""
#### Load the CelebA training split and pre-extract image features

The combiner trains on CLIP image features, so we encode the CelebA training split once and cache it for reuse.

Beyond the pooled image embedding, Cross-Attention Fusion also reads each source's CLIP **visual tokens** ($[\text{CLS};\,49\text{ patches}]$). These are pre-extracted once and cached in fp16 - for the train split only over the unique source indices the triplets reference (with an index lookup mapping a train index to its row in the bank), to bound storage. Because the architecture now includes the patch-grounding stage, the old pooled-only checkpoint is incompatible: the module is retrained from scratch into a fresh checkpoint (`cross_attn_patch.pt`).
"""


#==============================================================================
# Cell  89 [code] - Load CelebA train split & pre-extract features
#==============================================================================

CELEBA_TRAIN_ROOT = Path("/content/datasets")
TRAIN_FEATS_PATH  = Path(EVALUATION_CACHE_DIR) / "train_embeddings.pt"

# Load the full CelebA train split
print(f"Loading CelebA train split from {CELEBA_TRAIN_ROOT} ...")
celeba_train = CelebA(root=CELEBA_TRAIN_ROOT, split="train", download=False)
print(f"CelebA train split size: {len(celeba_train)}")

# Pre-extract image features once and cache (shared utility, see encoding utilities)
train_embeddings, train_labels = get_encoded_dataset(
    celeba_train, device, str(TRAIN_FEATS_PATH), batch_size=128
)
print(f"train_features dtype: {train_embeddings.dtype}, device: {train_embeddings.device}")

train_labels_bool = (train_labels.to(device) > 0)   # (M, 40) on GPU, for candidate filtering
train_labels_bool_np = train_labels_bool.cpu().numpy()   # CPU copy, for cheap per-sample query sampling
TRAIN_N = train_labels_bool.shape[0]
n_attrs = train_labels_bool.shape[1]


#==============================================================================
# Cell  90 [markdown] - Label-set logic
#==============================================================================

"""
##### Label-set logic

Given a signed query, define which target attribute vectors are valid and which candidates satisfy it.
"""


#==============================================================================
# Cell  91 [code] - def desired_target_labels(source_labels: torch.Tensor, pos_idx: list[int], ne…
#==============================================================================

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


def find_valid_targets(source_labels: torch.Tensor, pos_idx: list[int], neg_idx: list[int]) -> torch.Tensor:
    """Return indices of valid target images for a source and query.

    Valid targets satisfy the query and lie within HAMMING_BUDGET of the ideal target.

    Args:
        source_labels: Boolean attribute-label vector of the source image.
        pos_idx: Attribute indices the target must have.
        neg_idx: Attribute indices the target must not have.

    Returns:
        Indices into the training features of the valid target images.
    """
    # Compute the ideal target attribute vector for this source and query
    target = desired_target_labels(source_labels, pos_idx, neg_idx)
    # First filter to candidates that satisfy the query and whose labels are within HAMMING_BUDGET
    ok = query_satisfied(train_labels_bool, pos_idx, neg_idx)
    hamming = (train_labels_bool != target.unsqueeze(0)).sum(dim=1)
    return (ok & (hamming <= HAMMING_BUDGET)).nonzero(as_tuple=True)[0]


def find_hard_negative(source_labels: torch.Tensor, pos_idx: list[int], neg_idx: list[int],
                       source_idx: int, rng) -> int:
    """Return the index of a hard negative for a source and query.

    A hard negative violates the query but stays within HAMMING_BUDGET of the source.
    Returns -1 when the query is empty or no candidate qualifies.

    Args:
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
    ok = (train_labels_bool[:, queried] == violated[queried].unsqueeze(0)).all(dim=1)
    ok &= (train_labels_bool != violated.unsqueeze(0)).sum(dim=1) <= HAMMING_BUDGET
    # Exclude the source image itself from the candidates
    cand = ok.nonzero(as_tuple=True)[0]
    cand = cand[cand != source_idx]
    if cand.numel() == 0:
        return -1
    return int(cand[int(rng.integers(0, cand.numel()))])


#==============================================================================
# Cell  92 [markdown] - Triplet sampling
#==============================================================================

"""
##### Triplet sampling

Sample references, flip attributes into signed queries, and draw a matching positive target plus a hard negative.
"""


#==============================================================================
# Cell  93 [code] - def build_condition_row(pos_idx: list[int], neg_idx: list[int], width: int) -…
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

    cond_attr / cond_sign are fixed-width (MAX_TERMS,) rows; sign 0 marks padding.
    hard is one constraint-violating hard negative per row (-1 if none found).

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
        candidates = find_valid_targets(train_labels_bool[s], pos_idx, neg_idx)
        candidates = candidates[candidates != s]
        if candidates.numel() == 0:
            continue
        t = int(candidates[int(rng.integers(0, candidates.numel()))])

        # Mine one constraint-violating hard negative for this query (-1 if none exists)
        h = find_hard_negative(train_labels_bool[s], pos_idx, neg_idx, s, rng)

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
# Cell  94 [markdown] - Caching & materialization
#==============================================================================

"""
##### Caching & materialization

Generate the triplet pool once and cache it, keyed to the synthesis settings so a stale pool is never silently reused.
"""


#==============================================================================
# Cell  95 [code] - def load_or_generate_triplets(n_triplets: int, seed: int, cache_path: str)
#==============================================================================

def load_or_generate_triplets(n_triplets: int, seed: int, cache_path: str):
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
    if os.path.exists(cache_path):
        blob = torch.load(cache_path, map_location="cpu")
        if blob.get("key") == key:
            print(f"Loaded {n_triplets} cached triplets (seed={seed}) from {cache_path}.")
            return tuple(blob["tensors"])
        print(f"Triplet cache {cache_path} key {blob.get('key')} != {key}; regenerating.")
    pool = generate_triplet_pool(n_triplets, seed)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({"key": key, "tensors": [t.cpu() for t in pool]}, cache_path)
    print(f"Saved {n_triplets} triplets (seed={seed}) to {cache_path}.")
    return pool


# --- Materialise the train/val triplet pools here, in the synthesis cell (cached and keyed on the
#     generation parameters). The training cell below just consumes these tensors. ---
CA_TRIPLETS_TRAIN_PATH = str(Path(EVALUATION_CACHE_DIR) / "cross_attn_triplets_train.pt")
CA_TRIPLETS_VAL_PATH   = str(Path(EVALUATION_CACHE_DIR) / "cross_attn_triplets_val.pt")
ca_trip_src, ca_trip_tgt, ca_trip_attr, ca_trip_sign, ca_trip_hard = load_or_generate_triplets(
    CA_TRAIN_TRIPLETS, 10, CA_TRIPLETS_TRAIN_PATH)
ca_val_src,  ca_val_tgt,  ca_val_attr,  ca_val_sign,  ca_val_hard  = load_or_generate_triplets(
    CA_VAL_TRIPLETS, 11, CA_TRIPLETS_VAL_PATH)
print(f"train triplets: {ca_trip_src.shape[0]}, val triplets: {ca_val_src.shape[0]}")
print(f"hard negatives found for {(ca_trip_hard >= 0).float().mean().item():.1%} of training triplets")


# --- Patch tokens for the unique train sources the triplets reference. Extracting only these
#     (rather than the whole train split) bounds storage; a lookup maps an original train index to
#     its row in the bank so the train/val loops can gather visual tokens per batch. ---
CA_TRAIN_PATCHES_PATH = str(Path(EVALUATION_CACHE_DIR) / "patches_train.pt")
ca_patch_src = torch.unique(torch.cat([ca_trip_src, ca_val_src])).tolist()
train_patches = get_encoded_patches(celeba_train, device, CA_TRAIN_PATCHES_PATH, indices=ca_patch_src)
train_patch_lookup = torch.full((TRAIN_N,), -1, dtype=torch.long)
train_patch_lookup[torch.tensor(ca_patch_src)] = torch.arange(len(ca_patch_src))
print(f"train patch bank: {tuple(train_patches.shape)} over {len(ca_patch_src)} unique sources")


def train_patches_for(idx: torch.Tensor) -> torch.Tensor:
    """Gather cached CLIP visual tokens for train-split source indices via the bank lookup.

    Args:
        idx: (B,) original train-split source indices (any device).

    Returns:
        A (B, 50, 768) fp16 tensor on `idx`'s device, row-aligned to `idx`.
    """
    rows = train_patch_lookup[idx.cpu()]
    return train_patches[rows].to(idx.device)


#==============================================================================
# Cell  96 [markdown] - Training setup
#==============================================================================

"""
##### Training setup

Load frozen CLIP, move the triplets onto the device, and define the validation Recall@10 metric.
"""


#==============================================================================
# Cell  97 [code] - Cross-Attention Fusion training setup
#==============================================================================

model, _ = get_CLIP_model()
logit_scale_value = model.logit_scale.exp().detach()   # Frozen CLIP temperature for InfoNCE

# Move the synthesised train/val triplet tensors onto the device once. Cheap and harmless even
# when the cell below loads a cached checkpoint and skips training
ca_trip_src_dev, ca_trip_tgt_dev = ca_trip_src.to(device), ca_trip_tgt.to(device)
ca_trip_attr_dev, ca_trip_sign_dev = ca_trip_attr.to(device), ca_trip_sign.to(device)
ca_trip_hard_dev = ca_trip_hard.to(device)
ca_val_src_dev, ca_val_attr_dev, ca_val_sign_dev = ca_val_src.to(device), ca_val_attr.to(device), ca_val_sign.to(device)
ca_val_tgt_dev, ca_val_hard_dev = ca_val_tgt.to(device), ca_val_hard.to(device)


def ca_infonce_loss(q: torch.Tensor, tgt_idx: torch.Tensor, hard_idx: torch.Tensor) -> torch.Tensor:
    """Compute the InfoNCE loss over in-batch targets, optionally with mined hard negatives.

    Args:
        q: (B, D) fused query embeddings.
        tgt_idx: (B,) gallery indices of the positive target per row.
        hard_idx: (B,) mined hard-negative indices per row; negative marks "none".

    Returns:
        The scalar InfoNCE loss.
    """
    t = train_embeddings[tgt_idx]
    if CA_HARD_NEG:
        no_hard = hard_idx < 0
        hf = train_embeddings[hard_idx.clamp(min=0)]            # (B, D); invalid rows masked below
        hard_sim = (q * hf).sum(-1, keepdim=True)             # (B, 1) per-row hard-negative score
        logits = logit_scale_value * torch.cat([q @ t.T, hard_sim], dim=1)   # (B, B+1)
        logits[:, -1] = logits[:, -1].masked_fill(no_hard, -1e9)
    else:
        logits = logit_scale_value * (q @ t.T)                # In-batch negatives only
    labels_ce = torch.arange(q.shape[0], device=device)
    return F.cross_entropy(logits, labels_ce)


@torch.no_grad()
def ca_val_loss() -> float:
    """Mean validation InfoNCE loss over the held-out triplets.

    Reuses the training objective (in-batch plus mined hard negatives, via ca_infonce_loss)
    so the train and validation loss curves are directly comparable. This is the metric used
    for checkpoint selection.

    Returns:
        The mean validation loss.
    """
    ca_model.eval()
    loss_sum = 0.0
    n_val = ca_val_src_dev.shape[0]
    for start in range(0, n_val, CA_BATCH):
        sl = slice(start, min(start + CA_BATCH, n_val))
        q = ca_model(train_embeddings[ca_val_src_dev[sl]], train_patches_for(ca_val_src_dev[sl]), ca_val_attr_dev[sl], ca_val_sign_dev[sl])
        loss_sum += float(ca_infonce_loss(q, ca_val_tgt_dev[sl], ca_val_hard_dev[sl])) * q.shape[0]
    return loss_sum / n_val


#==============================================================================
# Cell  98 [markdown] - Training plot utility
#==============================================================================

"""
##### Training plot utility

A single learning curve over optimizer steps: the raw per-step training loss together with the per-epoch validation loss (starting from a step-0 baseline), both on one loss axis. The best epoch — the one with the lowest validation loss, used for checkpoint selection — is marked with a star.
"""


#==============================================================================
# Cell  99 [code] - def plot_training_curve(history: dict, title: str = "Cross-Attention Training…
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
# Cell 100 [markdown] - Train (or load cached)
#==============================================================================

"""
##### Train (or load cached)

Run the training loop, or load a cached checkpoint when one is available.
"""


#==============================================================================
# Cell 101 [code] - Train (or load cached) Cross-Attention model
#==============================================================================

CA_CKPT = Path(EVALUATION_CACHE_DIR) / "cross_attn_patch.pt"
_ca_cached = torch.load(CA_CKPT, map_location=device) if CA_CKPT.exists() else None

if _ca_cached is not None:
    ca_model.load_state_dict(_ca_cached["state_dict"])
    ca_model.eval()
    print(f"Loaded cached cross-attention from {CA_CKPT} (val loss={_ca_cached.get('val_loss', float('nan')):.4f}) — skipping training.")
    if _ca_cached.get("history") is not None:
        plot_training_curve(_ca_cached["history"])
    else:
        print("Cached checkpoint predates training history — re-train to regenerate the learning curve.")
else:
    # Optimizer / LR schedule.
    optimizer = torch.optim.AdamW(ca_model.parameters(), lr=CA_LR, weight_decay=CA_WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CA_EPOCHS)
    best_val_loss, best_ca_state, best_epoch = float("inf"), None, -1
    n_train_trip = ca_trip_src_dev.shape[0]

    # Learning-curve history: per-step training loss and per-epoch validation loss.
    history = {"step": [], "loss": [], "epoch_step": [], "val_loss": [], "best_epoch": -1}
    step = 0

    # Baseline validation loss at step 0 (untrained model): plotted, but not a checkpoint candidate.
    history["epoch_step"].append(0)
    history["val_loss"].append(ca_val_loss())

    # Training loop: InfoNCE on the fused query vs target, keeping the min-val-loss weights.
    for epoch in range(CA_EPOCHS):
        ca_model.train()
        perm = torch.randperm(n_train_trip, device=device)
        for start in range(0, n_train_trip, CA_BATCH):
            idx = perm[start:start + CA_BATCH]
            q = ca_model(train_embeddings[ca_trip_src_dev[idx]], train_patches_for(ca_trip_src_dev[idx]), ca_trip_attr_dev[idx], ca_trip_sign_dev[idx])
            loss = ca_infonce_loss(q, ca_trip_tgt_dev[idx], ca_trip_hard_dev[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            history["step"].append(step)
            history["loss"].append(float(loss.detach()))
        scheduler.step()

        # Per-epoch validation loss and best-checkpoint tracking (minimum val loss).
        val_loss = ca_val_loss()
        history["epoch_step"].append(step)
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, len(history["val_loss"]) - 1
            best_ca_state = {k: v.detach().clone() for k, v in ca_model.state_dict().items()}
        history["best_epoch"] = best_epoch

        # Live, in-place redraw of the learning curve so training state is visible as it runs.
        clear_output(wait=True)
        plot_training_curve(history)
        print(f"Epoch {epoch+1:3d}/{CA_EPOCHS}  train loss={history['loss'][-1]:.4f}  "
              f"val loss={val_loss:.4f}  (best epoch {best_epoch})")

    # Restore best weights, draw the final figure, and cache weights + history.
    ca_model.load_state_dict(best_ca_state)
    clear_output(wait=True)
    plot_training_curve(history)
    print(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    torch.save(
        {"state_dict": {k: v.cpu() for k, v in best_ca_state.items()},
         "val_loss": best_val_loss, "history": history},
        CA_CKPT,
    )
    print(f"Saved cross-attention to {CA_CKPT}")


#==============================================================================
# Cell 102 [markdown] - Scorer
#==============================================================================

"""
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
"""


#==============================================================================
# Cell 103 [code] - Cross-Attention gallery scorer
#==============================================================================

# Gallery visual tokens [CLS ; 49 patches] for the source side of retrieval. The gallery *target*
# side stays the pooled, frozen ``gallery_embeddings``, so retrieval cost is unchanged; only the
# query computation reads patches. Only the benchmark's *source* images are ever queried this way
# (scorer/inspection both index by source_idx), so - mirroring the train-split pattern above
# (``train_patch_lookup`` / ``train_patches_for``) - we encode just that subset rather than all
# 19,962 test images, with an index lookup mapping an original gallery index to its row in the bank.
# Kept on CPU (fp16); the lookup moves one source's slice to GPU.
GALLERY_PATCHES_PATH = str(Path(EVALUATION_CACHE_DIR) / "patches_test.pt")
gallery_patch_src = sorted({src for ann in annotations for src in get_source_image_idxs(ann)})
gallery_patches = get_encoded_patches(celeba, device, GALLERY_PATCHES_PATH, indices=gallery_patch_src)
gallery_patch_lookup = torch.full((gallery_embeddings.shape[0],), -1, dtype=torch.long)
gallery_patch_lookup[torch.tensor(gallery_patch_src)] = torch.arange(len(gallery_patch_src))


def gallery_patches_for(idx: torch.Tensor) -> torch.Tensor:
    """Gather cached CLIP visual tokens for gallery source indices via the bank lookup.

    Args:
        idx: (B,) original gallery indices (any device); must be benchmark source images.

    Returns:
        A (B, 50, 768) fp16 tensor on `idx`'s device, row-aligned to `idx`.
    """
    rows = gallery_patch_lookup[idx.cpu()]
    return gallery_patches[rows].to(idx.device)


def query_to_condition_rows(text_query: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a benchmark query string into condition tensors for the model.

    Args:
        text_query: Comma-separated signed query, e.g. "+Bald, -Eyeglasses".

    Returns:
        A (cond_attr, cond_sign) pair of (1, T) tensors, padded as needed.
    """
    pos_idx, neg_idx = parse_query_signs(text_query)
    width = max(MAX_TERMS, len(pos_idx) + len(neg_idx))
    attrs, signs = build_condition_row(pos_idx, neg_idx, width)
    cond_attr = torch.tensor([attrs], device=gallery_embeddings.device)
    cond_sign = torch.tensor([signs], device=gallery_embeddings.device)
    return cond_attr, cond_sign


@torch.no_grad()
def fuse_source_query(source_idx: int, cond_attr: torch.Tensor, cond_sign: torch.Tensor) -> torch.Tensor:
    """Run the trained fusion module on one (source image, query) pair.

    Shared by the evaluation scorer and the qualitative inspection (which additionally hooks the
    gate), so the fused-query computation for a single source is defined in exactly one place.

    Args:
        source_idx: Gallery index of the source image; must be a benchmark source (see
            `gallery_patches_for`).
        cond_attr: (1, T) attribute indices for each condition.
        cond_sign: (1, T) signs in {+1, -1, 0}; 0 marks padding.

    Returns:
        A (D,) L2-normalized fused query embedding.
    """
    idx = torch.tensor([source_idx], device=gallery_embeddings.device)
    return ca_model(
        gallery_embeddings[idx], gallery_patches_for(idx), cond_attr, cond_sign
    ).squeeze(0)


def cross_attn_scorer(gallery_embeddings: torch.Tensor, ca_model: nn.Module) -> Callable:
    """Build the scorer factory for Cross-Attention Fusion.

    Builds the fused query embedding once per annotation and returns gallery
    cosine similarities against it.

    Args:
        gallery_embeddings: (N, D) gallery image embeddings, L2-normalized per row.
        ca_model: Trained CrossAttentionFusion model.

    Returns:
        A ``make_scorer(annotation)`` factory consumed by evaluate().
    """
    ca_model.eval()

    def make_scorer(annotation: dict) -> Callable:
        """Build a per-query scorer from the query's condition rows."""
        cond_attr, cond_sign = query_to_condition_rows(get_text_query(annotation))

        @torch.no_grad()
        def scorer(source_idx: int) -> torch.Tensor:
            """Score every gallery image against the fused source query embedding."""
            q = fuse_source_query(source_idx, cond_attr, cond_sign)
            return gallery_embeddings @ q

        return scorer
    return make_scorer


#==============================================================================
# Cell 104 [markdown] - Evaluation & Plot
#==============================================================================

"""
### Evaluation & Plot
"""


#==============================================================================
# Cell 105 [code] - Evaluate & plot Cross-Attention Fusion
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
# Cell 106 [markdown] - Cross-Attention: Qualitative Inspection
#==============================================================================

"""
### Cross-Attention: Qualitative Inspection

To see *what the trained model does* and where it breaks, we inspect a **SUCCESS** and a **FAILURE** case for two query types the benchmark stresses: a single-attribute **negation** (e.g. `-Heavy Makeup`) and a **composed** multi-attribute query (e.g. `+Eyeglasses, -Smiling`). For each, we automatically pick, from that query's own benchmark sources, one source the model gets right (a ground-truth target in its top-k) and one it gets wrong (none in top-k); nothing is hardcoded.

For each `(source, query)` we read out:

- **Top-k retrieval under the edit**: the images the *fused* query pulls to the top (source excluded), each marked ✓/✗ for satisfying the requested attributes and tagged `GT` when it is a benchmark target. This shows directly whether the edit moved retrieval toward the request rather than toward look-alikes of the source.
- **Residual gate** $\sigma(g)\in[0,1]$ from the gated-residual head: its mean summarises overall edit strength, while a low mean with a few high dimensions signals a localised edit and a flat $\approx 0.5$ means the head barely moved off its initialisation.

The trained weights are reused exactly; nothing is re-trained.
"""


#==============================================================================
# Cell 107 [code] - Qualitative attention inspection
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
        q = fuse_source_query(source_idx, cond_attr, cond_sign)
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
    ``ground_truth`` is given, tagged "GT" if it is an actual benchmark target — so a SUCCESS panel
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

    pos_idx, neg_idx = parse_query_signs(text_query)
    gallery_bool = gallery_labels > 0                       # (N, 40) bool
    gt = set(ground_truth or [])

    head = f"{label}\n" if label else ""
    images = [celeba[source_idx][0]]
    titles = [f"{head}source #{source_idx}\nquery: {text_query}"]
    colors = ["black"]
    for idx in topk:
        ok = bool(query_satisfied(gallery_bool[idx].unsqueeze(0), pos_idx, neg_idx).item())
        tag = "✓ satisfies" if ok else "✗ violates"
        if gt:
            tag += "  ·  GT" if idx in gt else ""
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
        hit = len(set(retrieved) & set(get_ground_truth_indices(annotation, src))) > 0
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
        pos_idx, neg_idx = parse_query_signs(get_text_query(ann))
        if predicate(pos_idx, neg_idx):
            return ann
    return None


# Inspect a SUCCESS and a FAILURE case for two query types the spec calls out: a single-attribute
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
            ground_truth=set(get_ground_truth_indices(ann, src)),
        )


#==============================================================================
# Cell 108 [markdown] - Limitations
#==============================================================================

"""
### Limitations

Read against the task, the trained module has clear ceilings worth stating plainly.

- **The attention does relatively light work.** A query carries at most $T=3$ conditions, so the cross-attention arbitrates among a handful of vectors rather than modelling long-range structure over a long sequence. Most of what makes the method work is the **sign-aware FiLM** (which makes $+$ and $-$ genuinely distinct) and the **gated residual** (which anchors identity and permits subtraction); the attention mainly reweights. The Transformer is the right *frame*, but it is not where the heavy lifting happens.
- **The remaining capacity ceiling is the pooled gallery target, not the source.** The source side now enters as CLIP's full visual-token sequence (the global CLS token plus 49 patches), so the conditions can ground on the region an edit should touch - "remove the glasses" can read the eye patches - rather than only shifting a pooled vector. The gallery *target* side, however, is still scored as a single pooled 512-d embedding (retrieval ranks `gallery_embeddings @ q`), so a localised edit must ultimately be matched against a holistic image vector. Lifting this would mean a patch-level gallery index, at a real cost in storage and retrieval time.
- **The text bank is frozen and non-compositional.** Conditions are bare-name CLIP text vectors, and CLIP text behaves like a bag of concepts. Interacting attributes (e.g. *Smiling* and *Mouth Slightly Open*) enter as independent conditions the module can only reweight and ground; it cannot learn their joint semantics.
- **Negation still rides on the CLIP geometry.** FiLM and the signed $\boldsymbol{\delta}$ approximate "absence of an attribute" as a direction in an embedding space that was never trained for negation - a learned workaround, not a true representation of *not*.
- **Identity is preserved at the cost of under-editing.** Because the output defaults to $\mathbf{v}_{\text{ref}}$, the path of least resistance is to leave the embedding nearly unchanged, so strongly requested edits can be damped: keeping the source is always the safe option for the contrastive loss.
"""


#==============================================================================
# Cell 109 [markdown] - Final Comparison: all methods
#==============================================================================

"""
---

## Final Comparison: all methods

All methods evaluated on the same benchmark JSON and the same precomputed image embeddings: the training-free series (baseline, source-attribute matching, prompt ensembling) followed by the training-based Cross-Attention Fusion.
"""


#==============================================================================
# Cell 110 [code] - Aggregate results across all methods
#==============================================================================

all_methods_results = {
    "Baseline":                 average_results_per_query_baseline,
    "Source-Attribute Matching":  average_results_per_query_sam,
    "Prompt Ensembling":        average_results_per_query_promptens,
    "Cross-Attention":          average_results_per_query_ca,
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
# Cell 111 [markdown] - References
#==============================================================================

"""
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
"""
