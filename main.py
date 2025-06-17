from pathlib import Path
from typing import List, Dict, Tuple, Literal
import numpy as np
from numpy.typing import NDArray
import cv2
import csv
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
from scipy.stats import gaussian_kde, skew, kurtosis
from scipy.signal import find_peaks
import math
import io
import contextlib

# ─── DATA I/O ──────────────────────────────────────────────────────────────────

def read_data(directory: Path) -> Dict[str, NDArray[np.float32]]:
    data: Dict[str, NDArray[np.float32]] = {}
    for file in directory.iterdir():
        if file.suffix == '.npy':
            data[file.stem] = np.load(file)
        else:
            print(f'File type not supported: {file.suffix}')
    return data

def organize_data(
    data: Dict[str, NDArray[np.float32]]
) -> Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]]:
    prefixes = {k.split('_',1)[0] for k in data}
    organized: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]] = {}
    for p in sorted(prefixes, key=int):
        i = int(p)
        organized[i] = (data[f'{p}_target'], data[f'{p}_pred'])
    return organized

# ─── METRICS ───────────────────────────────────────────────────────────────────

def sum_pixelwise_product(
    data: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> Dict[int, float]:
    return {i: float(np.sum(t * p)) for i, (t, p) in data.items()}

def sum_pixelwise(
    data: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> Dict[int, float]:
    """
    Returns sum of all target pixels
    """
    return {i: float(np.sum(t)) for i, (t, p) in data.items()}

def sum_gt(
    data: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> Dict[int, float]:
    """
    Returns sum of ground-truth (target) pixels for each image.
    Parameters:
        data: Dict mapping ID → (gt, pred) image arrays (unit: intensity).
    """
    return {i: float(np.sum(gt)) for i, (gt, _) in data.items()}


def sum_pred(
    data: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> Dict[int, float]:
    """
    Returns sum of prediction pixels for each image.
    Parameters:
        data: Dict mapping ID → (gt, pred) image arrays (unit: intensity).
    """
    return {i: float(np.sum(pred)) for i, (_, pred) in data.items()}

def cosine_similarity(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    a_f = a.ravel()
    b_f = b.ravel()
    dot = float(a_f @ b_f)
    na = float(np.linalg.norm(a_f))
    nb = float(np.linalg.norm(b_f))
    return dot / (na * nb) if na and nb else 0.0

def mean_squared_error(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    diff = a - b
    return float(np.mean(diff * diff))

def root_mean_squared_error(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))

# ─── REPLACED: mean_absolute_error ──────────────────────────────────────────────
# def mean_absolute_error(
#     a: NDArray[np.float32], b: NDArray[np.float32]
# ) -> float:
#     return float(np.mean(np.abs(a - b)))

def matthews_correlation(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    """
    Compute Matthews Correlation Coefficient (MCC) between two binary masks.
    Both a and b are first thresholded at 0.5 to obtain {0,1} labels.
    Returns value in [-1, +1]. If denominator is zero, returns 0.0.
    """
    # Flatten and threshold
    a_bin = (a.ravel() > 0.05).astype(np.uint8)
    b_bin = (b.ravel() > 0.05).astype(np.uint8)

    # True positives, true negatives, false positives, false negatives
    tp = int(np.sum((a_bin == 1) & (b_bin == 1)))
    tn = int(np.sum((a_bin == 0) & (b_bin == 0)))
    fp = int(np.sum((a_bin == 0) & (b_bin == 1)))
    fn = int(np.sum((a_bin == 1) & (b_bin == 0)))

    num = tp * tn - fp * fn  # numerator
    denom_term = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom_term <= 0:
        return 0.0
    return float(num) / math.sqrt(denom_term)

def r_squared(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot else 1.0

def pearson_correlation(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    af = a.ravel()
    bf = b.ravel()
    a0 = af - af.mean()
    b0 = bf - bf.mean()
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    return float((a0 @ b0) / denom) if denom else 0.0

def soft_dice(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    num = 2 * np.sum(a * b)
    den = np.sum(a * a) + np.sum(b * b)
    return float(num / den) if den else 1.0

def soft_iou(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    inter = np.sum(a * b)
    uni = np.sum(a * a) + np.sum(b * b) - inter
    return float(inter / uni) if uni else 1.0

def ssim_index(
    a: np.ndarray, b: np.ndarray
) -> float:
    """
    Returns SSIM in [-1,1], with 1.0 for a perfect match.
    """
    if float(a.max() - a.min()) == 0 and float(b.max() - b.min()) == 0:
        return 1.0
    dmin = float(min(a.min(), b.min()))
    dmax = float(max(a.max(), b.max()))
    data_range = dmax - dmin
    score, _ = structural_similarity(
        a, b, data_range=data_range, full=True
    )
    return float(score)


# PIXELWISE CONFUSION MATRIX
def pixelwise_confusion_matrix(
    pred: NDArray[np.float32],
    gt: NDArray[np.float32],
    threshold: float = 0.3
) -> NDArray[np.int64]:
    """
    Compute a 2x2 pixel-wise confusion matrix between a predicted intensity map
    and a ground-truth intensity map.

    The returned matrix is:
        [[TP, FP],
         [FN, TN]]
    where each entry is the count of pixels in that category.

    Parameters:
        pred:      Predicted intensity map (float32, arbitrary range, e.g. [0,1]), shape (H, W).  // unitless
        gt:        Ground truth intensity map (float32, arbitrary range, e.g. [0,1]), shape (H, W).  // unitless
        threshold: Value in [0,1] at which to binarize both pred and gt.  
                   Pixels ≥ threshold → 1, else → 0.                       // unitless

    Returns:
        2x2 confusion matrix (dtype=int64):
            [[ TP_count,  FP_count ],
             [ FN_count,  TN_count ]]
    """

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    pred_bin = pred >= threshold
    gt_bin   = gt   >= threshold

    tp = int(np.sum(pred_bin & gt_bin))
    fp = int(np.sum(pred_bin & (~gt_bin)))
    fn = int(np.sum((~pred_bin) & gt_bin))
    tn = int(np.sum((~pred_bin) & (~gt_bin)))

    return np.array([[tp, fp],
                     [fn, tn]], dtype=np.int64)


def pixel_accuracy(conf_mat: NDArray[np.int64]) -> float:
    """
    Compute pixel accuracy from a 2x2 confusion matrix.

    Parameters:
        conf_mat: 2x2 matrix of ints:
                  [[TP, FP],
                   [FN, TN]]

    Returns:
        Pixel accuracy as a float in [0, 1]:
        (TP + TN) / (TP + FP + FN + TN)
    """
    tp, fp = int(conf_mat[0, 0]), int(conf_mat[0, 1])
    fn, tn = int(conf_mat[1, 0]), int(conf_mat[1, 1])

    total = tp + fp + fn + tn
    return (tp + tn) / total if total > 0 else 0.0


def precision(conf_mat: NDArray[np.int64]) -> float:
    """
    Compute precision from a 2x2 confusion matrix.

    Parameters:
        conf_mat: 2x2 matrix of ints:
                  [[TP, FP],
                   [FN, TN]]

    Returns:
        Precision = TP / (TP + FP), or 0.0 if denominator is zero.
    """
    tp, fp = int(conf_mat[0, 0]), int(conf_mat[0, 1])

    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def recall(conf_mat: NDArray[np.int64]) -> float:
    """
    Compute recall (sensitivity) from a 2x2 confusion matrix.

    Parameters:
        conf_mat: 2x2 matrix of ints:
                  [[TP, FP],
                   [FN, TN]]

    Returns:
        Recall = TP / (TP + FN), or 0.0 if denominator is zero.
    """
    tp, fn = int(conf_mat[0, 0]), int(conf_mat[1, 0])

    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def specificity(conf_mat: NDArray[np.int64]) -> float:
    """
    Compute specificity (true negative rate) from a 2x2 confusion matrix.

    Parameters:
        conf_mat: 2x2 matrix of ints:
                  [[TP, FP],
                   [FN, TN]]

    Returns:
        Specificity = TN / (TN + FP), or 0.0 if denominator is zero.
    """
    fp, tn = int(conf_mat[0, 1]), int(conf_mat[1, 1])

    denom = tn + fp
    return tn / denom if denom > 0 else 0.0


def f1_score(conf_mat: NDArray[np.int64]) -> float:
    """
    Compute F1 score from a 2x2 confusion matrix.

    Parameters:
        conf_mat: 2x2 matrix of ints:
                  [[TP, FP],
                   [FN, TN]]

    Returns:
        F1 = 2 * (precision * recall) / (precision + recall),
        or 0.0 if both precision and recall are zero.
    """
    # Reuse above functions for clarity
    p = precision(conf_mat)
    r = recall(conf_mat)

    denom = p + r
    return (2 * p * r) / denom if denom > 0 else 0.0

# ─── IMAGE OPS ─────────────────────────────────────────────────────────────────

def stack_image(
    pair: Tuple[NDArray[np.float32], NDArray[np.float32]]
) -> NDArray[np.float32]:
    i1, i2 = pair
    if i1.shape != i2.shape:
        raise ValueError(f"Shape mismatch {i1.shape} vs {i2.shape}")
    return np.concatenate((i1, i2), axis=1)

def insert_vertical_line(
    image: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    h, w = image.shape[:2]
    mid = w // 2
    if image.ndim == 2:
        return np.insert(image, mid, 255, axis=1)
    c = image.shape[2]
    line = np.zeros((h, 1, c), dtype=image.dtype)
    line[..., 0] = 255
    line[..., 1] = 255
    if c == 4:
        line[..., 3] = 255
    return np.insert(image, mid, line, axis=1)

def contrast_enhance(
    image: NDArray[np.uint8], alpha: float, beta: float
) -> NDArray[np.uint8]:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def enhance_in_place(
    images: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]],
    alpha: float, beta: float
) -> None:
    for key, (img1, img2) in list(images.items()):
        images[key] = (
            contrast_enhance(img1.astype(np.uint8), alpha, beta),
            contrast_enhance(img2.astype(np.uint8), alpha, beta)
        )

def normalize_image(
    image: NDArray[np.float32]
) -> NDArray[np.uint8]:
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    range_val = max_val - min_val
    if range_val == 0.0:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image - min_val) * (255.0 / range_val)
    inverted: NDArray[np.uint8] = (255.0 - scaled).astype(np.uint8)
    return inverted

def normalize_in_place(
    images: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> None:
    for key, (img1, img2) in list(images.items()):
        images[key] = (
            normalize_image(img1),
            normalize_image(img2)
        )

def strengthen_colormap(
    image: NDArray[np.float32], strength: float
) -> NDArray[np.float32]:
    return np.power(image, strength)

def strengthen_colormap_in_place(
    images: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]],
    mode: str, strength: float
) -> None:
    for key, (gt_img, pred_img) in images.items():
        new_gt = strengthen_colormap(gt_img, strength) if mode in ('gt', 'both') else gt_img
        new_pred = strengthen_colormap(pred_img, strength) if mode in ('pred', 'both') else pred_img
        images[key] = (new_gt, new_pred)

def kill_zeros_in_place(
    data: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]],
    thresh: float,
    mode: Literal['gt', 'pred', 'sum'] = 'gt'
) -> None:
    """
    Remove entries whose selected pixelwise sum is below `thresh`.
    Parameters:
        data:   Dict mapping ID → (gt, pred) image arrays (unit: intensity).
        thresh: Minimum allowed sum (unit: intensity).
        mode:   Which sum to use – 'gt' (ground truth), 'pred' (prediction),
                or 'sum' (gt + pred).
    """
    to_remove: List[int] = []
    for idx, (gt, pred) in data.items():
        if mode == 'gt':
            total: float = float(np.sum(gt))       # unit: intensity
        elif mode == 'pred':
            total: float = float(np.sum(pred))     # unit: intensity
        else:  # mode == 'sum'
            total = float(np.sum(gt) + np.sum(pred))  # unit: intensity

        if total < thresh:
            to_remove.append(idx)

    for idx in to_remove:
        data.pop(idx, None)


def save_and_kill_zeros_in_place(
    data: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]],
    thresh: float,
    zeros_dir: Path,
    mode: Literal['gt', 'pred', 'sum'] = 'gt'
) -> None:
    """
    Save and remove entries whose selected pixelwise sum < thresh.
    Parameters:
        data:      Dict mapping ID → (gt, pred) image arrays (unit: intensity).
        thresh:    Minimum allowed sum (unit: intensity).
        zeros_dir: Directory in which to save the low-sum images.
        mode:      Which sum to use – 'gt', 'pred', or 'sum'.
    """
    zeros_dir.mkdir(parents=True, exist_ok=True)
    to_remove: List[int] = []

    for idx, (gt, pred) in data.items():
        if mode == 'gt':
            total = float(np.sum(gt))
        elif mode == 'pred':
            total = float(np.sum(pred))
        else:
            total = float(np.sum(gt) + np.sum(pred))

        if total < thresh:
            # stack side-by-side and insert a white separator
            stacked = np.concatenate((gt, pred), axis=1)
            img8 = (stacked * 255).astype(np.uint8)
            h, w = img8.shape[:2]
            mid = w // 2
            if img8.ndim == 2:
                img8 = np.insert(img8, mid, 255, axis=1)
            else:
                c = img8.shape[2]
                line = np.ones((h, 1, c), dtype=np.uint8) * 255
                img8 = np.insert(img8, mid, line, axis=1)

            fname = f"{mode}_{total:.4f}_{idx}.png"
            cv2.imwrite(str(zeros_dir / fname), img8)
            to_remove.append(idx)

    for idx in to_remove:
        data.pop(idx, None)

# ─── PLOT OPS ─────────────────────────────────────────────────────────────────

def load_metrics(csv_path: Path) -> Dict[str, List[float]]:
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        metrics: Dict[str, List[float]] = {
            name: [] for name in reader.fieldnames if name != 'idx'
        }
        for row in reader:
            for k in metrics:
                metrics[k].append(float(row[k]))
    return metrics

def plot_metrics_density(
    metrics: Dict[str, List[float]], save_path: Path
) -> None:
    names = list(metrics.keys())
    n_metrics = len(names)
    n_cols = 5
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 4),
        squeeze=False,
        sharey=False
    )
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        vals = np.array(metrics[name], dtype=float)
        kde = gaussian_kde(vals)
        dens = kde(vals)
        ax.scatter(
            np.zeros_like(vals), vals,
            c=dens, s=12, cmap=colormap.jet, rasterized=True, zorder=10
        )
        ax.axvline(0.0, color='black', linewidth=1, zorder=5)
        ax.set_title(name, pad=6)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel('Value')
        ax.grid(axis='y', linestyle=':', linewidth=0.5, zorder=0)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)

def plot_metrics_histograms(
    metrics: Dict[str, List[float]], save_path: Path
) -> None:
    names: List[str] = list(metrics.keys())
    n_metrics: int = len(names)
    n_cols: int = 5
    n_rows: int = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3, n_rows * 4),
        squeeze=False,
        sharey=False
    )
    axes_flat = axes.flatten()

    for idx, name in enumerate(names):
        ax = axes_flat[idx]
        vals = np.array(metrics[name], dtype=float)
        mask = np.isfinite(vals)
        if not mask.all():
            print(f"[{name}] dropping {len(vals) - mask.sum()} non‐finite samples")
            vals = vals[mask]
        n = vals.size
        bins = min(50, int(math.sqrt(n))) if n else 10
        try:
            ax.hist(
                vals, bins=bins,
                edgecolor='black', linewidth=0.5, zorder=2
            )
        except Exception as e:
            print(f"Skipping histogram for {name} at index {idx}: {e}")
            ax.text(
                0.5, 0.5, "error",
                ha="center", va="center",
                transform=ax.transAxes,
                color="red"
            )
        ax.set_title(name, pad=6)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.grid(axis='y', linestyle=':', linewidth=0.5, zorder=1)

    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    save_path: Path
) -> None:
    """
    Plot a 2x2 confusion matrix in scikit-learn format,
    after swapping the TN (top-left) and TP (bottom-right) entries.

    Args:
        confusion_matrix: 2x2 array [[TN, FP],
                                     [FN, TP]]
        save_path:         Path to save the resulting figure
    """
    # copy and swap TN and TP
    cm = confusion_matrix.copy()
    cm[0, 0], cm[1, 1] = cm[1, 1], cm[0, 0]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

    ax.set_title("Pixelwise Confusion Matrix", pad=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    # natural-order tick labels
    labels = ["0", "1"]
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels)

    # annotate each cell
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        count = int(cm[i, j])
        color = "white" if cm[i, j] > thresh else "black"
        ax.text(
            j, i, f"{count}",
            ha="center", va="center",
            color=color, fontsize=12
        )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


# ─── STATS OPS ─────────────────────────────────────────────────────────────────

def print_pearson_bimodal_analysis(scores: List[float]) -> None:
    arr: np.ndarray = np.array(scores, dtype=float)
    n: int = arr.size

    mean: float = float(np.mean(arr))
    std: float = float(np.std(arr, ddof=0))
    med: float = float(np.median(arr))
    sk: float = float(skew(arr))
    kurt: float = float(kurtosis(arr))

    bins = min(50, int(math.sqrt(n))) if n else 10
    counts, bin_edges = np.histogram(arr, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    peak_idxs, _ = find_peaks(counts)
    if peak_idxs.size < 2:
        peak_idxs = np.argsort(counts)[-2:]
    top2 = peak_idxs[np.argsort(counts[peak_idxs])[::-1][:2]]

    mode_locs = [centers[i] for i in top2]
    mode_counts = [int(counts[i]) for i in top2]

    half_w: float = 0.5 * std
    integrals: List[int] = []
    for loc in mode_locs:
        low, high = loc - half_w, loc + half_w
        mask = (centers >= low) & (centers <= high)
        integrals.append(int(counts[mask].sum()))

    ratio: float = integrals[0] / integrals[1] if integrals[1] else float('inf')

    print("=== Bimodal Analysis of Pearson Correlation Scores ===")
    print(f"Samples: {n}")
    print(f"Mean: {mean:.4f}, Median: {med:.4f}, Std Dev: {std:.4f}")
    print(f"Skewness: {sk:.4f}, Kurtosis: {kurt:.4f}\n")

    print("Identified Modes:")
    for i, (loc, cnt) in enumerate(zip(mode_locs, mode_counts), start=1):
        print(f" Mode {i}: location = {loc:.4f}, peak count = {cnt}")
    print()

    print("Integrals in ±0.5σ window around each mode:")
    for i, integral in enumerate(integrals, start=1):
        print(f" Mode {i} integral (±0.5σ): {integral}")
    print(f"Integral‐based mode ratio: {ratio:.4f}\n")

    print("Percentage within ±kσ of each mode:")
    for i, loc in enumerate(mode_locs, start=1):
        print(f" Mode {i} (loc={loc:.4f}):")
        for k in (0.5, 1.0, 1.5):
            low, high = loc - k * std, loc + k * std
            pct = float(np.mean((arr >= low) & (arr <= high))) * 100.0
            print(f"   ±{k}σ: {pct:.2f}% (range {low:.4f} to {high:.4f})")
        print()

# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = Path(__file__).parent

    parent_dir = Path('/home/jackplum/Documents/projects/evalgbvd/results_focal_tversky_gbblur7_final')
    child_prefix = 'tversky_focal_float_blur_'

    for child in sorted(parent_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith(child_prefix):
            continue

        base_dir = child
        data_dir = base_dir / 'train'

        ### Preprocess parameters ###
        CONTRAST_ENHANCE = 1.0
        BRIGHTEN = 0
        NORMALIZE = False
        STRENGTH_PRED = 1.4
        STRENGTH_GT = 1.0
        KILL_ZEROS_THRESH = 3 # Threshold of minimum pixelwise sum below which data will be ignored

        out_dir = script_dir / 'output' / child.name
        out_dir.mkdir(exist_ok=True, parents=True)

        # ─── Load and organize ──────────────────────────────────────────────────
        raw = read_data(data_dir)
        print(f'[{child.name}] Loaded {len(raw)} files')

        td = organize_data(raw)
        print(f'[{child.name}] Found {len(td)} (target,pred) pairs')

        if STRENGTH_PRED != 1.0:
            strengthen_colormap_in_place(td, 'pred', STRENGTH_PRED)
        if STRENGTH_GT != 1.0:
            strengthen_colormap_in_place(td, 'gt', STRENGTH_GT)
        if NORMALIZE:
            normalize_in_place(td)
        if KILL_ZEROS_THRESH is not None:
            #kill_zeros_in_place(td, KILL_ZEROS_THRESH, mode='pred')
            zeros_dir = out_dir / 'zeros'
            save_and_kill_zeros_in_place(td, KILL_ZEROS_THRESH, zeros_dir, mode='gt')

        # ─── Compute metrics ────────────────────────────────────────────────────
        rho_dict = sum_pixelwise_product(td)
        cos_dict = {i: cosine_similarity(t, p) for i, (t, p) in td.items()}

        mse_dict = {i: mean_squared_error(t, p) for i, (t, p) in td.items()}
        rmse_dict = {i: root_mean_squared_error(t, p) for i, (t, p) in td.items()}
        mcc_dict = {i: matthews_correlation(t, p) for i, (t, p) in td.items()}
        r2_dict = {i: r_squared(t, p) for i, (t, p) in td.items()}
        pc_dict = {i: pearson_correlation(t, p) for i, (t, p) in td.items()}
        sd_dict = {i: soft_dice(t, p) for i, (t, p) in td.items()}
        si_dict = {i: soft_iou(t, p) for i, (t, p) in td.items()}
        ss_dict = {i: ssim_index(t, p) for i, (t, p) in td.items()}

        # Compute per-image confusion matrices and derived stats
        conf_mats: List[NDArray[np.int64]] = []
        pa_dict: Dict[int, float] = {}
        prec_dict: Dict[int, float] = {}
        rec_dict: Dict[int, float] = {}
        spec_dict: Dict[int, float] = {}
        f1_dict: Dict[int, float] = {}

        for i, (t, p) in td.items():
            cm = pixelwise_confusion_matrix(t, p, threshold=0.1)
            conf_mats.append(cm)
            pa_dict[i] = pixel_accuracy(cm)
            prec_dict[i] = precision(cm)
            rec_dict[i] = recall(cm)
            spec_dict[i] = specificity(cm)
            f1_dict[i] = f1_score(cm)

        # ─── Save summary CSV ───────────────────────────────────────────────────
        summary_csv = out_dir / 'metrics_summary.csv'
        with open(summary_csv, 'w', newline='') as f:
            w = csv.writer(f)
            header = [
                'idx',
                'rho', 'cosine', 'mse', 'rmse', 'mcc',
                'r2', 'pearson', 'soft_dice', 'soft_iou', 'ssim',
                'pixel_accuracy', 'precision', 'recall', 'specificity', 'f1'
            ]
            w.writerow(header)
            for i in sorted(td):
                w.writerow([
                    i,
                    f'{rho_dict[i]:.6f}',
                    f'{cos_dict[i]:.6f}',
                    f'{mse_dict[i]:.6f}',
                    f'{rmse_dict[i]:.6f}',
                    f'{mcc_dict[i]:.6f}',
                    f'{r2_dict[i]:.6f}',
                    f'{pc_dict[i]:.6f}',
                    f'{sd_dict[i]:.6f}',
                    f'{si_dict[i]:.6f}',
                    f'{ss_dict[i]:.6f}',
                    f'{pa_dict[i]:.6f}',
                    f'{prec_dict[i]:.6f}',
                    f'{rec_dict[i]:.6f}',
                    f'{spec_dict[i]:.6f}',
                    f'{f1_dict[i]:.6f}',
                ])
        print(f'[{child.name}] Wrote metrics_summary.csv')

        # ─── Save stacked images for each metric ────────────────────────────────
        metric_dicts: List[Tuple[str, Dict[int, float]]] = [
            ('rho', rho_dict), ('cosine', cos_dict),
            ('mse', mse_dict), ('rmse', rmse_dict),
            ('mcc', mcc_dict), ('r2', r2_dict),
            ('pearson', pc_dict), ('soft_dice', sd_dict),
            ('soft_iou', si_dict), ('ssim', ss_dict),
            ('pixel_accuracy', pa_dict), ('precision', prec_dict),
            ('recall', rec_dict), ('specificity', spec_dict),
            ('f1', f1_dict)
        ]
        for name, md in metric_dicts:
            metric_dir = out_dir / name
            metric_dir.mkdir(exist_ok=True)
            for i, (t, p) in td.items():
                img = stack_image((t, p))
                img8 = np.clip(
                    insert_vertical_line((img * 255).astype(np.uint8)),
                    0, 255
                )
                fname = f"{md[i]:.4f}_{i}.png"
                cv2.imwrite(str(metric_dir / fname), img8)

        # ─── Generate & save density plot ───────────────────────────────────────
        metrics = load_metrics(summary_csv)
        overview_path = out_dir / 'overview.png'
        plot_metrics_density(metrics, overview_path)
        print(f'[{child.name}] Saved overview density plot to {overview_path}')

        # ─── Generate & save histograms ────────────────────────────────────────
        histograms_path = out_dir / 'histogram.png'
        plot_metrics_histograms(metrics, histograms_path)
        print(f'[{child.name}] Saved histograms to {histograms_path}')

        # ─── Aggregate confusion matrices and plot ─────────────────────────────
        # Sum element-wise across all per-image 2x2 matrices
        stacked = np.stack(conf_mats, axis=0)  # shape = (N_images, 2, 2)
        total_cm = np.sum(stacked, axis=0)     # shape = (2, 2)
        cm_path = out_dir / 'confusion_matrix.png'
        plot_confusion_matrix(total_cm, cm_path)
        print(f'[{child.name}] Saved aggregated confusion matrix to {cm_path}')

        # ─── Bimodal Pearson analysis & save to bimodal.txt ────────────────────
        pearson_values = [pc for _, pc in sorted(pc_dict.items())]
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_pearson_bimodal_analysis(pearson_values)
        bimodal_txt = out_dir / 'bimodal.txt'
        with open(bimodal_txt, 'w') as f:
            f.write(buffer.getvalue())
        print(f'[{child.name}] Saved bimodal statistics to {bimodal_txt}')

        # ─── Pearson vs. pixelwise-sum density plot with best-fit lines ──
        sum_dict: Dict[int, float] = sum_pixelwise(td)
        idxs = sorted(td.keys())

        if len(idxs) >= 2:
            # Prepare arrays of Pearson (x) and pixel-sum (y)
            x_vals: np.ndarray = np.array([pc_dict[i] for i in idxs], dtype=float)
            y_vals: np.ndarray = np.array([sum_dict[i] for i in idxs], dtype=float)

            # Attempt a 2D KDE in (x, y)
            try:
                points = np.vstack((x_vals, y_vals))       # shape = (2, N)
                kde = gaussian_kde(points)                  # may raise if N < 2 or singular
                dens = kde(points)
            except Exception:
                # Fall back to uniform density if KDE fails
                dens = np.ones_like(x_vals, dtype=float)

            plt.figure(figsize=(6, 6))
            plt.scatter(
                x_vals, y_vals,
                c=dens,            # color by estimated density or constant
                s=16,              # marker size
                cmap=colormap.jet,
                rasterized=True,
                edgecolors='none'
            )

            # Linear fit: y = m*x + b
            lin_coeffs: np.ndarray = np.polyfit(x_vals, y_vals, 1)
            m, b = float(lin_coeffs[0]), float(lin_coeffs[1])

            # Quadratic fit: y = a*x^2 + b*x + c
            quad_coeffs: np.ndarray = np.polyfit(x_vals, y_vals, 2)
            a_q, b_q, c_q = map(float, quad_coeffs)

            x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
            x_fit: np.ndarray = np.linspace(x_min, x_max, 200)

            y_lin_fit: np.ndarray = m * x_fit + b
            y_quad_fit: np.ndarray = a_q * x_fit**2 + b_q * x_fit + c_q

            plt.plot(
                x_fit, y_lin_fit,
                color='black', linewidth=2, label='Linear fit'
            )
            plt.plot(
                x_fit, y_quad_fit,
                color='black', linewidth=2, linestyle='--', label='Quadratic fit'
            )
            plt.legend()

            plt.xlabel('Pearson Correlation', fontsize=12)     # unitless
            plt.ylabel('Pixelwise Sum (target)', fontsize=12)   # intensity units summed
            plt.title('Pearson vs. Target Pixelwise Sum (Density with Fits)', fontsize=14)
            plt.grid(linestyle=':', linewidth=0.5)

            newfig_path = out_dir / 'newfig.png'
            plt.tight_layout()
            plt.savefig(newfig_path, dpi=250)
            plt.close()
            print(f'[{child.name}] Saved Pearson vs. pixelwise-sum density + fits to {newfig_path}')

        elif len(idxs) == 1:
            # Only a single (x, y) pair exists: plot it in a blank figure (no KDE, no fit)
            i0 = idxs[0]
            x0 = float(pc_dict[i0])
            y0 = float(sum_dict[i0])

            plt.figure(figsize=(6, 6))
            plt.scatter(
                [x0], [y0],
                c='gray',  # single‐point color
                s=50,
                edgecolors='black'
            )
            plt.xlabel('Pearson Correlation', fontsize=12)
            plt.ylabel('Pixelwise Sum (target)', fontsize=12)
            plt.title('Pearson vs. Target Pixelwise Sum (Single Sample)', fontsize=14)
            plt.grid(linestyle=':', linewidth=0.5)

            newfig_path = out_dir / 'newfig.png'
            plt.tight_layout()
            plt.savefig(newfig_path, dpi=250)
            plt.close()
            print(f'[{child.name}] Only one sample—saved single‐point plot to {newfig_path}')

        else:
            # No data at all: skip this plot
            print(f'[{child.name}] No data for Pearson vs. pixelwise-sum plot (skipping).')

if __name__ == '__main__':
    main()