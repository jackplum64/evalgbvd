from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
import csv
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde, skew, kurtosis
from scipy.signal import find_peaks
import math


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
    return {i: float(np.sum(t*p)) for i,(t,p) in data.items()}


def cosine_similarity(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    a_f = a.ravel(); b_f = b.ravel()
    dot = float(a_f @ b_f)
    na  = float(np.linalg.norm(a_f))
    nb  = float(np.linalg.norm(b_f))
    return dot/(na*nb) if na and nb else 0.0


def mean_squared_error(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    diff = a - b
    return float(np.mean(diff*diff))


def root_mean_squared_error(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    return float(np.sqrt(mean_squared_error(a,b)))


def mean_absolute_error(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    return float(np.mean(np.abs(a-b)))


def r_squared(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    ss_res = np.sum((a-b)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    return 1 - ss_res/ss_tot if ss_tot else 1.0


def pearson_correlation(
    a: NDArray[np.float32],
    b: NDArray[np.float32]
                        ) -> float:
    
    af = a.ravel() # Flatten
    bf = b.ravel()

    a0 = af - af.mean() # Center mean to zero
    b0 = bf - bf.mean()

    denom = np.linalg.norm(a0)*np.linalg.norm(b0) # Compute normalization factor

    return float((np.dot(a0, b0))/denom) if denom else 0.0


def soft_dice(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    num = 2*np.sum(a*b)
    den = np.sum(a*a) + np.sum(b*b)
    return float(num/den) if den else 1.0


def soft_iou(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    inter = np.sum(a*b)
    uni   = np.sum(a*a) + np.sum(b*b) - inter
    return float(inter/uni) if uni else 1.0


def ssim_index(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Returns SSIM in [-1,1], with 1.0 for a perfect match.
    Automatically computes data_range for floating‐point images.
    """
    # if both images are constant, they’re “identical”
    if float(a.max() - a.min()) == 0 and float(b.max() - b.min()) == 0:
        return 1.0

    # overall range across both images
    dmin = float(min(a.min(), b.min()))
    dmax = float(max(a.max(), b.max()))
    data_range = dmax - dmin

    score, _ = structural_similarity(
        a, b,
        data_range=data_range,
        full=True
    )
    return float(score)


# ─── IMAGE OPS ─────────────────────────────────────────────────────────────────

def stack_image(
    pair: Tuple[NDArray[np.float32], NDArray[np.float32]]
) -> NDArray[np.float32]:
    i1,i2 = pair
    if i1.shape!=i2.shape:
        raise ValueError(f"Shape mismatch {i1.shape} vs {i2.shape}")
    return np.concatenate((i1,i2), axis=1)


def insert_vertical_line(
    image: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    h,w = image.shape[:2]
    mid = w//2
    if image.ndim==2:
        return np.insert(image, mid, 255, axis=1)
    c = image.shape[2]
    line = np.zeros((h,1,c), dtype=image.dtype)
    line[...,0]=255; line[...,1]=255
    if c==4: line[...,3]=255
    return np.insert(image, mid, line, axis=1)

def contrast_enhance(
    image: NDArray[np.uint8],
    alpha: float,
    beta: float
) -> NDArray[np.uint8]:
    """
    Increase the contrast of an image using alpha (contrast) and beta (brightness).
    - alpha: 1.0 means no change, >1.0 means higher contrast
    - beta:  positive values to brighten, negative values to darken
    """
    # cv2.convertScaleAbs performs the following operation:
    # output = alpha * image + beta
    # (then clamps values to the 8-bit range [0, 255])
    contrasted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return contrasted

def enhance_in_place(
    images: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]],
    alpha: float,
    beta: float
) -> None:
    """
    Apply contrast_enhance to each image in the dict in-place.
    """
    for key, (img1, img2) in list(images.items()):
        images[key] = (
            contrast_enhance(img1, alpha, beta),
            contrast_enhance(img2, alpha, beta)
        )

def normalize_image(
    image: NDArray[np.float32]
) -> NDArray[np.uint8]:
    """
    Normalize pixel intensities so the minimum maps to 0 and the maximum to 255.

    Parameters:
        image: Input image array with arbitrary intensity range (unit: intensity).

    Returns:
        8-bit image array normalized to [0, 255] (unit: intensity).
    """
    min_val = float(np.min(image))       # intensity
    max_val = float(np.max(image))       # intensity
    range_val = max_val - min_val        # intensity

    if range_val == 0.0:
        # Avoid division by zero: return a black image
        return np.zeros_like(image, dtype=np.uint8)

    # Vectorized scaling to [0, 255]
    scaled = (image - min_val) * (255.0 / range_val)
    inverted: NDArray[np.uint8] = (255.0 - scaled).astype(np.uint8)

    return inverted

def normalize_in_place(
    images: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> None:
    """
    Apply normalize_image to each pair of images in the dict in-place.

    Parameters:
        images: Dict mapping an integer ID to a (target, pred) tuple of image arrays (unit: intensity).
    """
    for key, (img1, img2) in list(images.items()):
        images[key] = (
            normalize_image(img1),
            normalize_image(img2)
        )

def strengthen_colormap(
        image: NDArray[np.float32],
        strength: float
) -> NDArray[np.float32]:
    """
    Apply a nonlinear transformation function to each pair of images in place

    Parameters:
        image: Input image array with arbitrary intensity range (unit: intensity).

    Returns:
        8-bit image array normalized to [0, 255] (unit: intensity).
    """

    strengthened = np.power(image, strength)


    return strengthened


def strengthen_colormap_in_place(
    images: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]],
    mode: str,
    strength: float
) -> None:
    """
    Apply strengthen_colormap to ground truth, prediction, or both images in pair.

    Parameters:
        images: Dict mapping an integer ID to a (target, pred) tuple of image arrays (unit: intensity).
        mode: Str 'gt', 'pred', or 'both' for ground truth, prediction, or both images
        strength: colormap strength factor (unitless).
    """
    for key, (gt_img, pred_img) in images.items():
        # decide whether to process each image or leave it unchanged
        new_gt   = strengthen_colormap(gt_img, strength)   if mode in ('gt',   'both') else gt_img
        new_pred = strengthen_colormap(pred_img, strength) if mode in ('pred', 'both') else pred_img

        images[key] = (new_gt, new_pred)



# ─── PLOT OPS ─────────────────────────────────────────────────────────────────

def load_metrics(csv_path: Path) -> Dict[str, List[float]]:
    """
    Reads 'metrics_summary.csv' and returns a dict
    mapping metric name → list of values.
    """
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        metrics: Dict[str, List[float]] = {name: [] for name in reader.fieldnames if name!='idx'}
        for row in reader:
            for k in metrics:
                metrics[k].append(float(row[k]))
    return metrics

def plot_metrics_density(
    metrics: Dict[str, List[float]],
    save_path: Path
) -> None:
    """
    For each metric, plot its samples along a vertical line (x=0),
    coloring each dot by its local 1D kde-based density,
    then save the whole figure to save_path (PNG).
    """
    names = list(metrics.keys())
    n_metrics = len(names)

    # grid: 5 columns, enough rows
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

        # 1D density estimate
        kde = gaussian_kde(vals)
        dens = kde(vals)

        # plot points at x=0
        ax.scatter(
            np.zeros_like(vals), vals,
            c=dens,
            s=12,
            cmap=cm.jet,
            rasterized=True,
            zorder=10
        )
        # central line
        ax.axvline(0.0, color='black', linewidth=1, zorder=5)

        ax.set_title(name, pad=6)
        ax.set_xticks([])
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylabel('Value')
        ax.grid(axis='y', linestyle=':', linewidth=0.5, zorder=0)

    # hide unused axes
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


def plot_metrics_histograms(
    metrics: Dict[str, List[float]],
    save_path: Path
) -> None:
    """
    For each metric in `metrics`, plot its sample values as a histogram
    and save the combined figure to `save_path`.
    """
    names: List[str] = list(metrics.keys())
    n_metrics: int = len(names)

    # 5 columns, as before
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
        vals = np.array(metrics[name], dtype=float)  # unitless samples

        # — drop NaNs / ±Inf —
        mask = np.isfinite(vals)
        if not mask.all():
            # optional: report how many
            print(f"[{name}] dropping {len(vals) - mask.sum()} non‐finite samples")
            vals = vals[mask]

        # — pick a safe bin count —
        n = vals.size
        bins = min(50, int(math.sqrt(n))) if n else 10

        # — wrap in try/except so we catch any surprise crashes —
        try:
            ax.hist(
                vals,
                bins=bins,
                edgecolor='black',
                linewidth=0.5,
                zorder=2
            )
        except Exception as e:
            print(f"Skipping histogram for {name} at index {idx}: {e}")
            ax.text(
                0.5, 0.5,
                "error",
                ha="center", va="center",
                transform=ax.transAxes,
                color="red"
            )

        ax.set_title(name, pad=6)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.grid(axis='y', linestyle=':', linewidth=0.5, zorder=1)

    # hide any unused axes
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250)
    plt.close(fig)


# ─── STATS OPS ─────────────────────────────────────────────────────────────────

def print_pearson_bimodal_analysis(scores: List[float]) -> None:
    """
    Print a bimodal statistical analysis of Pearson correlation scores.

    Parameters:
        scores: List of Pearson correlation coefficients (unitless).
    """
    arr: np.ndarray = np.array(scores, dtype=float)    # unitless scores
    n: int = arr.size                                 # number of samples

    # Global summary statistics
    mean: float = float(np.mean(arr))                  # unitless
    std: float  = float(np.std(arr, ddof=0))           # unitless
    med: float  = float(np.median(arr))                # unitless
    sk: float   = float(skew(arr))                     # unitless
    kurt: float = float(kurtosis(arr))                 # unitless

    # Build histogram
    bins = min(50, int(math.sqrt(n))) if n else 10
    counts, bin_edges = np.histogram(arr, bins=bins)    # counts per bin
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0   # bin centers

    # Find local maxima in the bin counts
    peak_idxs, _ = find_peaks(counts)
    if peak_idxs.size < 2:
        # fallback if <2 local peaks: take two largest bins
        peak_idxs = np.argsort(counts)[-2:]
    # pick the top two peaks by height
    top2 = peak_idxs[np.argsort(counts[peak_idxs])[::-1][:2]]

    mode_locs = [centers[i] for i in top2]            # unitless
    mode_mags = [int(counts[i]) for i in top2]        # bin counts

    ratio: float = mode_mags[0] / mode_mags[1] if mode_mags[1] else float('inf')

    # Print summary
    print("=== Bimodal Analysis of Pearson Correlation Scores ===")
    print(f"Samples: {n}")
    print(f"Mean: {mean:.4f}, Median: {med:.4f}, Std Dev: {std:.4f}")
    print(f"Skewness: {sk:.4f}, Kurtosis: {kurt:.4f}\n")

    print("Identified Modes:")
    for idx, (loc, mag) in enumerate(zip(mode_locs, mode_mags), start=1):
        print(f" Mode {idx}: location = {loc:.4f}, count = {mag}")
    print(f"Mode ratio (Mode 1 / Mode 2): {ratio:.4f}\n")

    # % within ±0.5σ, ±1σ, ±1.5σ of each mode
    for idx, loc in enumerate(mode_locs, start=1):
        print(f"Mode {idx} intervals around {loc:.4f}:")
        for k in (0.5, 1.0, 1.5):
            low, high = loc - k*std, loc + k*std
            pct = float(np.mean((arr >= low) & (arr <= high))) * 100.0
            print(f"  ±{k}σ: {pct:.2f}% (from {low:.4f} to {high:.4f})")
        print()


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = Path(__file__).parent
    base_dir   = Path(
        './resultsnew'
        '/tversky_focal_float_blur_5_1_001_5_0.0_1.0_0.75'
    )
    data_dir   = base_dir / 'train'


    ### Preprocess ###
    CONTRAST_ENHANCE = 1.0 # 1.0 means no change
    BRIGHTEN         = 0   # 0 means no change
    NORMALIZE        = False # normalize image to [0, 255]
    STRENGTH_PRED    = 1.35 # apply strengthening function to predictions. 1.0 means no change
    STRENGTH_GT      = 1.0 # apply strengthening function to ground truth. 1.0 means no change


    ### DO NOT EDIT BELOW THIS LINE ###

    out_dir    = script_dir / 'output'
    out_dir.mkdir(exist_ok=True, parents=True)

    raw = read_data(data_dir)
    print(f'Loaded {len(raw)} files')

    td  = organize_data(raw)
    print(f'Found {len(td)} (target,pred) pairs')

    #enhance_in_place(td, CONTRAST_ENHANCE, BRIGHTEN)
    if STRENGTH_PRED != 1.0:
        strengthen_colormap_in_place(td, 'pred', STRENGTH_PRED)
    if STRENGTH_GT != 1.0:
        strengthen_colormap_in_place(td, 'gt', STRENGTH_GT)

    if NORMALIZE: normalize_in_place(td)

    # compute everything
    rho_dict = sum_pixelwise_product(td)
    cos_dict = {i: cosine_similarity(t,p) for i,(t,p) in td.items()}

    mse_dict  = {i: mean_squared_error   (t,p) for i,(t,p) in td.items()}
    rmse_dict = {i: root_mean_squared_error(t,p) for i,(t,p) in td.items()}
    mae_dict  = {i: mean_absolute_error   (t,p) for i,(t,p) in td.items()}
    r2_dict   = {i: r_squared            (t,p) for i,(t,p) in td.items()}
    pc_dict   = {i: pearson_correlation  (t,p) for i,(t,p) in td.items()}
    sd_dict   = {i: soft_dice            (t,p) for i,(t,p) in td.items()}
    si_dict   = {i: soft_iou             (t,p) for i,(t,p) in td.items()}
    ss_dict   = {i: ssim_index           (t,p) for i,(t,p) in td.items()}

    # save summary CSV
    import csv
    with open(out_dir/'metrics_summary.csv','w',newline='') as f:
        w = csv.writer(f)
        header = [
            'idx','rho','cosine','mse','rmse','mae',
            'r2','pearson','soft_dice','soft_iou','ssim'
        ]
        w.writerow(header)
        for i in sorted(td):
            w.writerow([
                i,
                f'{rho_dict[i]:.6f}',
                f'{cos_dict[i]:.6f}',
                f'{mse_dict[i]:.6f}',
                f'{rmse_dict[i]:.6f}',
                f'{mae_dict[i]:.6f}',
                f'{r2_dict[i]:.6f}',
                f'{pc_dict[i]:.6f}',
                f'{sd_dict[i]:.6f}',
                f'{si_dict[i]:.6f}',
                f'{ss_dict[i]:.6f}',
            ])
    print('Wrote metrics_summary.csv')

    # (optional) save stacked images with yellow separator under each metric dir
    for name, md in [
        ('rho', rho_dict), ('cosine', cos_dict),
        ('mse', mse_dict), ('rmse', rmse_dict),
        ('mae', mae_dict), ('r2', r2_dict),
        ('pearson', pc_dict), ('soft_dice', sd_dict),
        ('soft_iou', si_dict), ('ssim', ss_dict),
    ]:
        d = out_dir/name; d.mkdir(exist_ok=True)
        for i,(t,p) in td.items():
            img = stack_image((t,p))
            img8 = (np.clip(insert_vertical_line((img*255).astype(np.uint8)),0,255))
            fname = f"{md[i]:.4f}_{i}.png"
            cv2.imwrite(str(d/fname), img8)


    # Generate and save boxplots
    csv_file = Path('output/metrics_summary.csv')
    metrics = load_metrics(csv_file)

    overview_path = out_dir / 'overview.png'
    plot_metrics_density(metrics, overview_path)
    print(f"Saved overview density plot to {overview_path}")

    # Generate and save histograms
    histograms_path = out_dir / 'histogram.png'

    plot_metrics_histograms(metrics, histograms_path)
    print(f'Saved histograms to {histograms_path}')

    pc_dict = {i: pearson_correlation(t,p) for i,(t,p) in td.items()}

    # ── Bimodal analysis of Pearson scores ──
    print_pearson_bimodal_analysis(
        [pc for _, pc in sorted(pc_dict.items())]  # unitless scores
    )




if __name__ == '__main__':
    main()

