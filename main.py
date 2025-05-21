from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
import csv
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
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
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> float:
    af = a.ravel(); bf = b.ravel()
    a0 = af - af.mean(); b0 = bf - bf.mean()
    denom = np.linalg.norm(a0)*np.linalg.norm(b0)
    return float((a0@b0)/denom) if denom else 0.0


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


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = Path(__file__).parent
    base_dir   = Path(
        './results_focal_tversky_gbblur7'
        '/tversky_focal_float_blur_5_1_001_5_0.0_1.0'
    )
    data_dir   = base_dir / 'train'
    out_dir    = script_dir / 'output'
    out_dir.mkdir(exist_ok=True, parents=True)

    raw = read_data(data_dir)
    print(f'Loaded {len(raw)} files')
    td  = organize_data(raw)
    print(f'Found {len(td)} (target,pred) pairs')

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


if __name__ == '__main__':
    main()
