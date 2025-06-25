import math
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple
from skimage.metrics import structural_similarity

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

def main():
    print('metrics.py was run but there is no code in main()')

if __name__ == '__main__':
    main()