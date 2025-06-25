import numpy as np
from numpy.typing import NDArray

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

def main():
    print('confusionmatrix.py was run but there is no code in main()')

if __name__ == '__main__':
    main()