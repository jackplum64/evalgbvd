from pathlib import Path
import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Tuple, Literal, List

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

def main():
    print('imageops.py was run but there is no code in main()')

if __name__ == '__main__':
    main()