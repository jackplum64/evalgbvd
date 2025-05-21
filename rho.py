from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2

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
    """
    Return a dict mapping each numeric ID to (target, pred).
    Assumes keys like '12_target' and '12_pred'.
    """
    # find all unique numeric prefixes
    prefixes = {key.split('_', 1)[0] for key in data.keys()}
    organized: Dict[int, Tuple[NDArray[np.float32], NDArray[np.float32]]] = {}

    for prefix in sorted(prefixes, key=int):
        idx = int(prefix)
        t_key = f'{prefix}_target'
        p_key = f'{prefix}_pred'
        if t_key not in data or p_key not in data:
            raise KeyError(f'Missing {t_key} or {p_key} in data')
        organized[idx] = (data[t_key], data[p_key])

    return organized


def sum_pixelwise_product(
    data: Dict[int, tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> Dict[int, float]:
    """
    Map each numeric ID to the sum of target * pred over all pixels.
    """
    return {
        idx: float(np.sum(target * pred))
        for idx, (target, pred) in data.items()
    }


def stack_image(
    base_images: Tuple[NDArray[np.float32], NDArray[np.float32]]
) -> NDArray[np.float32]:
    img1, img2 = base_images
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes must match, got {img1.shape} vs {img2.shape}")
    # concatenate along width (axis=1)
    return np.concatenate((img1, img2), axis=1)


def insert_vertical_line(
    image: NDArray[np.uint8]
) -> NDArray[np.uint8]:
    """
    Insert a 1-px-wide vertical line down the center of an image.
    - For RGB/RGBA images (H*W*3 or H*W*4), the line is yellow (255,255,0[,255]).
    - For grayscale images (H*W), the line is white (255).
    """
    # dimensions: h: height (px), w: width (px)
    h, w = image.shape[:2]
    mid = w // 2  # insertion column index

    if image.ndim == 2:
        # grayscale → insert white column
        return np.insert(image, mid, 255, axis=1)

    # color image
    c = image.shape[2]  # channels
    # build line column
    if c == 3 or c == 4:
        line = np.zeros((h, 1, c), dtype=image.dtype)
        # yellow for RGB/RGB:
        line[..., 0] = 255  # R channel
        line[..., 1] = 255  # G channel
        # B stays 0; alpha stays 255 if present
        if c == 4:
            line[..., 3] = 255
        return np.insert(image, mid, line, axis=1)

    raise ValueError(f"Unsupported image shape: {image.shape}")


def cosine_similarity(
    a: NDArray[np.float32],
    b: NDArray[np.float32]
) -> float:
    """
    Compute cosine similarity between two intensity images.
    a, b: unitless intensity arrays of the same shape.
    returns: unitless score in [0,1].
    """
    if a.shape != b.shape:
        raise ValueError(f"Image shapes must match, got {a.shape} vs {b.shape}")

    a_flat = a.ravel()
    b_flat = b.ravel()

    dot_product: float = float(a_flat @ b_flat)         # intensity^2 summed
    norm_a: float = float(np.linalg.norm(a_flat))       # intensity magnitude
    norm_b: float = float(np.linalg.norm(b_flat))       # intensity magnitude

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def main():
    script_dir = Path(__file__).parent
    base_dir = Path(
        './results_focal_tversky_gbblur7'
        '/tversky_focal_float_blur_5_1_001_5_0.0_1.0'
    )
    out_dir = script_dir / 'output'
    out_dir.mkdir(exist_ok=True, parents=True)

    data_dir  = base_dir / 'train'
    data = read_data(data_dir)
    print(f'Loaded {len(data)} .npy files from {data_dir!s}')

    # organize into { idx: (target, pred) }
    target_pred_dict = organize_data(data)
    print(f'Organized into {len(target_pred_dict)} entries.')

    rho_products = sum_pixelwise_product(target_pred_dict)
    print(f'Computed scalar products for {len(rho_products)} entries.')

    cosine_similarities: Dict[int, float] = {
        idx: cosine_similarity(tgt, pred)
        for idx, (tgt, pred) in target_pred_dict.items()
    }
    print(f'Computed cosine similarities for {len(cosine_similarities)} entries.')

    stacked_images: Dict[int, NDArray[np.float32]] = {
        idx: stack_image(pair)
        for idx, pair in target_pred_dict.items()
    }
    print(f'Stacked {len(stacked_images)} image pairs.')

    # create output subdirectories
    cosine_dir = out_dir / 'cosine'
    rho_dir    = out_dir / 'rho'
    cosine_dir.mkdir(exist_ok=True, parents=True)
    rho_dir.mkdir(exist_ok=True, parents=True)

    for idx, (tgt, pred) in target_pred_dict.items():
        stacked = stack_image((tgt, pred))
        stacked_yellow = insert_vertical_line(stacked)
        # normalize to 0–255 and convert to uint8 for saving
        img_uint8 = (np.clip(stacked_yellow, 0.0, 1.0) * 255).astype(np.uint8)

        # save cosines to output/cosine
        cosine = cosine_similarities[idx]
        fname = f"{cosine:.4f}_{idx}.png"
        out_path = cosine_dir / fname
        cv2.imwrite(str(out_path), img_uint8)

        # save rho products to output/rho
        rho = rho_products[idx]
        fname = f"{rho:.4f}_{idx}.png"
        out_path = rho_dir / fname
        cv2.imwrite(str(out_path), img_uint8)

    


if __name__ == '__main__':
    main()
