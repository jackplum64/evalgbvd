import os
import cv2
import numpy as np
from typing import Generator, Tuple, Optional
import matplotlib  # Use the new colormaps API from matplotlib 3.7+

def load_image_pairs(directory: str) -> Generator[Tuple[int, Optional['cv2.Mat'], Optional['cv2.Mat']], None, None]:
    """
    Loads image pairs from a directory where files are named in the format 'x_y.png',
    where x is a number and y is either 'pred' or 'target'. For each unique x, both images are loaded.

    Args:
        directory (str): Path to the directory containing the PNG images.

    Yields:
        tuple: A tuple (x, pred_image, target_image) where:
            - x (int): The numeric identifier.
            - pred_image (cv2.Mat or None): The prediction image.
            - target_image (cv2.Mat or None): The target image.
              If an image fails to load, its corresponding value is None.
    """
    image_dict = {}
    for file_name in os.listdir(directory):
        if file_name.lower().endswith('.png'):
            try:
                x_str, label_ext = file_name.split('_')
                label = label_ext.split('.')[0]
                x = int(x_str)
            except (ValueError, IndexError):
                print(f"Skipping file (invalid naming): {file_name}")
                continue
            
            if x not in image_dict:
                image_dict[x] = {}
            image_dict[x][label] = file_name

    for x in sorted(image_dict.keys()):
        pred_filename = image_dict[x].get('pred')
        target_filename = image_dict[x].get('target')
        
        pred_image = None
        target_image = None
        
        if pred_filename:
            pred_path = os.path.join(directory, pred_filename)
            pred_image = cv2.imread(pred_path)
            if pred_image is None:
                print(f"Failed to load prediction image: {pred_path}")
        
        if target_filename:
            target_path = os.path.join(directory, target_filename)
            target_image = cv2.imread(target_path)
            if target_image is None:
                print(f"Failed to load target image: {target_path}")
        
        yield x, pred_image, target_image

def reverse_jet_colormap(colored_img: np.ndarray) -> np.ndarray:
    """
    Recovers the original normalized mask from an image produced by the jet colormap.
    This function assumes that the original mask (with values in [0,1]) was mapped
    to RGB using matplotlib’s jet colormap, then scaled to 0–255 and converted to uint8.

    Args:
        colored_img (np.ndarray): The input jet-colored image in BGR format.

    Returns:
        np.ndarray: A single-channel float32 image with values in [0,1] representing the original mask.
    """
    # Get the jet colormap using the new matplotlib API.
    jet_cmap = matplotlib.colormaps['jet']
    
    # Pre-compute a lookup table for 256 discretized mask values.
    # Each row is the RGB color corresponding to a normalized mask value.
    lut = np.empty((256, 3), dtype=np.uint8)
    mask_values = np.linspace(0, 1, 256)
    for i, mask_val in enumerate(mask_values):
        # Get the RGB value (first 3 channels) for the mask value.
        rgb_float = jet_cmap(mask_val)[:3]
        # Scale to 0-255 and round to nearest integer.
        lut[i] = (np.array(rgb_float) * 255).round().astype(np.uint8)
    
    # Convert the input image from BGR (OpenCV default) to RGB.
    rgb_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a 2D array of pixels, shape (num_pixels, 3).
    pixels = rgb_img.reshape(-1, 3)  # shape (N, 3)
    
    # Compute the absolute differences between each pixel and every LUT entry.
    # This yields an array of shape (N, 256, 3).
    diffs = np.abs(pixels[:, None, :] - lut[None, :, :])
    # Sum differences across the color channels to get a score per LUT entry.
    diff_sums = diffs.sum(axis=2)  # shape (N, 256)
    # For each pixel, find the index of the LUT entry with the smallest difference.
    best_indices = diff_sums.argmin(axis=1)  # shape (N,)
    
    # Normalize the indices to [0, 1] (0 corresponds to 0 and 255 corresponds to 1).
    recovered_mask_flat = best_indices.astype(np.float32) / 255.0
    # Reshape back to the original image dimensions.
    recovered_mask = recovered_mask_flat.reshape(rgb_img.shape[:2])
    
    return recovered_mask

def main() -> None:
    src_directory = 'results2/train'      # Source directory with jet-colored images.
    dest_directory = 'results2/trainPOST'   # Destination directory for the recovered mask images.
    
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    
    for x, pred_img, target_img in load_image_pairs(src_directory):
        print(f"Processing pair for x = {x}")
        
        if pred_img is not None:
            mask_pred = reverse_jet_colormap(pred_img)
            # Save the recovered mask. Here we scale to 16-bit for better precision.
            pred_save_path = os.path.join(dest_directory, f"{x}_pred_mask.png")
            cv2.imwrite(pred_save_path, (mask_pred * 65535).astype(np.uint16))
        
        if target_img is not None:
            mask_target = reverse_jet_colormap(target_img)
            target_save_path = os.path.join(dest_directory, f"{x}_target_mask.png")
            cv2.imwrite(target_save_path, (mask_target * 65535).astype(np.uint16))

if __name__ == "__main__":
    main()
