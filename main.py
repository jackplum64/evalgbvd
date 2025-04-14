import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import cv2

def compute_weighted_area(image_array: np.ndarray, strong_weight: bool = False) -> float:
    """
    Compute the weighted area for an image array.
    Each pixel contributes according to its inverted normalized intensity:
        weight = 1 - normalized_value    (if strong_weight is False)
        weight = (1 - normalized_value)^2  (if strong_weight is True)
    
    Args:
        image_array (np.ndarray): The image data.
        strong_weight (bool): Use quadratic weighting if True.
    
    Returns:
        float: The sum of the pixel weights.
    """
    if np.issubdtype(image_array.dtype, np.uint8):
        normalized = image_array.astype(np.float32) / 255.0
    elif np.issubdtype(image_array.dtype, np.floating):
        normalized = image_array
    else:
        normalized = image_array.astype(np.float32)
    
    if strong_weight:
        weights = (1.0 - normalized) ** 2
    else:
        weights = 1.0 - normalized
    
    return float(np.sum(weights))

def compute_weighted_centroid(image_array: np.ndarray, strong_weight: bool = False,
                              no_weighting_light_areas: bool = False) -> tuple[float, float]:
    """
    Compute the weighted centroid (x, y) for an image array.
    Weighting is done based on inverted normalized intensity:
        weight = 1 - normalized (linear) or (1 - normalized)^2 (if strong_weight is True)
    If no_weighting_light_areas is True then pixels with intensity below 64 (or 64/255 for normalized floats)
    are assigned zero weight.
    
    Args:
        image_array (np.ndarray): The image data.
        strong_weight (bool): Use quadratic weighting if True.
        no_weighting_light_areas (bool): If True, pixels with intensity below 64 are ignored.
    
    Returns:
        tuple[float, float]: The centroid coordinates (x, y). Returns (nan, nan) if total weight is zero.
    """
    if np.issubdtype(image_array.dtype, np.uint8):
        normalized = image_array.astype(np.float32) / 255.0
    elif np.issubdtype(image_array.dtype, np.floating):
        normalized = image_array
    else:
        normalized = image_array.astype(np.float32)
    
    if strong_weight:
        weights = (1.0 - normalized) ** 2
    else:
        weights = 1.0 - normalized

    # Apply the no-weighting condition for light areas if requested.
    if no_weighting_light_areas:
        # For images in uint8, use the original intensity; for floats (assuming normalized) use 64/255.
        if np.issubdtype(image_array.dtype, np.uint8):
            intensity_mask = image_array >= 64
        elif np.issubdtype(image_array.dtype, np.floating) and np.max(image_array) <= 1:
            intensity_mask = image_array >= (64/255.0)
        else:
            intensity_mask = image_array >= 64
        weights = weights * intensity_mask.astype(np.float32)
    
    total_weight = np.sum(weights)
    if total_weight == 0:
        return float('nan'), float('nan')
    
    height, width = image_array.shape[:2]
    y_indices, x_indices = np.indices((height, width))
    
    x_centroid = np.sum(x_indices * weights) / total_weight
    y_centroid = np.sum(y_indices * weights) / total_weight
    return x_centroid, y_centroid

def compute_best_prediction_centroid(image_array: np.ndarray, strong_weight: bool = False,
                                     no_weighting_light_areas: bool = False) -> tuple[float, float, list]:
    """
    Compute the centroid of the largest and darkest prediction region in the image,
    optionally using quadratic intensity weighting if strong_weight is True.
    If strong_weight is False, the centroid is computed using contour moments.
    When no_weighting_light_areas is True, any pixel with intensity below 64 (or below 64/255
    if normalized) does not contribute to the weighted centroid computation (only applied for
    the quadratic weighting branch).
    
    Args:
        image_array (np.ndarray): The image data.
        strong_weight (bool): Use quadratic intensity weighting if True.
        no_weighting_light_areas (bool): If True, pixels with intensity below 64 are ignored.
    
    Returns:
        tuple[float, float, list]: The computed centroid coordinates (x, y) and the list of relevant contours.
    """
    # Ensure the image is in 8-bit grayscale format.
    if image_array.dtype != np.uint8:
        img = (image_array * 255).astype(np.uint8) if np.max(image_array) <= 1 else image_array.astype(np.uint8)
    else:
        img = image_array.copy()
    
    threshold_value = 220
    ret, thresh = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Apply morphological opening to remove small noise blobs.
    kernel = np.ones((3, 3), np.uint8)
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, hierarchy = cv2.findContours(thresh_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out contours too close to image boundaries.
    margin = 10  # pixels
    height, width = img.shape[:2]
    filtered_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x > margin and y > margin and (x + w) < (width - margin) and (y + h) < (height - margin):
            filtered_contours.append(cnt)
    
    relevant_contours = filtered_contours if filtered_contours else contours
    if not relevant_contours:
        cx, cy = compute_weighted_centroid(image_array, strong_weight=strong_weight,
                                             no_weighting_light_areas=no_weighting_light_areas)
        return cx, cy, []
    
    # Select the contour with the largest area.
    largest_contour = max(relevant_contours, key=cv2.contourArea)
    
    if not strong_weight:
        # Use traditional contour moments.
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx, cy = compute_weighted_centroid(image_array, strong_weight=strong_weight,
                                               no_weighting_light_areas=no_weighting_light_areas)
    else:
        # Compute weighted centroid using quadratic weighting on the masked region.
        mask = np.zeros_like(img, dtype=np.float32)
        cv2.drawContours(mask, [largest_contour], -1, 1, thickness=-1)
        
        if np.issubdtype(img.dtype, np.uint8):
            normalized = img.astype(np.float32) / 255.0
        elif np.issubdtype(img.dtype, np.floating):
            normalized = img
        else:
            normalized = img.astype(np.float32)
        
        weights = (1.0 - normalized) ** 2  # Quadratic intensity weighting.
        # Apply the drawn contour mask.
        weights = weights * mask
        
        # Apply the no-weighting condition if requested.
        if no_weighting_light_areas:
            if np.issubdtype(img.dtype, np.uint8):
                intensity_mask = img >= 128
            elif np.issubdtype(img.dtype, np.floating) and np.max(img) <= 1:
                intensity_mask = img >= (128/255.0)
            else:
                intensity_mask = img >= 128
            weights = weights * intensity_mask.astype(np.float32)
        
        total_weight = np.sum(weights)
        if total_weight == 0:
            cx, cy = compute_weighted_centroid(image_array, strong_weight=strong_weight,
                                               no_weighting_light_areas=no_weighting_light_areas)
        else:
            y_indices, x_indices = np.indices(img.shape)
            cx = np.sum(x_indices * weights) / total_weight
            cy = np.sum(y_indices * weights) / total_weight
    
    return cx, cy, relevant_contours

def save_image_with_dot(image_array: np.ndarray, centroid: tuple[float, float],
                        out_filepath: str, contours: list = None) -> None:
    """
    Save the image array as a PNG file with an overlaid dot.
    A blue dot is drawn at the computed centroid (or a red dot at the center if the centroid is invalid).
    If contours are provided, they are drawn on the image in green.
    
    Args:
        image_array (np.ndarray): The image data.
        centroid (tuple[float, float]): The computed centroid coordinates (x, y).
        out_filepath (str): Path where the PNG image will be saved.
        contours (list, optional): List of contours to be drawn.
    """
    height, width = image_array.shape[:2]
    plt.figure()
    
    if contours is not None and len(contours) > 0:
        if image_array.ndim == 2:
            disp_img = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        else:
            disp_img = image_array.copy()
        cv2.drawContours(disp_img, contours, -1, (0, 255, 0), 1)
        disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2RGB)
        plt.imshow(disp_img, origin='upper')
    else:
        plt.imshow(image_array, cmap='gray', origin='upper')
    
    if np.isnan(centroid[0]) or np.isnan(centroid[1]):
        dot_x = width / 2
        dot_y = height / 2
        dot_color = 'red'
    else:
        dot_x, dot_y = centroid
        dot_color = 'blue'
    
    plt.scatter(dot_x, dot_y, c=dot_color, s=50, marker='o')
    plt.axis('off')
    
    plt.savefig(out_filepath, dpi=250, bbox_inches='tight')
    plt.close()

def process_directory(directory: str, base_data_dir: str, images_dir: str, plots_dir: str,
                      save_images: bool, best_prediction_only: bool,
                      strong_weight: bool, no_weighting_light_areas: bool) -> None:
    """
    Process prediction/target pairs of .npy files from a given sub-directory.
    Computations for weighted area and centroids use the strong_weight flag if enabled.
    When no_weighting_light_areas is True the centroid computations ignore pixels with
    intensity below 64.
    
    Args:
        directory (str): Sub-directory name (e.g., "test", "train", "val").
        base_data_dir (str): Base directory containing data.
        images_dir (str): Directory to save PNG images.
        plots_dir (str): Directory to save parity plots.
        save_images (bool): Toggle for saving individual images.
        best_prediction_only (bool): Toggle to compute the centroid from the largest/darkest region.
        strong_weight (bool): Toggle for quadratic intensity weighting.
        no_weighting_light_areas (bool): If True, pixels with intensity below 64 are not weighted.
    """
    print(f"Processing directory: {directory}")
    data_dir: str = os.path.join(base_data_dir, directory)
    
    pred_files = glob.glob(os.path.join(data_dir, "*_pred.npy"))
    pred_files.sort()
    
    if not pred_files:
        print(f"No prediction files found in directory: {data_dir}")
        return

    os.makedirs(plots_dir, exist_ok=True)
    if save_images:
        os.makedirs(images_dir, exist_ok=True)
    
    predicted_areas = []
    target_areas = []
    pred_centroid_x = []
    pred_centroid_y = []
    target_centroid_x = []
    target_centroid_y = []

    for pred_file in pred_files:
        base_name = os.path.basename(pred_file)
        index = base_name.split("_")[0]
        
        target_filename = f"{index}_target.npy"
        target_filepath = os.path.join(data_dir, target_filename)
        
        if not os.path.exists(target_filepath):
            print(f"Warning: Target file '{target_filepath}' does not exist. Skipping index {index}.")
            continue
        
        pred_image = np.load(pred_file)
        target_image = np.load(target_filepath)
        
        pred_area = compute_weighted_area(pred_image, strong_weight=strong_weight)
        target_area = compute_weighted_area(target_image, strong_weight=strong_weight)
        
        if best_prediction_only:
            pred_cx, pred_cy, pred_contours = compute_best_prediction_centroid(
                pred_image, strong_weight=strong_weight,
                no_weighting_light_areas=no_weighting_light_areas
            )
        else:
            pred_cx, pred_cy = compute_weighted_centroid(
                pred_image, strong_weight=strong_weight,
                no_weighting_light_areas=no_weighting_light_areas
            )
            pred_contours = None
        
        target_cx, target_cy = compute_weighted_centroid(
            target_image, strong_weight=strong_weight,
            no_weighting_light_areas=no_weighting_light_areas
        )
        
        height, width = pred_image.shape[:2]
        max_area = height * width
        print(f"Index {index} in {directory}:")
        print(f"  Predicted area      = {pred_area:.2f} (Max: {max_area})")
        print(f"  Target area         = {target_area:.2f} (Max: {max_area})")
        print(f"  Predicted centroid  = ({pred_cx:.2f}, {pred_cy:.2f})")
        print(f"  Target centroid     = ({target_cx:.2f}, {target_cy:.2f})")
        print()
        
        predicted_areas.append(pred_area)
        target_areas.append(target_area)
        pred_centroid_x.append(pred_cx)
        pred_centroid_y.append(pred_cy)
        target_centroid_x.append(target_cx)
        target_centroid_y.append(target_cy)
        
        if save_images:
            pred_out_filename = f"{directory}_{index}_pred.png"
            pred_out_filepath = os.path.join(images_dir, pred_out_filename)
            save_image_with_dot(pred_image, (pred_cx, pred_cy), pred_out_filepath, contours=pred_contours)
            
            target_out_filename = f"{directory}_{index}_target.png"
            target_out_filepath = os.path.join(images_dir, target_out_filename)
            save_image_with_dot(target_image, (target_cx, target_cy), target_out_filepath)
    
    predicted_areas = np.array(predicted_areas)
    target_areas = np.array(target_areas)
    pred_centroid_x = np.array(pred_centroid_x)
    pred_centroid_y = np.array(pred_centroid_y)
    target_centroid_x = np.array(target_centroid_x)
    target_centroid_y = np.array(target_centroid_y)
    
    # Plot: Weighted Area Comparison.
    if target_areas.size > 0 and predicted_areas.size > 0:
        coeffs = np.polyfit(target_areas, predicted_areas, 1)
        slope, intercept = coeffs
        print(f"Weighted Area Best fit in {directory}: predicted_area = {slope:.2f} * target_area + {intercept:.2f}")
        
        x_fit = np.linspace(target_areas.min(), target_areas.max(), 100)
        y_fit = slope * x_fit + intercept
        
        data_stack = np.vstack([target_areas, predicted_areas])
        density = gaussian_kde(data_stack)(data_stack)
        
        plt.figure(figsize=(8, 6))
        plt.scatter(target_areas, predicted_areas, c=density, s=4, cmap=cm.jet,
                    rasterized=True, label="Data points")
        plt.plot(x_fit, y_fit, color='black', linestyle='-', linewidth=2,
                 label="Best fit line")
        plt.xlabel("Target Weighted Area", fontsize=15)
        plt.ylabel("Predicted Weighted Area", fontsize=15)
        plt.title(f"Predicted vs. Target Weighted Areas ({directory.capitalize()})", fontsize=17)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        
        plot_path_area = os.path.join(plots_dir, f"parityplot_weighted_area_{directory}.png")
        plt.savefig(plot_path_area, dpi=250)
        plt.show()
    
    # Plot: Centroid Comparison.
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(14, 6))
    
    valid_x = ~np.isnan(target_centroid_x) & ~np.isnan(pred_centroid_x)
    if np.any(valid_x):
        coeffs_x = np.polyfit(target_centroid_x[valid_x], pred_centroid_x[valid_x], 1)
        slope_x, intercept_x = coeffs_x
        print(f"Centroid X Best fit in {directory}: predicted_x = {slope_x:.2f} * target_x + {intercept_x:.2f}")
        
        x_fit_x = np.linspace(target_centroid_x[valid_x].min(), target_centroid_x[valid_x].max(), 100)
        y_fit_x = slope_x * x_fit_x + intercept_x
        
        data_stack_x = np.vstack([target_centroid_x[valid_x], pred_centroid_x[valid_x]])
        density_x = gaussian_kde(data_stack_x)(data_stack_x)
        
        ax_x.scatter(target_centroid_x[valid_x], pred_centroid_x[valid_x],
                     c=density_x, s=4, cmap=cm.jet, rasterized=True, label="Data points")
        ax_x.plot(x_fit_x, y_fit_x, color='black', linestyle='-', linewidth=2,
                  label="Best fit line")
        ax_x.set_xlabel("Target X Centroid", fontsize=15)
        ax_x.set_ylabel("Predicted X Centroid", fontsize=15)
        ax_x.set_title(f"Centroid X Comparison ({directory.capitalize()})", fontsize=17)
        ax_x.legend(fontsize=12)
        ax_x.grid(True)
    else:
        ax_x.text(0.5, 0.5, "No valid data for Centroid X", ha="center", va="center")
        ax_x.axis('off')
    
    valid_y = ~np.isnan(target_centroid_y) & ~np.isnan(pred_centroid_y)
    if np.any(valid_y):
        coeffs_y = np.polyfit(target_centroid_y[valid_y], pred_centroid_y[valid_y], 1)
        slope_y, intercept_y = coeffs_y
        print(f"Centroid Y Best fit in {directory}: predicted_y = {slope_y:.2f} * target_y + {intercept_y:.2f}")
        
        x_fit_y = np.linspace(target_centroid_y[valid_y].min(), target_centroid_y[valid_y].max(), 100)
        y_fit_y = slope_y * x_fit_y + intercept_y
        
        data_stack_y = np.vstack([target_centroid_y[valid_y], pred_centroid_y[valid_y]])
        density_y = gaussian_kde(data_stack_y)(data_stack_y)
        
        ax_y.scatter(target_centroid_y[valid_y], pred_centroid_y[valid_y],
                     c=density_y, s=4, cmap=cm.jet, rasterized=True, label="Data points")
        ax_y.plot(x_fit_y, y_fit_y, color='black', linestyle='-', linewidth=2,
                  label="Best fit line")
        ax_y.set_xlabel("Target Y Centroid", fontsize=15)
        ax_y.set_ylabel("Predicted Y Centroid", fontsize=15)
        ax_y.set_title(f"Centroid Y Comparison ({directory.capitalize()})", fontsize=17)
        ax_y.legend(fontsize=12)
        ax_y.grid(True)
    else:
        ax_y.text(0.5, 0.5, "No valid data for Centroid Y", ha="center", va="center")
        ax_y.axis('off')
    
    fig.tight_layout()
    plot_path_centroid = os.path.join(plots_dir, f"parityplot_centroid_{directory}.png")
    plt.savefig(plot_path_centroid, dpi=250)
    plt.show()

def main() -> None:
    """
    Set control variables for data directories and toggles. The 'save_images' flag controls saving of individual PNG images.
    The 'best_prediction_only' flag selects whether the prediction centroid is computed from the largest/darkest region.
    The 'strong_weight' flag modifies the intensity weighting to use a quadratic scale.
    The 'no_weighting_light_areas' flag, when True, causes pixels with intensity below 64 to be ignored in the centroid calculation.
    """
    base_data_dir = os.path.join("4825newresults5", "tversky_focal_recons_float")
    plots_dir = "plots"
    images_dir = "output"
    directories = ["test", "train", "val"]
    
    save_images = True               # Toggle for saving individual image PNGs.
    best_prediction_only = False     # Toggle for using the largest/darkest region for prediction centroid.
    strong_weight = True             # Toggle for quadratic intensity weighting (set to False for standard linear weighting).
    no_weighting_light_areas = True  # When True, pixels with intensity below 64 are not considered in the centroid calculation.
    
    os.makedirs(plots_dir, exist_ok=True)
    
    for d in directories:
        process_directory(d, base_data_dir, images_dir, plots_dir, save_images,
                          best_prediction_only, strong_weight, no_weighting_light_areas)

if __name__ == "__main__":
    main()
