import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm

def compute_weighted_area(image_array: np.ndarray) -> float:
    """
    Compute the weighted area for an image array. Each pixel contributes
    according to its inverted normalized intensity:
        weight = 1 - normalized_value
    where normalized_value is the pixel value scaled between 0 and 1.
    
    Args:
        image_array (np.ndarray): The image data.
    
    Returns:
        float: The sum of the pixel weights.
    """
    if np.issubdtype(image_array.dtype, np.uint8):
        normalized = image_array.astype(np.float32) / 255.0
    elif np.issubdtype(image_array.dtype, np.floating):
        normalized = image_array
    else:
        normalized = image_array.astype(np.float32)
    
    weights = 1.0 - normalized
    return float(np.sum(weights))

def compute_weighted_centroid(image_array: np.ndarray) -> tuple[float, float]:
    """
    Compute the weighted centroid (x, y) for an image array using the same
    weight strategy as compute_weighted_area.
    
    Each pixel contributes a weight of: weight = 1 - normalized_value.
    If the total weight is 0 (e.g. an entirely saturated image), the function
    returns (nan, nan).
    
    Returns:
        (x_centroid, y_centroid): The computed centroid coordinates.
    """
    if np.issubdtype(image_array.dtype, np.uint8):
        normalized = image_array.astype(np.float32) / 255.0
    elif np.issubdtype(image_array.dtype, np.floating):
        normalized = image_array
    else:
        normalized = image_array.astype(np.float32)
    
    weights = 1.0 - normalized
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        # Avoid division by zero: return nan coordinates if the image is completely saturated.
        return float('nan'), float('nan')
    
    height, width = image_array.shape[:2]
    # Create coordinate grids: y corresponds to rows, x to columns.
    y_indices, x_indices = np.indices((height, width))
    
    x_centroid = np.sum(x_indices * weights) / total_weight
    y_centroid = np.sum(y_indices * weights) / total_weight
    return x_centroid, y_centroid

def process_directory(directory: str) -> None:
    """
    Process each prediction/target pair of .npy files in a specified sub-directory,
    compute their weighted areas and centroids, generate scatter plots comparing 
    predicted and target values, and save the plots into the script directory.

    Args:
        directory (str): The name of the sub-directory (e.g., "test", "train", or "val").
    """
    print(f"Processing directory: {directory}")
    data_dir: str = os.path.join("4825newresults5", "tversky_focal_recons_float", directory)
    
    # Retrieve all prediction files
    pred_files = glob.glob(os.path.join(data_dir, "*_pred.npy"))
    pred_files.sort()  # Ensure consistent ordering
    
    if not pred_files:
        print(f"No prediction files found in directory: {data_dir}")
        return

    # Prepare lists to store computed values.
    predicted_areas = []
    target_areas = []
    pred_centroid_x = []
    pred_centroid_y = []
    target_centroid_x = []
    target_centroid_y = []

    # Process each prediction file.
    for pred_file in pred_files:
        base_name = os.path.basename(pred_file)
        index = base_name.split("_")[0]
        
        # Construct the corresponding target filename.
        target_filename = f"{index}_target.npy"
        target_filepath = os.path.join(data_dir, target_filename)
        
        if not os.path.exists(target_filepath):
            print(f"Warning: Target file '{target_filepath}' does not exist. Skipping index {index}.")
            continue
        
        # Load prediction and target images.
        pred_image = np.load(pred_file)
        target_image = np.load(target_filepath)
        
        # Compute weighted areas.
        pred_area = compute_weighted_area(pred_image)
        target_area = compute_weighted_area(target_image)
        
        # Compute weighted centroids.
        pred_cx, pred_cy = compute_weighted_centroid(pred_image)
        target_cx, target_cy = compute_weighted_centroid(target_image)
        
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
    
    # Convert lists to NumPy arrays.
    predicted_areas = np.array(predicted_areas)
    target_areas = np.array(target_areas)
    pred_centroid_x = np.array(pred_centroid_x)
    pred_centroid_y = np.array(pred_centroid_y)
    target_centroid_x = np.array(target_centroid_x)
    target_centroid_y = np.array(target_centroid_y)
    
    # ---------------------------
    # Plot 1: Weighted Area Comparison
    # ---------------------------
    if target_areas.size > 0 and predicted_areas.size > 0:
        coeffs = np.polyfit(target_areas, predicted_areas, 1)
        slope, intercept = coeffs
        print(f"Weighted Area Best fit line in {directory}: predicted_area = {slope:.2f} * target_area + {intercept:.2f}")
        
        x_fit = np.linspace(target_areas.min(), target_areas.max(), 100)
        y_fit = slope * x_fit + intercept
        
        # Compute density of the data points.
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
        plt.savefig(f"parityplot_weighted_area_{directory}.pdf", dpi=250)
        plt.show()
    
    # ---------------------------
    # Plot 2: Centroid Comparison
    # ---------------------------
    fig, (ax_x, ax_y) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Centroid X Comparison
    valid_x = ~np.isnan(target_centroid_x) & ~np.isnan(pred_centroid_x)
    if np.any(valid_x):
        coeffs_x = np.polyfit(target_centroid_x[valid_x], pred_centroid_x[valid_x], 1)
        slope_x, intercept_x = coeffs_x
        print(f"Centroid X Best fit line in {directory}: predicted_x = {slope_x:.2f} * target_x + {intercept_x:.2f}")
        
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
    
    # Centroid Y Comparison
    valid_y = ~np.isnan(target_centroid_y) & ~np.isnan(pred_centroid_y)
    if np.any(valid_y):
        coeffs_y = np.polyfit(target_centroid_y[valid_y], pred_centroid_y[valid_y], 1)
        slope_y, intercept_y = coeffs_y
        print(f"Centroid Y Best fit line in {directory}: predicted_y = {slope_y:.2f} * target_y + {intercept_y:.2f}")
        
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
    plt.savefig(f"parityplot_centroid_{directory}.pdf", dpi=250)
    plt.show()

def main() -> None:
    """
    Loop over the specified directories ("test", "train", and "val") and process each one.
    """
    directories = ["test", "train", "val"]
    for d in directories:
        process_directory(d)

if __name__ == "__main__":
    main()