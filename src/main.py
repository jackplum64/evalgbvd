from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import cv2
import csv
import matplotlib.pyplot as plt
from matplotlib import cm as colormap
import io
import contextlib
from scipy.stats import gaussian_kde
from dataio import read_data, organize_data
from datametrics import (sum_pixelwise_product, sum_pixelwise, sum_gt, sum_pred,
                         cosine_similarity, mean_squared_error, root_mean_squared_error,
                         matthews_correlation, r_squared, pearson_correlation, soft_dice,
                         soft_iou, ssim_index)
from confusionmatrix import (pixelwise_confusion_matrix, pixel_accuracy, precision,
                             recall, specificity, f1_score)
from imageops import (stack_image, insert_vertical_line, contrast_enhance, enhance_in_place,
                      normalize_image, normalize_in_place, strengthen_colormap,
                      strengthen_colormap_in_place, kill_zeros_in_place, save_and_kill_zeros_in_place)
from plotops import load_metrics, plot_metrics_density, plot_confusion_matrix, plot_metrics_histograms
from statsops import print_pearson_bimodal_analysis



def main() -> None:
    script_dir = Path(__file__).parent

    parent_dir = Path('/home/jackplum/Documents/projects/evalgbvd/data/results_focal_tversky_gbblur7_final')
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
        overview_path = out_dir / 'overview.pdf'
        plot_metrics_density(metrics, overview_path)
        print(f'[{child.name}] Saved overview density plot to {overview_path}')

        # ─── Generate & save histograms ────────────────────────────────────────
        histograms_path = out_dir / 'histogram.pdf'
        plot_metrics_histograms(metrics, histograms_path)
        print(f'[{child.name}] Saved histograms to {histograms_path}')

        # ─── Aggregate confusion matrices and plot ─────────────────────────────
        # Sum element-wise across all per-image 2x2 matrices
        stacked = np.stack(conf_mats, axis=0)  # shape = (N_images, 2, 2)
        total_cm = np.sum(stacked, axis=0)     # shape = (2, 2)
        cm_path = out_dir / 'confusion_matrix.pdf'
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

            newfig_path = out_dir / 'newfig.pdf'
            plt.tight_layout()
            plt.savefig(newfig_path, dpi=250, format="pdf")
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
                c='gray',  # single-point color
                s=50,
                edgecolors='black'
            )
            plt.xlabel('Pearson Correlation', fontsize=12)
            plt.ylabel('Pixelwise Sum (target)', fontsize=12)
            plt.title('Pearson vs. Target Pixelwise Sum (Single Sample)', fontsize=14)
            plt.grid(linestyle=':', linewidth=0.5)

            newfig_path = out_dir / 'newfig.pdf'
            plt.tight_layout()
            plt.savefig(newfig_path, dpi=250, format="pdf")
            plt.close()
            print(f'[{child.name}] Only one sample—saved single-point plot to {newfig_path}')

        else:
            # No data at all: skip this plot
            print(f'[{child.name}] No data for Pearson vs. pixelwise-sum plot (skipping).')

if __name__ == '__main__':
    main()