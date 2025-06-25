import numpy as np
from typing import List
import math
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

def print_pearson_bimodal_analysis(scores: List[float]) -> None:
    arr: np.ndarray = np.array(scores, dtype=float)
    n: int = arr.size

    mean: float = float(np.mean(arr))
    std: float = float(np.std(arr, ddof=0))
    med: float = float(np.median(arr))
    sk: float = float(skew(arr))
    kurt: float = float(kurtosis(arr))

    bins = min(50, int(math.sqrt(n))) if n else 10
    counts, bin_edges = np.histogram(arr, bins=bins)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    peak_idxs, _ = find_peaks(counts)
    if peak_idxs.size < 2:
        peak_idxs = np.argsort(counts)[-2:]
    top2 = peak_idxs[np.argsort(counts[peak_idxs])[::-1][:2]]

    mode_locs = [centers[i] for i in top2]
    mode_counts = [int(counts[i]) for i in top2]

    half_w: float = 0.5 * std
    integrals: List[int] = []
    for loc in mode_locs:
        low, high = loc - half_w, loc + half_w
        mask = (centers >= low) & (centers <= high)
        integrals.append(int(counts[mask].sum()))

    ratio: float = integrals[0] / integrals[1] if integrals[1] else float('inf')

    print("=== Bimodal Analysis of Pearson Correlation Scores ===")
    print(f"Samples: {n}")
    print(f"Mean: {mean:.4f}, Median: {med:.4f}, Std Dev: {std:.4f}")
    print(f"Skewness: {sk:.4f}, Kurtosis: {kurt:.4f}\n")

    print("Identified Modes:")
    for i, (loc, cnt) in enumerate(zip(mode_locs, mode_counts), start=1):
        print(f" Mode {i}: location = {loc:.4f}, peak count = {cnt}")
    print()

    print("Integrals in ±0.5σ window around each mode:")
    for i, integral in enumerate(integrals, start=1):
        print(f" Mode {i} integral (±0.5σ): {integral}")
    print(f"Integral-based mode ratio: {ratio:.4f}\n")

    print("Percentage within ±kσ of each mode:")
    for i, loc in enumerate(mode_locs, start=1):
        print(f" Mode {i} (loc={loc:.4f}):")
        for k in (0.5, 1.0, 1.5):
            low, high = loc - k * std, loc + k * std
            pct = float(np.mean((arr >= low) & (arr <= high))) * 100.0
            print(f"   ±{k}σ: {pct:.2f}% (range {low:.4f} to {high:.4f})")
        print()


def main():
    print('statsops.py was run but there is no code in main()')

if __name__ == '__main__':
    main()