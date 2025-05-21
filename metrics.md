Here’s a detailed rundown of each metric defined in your script, including its mathematical form, interpretation, typical range, and strengths/weaknesses.

---

### 1. Sum of Pixelwise Products (ρ)

* **Definition**
  Computes the sum over all pixels of the element-wise product of target and predicted images:

  $$
    \rho = \sum_{i} t_i \, p_i
  $$
* **Formula**
  In code: `np.sum(t * p)`.
* **Interpretation**
  – If both images have nonnegative intensities, a larger ρ indicates greater overall overlap in “energy” between target and prediction.
  – Sensitive to the absolute scale of intensities: brighter images yield larger ρ.
* **Units & Range**
  – Units: intensity² (if pixel values carry units).
  – Range: $[0, \sum t_i^2]$ for nonnegative data.
* **Pros/Cons**
  – **Pros:** Simple, fast, and captures raw correlation in magnitude.
  – **Cons:** Doesn’t normalize for brightness or contrast; hard to compare across images of different scale.

---

### 2. Cosine Similarity

* **Definition**
  Measures the cosine of the angle between the flattened target and prediction vectors:

  $$
    \cos(\theta) = \frac{\mathbf{t}\cdot \mathbf{p}}{\|\mathbf{t}\|\,\|\mathbf{p}\|}.
  $$
* **Formula**

  ```python
  dot = a_f @ b_f
  na  = ‖a_f‖;  nb = ‖b_f‖
  cosine = dot / (na * nb)  if na,nb≠0 else 0
  ```
* **Interpretation**
  – Values near 1 → images are “pointing” in the same direction (highly similar up to scaling).
  – Values near 0 → orthogonal (no similarity).
  – Negative only if negative pixel values are allowed.
* **Units & Range**
  – Unitless
  – Range: $[-1,1]$ (but for nonnegative images, $[0,1]$).
* **Pros/Cons**
  – **Pros:** Invariant to overall intensity scaling; highlights pattern similarity.
  – **Cons:** Doesn’t penalize magnitude differences; two images with different brightness but same pattern score 1.

---

### 3. Mean Squared Error (MSE)

* **Definition**
  The average of squared pixelwise differences:

  $$
    \text{MSE} = \frac{1}{N}\sum_i (t_i - p_i)^2.
  $$
* **Formula**

  ```python
  diff = a - b
  mse  = np.mean(diff * diff)
  ```
* **Interpretation**
  – Penalizes large errors more heavily (quadratic penalty).
  – A direct measure of reconstruction quality.
* **Units & Range**
  – Units: intensity²
  – Range: $[0, \infty)$; 0 indicates perfect match.
* **Pros/Cons**
  – **Pros:** Differentiable and convex—widely used as loss in optimization.
  – **Cons:** Sensitive to outliers; not intuitive in original intensity units.

---

### 4. Root Mean Squared Error (RMSE)

* **Definition**
  The square root of MSE, to bring error back into original units:

  $$
    \text{RMSE} = \sqrt{\text{MSE}}.
  $$
* **Formula**

  ```python
  rmse = np.sqrt(mean_squared_error(a,b))
  ```
* **Interpretation**
  – Directly comparable to pixel intensities: “on average, predictions are RMSE units away from targets.”
* **Units & Range**
  – Units: intensity
  – Range: $[0, \infty)$.
* **Pros/Cons**
  – **Pros:** More interpretable than MSE in original scale.
  – **Cons:** Still overly penalizes large deviations.

---

### 5. Mean Absolute Error (MAE)

* **Definition**
  The average of absolute pixelwise differences:

  $$
    \text{MAE} = \frac{1}{N}\sum_i \lvert t_i - p_i \rvert.
  $$
* **Formula**

  ```python
  mae = np.mean(np.abs(a - b))
  ```
* **Interpretation**
  – A linear penalty on errors; more robust to outliers than MSE/RMSE.
* **Units & Range**
  – Units: intensity
  – Range: $[0, \infty)$.
* **Pros/Cons**
  – **Pros:** Intuitive “average error,” less sensitive to large individual errors.
  – **Cons:** Not differentiable at zero (but subgradients exist).

---

### 6. Coefficient of Determination (R²)

* **Definition**
  Fraction of variance in target explained by prediction:

  $$
    R^2 = 1 - \frac{\sum_i (t_i - p_i)^2}{\sum_i (t_i - \bar t)^2}.
  $$
* **Formula**

  ```python
  ss_res = np.sum((a-b)**2)
  ss_tot = np.sum((a - a.mean())**2)
  r2     = 1 - ss_res/ss_tot  if ss_tot else 1
  ```
* **Interpretation**
  – 1 → perfect fit, 0 → as good as predicting the mean, negative → worse than mean.
* **Units & Range**
  – Unitless
  – Range: $(-\infty,\,1]$.
* **Pros/Cons**
  – **Pros:** Summarizes fit quality relative to variance.
  – **Cons:** Can be negative; misleading if data have near-zero variance.

---

### 7. Pearson Correlation Coefficient

* **Definition**
  Measures linear correlation between flattened images:

  $$
    r = \frac{\sum (t_i - \bar t)(p_i-\bar p)}{\sqrt{\sum(t_i-\bar t)^2}\,\sqrt{\sum(p_i-\bar p)^2}}.
  $$
* **Formula**

  ```python
  af = a.ravel()-a.mean()
  bf = b.ravel()-b.mean()
  pearson = (af@bf)/(‖af‖*‖bf‖)  if denom else 0
  ```
* **Interpretation**
  – 1 → perfect positive linear relationship, –1 → perfect negative, 0 → no linear correlation.
* **Units & Range**
  – Unitless
  – Range: $[-1,1]$.
* **Pros/Cons**
  – **Pros:** Normalizes for mean and variance; interprets strength/direction of linear relationship.
  – **Cons:** Only sensitive to linear relationships; ignores non‐linear similarity.

---

### 8. Soft Dice Coefficient

* **Definition**
  A “soft” (differentiable) version of the Dice score used for fuzzy segmentation:

  $$
    \text{Dice} = \frac{2\sum t_i p_i}{\sum t_i^2 + \sum p_i^2}.
  $$
* **Formula**

  ```python
  num = 2 * np.sum(a*b)
  den = np.sum(a*a) + np.sum(b*b)
  dice = num/den  if den else 1
  ```
* **Interpretation**
  – 1 → perfect overlap, 0 → no overlap.
  – Balances precision and recall in segmentation contexts.
* **Units & Range**
  – Unitless
  – Range: $[0,1]$.
* **Pros/Cons**
  – **Pros:** Differentiable surrogate of set-based Dice; robust to class imbalance.
  – **Cons:** Less intuitive outside segmentation tasks.

---

### 9. Soft Intersection-over-Union (IoU)

* **Definition**
  “Soft” (differentiable) analogue of Jaccard index:

  $$
    \text{IoU} = \frac{\sum t_i p_i}{\sum t_i^2 + \sum p_i^2 - \sum t_i p_i}.
  $$
* **Formula**

  ```python
  inter = np.sum(a*b)
  uni   = np.sum(a*a) + np.sum(b*b) - inter
  iou   = inter/uni  if uni else 1
  ```
* **Interpretation**
  – 1 → perfect overlap, 0 → no overlap.
  – More punitive than Dice on small overlaps.
* **Units & Range**
  – Unitless
  – Range: $[0,1]$.
* **Pros/Cons**
  – **Pros:** Common in segmentation benchmarks (VOC, COCO).
  – **Cons:** Harsh penalty on boundary mismatches.

---

### 10. Structural Similarity Index (SSIM)

* **Definition**
  Perceptual metric that compares local luminance, contrast, and structure:

  $$
    \text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
  $$
* **Implementation Details**
  – Uses `skimage.metrics.structural_similarity` with automatically computed `data_range`.
  – Returns a map of local SSIM values; code takes the global mean.
* **Interpretation**
  – 1 → perceptually identical images; lower values indicate structural/contrast/luminance differences.
* **Units & Range**
  – Unitless
  – Range: $(-1,1]$ (but typically $[0,1]$ for nonnegative images).
* **Pros/Cons**
  – **Pros:** Models human visual perception; sensitive to structural changes.
  – **Cons:** More expensive to compute; parameters (window size, constants) can affect results.

---

**Summary of When to Use Each**

* **Energy-based**: ρ, MSE/RMSE, MAE (simple magnitudes).
* **Angle/Correlation**: Cosine similarity, Pearson (pattern vs. linear relationship).
* **Segmentation overlap**: Soft Dice, Soft IoU.
* **Perceptual quality**: SSIM.
* **Regression fit**: R².

Choosing the right metric depends on whether you care about raw error magnitude, pattern similarity, perceptual quality, or semantic overlap.
