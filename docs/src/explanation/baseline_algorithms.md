# Baseline Algorithms

OpticalSpectroscopy.jl provides baseline correction algorithms that take different approaches and perform best in different situations. This page explains the tradeoffs to help you choose.

## The Baseline Problem

Spectroscopic signals sit on top of a slowly varying background caused by fluorescence, scattering, detector drift, or optical effects. Baseline correction estimates and removes this background so that peak parameters (position, width, area) can be measured accurately.

A good baseline algorithm should:
- Follow the slowly varying background
- Not be pulled upward by peaks
- Require minimal manual tuning

## Algorithm Summary

| Algorithm | Approach | Tuning | Best for |
|-----------|----------|--------|----------|
| **arPLS** | Penalized least squares with automatic weighting | ``\lambda`` only | General purpose (recommended default) |
| **SNIP** | Iterative morphological clipping | `iterations` | Spectra with many overlapping peaks |

ALS (the classic asymmetric least squares method) is not exported as its own function; arPLS supersedes it for practical work. The ALS algorithm is discussed below for context.

## arPLS (Asymmetrically Reweighted Penalized Least Squares)

```julia
baseline = arpls_baseline(y; λ=1e5)
```

arPLS is the **recommended default**. It improves on ALS by automatically determining the asymmetry weights from the statistics of negative residuals (points below the current baseline estimate). This eliminates the need to tune the `p` parameter.

**How it works:** At each iteration, points far above the baseline get low weight (they're peaks), while points near or below the baseline get high weight (they're background). The weighting uses a sigmoid function centered at ``-\mu + 2\sigma`` of the negative residuals, where ``\mu`` and ``\sigma`` are the mean and standard deviation.

**When to use:** Start here. Works well for FTIR, Raman, UV-vis, and most spectroscopy data.

**Tuning ``\lambda``:**
- Higher ``\lambda`` = smoother baseline (ignores local variations)
- Lower ``\lambda`` = baseline follows local structure more closely
- Typical range: ``10^4`` to ``10^8``
- If baseline cuts through peaks: increase ``\lambda``
- If baseline doesn't follow broad curvature: decrease ``\lambda``

**Reference:** Baek et al. (2015). *Analyst*, 140(1), 250-257.

## ALS (Asymmetric Least Squares)

ALS is the conceptual predecessor to arPLS. OpticalSpectroscopy.jl does not export a standalone `als_baseline` function — [`arpls_baseline`](@ref) is recommended instead and supersedes plain ALS for almost all practical work. The weighted-update idea that ALS introduced is what arPLS extends with automatic reweighting.

ALS uses a fixed asymmetry parameter `p` to weight residuals. Points above the baseline are weighted by `p` (small), points below by `1-p` (large). This pushes the baseline below peaks.

**How it works:** Minimizes a penalized objective:

```math
\sum_i w_i (y_i - z_i)^2 + \lambda \sum_i (\Delta^2 z_i)^2
```

The first term fits the data (with asymmetric weights), and the second term penalizes roughness (second differences of the baseline).

**Tuning (conceptual):**
- ``\lambda``: smoothness (same as arPLS)
- ``p``: asymmetry (0.001 to 0.05). Lower values keep the baseline further below peaks. If peaks are being clipped, decrease `p`.

The main drawback — needing to hand-tune `p` — is exactly what arPLS removes by deriving the weighting from residual statistics.

**Reference:** Eilers & Boelens (2005). Leiden University Medical Centre Report.

## SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping)

```julia
baseline = snip_baseline(y; iterations=40)
```

SNIP takes a fundamentally different approach: instead of fitting a smooth function, it iteratively clips peaks by replacing each point with the minimum of itself and the average of its neighbors at increasing distances.

**How it works:** For each iteration with window size ``k``:

```math
z_i = \min(z_i, (z_{i-k} + z_{i+k}) / 2)
```

By default, window sizes decrease from `iterations` down to 1. This handles broad features first, then refines around narrow peaks.

**When to use:** Spectra with many overlapping peaks where penalized methods struggle to find the true baseline. Common in Raman spectroscopy of complex materials, XRD patterns, and energy-dispersive X-ray spectra.

**Tuning `iterations`:** Should be roughly half the width (in data points) of the widest peak. More iterations produce a smoother baseline. For a 2048-point Raman spectrum with peaks ~80 points wide, try `iterations=40`.

**Advantages over PLS methods:**
- No smoothness parameter to tune
- Handles many overlapping peaks naturally
- Computationally simpler

**Disadvantages:**
- Less smooth baseline (can show small artifacts)
- May not follow gradual curvature as well

**Reference:** Ryan et al. (1988). *Nuclear Instruments and Methods B*, 34(3), 396-402.

## Choosing an Algorithm

```
Is this a routine analysis with isolated or moderately overlapping peaks?
  → Use arPLS (default)

Does arPLS cut through peaks or miss broad curvature?
  → Try adjusting λ first

Does the spectrum have many heavily overlapping peaks?
  → Use SNIP
```

## Practical Tips

1. **Always plot the baseline** overlaid on the raw spectrum before trusting the correction. All baseline functions return just the baseline vector, so you can inspect it:

   ```julia
   baseline = arpls_baseline(y)
   lines(x, y)
   lines!(x, baseline, color=:red)
   ```

2. **Use `correct_baseline` for convenience** — it returns both the corrected spectrum and the baseline in a named tuple:

   ```julia
   result = correct_baseline(x, y; method=:arpls)
   result.y        # corrected
   result.baseline  # the baseline
   ```

3. **Baseline correction before peak fitting** can improve results, but `fit_peaks` already includes a polynomial baseline in the fit model. For well-isolated peaks, the built-in baseline may be sufficient.

4. **Baseline correction for peak detection** is available directly in `find_peaks`:

   ```julia
   peaks = find_peaks(x, y; baseline=:arpls)
   ```

## References

1. Eilers, P. H. C., & Boelens, H. F. M. (2005). Baseline Correction with Asymmetric Least Squares Smoothing.
2. Baek, S.-J., Park, A., Ahn, Y.-J., & Choo, J. (2015). Baseline correction using asymmetrically reweighted penalized least squares smoothing. *Analyst*, 140(1), 250-257.
3. Ryan, C. G., et al. (1988). SNIP, a statistics-sensitive background treatment. *Nuclear Instruments and Methods B*, 34(3), 396-402.
4. Morháč, M., & Matoušek, V. (2008). Peak Clipping Algorithms for Background Estimation. *Applied Spectroscopy*, 62(1), 91-106.
