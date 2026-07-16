# Chirp Correction for Broadband Transient Absorption

## What is chirp and why does it matter?

In a broadband transient absorption (TA) experiment, the white-light continuum probe pulse passes through optical elements --- windows, filters, the sample cuvette, and the solvent itself --- before reaching the detector. Each of these materials has a wavelength-dependent refractive index, so different colors travel at different speeds. This effect is called **group velocity dispersion** (GVD), and the resulting wavelength-dependent arrival time is called **chirp**.

Concretely, if you look at a raw TA surface (signal vs. time and wavelength), you will see that the signal onset is not a vertical line at ``t = 0``. Instead it traces a curve --- typically with blue wavelengths arriving later than red ones (positive GVD in most materials). This curved onset is the chirp.

If you don't correct for chirp, your data has two problems:

1. **Spectra at early times are distorted.** A spectrum extracted at, say, ``t = 200`` fs doesn't represent a single moment in time --- it's a superposition of signals at different pump-probe delays for different wavelengths.
2. **Kinetic traces at different wavelengths have shifted time zeros.** Fitting these traces yields apparent time constants that are convolved with the chirp, biasing your results.

## Overview of correction methods

Several approaches exist for detecting and correcting chirp. They divide into two categories: methods that determine the chirp curve and correct it as a preprocessing step, and methods that fold chirp into the kinetic model itself.

### Preprocessing approaches

These all follow the same two-stage logic: first measure the wavelength-dependent time zero ``t_0(\lambda)``, then shift each wavelength trace in time to align them.

#### Coherent artifact / cross-phase modulation (XPM) detection

The most common approach. The sharp, non-resonant coherent artifact that appears around time zero in TA data is wavelength-dependent due to chirp. You find the time-zero position at each wavelength, then fit a smooth function to the resulting chirp curve. This is what OpticalSpectroscopy implements.

Several strategies exist for locating time zero at each wavelength:

- **Maximum of ``|\Delta A|``:** Find the time point where the absolute signal is largest in the early-time window. Simple but can be biased if real dynamics have large amplitude near time zero.
- **Steepest slope (maximum of ``d\Delta A/dt``):** Take the numerical derivative and find its extremum. This targets the *rising edge* of the signal rather than the peak, making it less sensitive to overlapping dynamics. Probably the most widely used variant.
- **Half-rise point:** Find where each trace crosses 50% of its early-time extremum. Less sensitive to noise than the derivative method, but assumes a monotonic rise.
- **Fit a step function convolved with a Gaussian:** At each wavelength, fit ``A \cdot \text{erfc}((t - t_0) / \sigma) + \text{offset}``. This gives ``t_0`` and the instrument response width ``\sigma`` simultaneously. Most rigorous, but slow when applied at every wavelength.
- **Cross-correlation of onset gradients:** Cross-correlate the absolute gradient of each wavelength bin against a reference bin (the one with the strongest signal). Since the absolute gradient produces a spike at the onset regardless of whether the signal is positive (GSB) or negative (ESA), this is polarity-independent. Parabolic interpolation on the cross-correlation peak gives sub-time-step precision. This is the default `:xcorr` method in OpticalSpectroscopy.

#### Optical Kerr effect (OKE) reference measurement

Measure the optical Kerr effect in a non-resonant medium (pure solvent, glass) placed at the sample position. The OKE signal is instantaneous, so its peak position vs. wavelength directly traces the chirp curve. This is considered the gold standard because it is independent of sample dynamics, but it requires a separate measurement with a different detection geometry (crossed polarizers).

#### Two-photon absorption (TPA) reference

Use a thin semiconductor or dye solution where two-photon absorption provides a sharp, instantaneous response. The TPA signal onset traces the chirp curve. Less common than OKE, but useful when the setup geometry makes OKE difficult.

#### Sellmeier / dispersion calculation

Calculate the expected GVD analytically from the known optical elements in the probe path using Sellmeier equations for each material's refractive index. This gives a physics-based chirp curve without any calibration measurement. In practice this is used as a sanity check or starting guess rather than a primary correction, because small alignment differences and unaccounted optics introduce errors.

#### Hardware pre-compensation

Use chirped mirrors, prism compressors, or grism (grating + prism) compressors to pre-compensate the probe GVD so the pulse is nearly transform-limited at the sample. This minimizes chirp before acquisition rather than correcting it afterward. Often combined with a small software correction for residual chirp.

### Model-based approaches

#### Global analysis with chirp as a free parameter

In global/target analysis (e.g., Glotaran), the chirp parameters are included as free parameters in the kinetic fit. The chirp curve, typically parameterized as a low-order polynomial, is optimized simultaneously with the rate constants and spectral amplitudes.

This is becoming increasingly popular because:

- It avoids sequential error propagation --- errors in the chirp detection step don't get baked into the "corrected" data before the kinetic model sees it.
- The coherent artifact often overlaps with genuine ultrafast dynamics (solvation, vibrational cooling) in the first few hundred femtoseconds, making algorithmic separation ambiguous.
- Software like Glotaran and pyglotaran has made it accessible without custom code.
- It is self-consistent for publication: one coherent set of parameters rather than a preprocessing step with hidden assumptions.

The main downside is that the chirp estimate is coupled to the kinetic model. A wrong model can give wrong chirp, and the optimization landscape is more complex. Most groups still do an initial correction as preprocessing, then let the global fit refine the chirp parameters.

## Step-by-step correction as a preprocessing step

This section walks through the standard workflow in detail.

### Step 1: Subtract the pre-pump background

TA is already a difference measurement, so the signal before the pump arrives should be zero. Any residual offset is systematic background (detector dark current, scattered pump light, etc.). Subtracting it removes a constant bias that otherwise degrades detection --- and for `:threshold` it is effectively required: on a matrix that still carries its background, every bin crosses the half-maximum at the first sample, and detection collapses to a flat, useless calibration (`detect_chirp` warns and reports `R^2` as `NaN`).

```julia
using OpticalSpectroscopy

matrix_bg = subtract_background(matrix)
```

[`subtract_background`](@ref) averages the signal in the pre-pump region (auto-detected or specified via `t_range`) and subtracts that baseline from every time point.

### Step 2: Detect the chirp curve

OpticalSpectroscopy provides two detection methods. Both bin the wavelength axis for noise reduction, smooth each bin with a Savitzky-Golay filter, then locate the signal onset.

```julia
cal = detect_chirp(matrix_bg)
report(cal)
```

The default `:xcorr` method cross-correlates onset gradients against the strongest-signal bin. Use `:threshold` for simpler half-maximum onset detection:

```julia
cal_thr = detect_chirp(matrix_bg; method=:threshold)
```

Key parameters to adjust:

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `bin_width` | 8 | Increase for high-pixel-count CCDs (e.g. 2048 px). `16` or `32` reduces noise. |
| `min_signal` | 0.2 | Lower to include weak-signal spectral regions (noisier detection). |
| `order` | 3 | A 2nd-order polynomial suffices for modest chirp. Use 3 for broadband (> 200 nm) coverage. |
| `smooth_window` | 15 | Increase in noisy data. Must be odd. |
| `threshold` | 3.0 | MAD multiplier for outlier rejection. Lower = more aggressive rejection. |

### Step 3: Inspect the detected points

Before trusting the calibration, check that the detected chirp points are sensible. The `ChirpCalibration` stores both the raw detected points and the polynomial fit:

```julia
cal.wavelength     # detected wavelength points (nm)
cal.time_offset    # detected time offset at each point (ps)
cal.poly_coeffs    # polynomial coefficients (ascending order)
cal.r_squared      # how well the polynomial fits the detected points
cal.metadata[:n_unmeasurable]  # bins discarded as having no measurable shift

poly = polynomial(cal)  # callable: t_shift = poly(lambda)
poly(500.0)             # chirp offset at 500 nm
```

**Do not judge the calibration by ``R^2``.** It measures only whether a polynomial describes the points that were detected, and says nothing about whether those points are the chirp. Because it is normalised by the variance of those points, a few extreme values push it *up*: a curve bending to chase them scores well precisely because they are extreme. A real 2048-pixel CCD scan scores ``R^2 = 0.96`` on a chirp that isn't there, and 0.27 once eleven junk bins are removed.

Check these instead:

- **Is the chirp resolvable at all?** Compare `maximum(cal.time_offset) - minimum(cal.time_offset)` against your delay step. Chirp across the visible is typically 1--3 ps, so a scan stepping coarser than a few hundred fs cannot see it, and whatever comes back is noise. This is a property of the measurement, not something a parameter can fix.
- **How many bins were discarded?** `cal.metadata[:n_unmeasurable]` counts bins whose correlation peak hit the lag-search bound (`detect_chirp` also warns). A few at the spectral edges is normal; a large fraction means the time axis is too coarse or the edges are noise-dominated.
- **Do the points scatter smoothly around the curve** without systematic deviation?

If weak spectral edges are being dropped, raise `bin_width` to average more pixels per bin. Lowering `min_signal` admits those regions but makes detection noisier there --- it is not a way to rescue a scan that cannot resolve the chirp.

### Step 4: Apply the correction

[`correct_chirp`](@ref) shifts each wavelength column in time using cubic spline interpolation:

```julia
matrix_corrected = correct_chirp(matrix_bg, cal)
```

For each wavelength ``\lambda_j``, the corrected signal at time ``t_i`` is the original signal evaluated at ``t_i + t_{\text{shift}}(\lambda_j)``, where ``t_{\text{shift}}`` comes from the calibration polynomial. Cubic B-spline interpolation ensures sharp features (the coherent artifact, fast rise times) are preserved without the broadening that linear interpolation would introduce.

At the edges of the time axis, where the shifted time falls outside the measured range, the signal is extrapolated as flat (constant value equal to the boundary). Make sure your time window extends well beyond time zero so that the correction doesn't push any wavelength off the edge.

### Step 5: Validate

After correction, check three things:

1. **The TA surface.** Plot the corrected ``\Delta A(t, \lambda)`` heatmap. The coherent artifact should now appear as a vertical line at ``t = 0``, not a curved streak.

2. **Early-time spectra.** Extract spectra at several early delays (e.g. ``t = 0.1, 0.2, 0.5`` ps). Before correction these are distorted by chirp; after correction they should look like physically reasonable spectra without the characteristic dispersive artifact shape.

3. **Kinetics at the spectral extremes.** Compare kinetic traces at the blue and red edges. They should now show simultaneous rise times within the instrument response. If one edge is systematically shifted, the polynomial order may be too low or the detection failed at the edges.

### Step 6: Save the calibration

Store the chirp calibration for reproducibility and for applying to other datasets taken under the same optical conditions:

```julia
save_chirp("chirp_cal_2024-01-15.json", cal)

# Later, or in a different script:
cal = load_chirp("chirp_cal_2024-01-15.json")
matrix_corrected = correct_chirp(new_matrix, cal)
```

The JSON file stores the polynomial coefficients, detected points, and all detection parameters so the calibration is fully reproducible.

## Common pitfalls

**Using too narrow a time window for detection.**
If the detection window doesn't capture the full signal rise at all wavelengths (because of the chirp itself), edge wavelengths get incorrect ``t_0`` values. The auto-detection in OpticalSpectroscopy accounts for this, but if you set `t_range` manually, make it generous.

**Getting the sign convention wrong.**
Does ``t_0 > 0`` mean the blue arrives early or late? Getting this backwards flips the correction and doubles the chirp instead of removing it. In normal materials, the blue arrives later (positive GVD). After correction, check that the surface looks *less* curved, not more.

**Correcting twice.**
If the acquisition software already applies a partial chirp correction, applying a second correction on top will overcorrect. Check the raw data before starting.

**Overfitting the polynomial.**
The chirp curve should be smooth. Any high-frequency structure in ``t_0(\lambda)`` is noise. Use the minimum polynomial order that captures the curvature --- 2nd or 3rd order is almost always sufficient. OpticalSpectroscopy uses MAD-based outlier rejection to guard against this, but inspecting the residuals is still good practice.

**Ignoring the wavelength-dependent IRF.**
GVD doesn't just shift time zero --- it also temporally broadens the probe pulse at wavelengths far from the continuum center. The correction procedure above fixes the shift but not the broadening. For the most accurate kinetics, you may need a wavelength-dependent instrument response function in your fitting model.

## Full example

```julia
using OpticalSpectroscopy

# Assume `matrix` is a TimeResolvedMatrix loaded from your data
# Step 1: Background subtraction
matrix_bg = subtract_background(matrix)

# Step 2: Detect chirp
cal = detect_chirp(matrix_bg; bin_width=16, order=3)
report(cal)

# Step 3: Inspect
println("R-squared: ", cal.r_squared)
println("Number of detected points: ", length(cal.wavelength))

# Step 4: Correct
matrix_corrected = correct_chirp(matrix_bg, cal)

# Step 5: Save for reuse
save_chirp("chirp_calibration.json", cal)
```

## API Reference

See the [Additional API Reference](@ref api-reference) page for full docstrings of all chirp functions.
