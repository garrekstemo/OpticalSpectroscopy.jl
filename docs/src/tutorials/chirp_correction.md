# Chirp Correction for Broadband Transient Absorption

## What is chirp and why does it matter?

In a broadband transient absorption (TA) experiment, the white-light continuum probe pulse passes through optical elements --- windows, filters, the sample cuvette, and the solvent itself --- before reaching the detector. Each of these materials has a wavelength-dependent refractive index, so different colors travel at different speeds. This effect is called **group velocity dispersion** (GVD), and the resulting wavelength-dependent arrival time is called **chirp**.

Concretely, if you look at a raw TA surface (signal vs. time and wavelength), you will see that the signal onset is not a vertical line at ``t = 0``. Instead it traces a curve --- typically with blue wavelengths arriving later than red ones (positive GVD in most materials). This curved onset is the chirp.

If you don't correct for chirp, your data has two problems:

1. **Spectra at early times are distorted.** A spectrum extracted at, say, ``t = 200`` fs doesn't represent a single moment in time --- it's a superposition of signals at different pump-probe delays for different wavelengths.
2. **Kinetic traces at different wavelengths have shifted time zeros.** Fitting these traces yields apparent time constants that are convolved with the chirp, biasing your results.

## Two ways to get the chirp curve

OpticalSpectroscopy provides two producers of a [`ChirpCalibration`](@ref):

- **[`calibrate_chirp`](@ref) — measure it (recommended).** Run a dedicated optical Kerr effect (OKE) cross-correlation measurement and fit the per-wavelength peak position. This is the standard calibration: it is independent of any sample dynamics.
- **[`detect_chirp`](@ref) — infer it from the sample data (fallback).** Estimate ``t_0(\lambda)`` from the sample matrix's own signal onsets. Quick, needs no extra measurement, but only unbiased under conditions stated below.

Either way, the resulting calibration feeds the same [`correct_chirp`](@ref) step and the same [`save_chirp`](@ref)/[`load_chirp`](@ref) persistence.

## The recommended path: calibrate from an OKE run

Measure the optical Kerr effect in a non-resonant medium (pure solvent, glass) placed at the sample position, with crossed polarizers. The electronic Kerr response is non-resonant and effectively instantaneous, so the per-wavelength peak of the pump--probe cross-correlation traces the true ``t_0(\lambda)`` independent of any sample dynamics --- and the per-wavelength width of that peak is the instrument response function IRF(``\lambda``) for free. This is the gold standard because nothing about your sample's kinetics can bias it.

```julia
using OpticalSpectroscopy

oke = ...                        # TimeResolvedMatrix from the OKE run
cal = calibrate_chirp(oke)
report(cal)

corrected = correct_chirp(sample_matrix, cal)   # apply to the sample run(s)
save_chirp("chirp_oke_2026-07-23.json", cal)
```

For each wavelength column, `calibrate_chirp` locates the cross-correlation peak:

- `method=:gaussian` (default) fits a Gaussian with vertical offset, yielding ``t_0`` **and** the IRF width ``\sigma`` (a standard deviation) per wavelength.
- `method=:peak` uses the argmax with parabolic sub-sample interpolation --- faster, for high-SNR runs, with no IRF widths.

Columns whose peak-to-peak amplitude falls below `min_amplitude` of the strongest column (5% by default) are masked out, as are columns whose per-column fit fails. The surviving ``(\lambda, t_0)`` points are fitted with a polynomial using the same MAD-based outlier rejection as `detect_chirp`.

Key parameters:

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `method` | `:gaussian` | `:peak` for a fast pass on clean, high-SNR runs. |
| `order` | 3 | 2 suffices for modest chirp; 3 for broadband (> 200 nm) coverage. |
| `reference` | `:center` | Wavelength (nm) where the polynomial is zero. `:center` uses the midpoint of the surviving coverage. |
| `wl_range` | `nothing` | `(λmin, λmax)` to exclude spectral regions (e.g. pump scatter). |
| `t_range` | `nothing` | `(tmin, tmax)` to restrict the per-column fit window. |
| `min_amplitude` | 0.05 | Raise to reject more weak-signal columns; `0` disables the mask. |

### What the calibration carries

Beyond the polynomial, the calibration's `metadata` records provenance that downstream tools rely on:

- `:source => :oke` and `:source_file` --- where the calibration came from.
- `:t0_at_reference` --- the **absolute** fitted ``t_0`` at the reference wavelength. The stored polynomial is normalized to zero at `reference_λ`, so this is what connects the calibration back to the delay-stage axis (e.g. to seed a time-zero correction).
- `:irf_sigma` --- per-wavelength IRF standard deviations, aligned with `cal.wavelength` (`:gaussian` method only). Use these for wavelength-dependent IRF deconvolution in kinetic fits.
- `:lambda_range` --- the wavelength coverage actually kept in the fit, which controls clamping (next section).

### Coverage and clamping

A calibration is only trustworthy over the wavelengths it actually measured. When the spectrograph center changes between runs, the wavelength axes shift, and a sample run can extend past the calibration's coverage. A polynomial extrapolated beyond its data is garbage --- a cubic can swing by picoseconds a few tens of nm past the measured window.

`correct_chirp` therefore evaluates the shift polynomial at `clamp(λ, λmin, λmax)` whenever the calibration carries `:lambda_range`: outside the measured window, the shift is held flat at the endpoint value. That is bounded and honest, but it is still not a measurement --- if a spectral region you care about lies outside the calibration's coverage, re-measure the OKE run on that spectrograph window. Calibrations from `detect_chirp` measure the same matrix they correct, so they don't set the key and are evaluated everywhere, as before.

## Fallback: detecting the chirp from the sample matrix

Without an OKE run you can bootstrap a calibration from the sample matrix itself: `detect_chirp` estimates ``t_0(\lambda)`` from the wavelength-resolved signal onsets. **This is only unbiased when both of the following hold:**

1. The data contains a genuinely instantaneous marker of pump--probe overlap at every wavelength (a sharp coherent artifact, or signal rise limited only by the IRF), and
2. early-time kinetics are slow compared to the IRF.

Whenever rise times vary with wavelength --- hot-carrier cooling, energy transfer, spectral diffusion --- onset detection absorbs those real dynamics into the time axis and manufactures false simultaneity. Treat it as a quick-look bootstrap, not a calibration.

### Step 1: Subtract the pre-pump background

TA is already a difference measurement, so the signal before the pump arrives should be zero. Any residual offset is systematic background (detector dark current, scattered pump light, etc.). Subtracting it removes a constant bias that otherwise degrades detection --- and for `:threshold` it is effectively required: on a matrix that still carries its background, every bin crosses the half-maximum at the first sample, and detection collapses to a flat, useless calibration (`detect_chirp` warns and reports ``R^2`` as `NaN`).

```julia
matrix_bg = subtract_background(matrix)
```

[`subtract_background`](@ref) averages the signal in the pre-pump region (auto-detected or specified via `t_range`) and subtracts that baseline from every time point.

### Step 2: Detect the chirp curve

Both detection methods bin the wavelength axis for noise reduction, smooth each bin with a Savitzky-Golay filter, then locate the signal onset.

```julia
cal = detect_chirp(matrix_bg)               # :xcorr (default)
cal_thr = detect_chirp(matrix_bg; method=:threshold)
```

The default `:xcorr` method cross-correlates onset gradients against the strongest-signal bin; since the absolute gradient produces a spike at the onset regardless of signal polarity, it handles mixed ESA/GSB regions. `:threshold` finds where each bin crosses half of its extremum --- simpler, and assumes a monotonic rise.

Key parameters to adjust:

| Parameter | Default | When to change |
|-----------|---------|----------------|
| `bin_width` | 8 | Increase for high-pixel-count CCDs (e.g. 2048 px). `16` or `32` reduces noise. |
| `min_signal` | 0.2 | Lower to include weak-signal spectral regions (noisier detection). |
| `order` | 3 | A 2nd-order polynomial suffices for modest chirp. Use 3 for broadband (> 200 nm) coverage. |
| `smooth_window` | 15 | Increase in noisy data. Must be odd. |
| `threshold` | 3.0 | MAD multiplier for outlier rejection. Lower = more aggressive rejection. |

### Step 3: Inspect the detected points

Before trusting a detected calibration, check that the chirp points are sensible:

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

## Applying the correction

[`correct_chirp`](@ref) shifts each wavelength column in time, whichever way the calibration was produced:

```julia
matrix_corrected = correct_chirp(matrix_bg, cal)
```

For each wavelength ``\lambda_j``, the corrected signal at time ``t_i`` is the original signal evaluated at ``t_i + t_{\text{shift}}(\lambda_j)``, where ``t_{\text{shift}}`` comes from the calibration polynomial (clamped to the calibration's `:lambda_range` when present). Cubic B-spline interpolation ensures sharp features (the coherent artifact, fast rise times) are preserved without the broadening that linear interpolation would introduce; on a non-uniform delay axis a gridded-linear fallback keeps the correction quantitatively correct.

At the edges of the time axis, samples whose shifted time falls outside the measured range come out as `NaN` --- no data was recorded there, and extrapolated values would masquerade as signal. Make sure your acquisition window extends well beyond time zero so the correction doesn't push any wavelength of interest off the edge.

### Validate

After correction, check three things:

1. **The TA surface.** Plot the corrected ``\Delta A(t, \lambda)`` heatmap. The coherent artifact should now appear as a vertical line at ``t = 0``, not a curved streak.

2. **Early-time spectra.** Extract spectra at several early delays (e.g. ``t = 0.1, 0.2, 0.5`` ps). Before correction these are distorted by chirp; after correction they should look like physically reasonable spectra without the characteristic dispersive artifact shape.

3. **Kinetics at the spectral extremes.** Compare kinetic traces at the blue and red edges. They should now show simultaneous rise times within the instrument response. If one edge is systematically shifted, the polynomial order may be too low or the detection failed at the edges.

## Saving and reusing calibrations

Store the chirp calibration for reproducibility and for applying to other datasets taken under the same optical conditions:

```julia
save_chirp("chirp_oke_2026-07-23.json", cal)

# Later, or in a different script:
cal = load_chirp("chirp_oke_2026-07-23.json")
matrix_corrected = correct_chirp(new_matrix, cal)
```

The JSON file stores the polynomial coefficients, the fitted points, and all calibration parameters (including the OKE provenance metadata), so the calibration is fully reproducible. The chirp is a property of the probe path, not the sample --- one OKE calibration serves every sample run measured with the same optics and spectrograph window.

## Other approaches

For completeness, other strategies you will meet in the literature:

- **Two-photon absorption (TPA) reference.** A thin semiconductor or dye solution where two-photon absorption provides a sharp, instantaneous response tracing the chirp. Less common than OKE, but useful when the setup geometry makes crossed-polarizer detection difficult.
- **Sellmeier / dispersion calculation.** Compute the expected GVD analytically from the known optical elements in the probe path. Useful as a sanity check or starting guess; in practice small alignment differences and unaccounted optics limit its accuracy.
- **Hardware pre-compensation.** Chirped mirrors, prism compressors, or grisms minimize chirp before acquisition. Often combined with a small software correction for the residual.
- **Global analysis with chirp as a free parameter.** Software like Glotaran folds the chirp polynomial into the kinetic fit itself, avoiding sequential error propagation --- at the cost of coupling the chirp estimate to the kinetic model. Most groups still correct as preprocessing, then let the global fit refine.

## Common pitfalls

**Using too narrow a time window.**
If the window doesn't capture the full peak (OKE) or signal rise (detection) at all wavelengths --- because of the chirp itself --- edge wavelengths get incorrect ``t_0`` values. If you set `t_range` manually, make it generous.

**Getting the sign convention wrong.**
Does ``t_0 > 0`` mean the blue arrives early or late? Getting this backwards flips the correction and doubles the chirp instead of removing it. In normal materials, the blue arrives later (positive GVD). After correction, check that the surface looks *less* curved, not more.

**Correcting twice.**
If the acquisition software already applies a partial chirp correction, applying a second correction on top will overcorrect. Check the raw data before starting.

**Applying a calibration outside its wavelength coverage.**
The clamp keeps the correction bounded outside `:lambda_range`, but flat is not measured. If your sample run extends meaningfully past the calibration's coverage, re-measure the OKE run on the matching spectrograph window.

**Overfitting the polynomial.**
The chirp curve should be smooth. Any high-frequency structure in ``t_0(\lambda)`` is noise. Use the minimum polynomial order that captures the curvature --- 2nd or 3rd order is almost always sufficient. MAD-based outlier rejection guards against this, but inspecting the residuals is still good practice.

**Ignoring the wavelength-dependent IRF.**
GVD doesn't just shift time zero --- it also temporally broadens the probe pulse at wavelengths far from the continuum center. Chirp correction fixes the shift but not the broadening. `calibrate_chirp` with `method=:gaussian` hands you IRF(``\lambda``) in `metadata[:irf_sigma]`; use it for a wavelength-dependent instrument response in your kinetic fits.

## Full example

```julia
using OpticalSpectroscopy

# --- Calibrate once per optical configuration, on the OKE run ---
oke = ...                                  # TimeResolvedMatrix from the OKE run
cal = calibrate_chirp(oke; order=3)
report(cal)
save_chirp("chirp_oke.json", cal)

# --- Apply to sample runs measured with the same optics ---
matrix_bg = subtract_background(matrix)
matrix_corrected = correct_chirp(matrix_bg, cal)

# --- No OKE run available? Fall back to in-data detection ---
cal_fallback = detect_chirp(matrix_bg; bin_width=16, order=3)
report(cal_fallback)
matrix_corrected = correct_chirp(matrix_bg, cal_fallback)
```

## API Reference

See the [Additional API Reference](@ref api-reference) page for full docstrings of all chirp functions.
