# Tutorial: Raman-Style Peak Fitting

This tutorial walks through a Raman analysis workflow: detecting peaks in a spectrum, fitting a single peak, and decomposing overlapping peaks. By the end you will have fit parameters with uncertainties and a multi-peak decomposition figure.

We simulate a Raman spectrum inline so the whole workflow is reproducible. For real measurements, replace the simulated `(x, y)` vectors with data loaded from your preferred file loader.

## Prerequisites

```julia
using OpticalSpectroscopy
using CurveFitModels  # for `lorentzian` model
using CairoMakie      # or GLMakie for interactive exploration
using Random
```

## 1. Simulate a Raman Spectrum

Build a spectrum that resembles MoSe₂ — with the E¹₂g mode near 240 cm⁻¹, the A₁g mode near 170 cm⁻¹, and a weaker secondary feature around 280 cm⁻¹:

```julia
Random.seed!(2024)

x = collect(range(100.0, 400.0, length=2048))

# Each peak: [amplitude, center, fwhm, baseline_offset]
y_a1g   = lorentzian([650.0, 170.0, 6.0, 0.0], x)
y_e2g   = lorentzian([900.0, 240.0, 8.0, 0.0], x)
y_extra = lorentzian([180.0, 283.0, 9.0, 0.0], x)

# Slowly varying background + Gaussian noise
baseline_slope = 50.0 .+ 0.05 .* x
noise = 8.0 .* randn(length(x))

y = y_a1g .+ y_e2g .+ y_extra .+ baseline_slope .+ noise
```

Plot it:

```julia
fig = Figure(size=(700, 400))
ax = Axis(fig[1, 1],
    xlabel="Raman Shift (cm⁻¹)",
    ylabel="Intensity",
    title="Simulated Raman spectrum"
)
lines!(ax, x, y, color=:black)
save("figures/raman_raw.pdf", fig)
```

## 2. Detect Peaks

Use `find_peaks` to locate all prominent peaks. The `min_prominence` keyword controls detection sensitivity as a fraction of the data range:

```julia
peaks = find_peaks(x, y; min_prominence=0.05)
println("Detected $(length(peaks)) peaks")
```

Print a formatted summary:

```julia
println(peak_table(peaks))
```

Each `PeakInfo` has fields you can use to select peaks for fitting:

```julia
peak = peaks[1]
peak.position       # peak center in cm⁻¹
peak.intensity      # peak height
peak.prominence     # how much the peak stands out from surrounding baseline
peak.width          # full width at half prominence
peak.bounds         # (left, right) x-values at half prominence
peak.index          # index in the original data array
```

!!! tip "Baseline correction during detection"
    If your spectrum has a slowly varying background, enable baseline correction
    inside `find_peaks`:
    ```julia
    peaks = find_peaks(x, y; baseline=:arpls, min_prominence=0.05)
    ```
    Available methods include `:arpls`, `:snip`, `:rubberband`, `:imodpoly`,
    `:rolling_ball`. This corrects a working copy — the returned `PeakInfo`
    positions still refer to the original x-axis.

Plot the spectrum with peak markers:

```julia
fig = Figure(size=(700, 400))
ax = Axis(fig[1, 1],
    xlabel="Raman Shift (cm⁻¹)",
    ylabel="Intensity",
    title="Detected peaks"
)
lines!(ax, x, y, color=:black, label="Data")
scatter!(ax, [p.position for p in peaks], [p.intensity for p in peaks],
    color=:red, markersize=10, label="Detected")
axislegend(ax)
save("figures/raman_peaks.pdf", fig)
```

## 3. Fit a Single Peak

Pick the most prominent peak and define a fitting region around it. Using the peak's `bounds` with a width-sized margin gives a stable default:

```julia
peak = argmax(p -> p.prominence, peaks)
margin = peak.width
lo = peak.bounds[1] - margin
hi = peak.bounds[2] + margin

mask = lo .<= x .<= hi
result = fit_peaks(x[mask], y[mask])
```

`fit_peaks` detects peaks inside the region, fits a Lorentzian (default) plus a linear baseline, and returns a `MultiPeakFitResult`.

### Inspect the results

```julia
report(result)
```

Access individual parameters:

```julia
pk = result[1]                 # first (and only) peak in this fit
pk[:center].value              # peak position in cm⁻¹
pk[:center].err                # standard error
pk[:fwhm].value                # full width at half maximum
pk[:fwhm].ci                   # 95% confidence interval (lo, hi)
pk[:amplitude].value           # peak height
```

### Plot data + fit + residuals

Build a two-panel figure with the fit above and residuals below:

```julia
fig = Figure(size=(700, 600))

ax_fit = Axis(fig[1, 1],
    ylabel="Intensity",
    title="Single-peak fit"
)
scatter!(ax_fit, result._x, result._y, label="Data", markersize=5)
lines!(ax_fit, result._x, predict(result), color=:red, linewidth=2, label="Fit")
axislegend(ax_fit, position=:rt)
hidexdecorations!(ax_fit, grid=false)

ax_res = Axis(fig[2, 1],
    xlabel="Raman Shift (cm⁻¹)",
    ylabel="Residuals"
)
scatter!(ax_res, result._x, residuals(result), markersize=4)
hlines!(ax_res, 0, color=:black, linestyle=:dash)

rowsize!(fig.layout, 2, Relative(0.3))
save("figures/raman_fit.pdf", fig)
```

Check that:

- The fit line passes through the data
- Residuals scatter randomly around zero (no systematic pattern)

## 4. Multi-Peak Decomposition

When peaks overlap or sit close together, fit them simultaneously. Select the two most prominent peaks and define a region that covers both:

```julia
sorted_peaks = sort(peaks, by=p -> p.prominence, rev=true)
top2 = sort(sorted_peaks[1:2], by=p -> p.position)

lo = top2[1].bounds[1] - top2[1].width
hi = top2[2].bounds[2] + top2[2].width

mask = lo .<= x .<= hi
result2 = fit_peaks(x[mask], y[mask]; n_peaks=2)
report(result2)
```

The `n_peaks=2` keyword tells the fitter to model two peaks plus a shared baseline.

### Visualize the decomposition

Overlay the composite fit, the individual peak curves, and the baseline:

```julia
fig = Figure(size=(750, 500))
ax = Axis(fig[1, 1],
    xlabel="Raman Shift (cm⁻¹)",
    ylabel="Intensity",
    title="Two-peak decomposition"
)
scatter!(ax, result2._x, result2._y, label="Data", markersize=4, color=:black)
lines!(ax, result2._x, predict(result2), color=:red, linewidth=2, label="Composite fit")

# Individual peaks + baseline
baseline = predict_baseline(result2)
for i in 1:2
    pk_curve = predict_peak(result2, i) .+ baseline
    lines!(ax, result2._x, pk_curve, linestyle=:dash, linewidth=1.5,
           label="Peak $i")
end
lines!(ax, result2._x, baseline, color=:gray, linestyle=:dot, label="Baseline")

axislegend(ax, position=:rt)
save("figures/raman_multipeak.pdf", fig)
```

### Access individual peak parameters

```julia
result2[1][:center].value      # first peak center
result2[2][:center].value      # second peak center
predict_peak(result2, 1)       # curve for peak 1 only (without baseline)
predict_peak(result2, 2)       # curve for peak 2 only
predict_baseline(result2)      # baseline polynomial evaluated on _x
```

## 5. Trying a Different Model

The default peak shape is `lorentzian`. Swap to Gaussian or Pseudo-Voigt by passing `model`:

```julia
result_g = fit_peaks(x[mask], y[mask]; n_peaks=2, model=gaussian)
result_v = fit_peaks(x[mask], y[mask]; n_peaks=2, model=pseudo_voigt)
```

Compare R² values:

```julia
println("Lorentzian R² = ", round(result2.r_squared, digits=5))
println("Gaussian R²   = ", round(result_g.r_squared, digits=5))
println("Voigt R²      = ", round(result_v.r_squared, digits=5))
```

R² alone is not enough when models have different parameter counts — pseudo-Voigt adds a mixing parameter per peak. For that comparison use AIC or BIC (see the FTIR tutorial for the formulas).

## 6. When Detection Misses a Peak

`find_peaks` is local-maximum based and can miss shoulders, heavily overlapping peaks, or peaks buried under a sloped baseline. Two strategies help:

**Lower `min_prominence`**:
```julia
peaks = find_peaks(x, y; min_prominence=0.01)
```
This exposes weaker features but also surfaces noise — inspect the result visually.

**Tell `fit_peaks` the number of peaks directly**:
```julia
result = fit_peaks(x[mask], y[mask]; n_peaks=3)
```
If detection returns fewer peaks than `n_peaks`, the fitter synthesizes initial guesses by evenly spacing centers across the region.

**Or supply explicit initial parameters**:
```julia
p0 = [
    650.0, 170.0, 6.0,   # peak 1: amplitude, center, fwhm
    900.0, 240.0, 8.0,   # peak 2
    180.0, 283.0, 9.0,   # peak 3
    0.0, 0.0             # linear baseline (slope, intercept order depends on baseline_order)
]
result = fit_peaks(x[mask], y[mask]; p0=p0, n_peaks=3)
```

## Summary

| Step | Function | What it does |
|------|----------|-------------|
| Detect | `find_peaks(x, y; min_prominence)` | Find peaks, return `PeakInfo` vector |
| Inspect | `peak_table(peaks)` | Print peak summary |
| Fit | `fit_peaks(x, y; n_peaks, model)` | Fit peak(s) + polynomial baseline |
| Report | `report(result)` | Print formatted parameters with uncertainties |
| Evaluate | `predict(result)`, `predict_peak(r, i)`, `predict_baseline(r)` | Model curves |
| Residuals | `residuals(result)` | Data minus fit |

## Next Steps

- [FTIR Peak Fitting](@ref "Tutorial: FTIR-Style Peak Fitting") — model comparison and residual interpretation
- Experiment with `min_prominence`, `min_width`, `max_width` to tune detection
- Try baseline methods `:arpls`, `:snip`, `:rubberband` on spectra with strong backgrounds
- Use `pseudo_voigt` for lines that are neither purely Lorentzian nor Gaussian
