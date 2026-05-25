# Fit Overlapping Peaks

When two or more peaks overlap, fitting them individually gives wrong widths and areas. Fit them simultaneously instead.

## Problem

A spectral region contains multiple peaks that are not fully resolved.

## Solution

Construct an example with two overlapping Lorentzians:

```julia
using OpticalSpectroscopy
using CurveFitModels
using CairoMakie

x = collect(range(1900, 2200, length=800))
y = lorentzian([0.35, 2040.0, 18.0, 0.0], x) .+
    lorentzian([0.50, 2070.0, 22.0, 0.0], x) .+
    0.01 .+ 0.004 .* randn(length(x))
```

### Specify the Number of Peaks

Use `n_peaks` to tell `fit_peaks` how many peaks to fit:

```julia
result = fit_peaks(x, y, (1900, 2200); n_peaks=2)
```

The fitter detects peaks automatically for initial guesses and keeps the `n_peaks` most prominent ones. If auto-detection finds fewer peaks than requested, synthetic guesses fill the gaps.

### Inspect the Decomposition

Each peak is accessible by index:

```julia
result[1][:center].value    # first peak position
result[2][:center].value    # second peak position

predict_peak(result, 1)     # curve for peak 1
predict_peak(result, 2)     # curve for peak 2
predict_baseline(result)    # baseline only
```

### Visualize Individual Peaks

Overlay individual peak curves using `predict_peak` and `predict_baseline`:

```julia
using CairoMakie

fig = Figure(size=(700, 500))
ax = Axis(fig[1, 1], xlabel="Wavenumber (cm⁻¹)", ylabel="Intensity")
scatter!(ax, x, y, label="Data", markersize=5)
lines!(ax, x, predict(result, x), label="Composite fit", linewidth=2)
lines!(ax, x, predict_peak(result, 1, x), linestyle=:dash, label="Peak 1")
lines!(ax, x, predict_peak(result, 2, x), linestyle=:dash, label="Peak 2")
lines!(ax, x, predict_baseline(result, x), linestyle=:dot, label="Baseline")
axislegend(ax)
fig
```

### Control the Baseline

The polynomial baseline order defaults to 1 (linear). For a constant baseline or higher-order polynomial:

```julia
result = fit_peaks(x, y, (1900, 2200); n_peaks=2, baseline_order=0)  # constant
result = fit_peaks(x, y, (1900, 2200); n_peaks=2, baseline_order=2)  # quadratic
```

## Tips

- Start with `n_peaks=1` and increase only if residuals show systematic patterns
- If the fit doesn't converge well, try providing manual initial guesses (see [Provide Manual Initial Guesses](@ref))
- Use `report(result)` to see all peak parameters and fit quality at a glance

## See Also

- [`fit_peaks`](@ref) — full API reference
- [`predict_peak`](@ref), [`predict_baseline`](@ref) — evaluate individual fit components
- [Choose a Peak Model](@ref) — Lorentzian vs Gaussian vs Pseudo-Voigt
