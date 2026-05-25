# Tutorial: FTIR-Style Peak Fitting

This tutorial walks through an FTIR-style analysis workflow: fitting a vibrational band with a Lorentzian model, comparing Lorentzian vs Gaussian line shapes, and interpreting residuals. The focus is on model selection, which is one of the most common decisions in vibrational spectroscopy.

We simulate an FTIR absorbance spectrum inline so the workflow is fully reproducible. When you apply this to real data, replace the simulated `(x, y)` vectors with arrays loaded from your preferred file loader.

## Prerequisites

```julia
using OpticalSpectroscopy
using CurveFitModels  # for `lorentzian`, `gaussian` models
using CairoMakie
using Random
```

## 1. Simulate an FTIR Spectrum

Generate an absorbance spectrum with a single Lorentzian band near 2060 cm⁻¹ (the CN stretch of SCN⁻ is a good reference example):

```julia
Random.seed!(42)

x = collect(range(1900, 2200, length=600))

# Lorentzian band: [amplitude, center, fwhm, baseline_offset]
y_clean = lorentzian([0.45, 2062.0, 24.0, 0.01], x)

# Add realistic noise
y = y_clean .+ 0.002 .* randn(length(x))
```

Plot the raw data to orient yourself:

```julia
fig = Figure(size=(700, 400))
ax = Axis(fig[1, 1],
    xlabel="Wavenumber (cm⁻¹)",
    ylabel="Absorbance",
    title="Simulated FTIR spectrum",
    xreversed=true
)
lines!(ax, x, y, color=:black)
save("figures/ftir_raw.pdf", fig)
```

FTIR is conventionally plotted with wavenumber decreasing to the right; use `xreversed=true` on the axis.

## 2. Fit the CN Stretch

Define a fitting region that brackets the band and call `fit_peaks`:

```julia
result = fit_peaks(x, y; min_prominence=0.05)    # full-range
```

Or, more commonly, restrict to a region around the peak of interest:

```julia
region = (1950.0, 2150.0)
mask = region[1] .<= x .<= region[2]
result = fit_peaks(x[mask], y[mask])
report(result)
```

The report shows the center, FWHM, amplitude, and baseline parameters with uncertainties.

Visualize the fit with a residuals panel:

```julia
fig = Figure(size=(700, 600))

ax_fit = Axis(fig[1, 1],
    ylabel="Absorbance",
    title="Lorentzian fit",
    xreversed=true
)
scatter!(ax_fit, result._x, result._y, label="Data", markersize=5)
lines!(ax_fit, result._x, predict(result), color=:red, linewidth=2, label="Fit")
axislegend(ax_fit, position=:lt)
hidexdecorations!(ax_fit, grid=false)

ax_res = Axis(fig[2, 1],
    xlabel="Wavenumber (cm⁻¹)",
    ylabel="Residuals",
    xreversed=true
)
scatter!(ax_res, result._x, residuals(result), markersize=4)
hlines!(ax_res, 0, color=:black, linestyle=:dash)

rowsize!(fig.layout, 2, Relative(0.3))
save("figures/ftir_fit.pdf", fig)
```

Check the residuals for systematic patterns. Random scatter around zero indicates a good fit. An S-shaped residual means the peak shape is wrong; large deviations in the wings suggest the baseline order is too low or the region is too narrow.

## 3. Accessing Fit Parameters

Each peak in the result exposes named parameters with values, errors, and confidence intervals:

```julia
pk = result[1]                 # first peak
pk[:center].value              # peak position in cm⁻¹
pk[:center].err                # standard error
pk[:center].ci                 # 95% confidence interval (lo, hi)
pk[:fwhm].value                # full width at half maximum
pk[:amplitude].value           # peak height
```

Global fit statistics:

```julia
result.r_squared               # coefficient of determination
result.rss                     # residual sum of squares
result.mse                     # mean squared error
```

## 4. Model Comparison: Lorentzian vs Gaussian

Different physical broadening mechanisms produce different line shapes:

- **Lorentzian** — homogeneous broadening (collisions, lifetime). Typical for solution-phase vibrational bands.
- **Gaussian** — inhomogeneous broadening (environment heterogeneity, Doppler). Typical for solid-state or highly disordered systems.
- **Pseudo-Voigt** — mixture of both. Useful when neither pure shape fits well.

Fit the same data with both models:

```julia
result_lor = fit_peaks(x[mask], y[mask])                    # Lorentzian (default)
result_gau = fit_peaks(x[mask], y[mask]; model=gaussian)    # Gaussian
```

### Compare fit quality

R² and RSS give a first look:

```julia
println("Lorentzian: R² = ", round(result_lor.r_squared, digits=5),
        "  RSS = ", round(result_lor.rss, digits=6))
println("Gaussian:   R² = ", round(result_gau.r_squared, digits=5),
        "  RSS = ", round(result_gau.rss, digits=6))
```

For simulated Lorentzian data, you should see the Lorentzian fit come out slightly ahead on both metrics. But raw R² is not enough when comparing models with different numbers of parameters.

### AIC / BIC

The Akaike and Bayesian Information Criteria penalize model complexity:

- **AIC = 2k + n·ln(RSS/n)**
- **BIC = k·ln(n) + n·ln(RSS/n)**

where `k` is the number of fit parameters and `n` is the number of data points. Lower is better. BIC penalizes extra parameters more heavily than AIC.

```julia
function aic_bic(res)
    n = res.npoints
    k = length(res._coef)
    aic = 2k + n * log(res.rss / n)
    bic = k * log(n) + n * log(res.rss / n)
    return (aic=aic, bic=bic)
end

for (name, r) in [("Lorentzian", result_lor), ("Gaussian", result_gau)]
    s = aic_bic(r)
    println(rpad(name, 12),
            "AIC = ", round(s.aic, digits=2),
            "   BIC = ", round(s.bic, digits=2))
end
```

For solution-phase FTIR, Lorentzian line shapes are often a better description because the dominant broadening mechanism is collisional (homogeneous). For solid samples or dry films, Gaussian or pseudo-Voigt may fit better.

### Print both reports

```julia
println("=== Lorentzian ===")
report(result_lor)

println("=== Gaussian ===")
report(result_gau)
```

## 5. Interpreting Residuals

Residuals are the diagnostic signal that tells you whether the model is adequate. Read them for pattern:

- **Random scatter around zero** — fit is good; residuals are driven by measurement noise.
- **S-shape through the peak** — line shape is wrong (e.g., Gaussian for a Lorentzian band).
- **Parabolic curvature** — baseline order is too low. Try `baseline_order=2`.
- **Sharp spike in residuals** — an unmodeled narrow feature (cosmic ray, impurity line, or second peak).

Plot both residuals side by side:

```julia
fig = Figure(size=(900, 400))

ax1 = Axis(fig[1, 1],
    xlabel="Wavenumber (cm⁻¹)", ylabel="Residuals",
    title="Lorentzian residuals", xreversed=true
)
scatter!(ax1, result_lor._x, residuals(result_lor), markersize=4)
hlines!(ax1, 0, color=:black, linestyle=:dash)

ax2 = Axis(fig[1, 2],
    xlabel="Wavenumber (cm⁻¹)", ylabel="Residuals",
    title="Gaussian residuals", xreversed=true
)
scatter!(ax2, result_gau._x, residuals(result_gau), markersize=4)
hlines!(ax2, 0, color=:black, linestyle=:dash)

linkyaxes!(ax1, ax2)
save("figures/ftir_residuals_compare.pdf", fig)
```

When the data generating process is Lorentzian (as in this tutorial), the Gaussian residuals should show a characteristic S-shape through the band center — a clear signature of line-shape mismatch.

## 6. Subtracting a Reference Spectrum

Vibrational spectra of solutions contain both solute and solvent absorption. If you have a pure-solvent reference measured under the same conditions, `subtract_spectrum` removes the reference contribution from the sample (vectors or spectrum objects both work):

```julia
# Simulate a solvent reference: a broad band near 2000 cm⁻¹
y_ref = lorentzian([0.08, 2000.0, 80.0, 0.0], x) .+ 0.001 .* randn(length(x))

# Subtract
corrected = subtract_spectrum((x=x, y=y), (x=x, y=y_ref))
y_corr = corrected.y
```

Pass `scale` when the reference was measured at a different pathlength or concentration:

```julia
corrected = subtract_spectrum((x=x, y=y), (x=x, y=y_ref); scale=0.95)
```

Now refit the corrected spectrum and compare the band parameters before and after correction — the center should be unchanged if the reference is well-matched, but the amplitude and baseline will shift.

```julia
result_sub = fit_peaks(x[mask], y_corr[mask])

println("Before subtraction:  center = ",
        round(result_lor[1][:center].value, digits=1), " cm⁻¹, ",
        "fwhm = ", round(result_lor[1][:fwhm].value, digits=1), " cm⁻¹")
println("After subtraction:   center = ",
        round(result_sub[1][:center].value, digits=1), " cm⁻¹, ",
        "fwhm = ", round(result_sub[1][:fwhm].value, digits=1), " cm⁻¹")
```

## 7. Publication-Style Figure

Combine the fit and residuals into a single multi-panel figure suitable for papers:

```julia
fig = Figure(size=(900, 400))

ax_a = Axis(fig[1, 1],
    xlabel="Wavenumber (cm⁻¹)",
    ylabel="Absorbance",
    title="(a) Fit",
    xreversed=true
)
scatter!(ax_a, result_lor._x, result_lor._y, label="Data", markersize=5)
lines!(ax_a, result_lor._x, predict(result_lor), color=:red, linewidth=2, label="Fit")
axislegend(ax_a, position=:lt)

ax_b = Axis(fig[1, 2],
    xlabel="Wavenumber (cm⁻¹)",
    ylabel="Residuals",
    title="(b) Residuals",
    xreversed=true
)
scatter!(ax_b, result_lor._x, residuals(result_lor), markersize=4)
hlines!(ax_b, 0, color=:black, linestyle=:dash)

save("figures/ftir_publication.pdf", fig)
```

## Summary

| Step | Function | What it does |
|------|----------|-------------|
| Detect | `find_peaks(x, y)` | Locate peaks and return `PeakInfo` |
| Fit | `fit_peaks(x, y; model, n_peaks)` | Fit one or more peaks + polynomial baseline |
| Report | `report(result)` | Print formatted parameters with uncertainties |
| Evaluate | `predict(result)`, `residuals(result)` | Model curve and residuals |
| Compare | `fit_peaks(...; model=gaussian)` | Try different line shapes |
| Subtract | `subtract_spectrum(sample, reference; scale)` | Remove reference background |

## Next Steps

- [Raman Peak Fitting](@ref "Tutorial: Raman-Style Peak Fitting") — detecting peaks and decomposing overlapping bands
- Explore `pseudo_voigt` when neither Lorentzian nor Gaussian is adequate
- Use `baseline_order=2` or higher for curved backgrounds
- For broader background removal, see `arpls_baseline`, `snip_baseline`, `rubberband_baseline`
