# Compare Fits Across Samples

Extract and compare peak parameters from multiple samples or conditions.

## Problem

You have several spectra (different concentrations, temperatures, etc.) and want to track how a peak parameter changes.

## Solution

### Fit Each Sample

Assume you have a collection of `(x, y)` pairs — one per sample — keyed by a label:

```julia
using OpticalSpectroscopy
using CairoMakie

# spectra::Dict{String, Tuple{Vector{Float64}, Vector{Float64}}}
# e.g. spectra["1.0M"] = (x, y)

labels = ["0.5M", "1.0M", "2.0M"]
results = Dict{String, MultiPeakFitResult}()

for label in labels
    x, y = spectra[label]
    results[label] = fit_peaks(x, y, (1950, 2150))
end
```

### Extract Parameters

Build vectors of the parameter you want to compare:

```julia
centers = [results[c][1][:center].value for c in labels]
center_errs = [results[c][1][:center].err for c in labels]
fwhms = [results[c][1][:fwhm].value for c in labels]
fwhm_errs = [results[c][1][:fwhm].err for c in labels]
```

### Print a Comparison Table

```julia
println(rpad("Label", 8), rpad("Center", 12), rpad("FWHM", 12), "R²")
for c in labels
    r = results[c]
    pk = r[1]
    println(
        rpad(c, 8),
        rpad(string(round(pk[:center].value, digits=1)), 12),
        rpad(string(round(pk[:fwhm].value, digits=1), " ± ",
                    round(pk[:fwhm].err, digits=1)), 12),
        round(r.r_squared, digits=5)
    )
end
```

### Plot the Trend

```julia
conc_values = [0.5, 1.0, 2.0]

fig = Figure(size=(500, 400))
ax = Axis(fig[1, 1],
    xlabel="Concentration (M)",
    ylabel="Peak center (cm⁻¹)"
)
errorbars!(ax, conc_values, centers, center_errs)
scatter!(ax, conc_values, centers)
fig
```

## Tips

- Use `report(result)` for each sample to quickly inspect individual fits
- Store results in a `Dict` or `Vector` for easy iteration

## See Also

- [`fit_peaks`](@ref) — full API reference
- [Fitting Statistics](@ref "Fitting Statistics Reference") — interpreting uncertainties
