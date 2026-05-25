# Tune Peak Detection Sensitivity

`find_peaks` has several parameters that control which peaks are reported. This guide shows how to adjust them.

## Problem

The default settings detect too many peaks (noise spikes) or too few (real peaks missed).

## Solution

Construct some example data with two Lorentzian peaks and noise:

```julia
using OpticalSpectroscopy
using CurveFitModels

x = collect(range(2000, 2150, length=600))
y = lorentzian([0.45, 2062.0, 24.0, 0.0], x) .+
    lorentzian([0.30, 2110.0, 18.0, 0.0], x) .+
    0.005 .* randn(length(x))
```

### Filter by Prominence

Prominence measures how much a peak stands out from its surroundings. The `min_prominence` parameter is a fraction of the data range (0 to 1):

```julia
# Default: keep peaks with prominence > 5% of data range
peaks = find_peaks(x, y)

# Stricter: only prominent peaks
peaks = find_peaks(x, y, min_prominence=0.1)

# More sensitive: catch small peaks too
peaks = find_peaks(x, y, min_prominence=0.02)
```

### Filter by Width

Remove narrow spikes or overly broad features:

```julia
# Minimum width of 5 cm⁻¹
peaks = find_peaks(x, y, min_width=5.0)

# Both minimum and maximum width
peaks = find_peaks(x, y, min_width=5.0, max_width=100.0)
```

### Adjust the Detection Window

The `window` parameter controls how many neighboring points a peak must exceed to count as a local maximum:

```julia
# Default: each peak must be higher than its immediate neighbors
peaks = find_peaks(x, y, window=1)

# Wider window: requires peak to dominate over ±3 points
peaks = find_peaks(x, y, window=3)
```

Larger windows suppress noise but may merge closely spaced peaks.

### Apply Baseline Correction

Fluorescence background can create false peaks. Correct the baseline before detection:

```julia
peaks = find_peaks(x, y, baseline=:arpls)

# With custom baseline parameters
peaks = find_peaks(x, y, baseline=:snip, baseline_kw=(iterations=50,))
```

## Iterative Workflow

Use `find_peaks` interactively to tune parameters before fitting:

```julia
# Start loose and tighten
peaks = find_peaks(x, y, baseline=:arpls)
println("$(length(peaks)) peaks found")
println(peak_table(peaks))

# Too many? Increase prominence
peaks = find_peaks(x, y, baseline=:arpls, min_prominence=0.1)
println("$(length(peaks)) peaks found")
```

## See Also

- [`find_peaks`](@ref) — full API reference
