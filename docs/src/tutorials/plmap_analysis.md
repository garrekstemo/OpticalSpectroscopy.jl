# Tutorial: PL / Raman Map Analysis

This tutorial walks through the PLMap analysis workflow: extracting spectra at spatial positions, choosing a PL pixel range, subtracting a background spectrum, detecting and removing cosmic rays, and mapping peak intensity and peak center across the spatial grid.

`PLMap` is a 2D spatial grid where each point stores a full CCD spectrum — a data cube with shape `(nx, ny, n_pixel)`. It is the general container for PL and Raman raster scans.

Assuming you have a `PLMap` (load one from your preferred file loader, or construct one directly; see [Constructing a PLMap](#constructing-a-plmap) below).

## Prerequisites

```julia
using OpticalSpectroscopy
using CairoMakie
using Statistics
using Random
```

## Constructing a PLMap

If you are not using a lab-specific loader, you can build a `PLMap` directly from arrays. The constructor takes:

- `intensity::Matrix{Float64}` — integrated PL at each position `(nx, ny)`
- `spectra::Array{Float64,3}` — full CCD data cube `(nx, ny, n_pixel)`
- `x::Vector{Float64}` — spatial x positions (μm)
- `y::Vector{Float64}` — spatial y positions (μm)
- `pixel::Vector{Float64}` — pixel indices (or calibrated wavelength)
- `metadata::Dict{String,Any}` — arbitrary metadata; store `"pixel_range"` here so that downstream operations (`subtract_background`, `peak_centers`, `fit_map`) know which pixel window to use

For a minimal end-to-end test, simulate a 51×51 scan with a Gaussian PL patch in the centre:

```julia
Random.seed!(11)

nx, ny, np = 51, 51, 2000
x = collect(range(-54.0, 54.0, length=nx))
y = collect(range(-54.0, 54.0, length=ny))
pixel = collect(1.0:np)

# Background spectrum: strong Rayleigh tail + flat baseline
bg = 1200.0 .* exp.(-(pixel .- 50.0).^2 ./ (2 * 30.0^2)) .+ 50.0

# Gaussian "flake" envelope in spatial coordinates
flake = [exp(-((xi^2 + yi^2) / (2 * 18.0^2))) for xi in x, yi in y]

# Build the data cube: bg (everywhere) + PL band near pixel 1030 (scaled by flake)
spectra = Array{Float64,3}(undef, nx, ny, np)
pl_profile = 300.0 .* exp.(-(pixel .- 1030.0).^2 ./ (2 * 22.0^2))
for ix in 1:nx, iy in 1:ny
    spectra[ix, iy, :] = bg .+ flake[ix, iy] .* pl_profile .+ 5.0 .* randn(np)
end

# Sprinkle in a few cosmic ray spikes
for _ in 1:12
    ix = rand(1:nx); iy = rand(1:ny); ip = rand(200:1800)
    spectra[ix, iy, ip] += 4000.0
end

# Initial integrated intensity (full range)
intensity = dropdims(sum(spectra; dims=3); dims=3)

metadata = Dict{String,Any}(
    "source_file" => "simulated.lvm",
    "step_size"   => 2.16,
)

m_raw = PLMap(intensity, spectra, x, y, pixel, metadata)
println(m_raw)
```

## 1. Inspect Spectra to Find the PL Emission

Before making an intensity map, look at individual spectra. This tells you which pixel range contains the PL emission.

A typical CCD spectrum from a Raman/PL measurement has:

- **Low pixels**: strong laser/Rayleigh scatter (dominates total counts)
- **Middle pixels**: Raman bands and/or PL emission
- **High pixels**: baseline noise

If you integrate over ALL pixels, the laser scatter dominates and the PL signal is buried. You need to identify the PL peak pixel range.

```julia
# Extract the spectrum at a single spatial position
spec_center = extract_spectrum(m_raw; x=0.0, y=0.0)
spec_edge   = extract_spectrum(m_raw; x=40.0, y=40.0)

fig = Figure(size=(800, 400))

ax1 = Axis(fig[1, 1],
    xlabel="Pixel", ylabel="CCD counts",
    title="Full CCD spectra"
)
lines!(ax1, spec_center.pixel, spec_center.signal, label="Center (on flake)")
lines!(ax1, spec_edge.pixel,   spec_edge.signal,   label="Edge (off flake)")
axislegend(ax1, position=:rt)

ax2 = Axis(fig[1, 2],
    xlabel="Pixel", ylabel="CCD counts",
    title="Zoomed to PL region"
)
lines!(ax2, spec_center.pixel, spec_center.signal, label="Center")
lines!(ax2, spec_edge.pixel,   spec_edge.signal,   label="Edge")
xlims!(ax2, 900, 1150)
axislegend(ax2, position=:rt)

save("figures/plmap_spectra.png", fig)
```

`extract_spectrum(m; x, y)` snaps to the nearest grid point. The returned named tuple includes `.signal`, `.pixel`, and the actual position `.x, .y`. You can also index directly by grid index:

```julia
spec = extract_spectrum(m_raw, 26, 26)      # center of a 51×51 grid
spec.signal                                  # CCD counts vector
```

From the zoomed plot, the PL emission sits around pixels 950–1100. This is the range we integrate over.

## 2. Three Processing Approaches

We compare three increasingly refined ways to build a PL intensity map. Each removes a different source of background.

### Approach 1: Raw (all pixels)

Sum all CCD pixels per spectrum. The laser/Rayleigh scatter at low pixels dominates as a large DC offset, compressing PL contrast. This is `m_raw` as constructed above.

### Approach 2: PL pixel range only

Integrate only over the PL emission pixels (950–1100). This removes the laser scatter and dramatically improves contrast. Store the pixel range in metadata so later steps pick it up automatically:

```julia
pr = (950, 1100)
intensity_pr = dropdims(sum(m_raw.spectra[:, :, pr[1]:pr[2]]; dims=3); dims=3)

md_pr = copy(m_raw.metadata)
md_pr["pixel_range"] = pr

m_pr = PLMap(intensity_pr, m_raw.spectra, m_raw.x, m_raw.y, m_raw.pixel, md_pr)
```

You can always recompute the integrated intensity with `integrated_intensity`:

```julia
intensity_pr = integrated_intensity(m_pr)                     # uses metadata pixel_range
intensity_alt = integrated_intensity(m_pr; pixel_range=(900, 1200))
```

### Approach 3: Pixel range + background subtraction

Subtract a reference spectrum (averaged from off-flake positions) from every grid point, then integrate. This zeros out the per-pixel CCD baseline so off-flake regions have near-zero intensity.

Auto mode averages the bottom corners of the map:

```julia
m_bg = subtract_background(m_pr)
```

Or supply explicit off-flake positions:

```julia
m_bg = subtract_background(m_pr; positions=[(-40.0, -40.0), (40.0, -40.0)])
```

`subtract_background` preserves the original `pixel_range` metadata and recomputes the intensity field using that same window.

### Normalize and compare

```julia
m_raw_norm = normalize_intensity(m_raw)
m_pr_norm  = normalize_intensity(m_pr)
m_bg_norm  = normalize_m_bg.intensity
```

After min-max normalization to `[0, 1]`, approaches 2 and 3 look visually identical — normalization already removes the DC offset. Background subtraction matters when you need absolute PL intensities (comparing samples, correlating with excitation power), not for contrast alone.

## 3. SNR Analysis

Quantify the improvement from each processing step. SNR = (mean_flake − mean_background) / std_background.

```julia
on_x, on_y   = 21:31, 21:31   # center of map (on flake)
off_x, off_y = 1:10,  1:10    # bottom-left corner (off flake)

for (name, map) in [("Raw (all pixels)",       m_raw),
                    ("PL pixel range",         m_pr),
                    ("Pixel range + bg sub",   m_bg)]
    sig    = mean(map.intensity[on_x, on_y])
    bg     = mean(map.intensity[off_x, off_y])
    bg_std = std(map.intensity[off_x, off_y])
    snr    = (sig - bg) / bg_std
    println(rpad(name, 28), "SNR = ", round(snr, digits=1))
end
```

The pixel-range step gives a large contrast improvement over raw. Background subtraction gives *identical* SNR because it only shifts the mean — subtracting a constant from every point doesn't change variance. Its value is in producing physically meaningful absolute intensities.

## 4. Publication-Quality PL Map

Make the spatial heatmap with Makie directly:

```julia
fig = Figure(size=(500, 450))
ax = Axis(fig[1, 1],
    xlabel="X (μm)", ylabel="Y (μm)",
    title="PL Intensity", aspect=DataAspect()
)
hm = heatmap!(ax, m_pr_norm.x, m_pr_norm.y, m_pr_norm.intensity; colormap=:hot)
Colorbar(fig[1, 2], hm, label="Normalized PL")
save("figures/pl_map.pdf", fig)
```

### Multi-panel comparison

```julia
fig = Figure(size=(1400, 450))
ax1 = Axis(fig[1, 1], xlabel="X (μm)", ylabel="Y (μm)",
           title="Raw (all pixels)", aspect=DataAspect())
ax2 = Axis(fig[1, 2], xlabel="X (μm)", ylabel="Y (μm)",
           title="PL pixel range only", aspect=DataAspect())
ax3 = Axis(fig[1, 3], xlabel="X (μm)", ylabel="Y (μm)",
           title="Pixel range + background sub", aspect=DataAspect())

heatmap!(ax1, m_raw_norm.x, m_raw_norm.y, m_raw_norm.intensity; colormap=:hot)
heatmap!(ax2, m_pr_norm.x,  m_pr_norm.y,  m_pr_norm.intensity;  colormap=:hot)
hm3 = heatmap!(ax3, m_bg_norm.x, m_bg_norm.y, m_bg_norm.intensity; colormap=:hot)
Colorbar(fig[1, 4], hm3, label="Normalized PL")

save("figures/pl_comparison.pdf", fig)
```

## 5. Cosmic Ray Detection and Removal

CCD detectors occasionally register high-energy particle hits as sharp, narrow spikes in individual spectra. These "cosmic rays" affect a single spatial pixel at a random spectral channel and don't correlate with spatial neighbors. They distort integrated intensity, shift peak center calculations, and can cause fitting failures.

OpticalSpectroscopy uses modified z-scores on first differences (Whitaker–Hayes method) for detection. For `PLMap` data, spatial validation prevents real spectral features (shared across neighbors) from being falsely flagged.

### Detect and inspect

```julia
cr = detect_cosmic_rays(m_bg; threshold=5.0)
println("Found $(cr.count) cosmic ray spikes in $(cr.affected_spectra) spectra")
```

The result is a `CosmicRayMapResult` with:

- `mask` — 3D `BitArray` `(nx, ny, n_pixel)`, `true` at spike locations
- `count` — total number of flagged voxels
- `affected_spectra` — number of spectra with at least one spike
- `channel_counts` — spike count per spectral channel

### Visualize spike locations

```julia
cr_counts = dropdims(sum(cr.mask; dims=3); dims=3)

fig = Figure(size=(500, 450))
ax = Axis(fig[1, 1], xlabel="X (μm)", ylabel="Y (μm)",
    title="Cosmic Ray Count per Pixel", aspect=DataAspect())
hm = heatmap!(ax, m_bg.x, m_bg.y, cr_counts; colormap=:inferno)
Colorbar(fig[1, 2], hm, label="Spike Count")
save("figures/cosmic_ray_map.png", fig)
```

### Remove and compare

```julia
m_clean = remove_cosmic_rays(m_bg, cr)
```

Flagged voxels are replaced with the median value from non-flagged spatial neighbors. Edge pixels with no valid neighbors fall back to spectral interpolation. The cleaned `PLMap` has a recomputed intensity field.

```julia
fig = Figure(size=(1000, 400))
ax1 = Axis(fig[1, 1], xlabel="X (μm)", ylabel="Y (μm)",
    title="Before", aspect=DataAspect())
heatmap!(ax1, m_bg.x, m_bg.y, m_bg.intensity; colormap=:hot)
ax2 = Axis(fig[1, 2], xlabel="X (μm)", ylabel="Y (μm)",
    title="After Cosmic Ray Removal", aspect=DataAspect())
hm2 = heatmap!(ax2, m_clean.x, m_clean.y, m_clean.intensity; colormap=:hot)
Colorbar(fig[1, 3], hm2, label="PL Intensity")
save("figures/cosmic_ray_comparison.png", fig)
```

### Single-spectrum removal

The same functions work on any 1D signal:

```julia
spec = extract_spectrum(m_bg, 26, 26)
cr_1d = detect_cosmic_rays(spec.signal; threshold=5.0)
cleaned = remove_cosmic_rays(spec.signal, cr_1d)
```

### Threshold tuning

Lower values detect more spikes but may flag real features; higher values are conservative. Default 5.0 works well for typical CCD data.

```julia
for thresh in [3.0, 4.0, 5.0, 6.0, 7.0]
    cr_t = detect_cosmic_rays(m_bg; threshold=thresh)
    println("threshold=$thresh → $(cr_t.count) spikes in $(cr_t.affected_spectra) spectra")
end
```

### Processing order

When combining corrections:

1. Background subtract (improves spike contrast)
2. Detect cosmic rays
3. Remove cosmic rays
4. Normalize last

```julia
m = subtract_background(m_pr)
cr = detect_cosmic_rays(m; threshold=5.0)
m = remove_cosmic_rays(m, cr)
m = normalize_intensity(m)
```

## 6. Peak Center Map

The PL intensity map tells you *where* emission is strong; a peak center map tells you *what wavelength* (pixel position) the emission peaks at. Spatial variation in the peak center reveals strain, composition gradients, or charge transfer.

`peak_centers` computes the intensity-weighted centroid pixel at each grid point. Points with PL intensity below a threshold fraction of the map maximum are masked as `NaN`:

```julia
centers = peak_centers(m_clean)
```

The masking uses the PLMap's intensity field, so the peak center map automatically matches the flake shape in the intensity map. Adjust threshold for weak flakes:

```julia
centers_strict = peak_centers(m_clean; threshold=0.10)   # 10% — brightest regions only
centers_all    = peak_centers(m_clean; threshold=0.0)    # no masking (noisy off-flake)
```

Plot intensity and peak centre side by side:

```julia
fig = Figure(size=(1000, 450))

m_norm = normalize_m_clean.intensity

ax1 = Axis(fig[1, 1], xlabel="X (μm)", ylabel="Y (μm)",
           title="PL Intensity", aspect=DataAspect())
hm1 = heatmap!(ax1, m_norm.x, m_norm.y, m_norm.intensity; colormap=:hot)
Colorbar(fig[1, 2], hm1, label="Normalized PL")

ax2 = Axis(fig[1, 3], xlabel="X (μm)", ylabel="Y (μm)",
           title="Peak Center", aspect=DataAspect())
hm2 = heatmap!(ax2, m_clean.x, m_clean.y, centers;
               colormap=:viridis, nan_color=:transparent)
Colorbar(fig[1, 4], hm2, label="Peak Position (pixel)")

save("figures/peak_centers.png", fig)
```

`:viridis` with `nan_color=:transparent` is the standard convention — viridis gives good perceptual uniformity for small variations, and transparent NaN values let the background show through for off-flake regions.

## 7. Batch Peak Fitting with `fit_map`

For every on-flake position, fit a model peak in the spectrum and collect parameter maps:

```julia
result = fit_map(m_clean;
    model = lorentzian,
    n_peaks = 1,
    region = (950, 1100),
    threshold = 0.05
)

# Summary arrays for heatmaps
heatmap(m_clean.x, m_clean.y, result.centers)
heatmap(m_clean.x, m_clean.y, result.fwhms)

# Per-pixel result
result[26, 26]   # a MultiPeakFitResult (or nothing, if excluded)
```

`fit_map` uses **reference pixel seeding**: the brightest included pixel is fit first with auto-detection, and its converged parameters seed all remaining pixels. This is crucial for broad or noisy peaks where per-pixel auto-detection would be unreliable. Fitting is multi-threaded when Julia is started with `julia --threads=auto`.

## Summary

| Step | Function | What it does |
|------|----------|-------------|
| Extract | `extract_spectrum(m; x, y)` or `extract_spectrum(m, ix, iy)` | Pull CCD spectrum at a position |
| Integrate | `integrated_intensity(m; pixel_range)` | Sum spectra over a pixel window |
| Background | `subtract_background(m; positions)` | Remove per-pixel CCD baseline |
| Normalize | `normalize_intensity(m)` | Scale intensity to `[0, 1]` |
| Cosmic rays | `detect_cosmic_rays(m; threshold)` | Find cosmic ray spikes |
| Remove CRs | `remove_cosmic_rays(m, cr)` | Replace spikes with neighbor median |
| Peak centers | `peak_centers(m; threshold)` | Centroid pixel at each grid point |
| Mask | `intensity_mask(m; threshold, exclude)` | Boolean mask over the grid |
| Fit map | `fit_map(m; model, region)` | Batch per-pixel peak fitting |

## Next Steps

- Experiment with different `pixel_range` values to isolate Raman bands vs PL
- Use `extract_spectrum` to compare spectra at high- and low-intensity positions
- Try different colormaps (`:hot`, `:inferno`, `:viridis`) for different emphasis
- Run `detect_cosmic_rays` before any peak fitting to avoid fitting artifacts
- Try `pca_map` and `nmf_map` (exported) for unsupervised component analysis
