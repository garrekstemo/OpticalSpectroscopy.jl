# PL/Raman spatial mapping types and analysis

# =============================================================================
# PLMap type
# =============================================================================

"""
    PLMap <: AbstractSpectroscopyData

Photoluminescence intensity map from a CCD raster scan.

A 2D spatial grid where each point has a full CCD spectrum. The `intensity`
field holds the integrated PL signal at each position; the full spectra are
stored in `spectra` for extraction at individual positions.

# Fields
- `intensity::Matrix{Float64}` — Integrated PL intensity `(nx, ny)`
- `spectra::Array{Float64,3}` — Raw CCD counts `(nx, ny, n_pixel)`
- `x::Vector{Float64}` — Spatial x positions (μm)
- `y::Vector{Float64}` — Spatial y positions (μm)
- `pixel::Vector{Float64}` — Pixel indices (or wavelength if calibrated)
- `metadata::Dict{Symbol,Any}` — Symbol-keyed metadata token dict (shared with
  the rest of the `AbstractSpectroscopyData` family): provenance (`:source_file`,
  `:nx`, `:ny`, `:step_size`, `:pixel_range`) plus axis tokens (`:position_unit`
  for the spatial axes, `:yquantity`/`:yunit` for the intensity).
"""
struct PLMap <: AbstractSpectroscopyData
    intensity::Matrix{Float64}
    spectra::Array{Float64,3}
    x::Vector{Float64}
    y::Vector{Float64}
    pixel::Vector{Float64}
    metadata::Dict{Symbol,Any}
end

PLMap(intensity, spectra, x, y, pixel) =
    PLMap(intensity, spectra, x, y, pixel, Dict{Symbol,Any}())

# =============================================================================
# AbstractSpectroscopyData interface
# =============================================================================

# Spatial-axis label: "X"/"Y" identity with the unit derived from the
# :position_unit token (default :um), never hardcoded.
function _pl_position_label(axis::AbstractString, m::PLMap)
    u = Symbol(get(m.metadata, :position_unit, :um))
    ud = get(_UNIT_DISPLAY, u, String(u))
    return isempty(ud) ? String(axis) : "$axis ($ud)"
end

xdata(m::PLMap) = m.x
ydata(m::PLMap) = m.y
zdata(m::PLMap) = m.intensity
xlabel(m::PLMap) = _pl_position_label("X", m)
ylabel(m::PLMap) = _pl_position_label("Y", m)
zlabel(m::PLMap) = axis_label(Symbol(get(m.metadata, :yquantity, :intensity)),
                              Symbol(get(m.metadata, :yunit, :counts)))
_metadata(m::PLMap) = m.metadata
is_matrix(::PLMap) = true
npoints(m::PLMap) = (length(m.x), length(m.y))
source_file(m::PLMap) = get(m.metadata, :source_file, "unknown")
title(m::PLMap) = source_file(m)

# Semantic accessor
"""
    intensity(m::PLMap) -> Matrix{Float64}

Return the PL intensity matrix.
"""
intensity(m::PLMap) = m.intensity

function Base.show(io::IO, m::PLMap)
    nx, ny = length(m.x), length(m.y)
    np = length(m.pixel)
    print(io, "PLMap($(nx)×$(ny) grid, $(np) pixels)")
end

function Base.show(io::IO, ::MIME"text/plain", m::PLMap)
    nx, ny = length(m.x), length(m.y)
    np = length(m.pixel)
    println(io, "PLMap")
    println(io, "  Grid:     $(nx) × $(ny) spatial points")
    println(io, "  Pixels:   $(np) per spectrum")
    println(io, "  X range:  $(round(m.x[1], digits=1)) to $(round(m.x[end], digits=1)) μm")
    println(io, "  Y range:  $(round(m.y[1], digits=1)) to $(round(m.y[end], digits=1)) μm")
    print(io, "  Source:   $(source_file(m))")
end

# =============================================================================
# Spectrum extraction
# =============================================================================

"""
    extract_spectrum(m::PLMap, ix::Int, iy::Int) -> NamedTuple

Extract the CCD spectrum at grid index `(ix, iy)`.

Returns `(pixel=..., signal=..., x=..., y=...)`.
"""
function extract_spectrum(m::PLMap, ix::Int, iy::Int)
    1 <= ix <= length(m.x) || error("ix=$ix out of range 1:$(length(m.x))")
    1 <= iy <= length(m.y) || error("iy=$iy out of range 1:$(length(m.y))")
    return (pixel=m.pixel, signal=vec(m.spectra[ix, iy, :]),
            x=m.x[ix], y=m.y[iy])
end

"""
    extract_spectrum(m::PLMap; x::Real, y::Real) -> NamedTuple

Extract the CCD spectrum at the spatial position nearest to `(x, y)`.

Returns `(pixel=..., signal=..., x=..., y=..., ix=..., iy=...)`.
"""
function extract_spectrum(m::PLMap; x::Real, y::Real)
    ix = argmin(abs.(m.x .- x))
    iy = argmin(abs.(m.y .- y))
    spec = extract_spectrum(m, ix, iy)
    return (pixel=spec.pixel, signal=spec.signal,
            x=spec.x, y=spec.y, ix=ix, iy=iy)
end

# =============================================================================
# Background subtraction
# =============================================================================

"""
    subtract_background(m::PLMap; positions=nothing, margin=5) -> PLMap

Subtract a background spectrum from every grid point.

The background is the average CCD spectrum over the reference positions.
After subtraction, the intensity map is recomputed from the corrected spectra
using the same `pixel_range` as the original load (if any).

# Arguments
- `positions`: Vector of `(x, y)` spatial coordinate tuples (μm) for background
  reference points. These should be off-flake positions with no PL signal.
- `margin`: Number of grid points from each edge used for auto-detection when
  `positions` is not given. Auto mode averages the corners of the bottom half
  of the map (avoids top-row artifacts). Default: 5.

# Example
```julia
# spectra: (nx, ny, n_pixel) CCD counts; x, y: stage positions (μm)
intensity = dropdims(sum(spectra; dims=3); dims=3)
m = PLMap(intensity, spectra, x, y, collect(1.0:size(spectra, 3)))

# Explicit background positions
m_bg = subtract_background(m; positions=[(-40, -40), (40, -40), (-40, -20)])

# Auto mode (bottom corners)
m_bg = subtract_background(m)
```
"""
function subtract_background(m::PLMap; positions=nothing, margin::Int=5)
    nx, ny = length(m.x), length(m.y)

    if !isnothing(positions)
        # Explicit: average spectra at user-specified (x, y) positions
        bg_spectra = zeros(length(m.pixel))
        bg_positions = Tuple{Float64,Float64}[]
        for pos in positions
            spec = extract_spectrum(m; x=pos[1], y=pos[2])
            bg_spectra .+= spec.signal
            push!(bg_positions, (spec.x, spec.y))
        end
        bg_spectra ./= length(positions)
    else
        # Auto: average corners from the bottom half of the map
        bg_spectra = zeros(length(m.pixel))
        bg_positions = Tuple{Float64,Float64}[]
        for ix in [1:margin; (nx-margin+1):nx]
            for iy in 1:margin
                bg_spectra .+= vec(m.spectra[ix, iy, :])
                push!(bg_positions, (m.x[ix], m.y[iy]))
            end
        end
        bg_spectra ./= length(bg_positions)
        @info "Auto background: averaged $(length(bg_positions)) spectra from bottom corners " *
              "(ix=1:$margin and $(nx-margin+1):$nx, iy=1:$margin)"
    end

    # Subtract background spectrum from every grid point
    corrected = m.spectra .- reshape(bg_spectra, 1, 1, :)

    # Recompute intensity with the same pixel_range as the original
    pixel_range = get(m.metadata, :pixel_range, nothing)
    if !isnothing(pixel_range)
        p1, p2 = pixel_range
        intensity = dropdims(sum(corrected[:, :, p1:p2]; dims=3); dims=3)
    else
        intensity = dropdims(sum(corrected; dims=3); dims=3)
    end

    new_metadata = copy(m.metadata)
    new_metadata[:background_positions] = bg_positions
    return PLMap(intensity, corrected, m.x, m.y, m.pixel, new_metadata)
end

# =============================================================================
# Normalization
# =============================================================================

"""
    normalize_intensity(m::PLMap) -> PLMap

Return a new PLMap with intensity normalized to [0, 1].
"""
function normalize_intensity(m::PLMap)
    imin, imax = extrema(m.intensity)
    if imax == imin
        norm_intensity = zeros(size(m.intensity))
    else
        norm_intensity = (m.intensity .- imin) ./ (imax - imin)
    end
    # Normalized intensity is no longer in counts: retag the signal unit so the
    # derived z-label stays honest (`Intensity (arb. units)`).
    md = copy(m.metadata)
    md[:yquantity] = :intensity
    md[:yunit] = :arb
    return PLMap(norm_intensity, m.spectra, m.x, m.y, m.pixel, md)
end

# =============================================================================
# Integrated intensity
# =============================================================================

"""
    integrated_intensity(m::PLMap; pixel_range=nothing) -> Matrix{Float64}

Compute the integrated intensity at each grid point.

Without `pixel_range`, returns `m.intensity` (the precomputed full-range sum).
With `pixel_range`, sums `m.spectra[:, :, p1:p2]` over the given pixel window.

# Arguments
- `pixel_range`: `(start, stop)` pixel indices to integrate over.
  Falls back to the `pixel_range` stored in metadata, or uses `m.intensity` if unset.
"""
function integrated_intensity(m::PLMap; pixel_range::Union{Tuple{Int,Int},Nothing}=nothing)
    pr = !isnothing(pixel_range) ? pixel_range : get(m.metadata, :pixel_range, nothing)
    if !isnothing(pr)
        p1 = max(1, pr[1])
        p2 = min(length(m.pixel), pr[2])
        return dropdims(sum(m.spectra[:, :, p1:p2]; dims=3); dims=3)
    else
        return m.intensity
    end
end

# =============================================================================
# Intensity mask
# =============================================================================

"""
    intensity_mask(m::PLMap; pixel_range=nothing, threshold=0.05, exclude=nothing) -> NamedTuple

Compute a boolean mask over the PLMap grid based on integrated intensity.

Grid points where the integrated intensity is below `threshold` fraction of the
intensity range are excluded (`false`). Spatial regions listed in `exclude` are
also set to `false` regardless of intensity. This is useful for filtering out
off-sample regions and known artifacts (e.g., substrate bands) before batch
operations like fitting.

The threshold is computed as: `cutoff = min + threshold × (max - min)`.

# Arguments
- `pixel_range`: `(start, stop)` pixel indices for computing integrated intensity.
  If `nothing`, uses the full precomputed `m.intensity`.
- `threshold`: Fraction of the intensity range (0.0 to 1.0). Default `0.05` (5%).
- `exclude`: Spatial regions to exclude. Each region is a tuple of x and y ranges
  in spatial coordinates (μm): `((x_min, x_max), (y_min, y_max))`. Accepts a
  single region or a vector of regions. Grid points whose spatial coordinates
  fall within any excluded region are masked out.

# Returns
A named tuple with fields:
- `mask::BitMatrix` — `true` for included grid points
- `n_included::Int` — number of included points
- `n_total::Int` — total number of grid points
- `intensity_min::Float64` — minimum integrated intensity
- `intensity_max::Float64` — maximum integrated intensity
- `cutoff::Float64` — absolute intensity cutoff value

# Examples
```julia
m = PLMap(intensity, spectra, x, y, pixel)
m = subtract_background(m)

# Threshold only
result = intensity_mask(m; pixel_range=(950, 1100), threshold=0.1)

# Exclude a substrate band at the top of the map
result = intensity_mask(m; threshold=0.05,
    exclude=((-Inf, Inf), (40.0, Inf)))

# Multiple exclusion regions
result = intensity_mask(m; threshold=0.05,
    exclude=[
        ((-Inf, Inf), (40.0, Inf)),     # top substrate band
        ((-50.0, -40.0), (-50.0, -40.0))  # noisy corner
    ])
```
"""
function intensity_mask(m::PLMap; pixel_range::Union{Tuple{Int,Int},Nothing}=nothing,
                        threshold::Real=0.05,
                        exclude=nothing)
    ii = integrated_intensity(m; pixel_range=pixel_range)
    imin, imax = extrema(ii)
    cutoff = imin + Float64(threshold) * (imax - imin)
    mask = ii .>= cutoff

    if !isnothing(exclude)
        _apply_exclusions!(mask, m.x, m.y, exclude)
    end

    return (
        mask = mask,
        n_included = count(mask),
        n_total = length(mask),
        intensity_min = imin,
        intensity_max = imax,
        cutoff = cutoff,
    )
end

# Single region: ((x_min, x_max), (y_min, y_max))
function _apply_exclusions!(mask::BitMatrix, x::Vector{Float64}, y::Vector{Float64},
                            region::Tuple{Tuple{T1,T2}, Tuple{T3,T4}}) where {T1<:Real, T2<:Real, T3<:Real, T4<:Real}
    xr, yr = region
    for (ix, xv) in enumerate(x)
        for (iy, yv) in enumerate(y)
            if xr[1] <= xv <= xr[2] && yr[1] <= yv <= yr[2]
                mask[ix, iy] = false
            end
        end
    end
end

# Vector of regions
function _apply_exclusions!(mask::BitMatrix, x::Vector{Float64}, y::Vector{Float64},
                            regions::AbstractVector)
    for region in regions
        _apply_exclusions!(mask, x, y, region)
    end
end

# =============================================================================
# Peak center (centroid) map
# =============================================================================

"""
    peak_centers(m::PLMap; pixel_range=nothing, threshold=0.05) -> Matrix{Float64}

Compute the centroid (intensity-weighted average pixel) at each grid point.

Returns a `(nx, ny)` matrix of peak center positions in pixel units. Grid points
where the PL intensity (`m.intensity`) is below `threshold` × the map maximum
are set to `NaN` (renders as transparent in heatmaps with `nan_color=:transparent`).

Masking uses the PLMap's intensity field — the integrated PL signal already stored
in the map. This produces clean masks that match the intensity heatmap: off-flake
regions (low PL signal) are transparent, on-flake regions show centroid positions.

# Arguments
- `pixel_range`: `(start, stop)` pixel range to compute the centroid over.
  Falls back to the `pixel_range` stored in metadata, or all pixels if unset.
- `threshold`: Fraction of the maximum PL intensity below which a grid point is
  masked as `NaN`. Default `0.05` (5%). Set to `0` to disable masking.

# Example
```julia
m = PLMap(intensity, spectra, x, y, pixel)
m = subtract_background(m)
centers = peak_centers(m; pixel_range=(950, 1100))
heatmap(m.x, m.y, centers; colormap=:viridis, nan_color=:transparent)
```
"""
function peak_centers(m::PLMap; pixel_range::Union{Tuple{Int,Int},Nothing}=nothing,
                      threshold::Real=0.05)
    pr = !isnothing(pixel_range) ? pixel_range : get(m.metadata, :pixel_range, nothing)

    if !isnothing(pr)
        p1 = max(1, pr[1])
        p2 = min(length(m.pixel), pr[2])
        pixels = m.pixel[p1:p2]
        spectra_slice = @view m.spectra[:, :, p1:p2]
    else
        pixels = m.pixel
        spectra_slice = m.spectra
    end

    nx, ny = length(m.x), length(m.y)

    # Mask against the PL intensity map (integrated signal already in m.intensity)
    max_intensity = maximum(m.intensity)
    cutoff = max_intensity > 0 ? max_intensity * threshold : zero(max_intensity)

    centers = Matrix{Float64}(undef, nx, ny)
    Threads.@threads for idx in CartesianIndices((nx, ny))
        ix, iy = idx[1], idx[2]
        if m.intensity[ix, iy] <= cutoff
            centers[ix, iy] = NaN
        else
            sig = @view spectra_slice[ix, iy, :]
            total = sum(sig)
            centers[ix, iy] = sum(pixels[k] * sig[k] for k in eachindex(sig)) / total
        end
    end
    return centers
end

# =============================================================================
# Batch peak fitting
# =============================================================================

# FWHM of a fitted peak, converted according to the model actually used.
# Width parameterizations differ across CurveFitModels lineshapes:
#   lorentzian    Γ is already the FWHM (param :fwhm)
#   gaussian      σ is the standard deviation        -> FWHM = 2√(2ln2)·σ
#   pseudo_voigt  σ is the HWHM of BOTH components   -> FWHM = 2σ
#   voigt         σ Gaussian std dev, γ Lorentzian HWHM -> Thompson et al. (1987)
#   fano          asymmetric lineshape, FWHM not well defined -> NaN
function _peak_fwhm(pk::PeakFitResult)
    if haskey(pk, :fwhm)
        return abs(pk[:fwhm].value)
    elseif pk.model == "pseudo_voigt" && haskey(pk, :sigma)
        return 2 * abs(pk[:sigma].value)
    elseif haskey(pk, :sigma) && haskey(pk, :gamma)
        # Voigt FWHM approximation (Thompson et al., 1987)
        fwhm_g = abs(pk[:sigma].value) * 2 * sqrt(2 * log(2))
        fwhm_l = 2 * abs(pk[:gamma].value)
        return 0.5346 * fwhm_l + sqrt(0.2166 * fwhm_l^2 + fwhm_g^2)
    elseif haskey(pk, :sigma)
        # gaussian (and unknown models exposing :sigma): treat as std dev
        return abs(pk[:sigma].value) * 2 * sqrt(2 * log(2))
    else
        return NaN
    end
end

"""
    fit_map(m::PLMap; kwargs...) -> FitMapResult

Fit peaks at every grid point in a PLMap, returning a [`FitMapResult`](@ref)
with per-pixel fit results and summary arrays for heatmap visualization.

Uses **reference pixel seeding**: the brightest included pixel is fitted first
with automatic initial parameter detection, and its converged parameters are
used as starting values for all remaining pixels. This dramatically improves
convergence for broad or noisy peaks where auto-detection is unreliable, since
peak shapes vary smoothly across the spatial map.

Fitting is multi-threaded when Julia is started with multiple threads
(`julia --threads=auto`).

# Keyword Arguments
- `model::Function = gaussian` — Peak model (`gaussian`, `lorentzian`, `pseudo_voigt`)
- `n_peaks::Int = 1` — Number of peaks to fit
- `region::Union{Tuple{Real,Real}, Nothing} = nothing` — Spectral region to fit within
- `baseline_order::Int = 1` — Polynomial baseline order
- `mask::Union{BitMatrix, Nothing} = nothing` — Pre-computed fit mask.
  If provided, `threshold`/`pixel_range`/`exclude` are ignored.
- `threshold::Real = 0.0` — Intensity threshold for [`intensity_mask`](@ref) (0–1)
- `pixel_range::Union{Tuple{Int,Int}, Nothing} = nothing` — Pixel range for mask computation
- `exclude` — Spatial exclusion regions for [`intensity_mask`](@ref)
- `abort::Union{Threads.Atomic{Bool}, Nothing} = nothing` — Set to `true` to stop early
- `progress::Union{Function, Nothing} = nothing` — Callback `(n_done, n_total) -> nothing`

# Example
```julia
m = PLMap(intensity, spectra, x, y, pixel)
m = subtract_background(m)

result = fit_map(m;
    model = lorentzian,
    region = (950, 1050),
    threshold = 0.05)

# Summary arrays for heatmaps
heatmap(m.x, m.y, result.centers)
heatmap(m.x, m.y, result.fwhms)

# Per-pixel result
result[10, 15]  # MultiPeakFitResult or nothing
```
"""
function fit_map(m::PLMap;
                 model::Function = gaussian,
                 n_peaks::Int = 1,
                 region::Union{Tuple{Real,Real}, Nothing} = nothing,
                 baseline_order::Int = 1,
                 mask::Union{BitMatrix, Nothing} = nothing,
                 threshold::Real = 0.0,
                 pixel_range::Union{Tuple{Int,Int}, Nothing} = nothing,
                 exclude = nothing,
                 abort::Union{Threads.Atomic{Bool}, Nothing} = nothing,
                 progress::Union{Function, Nothing} = nothing,
                 ref_r2_threshold::Real = 0.80,
                 n_ref_candidates::Int = 5)

    nx, ny = length(m.x), length(m.y)
    results = Matrix{Any}(nothing, nx, ny)

    # Compute mask if not provided
    if isnothing(mask) && (threshold > 0 || !isnothing(exclude))
        mask = intensity_mask(m; pixel_range=pixel_range,
                              threshold=Float64(threshold),
                              exclude=exclude).mask
    end

    # Build flat list of pixels to fit
    pixels_to_fit = Tuple{Int,Int}[]
    n_skipped = 0
    for iy in 1:ny
        for ix in 1:nx
            if !isnothing(mask) && !mask[ix, iy]
                n_skipped += 1
            else
                push!(pixels_to_fit, (ix, iy))
            end
        end
    end

    n_to_fit = length(pixels_to_fit)

    # Helper to extract and optionally crop a spectrum
    function _extract_and_crop(ix, iy)
        spec = extract_spectrum(m, ix, iy)
        x, y = spec.pixel, spec.signal
        if !isnothing(region)
            rmask = region[1] .<= x .<= region[2]
            x = x[rmask]
            y = y[rmask]
        end
        return x, y
    end

    # Fit reference pixel to seed initial parameters.
    # Try the top-N brightest candidates and use the first with R² above threshold.
    # This avoids poisoning the seed with a CR-contaminated or anomalous pixel.
    p0_ref = nothing
    if !isempty(pixels_to_fit)
        sorted_by_intensity = sort(pixels_to_fit,
                                   by=(p -> m.intensity[p[1], p[2]]),
                                   rev=true)
        n_cand = min(n_ref_candidates, length(sorted_by_intensity))

        for (ref_ix, ref_iy) in sorted_by_intensity[1:n_cand]
            try
                x, y = _extract_and_crop(ref_ix, ref_iy)
                ref_result = fit_peaks(x, y; model=model, n_peaks=n_peaks,
                                       baseline_order=baseline_order)
                if ref_result.r_squared >= ref_r2_threshold
                    results[ref_ix, ref_iy] = ref_result
                    p0_ref = ref_result._coef
                    break
                end
            catch
            end
        end
    end

    # Thread-safe counters
    n_done = Threads.Atomic{Int}(!isnothing(p0_ref) ? 1 : 0)
    n_converged = Threads.Atomic{Int}(!isnothing(p0_ref) ? 1 : 0)
    n_failed = Threads.Atomic{Int}(0)

    if !isnothing(progress)
        progress(n_done[], n_to_fit)
    end

    # Parallel fitting with reference seeding
    Threads.@threads for idx in eachindex(pixels_to_fit)
        !isnothing(abort) && abort[] && continue

        ix, iy = pixels_to_fit[idx]
        !isnothing(results[ix, iy]) && continue  # skip reference pixel

        try
            x, y = _extract_and_crop(ix, iy)
            result = fit_peaks(x, y; model=model, n_peaks=n_peaks,
                               baseline_order=baseline_order, p0=p0_ref)
            results[ix, iy] = result
            Threads.atomic_add!(n_converged, 1)
        catch
            Threads.atomic_add!(n_failed, 1)
        end
        Threads.atomic_add!(n_done, 1)
        if !isnothing(progress)
            progress(n_done[], n_to_fit)
        end
    end

    # Build summary arrays for all peaks
    centers = fill(NaN, nx, ny, n_peaks)
    fwhms = fill(NaN, nx, ny, n_peaks)
    amplitudes = fill(NaN, nx, ny, n_peaks)
    r_squareds = fill(NaN, nx, ny)

    for iy in 1:ny
        for ix in 1:nx
            r = results[ix, iy]
            isnothing(r) && continue
            r_squareds[ix, iy] = r.r_squared
            for pk_idx in eachindex(r.peaks)
                pk_idx > n_peaks && break
                pk = r.peaks[pk_idx]
                if haskey(pk, :center)
                    centers[ix, iy, pk_idx] = pk[:center].value
                end
                fwhms[ix, iy, pk_idx] = _peak_fwhm(pk)
                if haskey(pk, :amplitude)
                    amplitudes[ix, iy, pk_idx] = pk[:amplitude].value
                end
            end
        end
    end

    valid_rsq = filter(!isnan, vec(r_squareds))
    med_rsq = isempty(valid_rsq) ? NaN : Statistics.median(valid_rsq)

    return FitMapResult(
        results, mask,
        centers, fwhms, amplitudes, r_squareds,
        n_peaks, n_converged[], n_failed[], n_skipped, med_rsq
    )
end
