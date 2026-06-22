# Cosmic ray detection and removal
#
# Cosmic rays are single-pixel spectral artifacts from high-energy particles
# hitting the CCD detector. They appear as sharp, narrow spikes that don't
# correlate with spatial neighbors.
#
# 1D algorithm: Whitaker-Hayes modified z-score on first differences with
# shoulder expansion for single spectra.
# Reference: D. A. Whitaker and K. Hayes, Chemom. Intell. Lab. Syst. 179, 82
# (2018), doi:10.1016/j.chemolab.2018.06.009.
#
# PLMap algorithm: Most Similar Neighbor (MSN) comparison. Each pixel's
# spectrum is compared to the 4-connected neighbor with the highest Pearson
# correlation. Channels where the residual is a significant positive
# outlier are flagged as cosmic rays. Using the most-similar neighbor
# avoids false positives at spatial boundaries where spectra change.

# =============================================================================
# Result types
# =============================================================================

"""
    CosmicRayResult

Result of cosmic ray detection on a 1D signal.

# Fields
- `indices::Vector{Int}` — flagged channel indices
- `count::Int` — number of flagged channels
"""
struct CosmicRayResult
    indices::Vector{Int}
    count::Int
end

"""
    CosmicRayMapResult

Result of cosmic ray detection on a PLMap.

# Fields
- `mask::BitArray{3}` — `(nx, ny, n_pixel)`, `true` = cosmic ray
- `count::Int` — total flagged voxels
- `affected_spectra::Int` — number of spectra with at least one cosmic ray
- `channel_counts::Vector{Int}` — cosmic ray count per spectral channel
"""
struct CosmicRayMapResult
    mask::BitArray{3}
    count::Int
    affected_spectra::Int
    channel_counts::Vector{Int}
end

# =============================================================================
# 1D detection and removal
# =============================================================================

"""
    detect_cosmic_rays(signal::AbstractVector; threshold=5.0) -> CosmicRayResult

Detect cosmic ray spikes in a 1D spectrum using modified z-scores on first
differences (Whitaker-Hayes method).

Computes first differences `Δ[k] = signal[k+1] - signal[k]`, then flags
channels where the modified z-score exceeds `threshold`. The modified z-score
uses the Median Absolute Deviation (MAD) for robust scale estimation.

# Arguments
- `signal`: 1D spectral signal
- `threshold`: z-score cutoff (default 5.0). Lower values detect more spikes
  but may flag real features.

# Returns
A [`CosmicRayResult`](@ref) with the flagged channel indices.

# Example
```julia
result = detect_cosmic_rays(spectrum; threshold=5.0)
println("Found \$(result.count) cosmic rays at indices \$(result.indices)")
```

# Reference
D. A. Whitaker and K. Hayes, "A simple algorithm for despiking Raman spectra",
Chemom. Intell. Lab. Syst. 179, 82 (2018).
[doi:10.1016/j.chemolab.2018.06.009](https://doi.org/10.1016/j.chemolab.2018.06.009)
"""
function detect_cosmic_rays(signal::AbstractVector; threshold::Real=5.0)
    n = length(signal)
    if n < 3
        return CosmicRayResult(Int[], 0)
    end

    # First differences
    diffs = diff(signal)

    med = median(diffs)
    mad_val = median(abs.(diffs .- med))

    if mad_val < eps(Float64)
        return CosmicRayResult(Int[], 0)
    end

    # Modified z-scores (0.6745 ≈ Φ⁻¹(0.75), makes MAD consistent with σ)
    z_scores = @. 0.6745 * (diffs - med) / mad_val

    # A cosmic ray spike creates a positive jump followed by a negative jump.
    # Flag the channel where the spike sits: if z[k] > threshold (upward jump),
    # the spike is at k+1; if z[k] < -threshold (downward jump from previous
    # spike), the spike was at k. We collect unique indices.
    flagged = Set{Int}()
    for k in eachindex(z_scores)
        if z_scores[k] > threshold
            push!(flagged, k + 1)  # spike at the "to" channel
        elseif z_scores[k] < -threshold
            push!(flagged, k)      # spike at the "from" channel
        end
    end

    # Expand flagged regions to capture spike shoulders (CCD charge spread)
    if !isempty(flagged)
        _expand_to_shoulders!(flagged, signal, mad_val)
    end

    indices = sort!(collect(flagged))
    # Clamp to valid range
    filter!(i -> 1 <= i <= n, indices)

    return CosmicRayResult(indices, length(indices))
end

"""
    _expand_to_shoulders!(flagged, signal, mad_val; factor=3.0)

Expand flagged regions outward to capture shoulder channels where CCD charge
has spread from the cosmic ray peak. A candidate neighbor is included if it
is elevated above the nearest non-flagged reference channel by more than
`factor × σ`, where σ is estimated from the MAD of first differences.
"""
function _expand_to_shoulders!(flagged::Set{Int}, signal::AbstractVector, mad_val::Real;
                               factor::Real=3.0)
    n = length(signal)
    noise_est = mad_val / 0.6745  # MAD → σ estimate

    changed = true
    while changed
        changed = false
        for idx in collect(flagged)
            for nb in (idx - 1, idx + 1)
                (1 <= nb <= n && !(nb in flagged)) || continue

                # Find nearest non-flagged channel on the outward side
                step = nb < idx ? -1 : 1
                ref = nb + step
                while 1 <= ref <= n && ref in flagged
                    ref += step
                end
                (1 <= ref <= n) || continue

                if signal[nb] - signal[ref] > factor * noise_est
                    push!(flagged, nb)
                    changed = true
                end
            end
        end
    end
end

"""
    remove_cosmic_rays(signal::AbstractVector, result::CosmicRayResult) -> Vector

Replace flagged cosmic ray channels with linearly interpolated values from
the nearest non-flagged neighbors.

Returns a new vector; does not mutate the input.

# Example
```julia
result = detect_cosmic_rays(spectrum)
cleaned = remove_cosmic_rays(spectrum, result)
```
"""
function remove_cosmic_rays(signal::AbstractVector, result::CosmicRayResult)
    cleaned = collect(Float64, signal)
    if result.count == 0
        return cleaned
    end

    n = length(cleaned)
    flagged_set = Set(result.indices)

    for idx in result.indices
        # Find nearest non-flagged neighbor to the left
        left_idx = idx - 1
        while left_idx >= 1 && left_idx in flagged_set
            left_idx -= 1
        end

        # Find nearest non-flagged neighbor to the right
        right_idx = idx + 1
        while right_idx <= n && right_idx in flagged_set
            right_idx += 1
        end

        if left_idx >= 1 && right_idx <= n
            # Linear interpolation
            t = (idx - left_idx) / (right_idx - left_idx)
            cleaned[idx] = cleaned[left_idx] * (1 - t) + cleaned[right_idx] * t
        elseif left_idx >= 1
            cleaned[idx] = cleaned[left_idx]
        elseif right_idx <= n
            cleaned[idx] = cleaned[right_idx]
        end
        # If both out of range (entire signal flagged), leave as-is
    end

    return cleaned
end

# =============================================================================
# PLMap detection and removal
# =============================================================================

"""
    _neighbor_spectra(spectra, ix, iy, nx, ny, p1, p2)

Collect spectral slices from 4-connected spatial neighbors of `(ix, iy)`,
restricted to channels `p1:p2`. Returns a vector of views (one per neighbor).
"""
function _neighbor_spectra(spectra::AbstractArray{<:Any,3}, ix::Int, iy::Int,
                           nx::Int, ny::Int, p1::Int, p2::Int)
    neighbors = typeof(@view spectra[1, 1, p1:p2])[]
    if ix > 1
        push!(neighbors, @view spectra[ix-1, iy, p1:p2])
    end
    if ix < nx
        push!(neighbors, @view spectra[ix+1, iy, p1:p2])
    end
    if iy > 1
        push!(neighbors, @view spectra[ix, iy-1, p1:p2])
    end
    if iy < ny
        push!(neighbors, @view spectra[ix, iy+1, p1:p2])
    end
    return neighbors
end

"""
    _most_similar_neighbor(signal, neighbors) -> (index, correlation)

Find the neighbor with the highest Pearson correlation to `signal`.
Returns the 1-based index into `neighbors` and the correlation value.
At spatial boundaries, the most-similar neighbor is on the same side,
so real spectral differences don't create false residuals.
"""
function _most_similar_neighbor(signal::AbstractVector, neighbors::Vector)
    best_idx = 1
    best_corr = -Inf
    n = length(signal)
    # Precompute signal stats
    s_mean = mean(signal)
    s_centered = signal .- s_mean
    s_ss = sqrt(sum(abs2, s_centered))

    for (i, nb) in enumerate(neighbors)
        nb_mean = mean(nb)
        nb_centered = nb .- nb_mean
        nb_ss = sqrt(sum(abs2, nb_centered))
        if s_ss < eps(Float64) || nb_ss < eps(Float64)
            continue
        end
        corr = sum(s_centered .* nb_centered) / (s_ss * nb_ss)
        if corr > best_corr
            best_corr = corr
            best_idx = i
        end
    end
    return (best_idx, best_corr)
end

"""
    _unflag_wide_runs!(mask, ix, iy, p1, p2, max_width)

Scan channels `p1:p2` for pixel `(ix, iy)` and unflag any contiguous run of
flagged channels that is wider than `max_width`. Cosmic ray spikes are narrow;
wider runs are spectral variation at spatial boundaries.
"""
function _unflag_wide_runs!(mask::BitArray{3}, ix::Int, iy::Int,
                            p1::Int, p2::Int, max_width::Int)
    k = p1
    while k <= p2
        if mask[ix, iy, k]
            run_start = k
            while k <= p2 && mask[ix, iy, k]
                k += 1
            end
            if k - run_start > max_width
                for j in run_start:(k - 1)
                    mask[ix, iy, j] = false
                end
            end
        else
            k += 1
        end
    end
end

"""
    detect_cosmic_rays(m::PLMap; threshold=5.0, pixel_range=nothing, max_spike_width=7) -> CosmicRayMapResult

Detect cosmic ray spikes across all spectra in a PLMap using Most Similar
Neighbor (MSN) comparison.

For each pixel, selects the 4-connected neighbor with the highest Pearson
correlation as the reference spectrum. The residual (pixel minus MSN) is then
tested for positive outliers using the MAD-based robust scale estimate. Channels
where the residual exceeds `threshold × σ` above the median residual are flagged.

Using the most-similar neighbor avoids false positives at spatial boundaries:
real spectral differences cancel because the MSN is on the same side of the
boundary. Only true cosmic ray spikes — absent from all neighbors — remain.

Additional filters:
- **Width filter**: contiguous flagged runs wider than `max_spike_width` are
  unflagged — cosmic rays span 1–5 channels; wider runs are spectral variation.
- **Fraction cap**: if more than 5% of channels are still flagged after the
  width filter, all flags for that pixel are cleared.

A global noise floor (median of per-pixel σ estimates) prevents over-sensitivity
in spatially homogeneous regions.

# Arguments
- `m`: PLMap with 3D spectra array `(nx, ny, n_pixel)`
- `threshold`: outlier cutoff in MAD-scaled units (default 5.0)
- `pixel_range`: `(start, stop)` channel indices to analyze. When set, only
  this subrange of each spectrum is checked for cosmic rays. Channels outside
  the range are never flagged. Defaults to the `pixel_range` in `m.metadata`,
  or the full spectrum if unset.
- `max_spike_width`: maximum width (in channels) of a contiguous flagged run
  to keep (default 7). Runs wider than this are unflagged as spectral variation.

# Returns
A [`CosmicRayMapResult`](@ref) with a 3D boolean mask and summary statistics.

# Example
```julia
cr = detect_cosmic_rays(plmap; threshold=5.0)
println("Found \$(cr.count) cosmic rays in \$(cr.affected_spectra) spectra")
```
"""
function detect_cosmic_rays(m::PLMap; threshold::Real=5.0,
                            pixel_range::Union{Tuple{Int,Int},Nothing}=nothing,
                            max_spike_width::Int=7)
    nx, ny, np = size(m.spectra)
    mask = falses(nx, ny, np)

    # Determine channel range for detection
    pr = !isnothing(pixel_range) ? pixel_range : get(m.metadata, :pixel_range, nothing)
    p1 = !isnothing(pr) ? max(1, Int(pr[1])) : 1
    p2 = !isnothing(pr) ? min(np, Int(pr[2])) : np

    n_ch = p2 - p1 + 1

    # Two-pass approach: first compute per-pixel noise estimates to establish
    # a global noise floor, then flag with max(local_σ, floor). This prevents
    # false positives in spatially homogeneous regions where local σ is tiny.
    sigmas = fill(NaN, nx, ny)
    Threads.@threads for idx in CartesianIndices((nx, ny))
        ix, iy = idx[1], idx[2]
        neighbors = _neighbor_spectra(m.spectra, ix, iy, nx, ny, p1, p2)
        isempty(neighbors) && continue
        signal = @view m.spectra[ix, iy, p1:p2]
        # Use most similar neighbor for noise estimation (consistent with detection)
        best_idx, _ = _most_similar_neighbor(signal, neighbors)
        ref = neighbors[best_idx]
        residual = Vector{Float64}(undef, n_ch)
        for k in 1:n_ch
            residual[k] = signal[k] - ref[k]
        end
        mad_r = median(abs.(residual .- median(residual)))
        if mad_r > eps(Float64)
            sigmas[ix, iy] = mad_r / 0.6745
        end
    end
    pixel_sigmas = filter(!isnan, vec(sigmas))
    noise_floor = isempty(pixel_sigmas) ? 0.0 : median(pixel_sigmas)

    Threads.@threads for idx in CartesianIndices((nx, ny))
        ix, iy = idx[1], idx[2]
        neighbors = _neighbor_spectra(m.spectra, ix, iy, nx, ny, p1, p2)
        isempty(neighbors) && continue

        signal = @view m.spectra[ix, iy, p1:p2]

        # Most Similar Neighbor (MSN) reference: compare to the neighbor with
        # the highest spectral correlation. At spatial boundaries, the MSN is
        # on the same side of the boundary, so real spectral differences vanish
        # in the residual. Only true cosmic ray spikes remain.
        best_idx, _ = _most_similar_neighbor(signal, neighbors)
        ref = neighbors[best_idx]

        residual = Vector{Float64}(undef, n_ch)
        for k in 1:n_ch
            residual[k] = signal[k] - ref[k]
        end

        # Flag positive outliers using MAD-based scale with noise floor
        med_r = median(residual)
        mad_r = median(abs.(residual .- med_r))
        mad_r < eps(Float64) && continue
        σ = max(mad_r / 0.6745, noise_floor)

        for k in 1:n_ch
            if residual[k] - med_r > Float64(threshold) * σ
                mask[ix, iy, k + p1 - 1] = true
            end
        end

        # Cosmic ray spikes are narrow (1–5 channels). Broad contiguous runs
        # of flagged channels indicate spectral variation at spatial boundaries,
        # not cosmic rays. Unflag any run wider than max_spike_width.
        _unflag_wide_runs!(mask, ix, iy, p1, p2, max_spike_width)

        # Safety: if too many channels remain flagged after the width filter,
        # it's spatial variation, not cosmic rays. Clear all flags.
        remaining = count(@view mask[ix, iy, p1:p2])
        if remaining > n_ch ÷ 20  # > 5% of channels
            for k in p1:p2
                mask[ix, iy, k] = false
            end
        end
    end

    total = count(mask)
    affected = 0
    for iy in 1:ny
        for ix in 1:nx
            if any(@view mask[ix, iy, :])
                affected += 1
            end
        end
    end

    channel_counts = zeros(Int, np)
    for k in 1:np
        channel_counts[k] = count(@view mask[:, :, k])
    end

    return CosmicRayMapResult(mask, total, affected, channel_counts)
end

"""
    remove_cosmic_rays(m::PLMap, result::CosmicRayMapResult) -> PLMap

Remove cosmic rays from a PLMap using scaled Most Similar Neighbor (MSN) replacement.

For each affected pixel, finds the 4-connected neighbor with the highest Pearson
correlation (the MSN). A scale factor is computed from non-flagged channels to match
the pixel's intensity level. Flagged channels are then replaced with `scale × MSN[k]`.
This preserves the pixel's overall intensity while removing only the spike shape,
avoiding the dark-patch artifacts that raw neighbor median replacement produces.

Falls back to spectral interpolation (1D linear from nearest non-flagged channels)
for edge pixels with no spatial neighbors.

Returns a new PLMap with recomputed intensity.

# Example
```julia
cr = detect_cosmic_rays(plmap)
cleaned = remove_cosmic_rays(plmap, cr)
```
"""
function remove_cosmic_rays(m::PLMap, result::CosmicRayMapResult)
    if result.count == 0
        return m
    end

    nx, ny, np = size(m.spectra)
    cleaned = copy(m.spectra)

    # Use same channel range as detection
    pr = get(m.metadata, :pixel_range, nothing)
    p1 = !isnothing(pr) ? max(1, Int(pr[1])) : 1
    p2 = !isnothing(pr) ? min(np, Int(pr[2])) : np
    n_ch = p2 - p1 + 1

    Threads.@threads for idx in CartesianIndices((nx, ny))
        ix, iy = idx[1], idx[2]
        any(@view result.mask[ix, iy, :]) || continue

        # Find the Most Similar Neighbor (MSN) for this pixel
        neighbors = _neighbor_spectra(m.spectra, ix, iy, nx, ny, p1, p2)

        if isempty(neighbors)
            # Edge case: no neighbors — fall back to spectral interpolation
            flagged_channels = findall(@view result.mask[ix, iy, :])
            cr_1d = CosmicRayResult(flagged_channels, length(flagged_channels))
            signal = @view m.spectra[ix, iy, :]
            interp = remove_cosmic_rays(signal, cr_1d)
            for ch in flagged_channels
                cleaned[ix, iy, ch] = interp[ch]
            end
            continue
        end

        signal = @view m.spectra[ix, iy, p1:p2]
        best_idx, _ = _most_similar_neighbor(signal, neighbors)
        msn = neighbors[best_idx]

        # Compute scale factor from non-flagged channels so the replacement
        # preserves this pixel's intensity level (Whitaker & Hayes method).
        sig_sum = 0.0
        msn_sum = 0.0
        for k in 1:n_ch
            if !result.mask[ix, iy, k + p1 - 1]
                sig_sum += signal[k]
                msn_sum += msn[k]
            end
        end
        scale = msn_sum > eps(Float64) ? sig_sum / msn_sum : 1.0

        # Replace flagged channels with scaled MSN values
        for k in 1:n_ch
            if result.mask[ix, iy, k + p1 - 1]
                cleaned[ix, iy, k + p1 - 1] = scale * msn[k]
            end
        end
    end

    # Recompute intensity
    pixel_range = get(m.metadata, :pixel_range, nothing)
    if !isnothing(pixel_range)
        rp1, rp2 = pixel_range
        new_intensity = dropdims(sum(cleaned[:, :, Int(rp1):Int(rp2)]; dims=3); dims=3)
    else
        new_intensity = dropdims(sum(cleaned; dims=3); dims=3)
    end

    return PLMap(new_intensity, cleaned, m.x, m.y, m.pixel, m.metadata)
end

# =============================================================================
# Helper: 3×3 median filter (used by TimeResolvedMatrix detection)
# =============================================================================

# 3×3 median filter with edge replication.
function _median_filter3(data::AbstractMatrix)
    nrow, ncol = size(data)
    out = similar(data, Float64)
    buf = Vector{Float64}(undef, 9)
    for j in 1:ncol, i in 1:nrow
        k = 0
        for dj in -1:1, di in -1:1
            k += 1
            buf[k] = data[clamp(i + di, 1, nrow), clamp(j + dj, 1, ncol)]
        end
        # buf is always fully overwritten with 9 values, so the median is the
        # 5th element after sorting — avoids median()'s per-call allocation.
        sort!(buf)
        out[i, j] = buf[5]
    end
    return out
end

# =============================================================================
# TimeResolvedMatrix detection and removal
# =============================================================================

"""
    CosmicRayMatrixResult

Result of cosmic ray detection on a [`TimeResolvedMatrix`](@ref).

# Fields
- `indices::Vector{CartesianIndex{2}}` — flagged (time, wavelength) pixels
- `count::Int` — number of flagged pixels
- `threshold::Float64` — detection threshold used
"""
struct CosmicRayMatrixResult
    indices::Vector{CartesianIndex{2}}
    count::Int
    threshold::Float64
end

"""
    detect_cosmic_rays(m::TimeResolvedMatrix; threshold=5.0, row_fraction_limit=0.25) -> CosmicRayMatrixResult

Detect cosmic ray hits in a 2D time × wavelength image.

Computes the residual of the image over its 3×3 median filter (which smooths
isolated spikes while preserving gradual spatial/temporal structure). The
residuals are then **Poisson-scaled** before the noise-floor computation: each
residual is divided by `sqrt(max(filtered, 1))`, so shot noise in bright bands
does not inflate the scale estimate. The `max(·, 1)` floor makes the scaling a
no-op on ΔA-scale data (where all values are small), and Poisson-correct for
photon-counting data (where `filtered ≈ N` counts). The noise scale is estimated
from the **positive scaled residuals only**: their median `med₊` and MAD `MAD₊`.
A pixel is flagged when

    scaled_resid > med₊ + threshold × MAD₊ / 0.6745

Only positive outliers are flagged — cosmic rays deposit charge, so they are
always bright.

# Arguments
- `m`: TimeResolvedMatrix with 2D data array `(n_time, n_wavelength)`
- `threshold`: outlier cutoff in MAD-scaled units (default 5.0). Lower values
  detect more spikes but may flag real features.
- `row_fraction_limit`: rows with more than this fraction of flagged channels
  are treated as real temporal structure, not cosmic rays (default 0.25).
  Run detection on unbinned data; on very narrow matrices (few wavelength columns)
  raise `row_fraction_limit`, since wide cosmic-ray streaks or hot rows spanning
  more than that fraction of a row are treated as temporal structure and left
  unflagged.

# Returns
A [`CosmicRayMatrixResult`](@ref) with the flagged pixel indices.

# Limitations
Returns zero detections when positive residuals are absent or have zero spread
(e.g. a near-flat background with fewer than ~3 distinct positive residual
values) — there is no noise floor to measure against in that regime.
"""
function detect_cosmic_rays(m::TimeResolvedMatrix; threshold::Real=5.0,
                            row_fraction_limit::Real=0.25)
    filtered = _median_filter3(m.data)
    # Poisson-aware scaling: counts data has sqrt(N) shot noise, so normalize
    # residuals by the local noise scale. The max(·, 1) floor makes this a
    # no-op for small-amplitude (e.g. ΔA) data.
    resid = (m.data .- filtered) ./ sqrt.(max.(filtered, 1.0))
    # Noise floor from POSITIVE residuals only: MAD over all residuals collapses
    # to 0 on smooth data (the median filter reproduces the background exactly,
    # so >50% of residuals are identically zero), which would disable detection
    # entirely. Cosmic rays are always positive, so positive residuals carry all
    # the signal of interest.
    positive_resid = filter(>(0.0), vec(resid))
    if isempty(positive_resid)
        return CosmicRayMatrixResult(CartesianIndex{2}[], 0, float(threshold))
    end
    med = median(positive_resid)
    mad_val = median(abs.(positive_resid .- med))
    if mad_val < eps(Float64)
        return CosmicRayMatrixResult(CartesianIndex{2}[], 0, float(threshold))
    end
    # Flag pixels where the residual is a large positive outlier
    flagged = findall(@. resid > med + Float64(threshold) * (mad_val / 0.6745))
    # Sharp temporal features (e.g. an IRF-limited rise at time zero) flag
    # entire time rows across wavelength; cosmic rays hit isolated pixels.
    # Drop rows where more than `row_fraction_limit` of channels are flagged.
    if !isempty(flagged)
        ncol = size(m.data, 2)
        row_counts = Dict{Int,Int}()
        for I in flagged
            row_counts[I[1]] = get(row_counts, I[1], 0) + 1
        end
        bad_rows = Set(r for (r, c) in row_counts if c > row_fraction_limit * ncol)
        isempty(bad_rows) || (flagged = [I for I in flagged if I[1] ∉ bad_rows])
    end
    return CosmicRayMatrixResult(flagged, length(flagged), float(threshold))
end

"""
    remove_cosmic_rays(m::TimeResolvedMatrix, result::CosmicRayMatrixResult) -> TimeResolvedMatrix

Replace flagged pixels with the median of their non-flagged 3×3 neighbors.
If all 8 neighbors of a flagged pixel are themselves flagged, falls back to
the global median of the image. Records `:cosmic_rays_removed` in metadata.

Returns a new TimeResolvedMatrix with cleaned data; does not mutate the input.
"""
function remove_cosmic_rays(m::TimeResolvedMatrix, result::CosmicRayMatrixResult)
    data = copy(m.data)
    flagged = Set(result.indices)
    nrow, ncol = size(data)
    for I in result.indices
        i, j = Tuple(I)
        neighbors = Float64[]
        for dj in -1:1, di in -1:1
            (di == 0 && dj == 0) && continue
            ni, nj = i + di, j + dj
            (1 <= ni <= nrow && 1 <= nj <= ncol) || continue
            CartesianIndex(ni, nj) in flagged && continue
            push!(neighbors, m.data[ni, nj])
        end
        data[I] = isempty(neighbors) ? median(m.data) : median(neighbors)
    end
    md = copy(m.metadata)
    md[:cosmic_rays_removed] = result.count
    return TimeResolvedMatrix(m.time, m.wavelength, data, md)
end
