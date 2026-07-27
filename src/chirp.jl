# Chirp correction for broadband transient absorption data
#
# GVD causes different probe wavelengths to arrive at different times,
# producing a diagonal "chirp" feature in the time-wavelength heatmap.
# This module provides calibration from a dedicated OKE run (the standard
# path), in-data detection (the fallback), and correction.

# =============================================================================
# ChirpCalibration type
# =============================================================================

"""
    ChirpCalibration

Stores the result of chirp detection: detected chirp points, polynomial fit,
and detection parameters for reproducibility.

# Fields
- `wavelength`: Wavelength points where chirp was detected (nm)
- `time_offset`: Detected chirp time at each wavelength (ps)
- `poly_coeffs`: Polynomial fit coefficients (constant term first, ascending order)
- `poly_order`: Polynomial order used
- `reference_λ`: Reference wavelength where chirp = 0 (nm)
- `r_squared`: How well the polynomial fits the points that were *detected*
- `metadata`: Detection parameters for reproducibility

!!! warning "R² does not measure detection accuracy"
    `r_squared` says only that a polynomial describes the detected points; it is
    silent on whether those points are the real chirp. Because R² is normalised
    by the variance of the points being fitted, a handful of extreme values pull
    it *up* — a curve bending to chase them scores well precisely because they
    are extreme. Judge detection by whether the offsets are resolvable (spread
    across several time steps, no pile-up at the search bound: see
    `metadata[:n_unmeasurable]`), not by R² alone.
"""
struct ChirpCalibration
    wavelength::Vector{Float64}
    time_offset::Vector{Float64}
    poly_coeffs::Vector{Float64}
    poly_order::Int
    reference_λ::Float64
    r_squared::Float64
    metadata::Dict{Symbol,Any}
end

"""
    polynomial(cal::ChirpCalibration) -> Function

Return a callable polynomial `t_shift = poly(λ)` from the calibration.
Coefficients are in ascending order: `c[1] + c[2]*λ + c[3]*λ² + ...`
"""
function polynomial(cal::ChirpCalibration)
    c = cal.poly_coeffs
    return λ -> _polyeval(c, λ)
end

function Base.show(io::IO, cal::ChirpCalibration)
    print(io, "ChirpCalibration: order $(cal.poly_order), R² = $(round(cal.r_squared, digits=4)), $(length(cal.wavelength)) points")
end

function Base.show(io::IO, ::MIME"text/plain", cal::ChirpCalibration)
    println(io, "ChirpCalibration")
    println(io, "  Polynomial order: ", cal.poly_order)
    println(io, "  Reference λ:      ", round(cal.reference_λ, digits=1), " nm")
    println(io, "  R²:               ", round(cal.r_squared, digits=6))
    println(io, "  Detection points:  ", length(cal.wavelength))
    print(io,   "  Coefficients:      ", [round(c, digits=6) for c in cal.poly_coeffs])
end

# =============================================================================
# Background subtraction
# =============================================================================

"""
    subtract_background(matrix::TimeResolvedMatrix; t_range=nothing) -> TimeResolvedMatrix

Subtract pre-pump background from a TA matrix by averaging and removing
the signal in the baseline region (before pump arrival).

NaN samples inside the baseline window are skipped: each wavelength column's
baseline is the mean of its finite values only. This keeps the correction
usable after [`correct_chirp`](@ref), which NaN-fills samples shifted past
the time-axis edges — for negative shifts that lands at the start of the
axis, inside typical pre-pump windows. A column with no finite values in
the window gets a NaN baseline (the whole column becomes NaN) and a
warning is emitted.
"""
function subtract_background(matrix::TimeResolvedMatrix; t_range::Union{Tuple,Nothing}=nothing)
    time = matrix.time
    data = matrix.data

    if isnothing(t_range)
        t_range = _auto_baseline_range(time, data)
    end

    # Find time indices in the baseline range
    mask = (time .>= t_range[1]) .& (time .<= t_range[2])
    n_baseline = sum(mask)
    if n_baseline < 2
        @warn "Only $n_baseline baseline points found in t_range=$t_range. Using first 5 rows."
        mask = falses(length(time))
        mask[1:min(5, length(time))] .= true
    end

    # Average baseline rows per wavelength column, skipping NaNs: correct_chirp
    # NaN-fills samples shifted past the time-axis edges, and for negative
    # shifts those land at the start of the axis — inside typical pre-pump
    # windows. A plain mean would turn one NaN row into an all-NaN column.
    baseline = map(eachcol(view(data, mask, :))) do col
        finite = filter(!isnan, col)
        isempty(finite) ? NaN : mean(finite)
    end
    n_allnan = count(isnan, baseline)
    if n_allnan > 0
        @warn "Baseline window t_range=$t_range is all-NaN for $n_allnan of $(length(baseline)) wavelength columns; those columns stay NaN"
    end

    # Subtract from every row
    corrected = data .- baseline'

    metadata = copy(matrix.metadata)
    metadata[:background_subtracted] = true
    metadata[:baseline_t_range] = t_range

    return TimeResolvedMatrix(copy(time), copy(matrix.wavelength), corrected, metadata)
end

"""
Auto-detect baseline region: everything before 80% of the way to signal onset.
Signal onset is found via the maximum of the column-averaged absolute gradient.
NaN samples (e.g. chirp-shifted edge rows) are skipped so they cannot blank
the onset search.
"""
function _auto_baseline_range(time, data)
    # Average absolute signal across all wavelengths, skipping NaNs
    avg_signal = map(eachrow(data)) do row
        finite = filter(!isnan, row)
        isempty(finite) ? NaN : mean(abs, finite)
    end

    # Gradient along time; a NaN gradient (all-NaN neighbor row) cannot win
    grad = diff(avg_signal)
    scores = [isnan(g) ? -Inf : abs(g) for g in grad]

    # Signal onset = time of maximum gradient
    onset_idx = argmax(scores)

    # Baseline ends at 80% of the way to onset
    baseline_end_idx = max(1, round(Int, 0.8 * onset_idx))

    return (time[1], time[baseline_end_idx])
end

# =============================================================================
# Chirp detection
# =============================================================================

"""
    detect_chirp(matrix::TimeResolvedMatrix; kwargs...) -> ChirpCalibration

Detect chirp (GVD) in a broadband TA matrix via cross-correlation (`:xcorr`)
or threshold crossing (`:threshold`). Returns a `ChirpCalibration` with polynomial fit.
"""
function detect_chirp(matrix::TimeResolvedMatrix;
    method::Symbol=:xcorr,
    order::Int=3,
    smooth_window::Int=15,
    reference::Union{Real,Symbol}=:center,
    threshold::Real=3.0,
    bin_width::Int=8,
    onset_frac::Real=0.5,
    min_signal::Real=0.2,
    t_range::Union{Tuple,Nothing}=nothing)

    time = matrix.time
    wavelength = matrix.wavelength
    data = matrix.data
    n_time, n_wl = size(data)

    # Input validation
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    bin_width >= 1 || throw(ArgumentError("bin_width must be >= 1, got $bin_width"))
    n_wl >= bin_width || throw(ArgumentError("bin_width ($bin_width) must be <= number of wavelengths ($n_wl)"))
    0 < min_signal <= 1 || throw(ArgumentError("min_signal must be in (0, 1], got $min_signal"))
    threshold > 0 || throw(ArgumentError("threshold must be positive, got $threshold"))
    if method === :threshold
        0 < onset_frac < 1 || throw(ArgumentError("onset_frac must be in (0, 1), got $onset_frac"))
    end

    _is_uniform(time) || @warn "detect_chirp assumes an evenly-spaced time axis; the detected axis is non-uniform, so the index-to-time (lag·dt) conversion may be inaccurate."

    # Ensure smooth_window is odd (required for SG filter)
    if !isodd(smooth_window)
        smooth_window += 1
        @info "Rounding smooth_window to $smooth_window (must be odd)"
    end

    # Detect chirp points using selected method
    n_unmeasurable = 0
    if method === :xcorr
        binned_wl, binned_chirp_times, n_unmeasurable = _detect_chirp_xcorr(
            time, wavelength, data, smooth_window, bin_width, min_signal)
        if n_unmeasurable > 0
            n_tried = n_unmeasurable + length(binned_wl)
            @warn "Discarded $n_unmeasurable of $n_tried wavelength bins with no measurable " *
                  "shift (correlation peak on the ±max_lag search bound, or a flat trace). " *
                  "A large fraction usually means the time axis is too coarse to resolve " *
                  "the chirp, or the spectral edges are noise-dominated."
        end
    elseif method === :threshold
        if isnothing(t_range)
            t_range = (time[1], time[end])
        end
        binned_wl, binned_chirp_times = _detect_chirp_threshold(
            time, wavelength, data, smooth_window, bin_width, t_range, onset_frac, min_signal)
    else
        throw(ArgumentError("Unknown chirp detection method: :$method. Use :xcorr or :threshold."))
    end

    # Determine reference wavelength
    ref_λ = reference === :center ? (minimum(wavelength) + maximum(wavelength)) / 2 : Float64(reference)

    # Fit polynomial with outlier rejection
    clean_wl, clean_times, coeffs, r2 = _fit_chirp_polynomial(
        binned_wl, binned_chirp_times, order, threshold, ref_λ)

    metadata = Dict{Symbol,Any}(
        :method => method,
        :order => order,
        :smooth_window => smooth_window,
        :mad_threshold => threshold,
        :bin_width => bin_width,
        :min_signal => min_signal,
        :n_points_raw => length(binned_wl),
        :n_points_clean => length(clean_wl),
        :n_outliers => length(binned_wl) - length(clean_wl),
        :n_unmeasurable => n_unmeasurable
    )

    if method === :threshold
        metadata[:onset_frac] = onset_frac
        metadata[:t_range] = something(t_range, (time[1], time[end]))
    end

    return ChirpCalibration(clean_wl, clean_times, coeffs, order, ref_λ, r2, metadata)
end

"""
Detect chirp via cross-correlation of absolute gradients.

Each bin's smoothed signal is differentiated and the absolute gradient is
cross-correlated against the reference (strongest signal). The onset gradient
spike is polarity-independent (works for both ESA and GSB), so this handles
mixed spectral regions. Parabolic interpolation gives sub-time-step precision.

Only the onset region is used (from data start to just past the signal peak)
to prevent later dynamics from dominating.

Bins whose lag is unmeasurable (see [`_xcorr_peak`](@ref)) are dropped rather
than fitted. Returns `(wavelengths, offsets, n_unmeasurable)`.
"""
function _detect_chirp_xcorr(time, wavelength, data, smooth_window, bin_width, min_signal)
    n_time, n_wl = size(data)
    dt = time[2] - time[1]
    global_max = maximum(abs.(data))

    # Window to onset region: start of data to just past the signal peak.
    avg_abs = vec(mean(abs.(data), dims=2))
    peak_idx = argmax(avg_abs)
    margin = max(1, n_time ÷ 10)
    window_end = min(n_time, peak_idx + margin)
    w_indices = 1:window_end

    # Bin, smooth, and compute absolute gradients (onset region only)
    n_bins = n_wl ÷ bin_width
    binned_grads = Vector{Vector{Float64}}(undef, n_bins)
    binned_wl = Vector{Float64}(undef, n_bins)
    bin_strength = Vector{Float64}(undef, n_bins)

    for b in 1:n_bins
        col_start = (b - 1) * bin_width + 1
        col_end = b * bin_width
        binned_wl[b] = mean(wavelength[col_start:col_end])
        col = vec(mean(data[w_indices, col_start:col_end], dims=2))

        win = min(smooth_window, length(col))
        win = isodd(win) ? win : win - 1
        if win >= 5
            col = _sg_filter(col, win, 2).y
        end

        # Absolute gradient: onset spike is positive regardless of ESA/GSB
        binned_grads[b] = abs.(diff(col))
        bin_strength[b] = maximum(abs.(col))
    end

    # Reference: strongest signal bin
    ref_idx = argmax(bin_strength)
    ref_grad = binned_grads[ref_idx]

    # Cross-correlate absolute gradients
    max_lag = length(w_indices) ÷ 4
    valid_wl = Float64[]
    offsets = Float64[]
    n_unmeasurable = 0

    for b in 1:n_bins
        if bin_strength[b] < min_signal * global_max
            continue
        end

        lag = _xcorr_peak(ref_grad, binned_grads[b], max_lag)
        if isnothing(lag)
            n_unmeasurable += 1
            continue
        end

        push!(valid_wl, binned_wl[b])
        push!(offsets, lag * dt)
    end

    return valid_wl, offsets, n_unmeasurable
end

"""
Compute the lag (with sub-sample parabolic interpolation) that maximizes the
absolute normalized cross-correlation between `ref` and `col`, or `nothing` when
no lag is measurable.

`nothing` is distinct from a measured lag of zero, and arises two ways:

- one of the traces is flat, so every lag correlates equally badly; or
- the correlation peak sits on the ±`max_lag` boundary, meaning the true peak
  lies outside the searched range or the correlation is noise.

Returning the boundary instead would pass the search limit off as a measurement.
Being by construction an extreme value, it then survives MAD rejection (the fit
bends toward it, shrinking its residual) and inflates R².
"""
function _xcorr_peak(ref, col, max_lag)
    n = length(ref)

    # Zero-mean; scale is handled per lag below
    ref_m = ref .- mean(ref)
    col_m = col .- mean(col)

    if sum(abs2, ref_m) < eps() || sum(abs2, col_m) < eps()
        return nothing
    end

    # Per-lag energy-normalized cross-correlation over the overlap window.
    # Dividing by the overlap count n − |lag| biases the peak toward large
    # lags (fewer samples inflate the average); normalizing each lag by the
    # energies inside its own overlap window removes both that bias and the
    # tilt the global mean subtraction leaves in the raw sums.
    n_lags = 2 * max_lag + 1
    corr = Vector{Float64}(undef, n_lags)

    for (i, lag) in enumerate(-max_lag:max_lag)
        s = 0.0
        e_ref = 0.0
        e_col = 0.0
        for t in max(1, 1 - lag):min(n, n - lag)
            r = ref_m[t]
            c = col_m[t + lag]
            s += r * c
            e_ref += r * r
            e_col += c * c
        end
        corr[i] = (e_ref > eps() && e_col > eps()) ? s / sqrt(e_ref * e_col) : 0.0
    end

    # Find peak of |correlation|
    abs_corr = abs.(corr)
    peak_i = argmax(abs_corr)

    # Peak pinned to either end of the lag window: unmeasurable, not a lag of
    # ±max_lag. This also leaves the interpolation below with both neighbours.
    (peak_i == 1 || peak_i == n_lags) && return nothing

    best_lag = peak_i - max_lag - 1

    # Parabolic interpolation for sub-sample precision: the vertex of the
    # parabola through (−1, y_m), (0, y₀), (+1, y_p) sits at
    # (y_p − y_m) / (2(2y₀ − y_m − y_p)).
    y_m = abs_corr[peak_i - 1]
    y_0 = abs_corr[peak_i]
    y_p = abs_corr[peak_i + 1]
    denom = 2 * (2 * y_0 - y_m - y_p)
    if abs(denom) > eps()
        delta = (y_p - y_m) / denom
        return best_lag + delta
    end

    return Float64(best_lag)
end

# -----------------------------------------------------------------------------

"""
Detect chirp via threshold crossing (half-maximum onset detection).

For each wavelength bin, finds the first time the smoothed absolute signal
exceeds `onset_frac` of that column's maximum.

Bins with maximum absolute signal below `min_signal` fraction of the global
maximum are skipped.
"""
function _detect_chirp_threshold(time, wavelength, data, smooth_window, bin_width, t_range, onset_frac, min_signal)
    n_time, n_wl = size(data)

    # Find time indices within the search window
    t_mask = (time .>= t_range[1]) .& (time .<= t_range[2])
    t_indices = findall(t_mask)
    if length(t_indices) < 3
        @warn "Chirp search window contains only $(length(t_indices)) time points. Expanding to full range."
        t_indices = collect(1:n_time)
    end

    # Global maximum for signal strength filtering
    global_max = maximum(abs.(data))

    # Bin wavelengths
    n_bins = n_wl ÷ bin_width
    binned_wl = Float64[]
    binned_times = Float64[]

    for b in 1:n_bins
        col_start = (b - 1) * bin_width + 1
        col_end = b * bin_width
        λ_center = mean(wavelength[col_start:col_end])

        # Average signal across the bin (full time axis for strength check)
        col_full = vec(mean(data[:, col_start:col_end], dims=2))

        # Skip bins with weak signal
        if maximum(abs.(col_full)) < min_signal * global_max
            continue
        end

        # Restrict to search window
        col_avg = vec(mean(data[t_indices, col_start:col_end], dims=2))

        # Smooth signal with Savitzky-Golay
        win = min(smooth_window, length(col_avg))
        win = isodd(win) ? win : win - 1
        if win >= 5
            col_smooth = _sg_filter(col_avg, win, 2).y
        else
            col_smooth = col_avg
        end

        # Threshold crossing on absolute smoothed signal
        abs_smooth = abs.(col_smooth)
        max_val = maximum(abs_smooth)
        threshold = onset_frac * max_val

        onset_idx = findfirst(x -> x > threshold, abs_smooth)
        if isnothing(onset_idx)
            onset_idx = argmax(abs_smooth)
        end

        # Map back to global time index
        global_idx = t_indices[onset_idx]
        chirp_time = time[global_idx]

        push!(binned_wl, λ_center)
        push!(binned_times, chirp_time)
    end

    return binned_wl, binned_times
end

"""
Fit polynomial to chirp points with MAD-based outlier rejection.
Returns (clean_wl, clean_times, coefficients, r_squared, ref_shift, keep):
`ref_shift` is the absolute polynomial value at `ref_λ` before the coefficients
are normalized to zero there (callers store it as `:t0_at_reference`), and
`keep` is the Boolean mask of input points that survived outlier rejection.
"""
function _fit_chirp_polynomial(wl, times, order, threshold, ref_λ)
    length(wl) > order || throw(ArgumentError(
        "Need more than $order points to fit order-$order polynomial, got $(length(wl))"))

    # First pass: fit polynomial
    coeffs = _polyfit(wl, times, order)
    residuals = times .- _polyeval(coeffs, wl)

    # MAD-based outlier rejection
    med_res = median(residuals)
    mad = median(abs.(residuals .- med_res))
    mad_scaled = MAD_TO_SIGMA * mad

    if mad_scaled > 0
        keep = abs.(residuals .- med_res) .<= threshold * mad_scaled
    else
        keep = trues(length(wl))
    end

    clean_wl = wl[keep]
    clean_times = times[keep]

    length(clean_wl) > order || throw(ArgumentError(
        "Only $(length(clean_wl)) points survived outlier rejection; need more than $order for order-$order polynomial"))

    # Second pass: refit on clean data
    coeffs = _polyfit(clean_wl, clean_times, order)

    # Shift so polynomial is zero at reference wavelength
    ref_shift = _polyeval(coeffs, ref_λ)
    coeffs[1] -= ref_shift
    clean_times = clean_times .- ref_shift

    # R². Zero ss_tot means every bin came back with the same offset: there is no
    # wavelength-dependent timing to explain, so R² is 0/0 — undefined, not 1.
    # Reporting a perfect fit here would flag the flattest possible result as the
    # best one. NaN is the honest value and fails any `r2 > threshold` check.
    # Both a chirp-free matrix and a total detection failure land here and are
    # indistinguishable from ss_tot alone, so warn rather than throw.
    fitted = _polyeval(coeffs, clean_wl)
    ss_res = sum((clean_times .- fitted).^2)
    ss_tot = sum((clean_times .- mean(clean_times)).^2)
    if ss_tot <= 0
        @warn "All $(length(clean_times)) wavelength bins returned the same time offset, " *
              "so R² is undefined (0/0) and is reported as NaN. Either the data carries no " *
              "chirp, or detection failed: with method=:threshold on a matrix that still " *
              "has its pre-pump background, every bin crosses the threshold at the first " *
              "sample. The calibration is flat, so correcting with it is a no-op."
        return clean_wl, clean_times, coeffs, NaN, ref_shift, keep
    end
    r2 = 1.0 - ss_res / ss_tot

    return clean_wl, clean_times, coeffs, r2, ref_shift, keep
end

# =============================================================================
# Chirp calibration from a dedicated OKE cross-correlation run
# =============================================================================

"""
    calibrate_chirp(oke::TimeResolvedMatrix; kwargs...) -> ChirpCalibration

Measure a chirp calibration from a dedicated OKE (optical Kerr effect)
cross-correlation run.

The electronic Kerr response of a non-resonant medium is effectively
instantaneous, so the per-wavelength peak of the pump–probe cross-correlation
traces the true t₀(λ) independent of any sample dynamics. This makes an OKE
run the standard source of chirp calibrations; use [`detect_chirp`](@ref),
which infers the curve from a sample matrix's own signal onsets, only when no
OKE run is available.

For each wavelength column the peak position is located with `method`:

- `:gaussian` (default): least-squares fit of a Gaussian with vertical offset,
  `p = [A, t₀, σ, y₀]`. Also yields the per-wavelength IRF width σ (a standard
  deviation, not a FWHM), stored in `metadata[:irf_sigma]`.
- `:peak`: argmax plus parabolic sub-sample interpolation. Fast path for
  high-SNR runs; provides no IRF widths.

Columns whose peak-to-peak amplitude is below `min_amplitude` of the strongest
column's, and columns whose per-column fit fails, are masked out. The surviving
(λ, t₀) points are fitted with an order-`order` polynomial with MAD outlier
rejection (shared with [`detect_chirp`](@ref)). The stored polynomial is
normalized to zero at `reference_λ`; the absolute overlap time there is kept in
`metadata[:t0_at_reference]`.

# Keywords
- `method::Symbol = :gaussian` — `:gaussian` or `:peak` (see above)
- `order::Int = 3` — chirp polynomial order
- `reference = :center` — wavelength (nm) where the polynomial is zero;
  `:center` uses the midpoint of the surviving wavelength coverage
- `wl_range = nothing` — `(λmin, λmax)`: restrict which columns are considered
- `t_range = nothing` — `(tmin, tmax)`: restrict the per-column fit window
- `min_amplitude::Real = 0.05` — amplitude mask threshold as a fraction of the
  strongest column's peak-to-peak amplitude; `0` disables the mask
- `source = nothing` — provenance string stored as `metadata[:source_file]`;
  defaults to the OKE matrix's own `metadata[:source]` when present

# Metadata
The returned calibration's `metadata` carries provenance and fit bookkeeping:
`:source => :oke`, `:source_file`, `:lambda_range` (the kept wavelength
coverage — [`correct_chirp`](@ref) clamps its polynomial evaluation to this
window), `:t0_at_reference`, `:irf_sigma` (`:gaussian` only, aligned with
`cal.wavelength`), plus `:method`, `:order`, `:mad_threshold`, point counts,
and the `wl_range`/`t_range` restrictions when given.

# Examples
```julia
oke = ...  # TimeResolvedMatrix from the OKE run (solvent/glass at sample position)
cal = calibrate_chirp(oke)
corrected = correct_chirp(sample_matrix, cal)
```
"""
function calibrate_chirp(oke::TimeResolvedMatrix;
    method::Symbol=:gaussian,
    order::Int=3,
    reference::Union{Real,Symbol}=:center,
    wl_range::Union{Tuple,Nothing}=nothing,
    t_range::Union{Tuple,Nothing}=nothing,
    min_amplitude::Real=0.05,
    source::Union{AbstractString,Nothing}=nothing)

    method in (:gaussian, :peak) || throw(ArgumentError(
        "Unknown calibration method: :$method. Use :gaussian or :peak."))
    order >= 1 || throw(ArgumentError("order must be >= 1, got $order"))
    0 <= min_amplitude < 1 || throw(ArgumentError(
        "min_amplitude must be in [0, 1), got $min_amplitude"))
    if !isnothing(wl_range)
        wl_range[1] < wl_range[2] || throw(ArgumentError(
            "wl_range must be ordered (λmin, λmax), got $wl_range"))
    end
    if !isnothing(t_range)
        t_range[1] < t_range[2] || throw(ArgumentError(
            "t_range must be ordered (tmin, tmax), got $t_range"))
    end
    if reference !== :center && !(reference isa Real)
        throw(ArgumentError(
            "reference must be :center or a wavelength in nm, got $reference"))
    end

    time = oke.time
    wavelength = oke.wavelength
    data = oke.data

    col_idx = isnothing(wl_range) ? collect(eachindex(wavelength)) :
        findall(λ -> wl_range[1] <= λ <= wl_range[2], wavelength)
    isempty(col_idx) && throw(ArgumentError(
        "wl_range $wl_range contains no wavelength columns " *
        "(axis spans $(extrema(wavelength)))"))

    row_idx = isnothing(t_range) ? collect(eachindex(time)) :
        findall(t -> t_range[1] <= t <= t_range[2], time)
    length(row_idx) >= 4 || throw(ArgumentError(
        "the" * (isnothing(t_range) ? " " : " t_range $t_range ") *
        "fit window contains only $(length(row_idx)) time samples; " *
        "need at least 4 to locate a peak"))

    t_win = time[row_idx]
    span = t_win[end] - t_win[1]

    # Peak-to-peak amplitude per column: polarity- and offset-independent
    amps = [let col = @view(data[row_idx, j])
                maximum(col) - minimum(col)
            end for j in col_idx]
    amp_max = maximum(amps)
    amp_max > 0 || throw(ArgumentError(
        "OKE matrix is flat over the selected window; no cross-correlation peaks to calibrate on"))
    strong = amps .>= min_amplitude * amp_max

    kept_wl = Float64[]
    kept_t0 = Float64[]
    kept_sigma = Float64[]
    n_failed = 0

    for (k, j) in enumerate(col_idx)
        strong[k] || continue
        col = data[row_idx, j]
        result = method === :gaussian ? _oke_gaussian_t0(t_win, col, span) :
                                        _oke_peak_t0(t_win, col)
        if isnothing(result)
            n_failed += 1
            continue
        end
        t0, σ = result
        push!(kept_wl, wavelength[j])
        push!(kept_t0, t0)
        method === :gaussian && push!(kept_sigma, σ)
    end

    n_kept = length(kept_wl)
    n_kept >= order + 2 || throw(ArgumentError(
        "Only $n_kept of $(length(col_idx)) wavelength columns survived amplitude " *
        "masking ($(count(!, strong)) weak) and per-column peak fitting ($n_failed " *
        "failed); need at least $(order + 2) for an order-$order chirp polynomial. " *
        "Lower min_amplitude or order, or widen wl_range/t_range."))

    ref_λ = reference === :center ? (minimum(kept_wl) + maximum(kept_wl)) / 2 :
                                    Float64(reference)

    clean_wl, clean_times, coeffs, r2, ref_shift, keep = _fit_chirp_polynomial(
        kept_wl, kept_t0, order, 3.0, ref_λ)

    src = isnothing(source) ? string(get(oke.metadata, :source, "")) : String(source)

    metadata = Dict{Symbol,Any}(
        :source => :oke,
        :source_file => src,
        :method => method,
        :order => order,
        :min_amplitude => min_amplitude,
        :mad_threshold => 3.0,
        :lambda_range => (minimum(clean_wl), maximum(clean_wl)),
        :t0_at_reference => ref_shift,
        :n_columns => length(col_idx),
        :n_weak => count(!, strong),
        :n_fit_failed => n_failed,
        :n_points_raw => n_kept,
        :n_points_clean => length(clean_wl),
        :n_outliers => n_kept - length(clean_wl),
    )
    isnothing(wl_range) || (metadata[:wl_range] = wl_range)
    isnothing(t_range) || (metadata[:t_range] = t_range)
    method === :gaussian && (metadata[:irf_sigma] = kept_sigma[keep])

    return ChirpCalibration(clean_wl, clean_times, coeffs, order, ref_λ, r2, metadata)
end

"""
Locate the cross-correlation peak in one OKE column by fitting a Gaussian with
offset, `p = [A, t₀, σ, y₀]`. Returns `(t₀, σ)` with σ a standard deviation,
or `nothing` when the fit fails or lands outside the fitted window (a center
off the data is a runaway on a noise column, not a measurement — the same
refuse-to-guess rule as [`_xcorr_peak`](@ref)).
"""
function _oke_gaussian_t0(t, col, span)
    y0_init = median(col)
    dev = col .- y0_init
    i_pk = argmax(abs.(dev))
    A_init = dev[i_pk]
    A_init == 0 && return nothing
    dt_mean = span / (length(t) - 1)
    n_half = count(d -> abs(d) >= abs(A_init) / 2, dev)
    σ_init = max(n_half * dt_mean / FWHM_FACTOR, dt_mean)

    sol = try
        solve(NonlinearCurveFitProblem(gaussian, [A_init, t[i_pk], σ_init, y0_init], t, col))
    catch
        return nothing
    end
    isconverged(sol) || return nothing

    A, t0, σ, _ = coef(sol)
    σ = abs(σ)  # the model only uses σ², so the sign is unconstrained
    all(isfinite, (A, t0, σ)) || return nothing
    t[1] <= t0 <= t[end] || return nothing
    0 < σ <= span || return nothing
    return (t0, σ)
end

"""
Locate the cross-correlation peak in one OKE column as the argmax of the
offset-corrected absolute signal, refined by parabolic interpolation through
the three samples around it (the technique [`_xcorr_peak`](@ref) uses on the
correlation function). Returns `(t₀, NaN)` — no width estimate — or `nothing`
when the peak sits on the window edge, where the true maximum may lie outside.
"""
function _oke_peak_t0(t, col)
    a = abs.(col .- median(col))
    i = argmax(a)
    (i == firstindex(a) || i == lastindex(a)) && return nothing
    t0 = _parabolic_vertex(t[i-1], t[i], t[i+1], a[i-1], a[i], a[i+1])
    return (t0, NaN)
end

"""
Vertex x-position of the parabola through three points. Works on arbitrary
(non-uniform) x spacing; falls back to `x2` when the points are collinear.
When `y2` is a strict local maximum the vertex lies inside `(x1, x3)`.
"""
function _parabolic_vertex(x1, x2, x3, y1, y2, y3)
    denom = y1 * (x2 - x3) + y2 * (x3 - x1) + y3 * (x1 - x2)
    abs(denom) <= eps(max(abs(y1), abs(y2), abs(y3), 1.0)) && return x2
    num = y1 * (x2^2 - x3^2) + y2 * (x3^2 - x1^2) + y3 * (x1^2 - x2^2)
    # With y2 the max of the three, the vertex lies in [x1, x3] exactly; the
    # clamp only absorbs floating-point noise in near-degenerate triples.
    return clamp(num / (2 * denom), min(x1, x3), max(x1, x3))
end

# =============================================================================
# Chirp correction
# =============================================================================

"""
Whether `t` is (approximately) evenly spaced. Cubic B-spline interpolation and
the lag→time conversion in chirp detection both assume a uniform time axis;
multi-segment / quasi-log delay axes are common in TA, so this gates the
fallbacks that keep those cases correct.
"""
function _is_uniform(t::AbstractVector; rtol::Real=1e-3)
    length(t) < 3 && return true
    dt = t[2] - t[1]
    dt == 0 && return false
    return maximum(abs.(diff(t) .- dt)) <= rtol * abs(dt)
end

"""
    correct_chirp(matrix::TimeResolvedMatrix, cal::ChirpCalibration) -> TimeResolvedMatrix

Apply chirp correction by interpolating each wavelength column onto its
`t_shift(λ)`-shifted time axis from the calibration polynomial. Uses cubic
B-splines on a uniform time axis; for a non-uniform axis it falls back to
gridded-linear interpolation against the actual sample times (cubic B-splines
require evenly-spaced knots) so the correction stays quantitatively correct.

Shifted samples that fall outside a column's measured time range are `NaN`:
no data was recorded there, and extrapolated values would masquerade as
signal. Keep the acquisition window generous around t₀ so the correction
doesn't push wavelengths of interest off the edge.

When the calibration carries `metadata[:lambda_range]` — set by
[`calibrate_chirp`](@ref) to its measured wavelength coverage — the shift
polynomial is evaluated at `clamp(λ, λmin, λmax)`: outside the measured window
the shift is held flat at the endpoint value instead of extrapolating the
polynomial into wavelengths it never saw. Calibrations from
[`detect_chirp`](@ref) don't set the key and are evaluated everywhere.
"""
function correct_chirp(matrix::TimeResolvedMatrix, cal::ChirpCalibration)
    time = matrix.time
    wavelength = matrix.wavelength
    data = matrix.data
    n_time, n_wl = size(data)

    poly = polynomial(cal)

    # Clamp evaluation to the calibration's measured wavelength coverage when
    # it records one (OKE runs are run-specific: a calibration measured on one
    # spectrograph window may be applied to a matrix that extends past it).
    lr = get(cal.metadata, :lambda_range, nothing)
    shift_at = if isnothing(lr)
        poly
    else
        λ_lo, λ_hi = minmax(Float64(first(lr)), Float64(last(lr)))
        λ -> poly(clamp(λ, λ_lo, λ_hi))
    end

    corrected = similar(data)

    uniform = _is_uniform(time)
    t_grid = range(time[1], time[end], length=n_time)

    for j in eachindex(wavelength)
        t_shift = shift_at(wavelength[j])

        # Interpolate this column, then read it at the shifted sample times.
        # Out-of-domain samples fill with NaN, never extrapolated values.
        col = @view data[:, j]
        if uniform
            itp = interpolate(col, BSpline(Cubic(Line(OnGrid()))))
            eitp = extrapolate(scale(itp, t_grid), NaN)
        else
            eitp = extrapolate(interpolate((time,), col, Gridded(Linear())), NaN)
        end

        # Evaluate at shifted time points (t + t_shift) inline, no allocation
        for i in eachindex(time)
            corrected[i, j] = eitp(time[i] + t_shift)
        end
    end

    metadata = copy(matrix.metadata)
    metadata[:chirp_corrected] = true
    metadata[:chirp_calibration] = cal

    return TimeResolvedMatrix(copy(time), copy(wavelength), corrected, metadata)
end

# =============================================================================
# Serialization (JSON)
# =============================================================================

"""
    save_chirp(path::String, cal::ChirpCalibration)

Save a chirp calibration to a JSON file.
"""
function save_chirp(path::String, cal::ChirpCalibration)
    d = Dict(
        "wavelength" => cal.wavelength,
        "time_offset" => cal.time_offset,
        "poly_coeffs" => cal.poly_coeffs,
        "poly_order" => cal.poly_order,
        "reference_lambda" => cal.reference_λ,
        # An undefined R² is NaN, which JSON has no literal for; `null` round-trips.
        "r_squared" => isnan(cal.r_squared) ? nothing : cal.r_squared,
        "metadata" => Dict(string(k) => v for (k, v) in cal.metadata)
    )
    open(path, "w") do io
        JSON.print(io, d, 2)
    end
end

"""
    load_chirp(path::String) -> ChirpCalibration

Load a chirp calibration from a JSON file.
"""
function load_chirp(path::String)
    d = JSON.parsefile(path)

    required = ("wavelength", "time_offset", "poly_coeffs", "poly_order", "reference_lambda", "r_squared", "metadata")
    for key in required
        haskey(d, key) || throw(ArgumentError("Malformed chirp JSON: missing required key \"$key\""))
    end

    metadata = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in d["metadata"])
    return ChirpCalibration(
        Float64.(d["wavelength"]),
        Float64.(d["time_offset"]),
        Float64.(d["poly_coeffs"]),
        Int(d["poly_order"]),
        Float64(d["reference_lambda"]),
        isnothing(d["r_squared"]) ? NaN : Float64(d["r_squared"]),
        metadata
    )
end

# =============================================================================
# SVD filtering for TA matrix denoising
# =============================================================================

"""
    svd_filter(matrix::TimeResolvedMatrix; n_components::Int=5) -> TimeResolvedMatrix

Denoise a TA matrix by keeping only the first `n_components` singular value
components. Higher-order components (dominated by noise) are discarded.

This is a standard preprocessing step for broadband TA data. Typical usage:
denoise first, then subtract background, detect chirp, and correct chirp.

# Arguments
- `matrix::TimeResolvedMatrix`: Input time × wavelength ΔA matrix

# Keywords
- `n_components::Int=5`: Number of singular value components to retain.
  Use [`singular_values`](@ref) to inspect the spectrum and choose.

# Returns
A new `TimeResolvedMatrix` with filtered data. Metadata includes `:svd_filtered => true`
and `:svd_n_components => n_components`.

# Examples
```julia
sv = singular_values(matrix)  # inspect singular value spectrum
filtered = svd_filter(matrix; n_components=3)
```
"""
function svd_filter(matrix::TimeResolvedMatrix; n_components::Int=5)
    n_time, n_wl = size(matrix.data)
    max_components = min(n_time, n_wl)
    n_components < 1 && throw(ArgumentError("n_components must be >= 1"))
    n_components > max_components && throw(ArgumentError(
        "n_components ($n_components) exceeds matrix rank ($max_components)"))

    F = svd(matrix.data)
    k = n_components
    filtered_data = @views F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]

    metadata = copy(matrix.metadata)
    metadata[:svd_filtered] = true
    metadata[:svd_n_components] = n_components

    return TimeResolvedMatrix(copy(matrix.time), copy(matrix.wavelength), filtered_data, metadata)
end

"""
    svd_filter(x::AbstractVector, y::AbstractVector, data::AbstractMatrix;
               n_components::Int=5) -> Matrix{Float64}

Denoise a raw data matrix by keeping only the first `n_components` singular
value components. Returns the filtered matrix.

# Arguments
- `x`: First axis (e.g., time)
- `y`: Second axis (e.g., wavelength)
- `data`: Matrix of size `(length(x), length(y))`

# Keywords
- `n_components::Int=5`: Number of components to retain
"""
function svd_filter(x::AbstractVector, y::AbstractVector, data::AbstractMatrix;
                    n_components::Int=5)
    size(data) == (length(x), length(y)) || throw(DimensionMismatch(
        "data size $(size(data)) doesn't match axes ($(length(x)), $(length(y)))"))
    max_components = min(size(data)...)
    n_components < 1 && throw(ArgumentError("n_components must be >= 1"))
    n_components > max_components && throw(ArgumentError(
        "n_components ($n_components) exceeds matrix rank ($max_components)"))

    F = svd(Float64.(data))
    k = n_components
    return @views F.U[:, 1:k] * Diagonal(F.S[1:k]) * F.Vt[1:k, :]
end

"""
    singular_values(matrix::TimeResolvedMatrix) -> Vector{Float64}

Return the singular values of the TA data matrix. Inspect these to choose
`n_components` for [`svd_filter`](@ref) — look for a gap between signal
and noise components.

# Examples
```julia
sv = singular_values(matrix)
# Plot sv to find the elbow, then filter:
filtered = svd_filter(matrix; n_components=3)
```
"""
function singular_values(matrix::TimeResolvedMatrix)
    return svd(matrix.data).S
end

"""
    singular_values(data::AbstractMatrix) -> Vector{Float64}

Return the singular values of a raw data matrix.
"""
function singular_values(data::AbstractMatrix)
    return svd(Float64.(data)).S
end

"""
    estimate_n_components(sv; ratio=10.0) -> Int

Estimate the number of significant components from a descending sequence of
singular values `sv` using an elbow heuristic: return the first index `i`
where `sv[i] / sv[i+1] > ratio`, marking a gap between signal and noise
components. Returns `length(sv)` when no such gap exists. Use with
[`singular_values`](@ref) to choose `n_components` for [`svd_filter`](@ref).
"""
function estimate_n_components(sv::AbstractVector; ratio::Real=10.0)
    for i in 1:(length(sv) - 1)
        if sv[i + 1] > 0 && sv[i] / sv[i + 1] > ratio
            return i
        end
    end
    return length(sv)
end
