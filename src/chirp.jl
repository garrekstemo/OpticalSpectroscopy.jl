# Chirp correction for broadband transient absorption data
#
# GVD causes different probe wavelengths to arrive at different times,
# producing a diagonal "chirp" feature in the time-wavelength heatmap.
# This module provides automatic detection and correction.

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
- `r_squared`: Polynomial fit quality
- `metadata`: Detection parameters for reproducibility
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

    # Average baseline rows per wavelength column
    baseline = vec(mean(data[mask, :], dims=1))

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
"""
function _auto_baseline_range(time, data)
    # Average absolute signal across all wavelengths
    avg_signal = vec(mean(abs.(data), dims=2))

    # Gradient along time
    grad = diff(avg_signal)

    # Signal onset = time of maximum gradient
    onset_idx = argmax(abs.(grad))

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
    if method === :xcorr
        binned_wl, binned_chirp_times = _detect_chirp_xcorr(
            time, wavelength, data, smooth_window, bin_width, min_signal)
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
        :n_outliers => length(binned_wl) - length(clean_wl)
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

    for b in 1:n_bins
        if bin_strength[b] < min_signal * global_max
            continue
        end

        lag = _xcorr_peak(ref_grad, binned_grads[b], max_lag)

        push!(valid_wl, binned_wl[b])
        push!(offsets, lag * dt)
    end

    return valid_wl, offsets
end

"""
Compute the lag (with sub-sample parabolic interpolation) that maximizes
the absolute normalized cross-correlation between `ref` and `col`.
"""
function _xcorr_peak(ref, col, max_lag)
    n = length(ref)

    # Normalize (zero mean, unit energy)
    ref_m = ref .- mean(ref)
    col_m = col .- mean(col)
    ref_e = sqrt(sum(ref_m .^ 2))
    col_e = sqrt(sum(col_m .^ 2))

    if ref_e < eps() || col_e < eps()
        return 0.0
    end

    ref_n = ref_m ./ ref_e
    col_n = col_m ./ col_e

    # Compute normalized cross-correlation at each lag
    n_lags = 2 * max_lag + 1
    corr = Vector{Float64}(undef, n_lags)

    for (i, lag) in enumerate(-max_lag:max_lag)
        s = 0.0
        count = 0
        for t in 1:n
            t2 = t + lag
            if 1 <= t2 <= n
                s += ref_n[t] * col_n[t2]
                count += 1
            end
        end
        corr[i] = count > 0 ? s / count : 0.0
    end

    # Find peak of |correlation|
    abs_corr = abs.(corr)
    peak_i = argmax(abs_corr)
    best_lag = peak_i - max_lag - 1

    # Parabolic interpolation for sub-sample precision
    if peak_i > 1 && peak_i < n_lags
        y_m = abs_corr[peak_i - 1]
        y_0 = abs_corr[peak_i]
        y_p = abs_corr[peak_i + 1]
        denom = 2 * (2 * y_0 - y_m - y_p)
        if abs(denom) > eps()
            delta = (y_m - y_p) / denom
            return best_lag + delta
        end
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
Returns (clean_wl, clean_times, coefficients, r_squared).
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
    mad_scaled = 1.4826 * mad  # Scale factor for normal distribution consistency

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

    # Compute R²
    fitted = _polyeval(coeffs, clean_wl)
    ss_res = sum((clean_times .- fitted).^2)
    ss_tot = sum((clean_times .- mean(clean_times)).^2)
    r2 = ss_tot > 0 ? 1.0 - ss_res / ss_tot : 1.0

    return clean_wl, clean_times, coeffs, r2
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
"""
function correct_chirp(matrix::TimeResolvedMatrix, cal::ChirpCalibration)
    time = matrix.time
    wavelength = matrix.wavelength
    data = matrix.data
    n_time, n_wl = size(data)

    poly = polynomial(cal)
    corrected = similar(data)

    uniform = _is_uniform(time)
    t_grid = range(time[1], time[end], length=n_time)

    for j in eachindex(wavelength)
        t_shift = poly(wavelength[j])

        # Interpolate this column, then read it at the shifted sample times.
        col = @view data[:, j]
        if uniform
            itp = interpolate(col, BSpline(Cubic(Line(OnGrid()))))
            eitp = extrapolate(scale(itp, t_grid), Flat())
        else
            eitp = extrapolate(interpolate((time,), col, Gridded(Linear())), Flat())
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
        "r_squared" => cal.r_squared,
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
        Float64(d["r_squared"]),
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
    S_filtered = copy(F.S)
    S_filtered[n_components+1:end] .= 0.0
    filtered_data = F.U * Diagonal(S_filtered) * F.Vt

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
    S_filtered = copy(F.S)
    S_filtered[n_components+1:end] .= 0.0
    return F.U * Diagonal(S_filtered) * F.Vt
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
