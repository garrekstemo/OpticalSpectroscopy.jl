# Core spectroscopy analysis functions

# ============================================================================
# BASIC UTILITY FUNCTIONS
# ============================================================================

"""
    normalize(x)

Normalize array by maximum absolute value. Returns zeros if max is zero.
"""
normalize(x) = (m = maximum(abs.(x)); m == 0 ? zero(x) : x ./ m)

"""
    time_index(times, t_target)

Find index closest to target time value.
"""
time_index(times, t_target) = _find_nearest_idx(times, t_target)

# ============================================================================
# TRANSMITTANCE <-> ABSORBANCE CONVERSIONS
# ============================================================================

"""
    transmittance_to_absorbance(T; percent=false)

Convert transmittance to absorbance: `A = -log10(T)`.
Input `T` is fractional (0 to 1). Use `percent=true` for percent transmittance.
"""
function transmittance_to_absorbance(T::Real; percent::Bool=false)
    T_frac = percent ? T / 100 : T
    T_frac > 0 || throw(ArgumentError("Transmittance must be positive (got $T_frac)"))
    return -log10(T_frac)
end

function transmittance_to_absorbance(T::AbstractVector; percent::Bool=false)
    return transmittance_to_absorbance.(T; percent=percent)
end

"""
    transmittance_to_absorbance(s::Spectrum; percent=false) -> Spectrum

Convert a transmittance [`Spectrum`](@ref) to absorbance. Sets
`metadata[:ylabel] = "Absorbance"` on the result, overwriting any prior value.
"""
function transmittance_to_absorbance(s::Spectrum; percent::Bool=false)
    md = copy(s.metadata)
    md[:ylabel] = "Absorbance"
    return Spectrum(s.x, transmittance_to_absorbance(s.y; percent=percent), md)
end

"""
    absorbance_to_transmittance(A; percent=false)

Convert absorbance to transmittance: `T = 10^(-A)`.
"""
function absorbance_to_transmittance(A::Real; percent::Bool=false)
    T = 10.0^(-A)
    return percent ? T * 100 : T
end

function absorbance_to_transmittance(A::AbstractVector; percent::Bool=false)
    return absorbance_to_transmittance.(A; percent=percent)
end

"""
    absorbance_to_transmittance(s::Spectrum; percent=false) -> Spectrum

Convert an absorbance [`Spectrum`](@ref) to transmittance. Sets
`metadata[:ylabel]` to `"Transmittance"` (or `"Transmittance (%)"` when
`percent=true`) on the result, overwriting any prior value.
"""
function absorbance_to_transmittance(s::Spectrum; percent::Bool=false)
    md = copy(s.metadata)
    md[:ylabel] = percent ? "Transmittance (%)" : "Transmittance"
    return Spectrum(s.x, absorbance_to_transmittance(s.y; percent=percent), md)
end

# ============================================================================
# SPECTRUM SUBTRACTION
# ============================================================================

"""
    subtract_spectrum(sample, reference; scale=1.0, interpolate=false)

Subtract a reference spectrum from a sample spectrum.

Accepts `AbstractSpectroscopyData` types (uses `xdata`/`ydata` interface)
or any objects with `.x` and `.y` fields.

Returns `(x=..., y=...)` NamedTuple.
"""
function subtract_spectrum(sample, reference; scale::Real=1.0, interpolate=false)
    a_y, b_y = _align_spectra((x=sample.x, y=sample.y), (x=reference.x, y=reference.y); interpolate)
    return (x=collect(sample.x), y=a_y .- scale .* b_y)
end

# Typed dispatch: AbstractSpectroscopyData → xdata/ydata interface
function subtract_spectrum(sample::AbstractSpectroscopyData,
                           reference::AbstractSpectroscopyData; kwargs...)
    _check_1d(sample, "subtract_spectrum"); _check_1d(reference, "subtract_spectrum")
    subtract_spectrum((x=xdata(sample), y=ydata(sample)),
                      (x=xdata(reference), y=ydata(reference)); kwargs...)
end

# ============================================================================
# SMOOTHING AND PEAK ANALYSIS
# ============================================================================

"""
    smooth_data(y; window=3)

Apply moving average smoothing to data.
"""
function smooth_data(y; window::Int=3)
    n = length(y)
    smoothed = similar(y)
    half_w = window ÷ 2

    for i in eachindex(y)
        left_extend = min(half_w, i - 1)
        right_extend = min(half_w, n - i)
        start_idx = i - left_extend
        end_idx = i + right_extend
        smoothed[i] = mean(@view y[start_idx:end_idx])
    end
    return smoothed
end

"""
    smooth_data(s::Spectrum; window=3) -> Spectrum

Moving-average smoothing of a [`Spectrum`](@ref). Returns a new `Spectrum`
with shallow-copied metadata.
"""
function smooth_data(s::Spectrum; window::Int=3)
    return Spectrum(s.x, smooth_data(s.y; window=window), copy(s.metadata))
end

"""
    calc_fwhm(x, y; smooth_window=5)
    calc_fwhm(spec; smooth_window=5)

Calculate full width at half maximum (FWHM) of the dominant positive peak.

The `spec` form accepts any 1D `AbstractSpectroscopyData` (uses `xdata`/`ydata`).
"""
function calc_fwhm(x, y; smooth_window=5)
    y_smooth = smooth_window > 1 ? _sg_filter(y, smooth_window, 2).y : y

    peak_idx = argmax(y_smooth)
    peak_val = y_smooth[peak_idx]
    half_max = peak_val / 2

    left_x = x[1]
    for i in (peak_idx-1):-1:1
        if y_smooth[i] <= half_max
            α = (half_max - y_smooth[i]) / (y_smooth[i+1] - y_smooth[i])
            left_x = x[i] + α * (x[i+1] - x[i])
            break
        end
    end

    right_x = x[end]
    for i in (peak_idx+1):length(y_smooth)
        if y_smooth[i] <= half_max
            α = (half_max - y_smooth[i-1]) / (y_smooth[i] - y_smooth[i-1])
            right_x = x[i-1] + α * (x[i] - x[i-1])
            break
        end
    end

    fwhm = abs(right_x - left_x)

    return (
        peak_position = x[peak_idx],
        peak_value = peak_val,
        fwhm = fwhm,
        bounds = (left_x, right_x)
    )
end

# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function calc_fwhm(spec::AbstractSpectroscopyData; kwargs...)
    _check_1d(spec, "calc_fwhm")
    return calc_fwhm(xdata(spec), ydata(spec); kwargs...)
end

# ============================================================================
# SPECTRAL MATH FUNCTIONS
# ============================================================================

"""
    savitzky_golay_smooth(y; window=11, order=3)

Apply Savitzky-Golay smoothing filter to data.

Uses polynomial fitting within a sliding window to smooth data while preserving
peak shape and width better than moving-average smoothing (Savitzky & Golay, 1964).

# Arguments
- `y::AbstractVector{<:Real}`: Input data vector.

# Keywords
- `window::Int=11`: Window size (must be odd and ≥ order + 2).
- `order::Int=3`: Polynomial order for the local fit.

# Returns
- `Vector{Float64}`: Smoothed data vector (same length as input).

# Examples
```julia
y_noisy = sin.(0:0.1:2π) .+ 0.1 * randn(63)
y_smooth = savitzky_golay_smooth(y_noisy; window=11, order=3)
```
"""
function savitzky_golay_smooth(y::AbstractVector{<:Real}; window::Int=11, order::Int=3)
    return _sg_filter(y, window, order).y
end

"""
    savitzky_golay_smooth(s::Spectrum; window=11, order=3) -> Spectrum

Savitzky-Golay smoothing of a [`Spectrum`](@ref). Returns a new `Spectrum`
with shallow-copied metadata.
"""
function savitzky_golay_smooth(s::Spectrum; window::Int=11, order::Int=3)
    return Spectrum(s.x, savitzky_golay_smooth(s.y; window=window, order=order), copy(s.metadata))
end

"""
    derivative(y; order=1, window=11, poly_order=3)
    derivative(x, y; order=1, window=11, poly_order=3)

Compute the derivative of a signal using Savitzky-Golay differentiation.

When `x` is provided, the derivative is correctly scaled by the x-spacing
(using the median point spacing as the rate parameter).

# Arguments
- `x::AbstractVector{<:Real}`: (optional) Independent variable (e.g., wavenumber, wavelength).
- `y::AbstractVector{<:Real}`: Signal to differentiate.

# Keywords
- `order::Int=1`: Derivative order (1 = first derivative, 2 = second, etc.).
- `window::Int=11`: Savitzky-Golay window size (must be odd, ≥ poly_order + 2).
- `poly_order::Int=3`: Polynomial order for the SG filter (must be ≥ `order`).

# Returns
- `Vector{Float64}`: Derivative of the input signal.

# Examples
```julia
x = 400.0:0.5:800.0
y = @. 100 * exp(-(x - 520)^2 / (2 * 20^2))
dy = derivative(x, y; order=1)          # First derivative (correctly scaled)
d2y = derivative(x, y; order=2)         # Second derivative
```
"""
function derivative(y::AbstractVector{<:Real}; order::Int=1, window::Int=11, poly_order::Int=3)
    return _sg_filter(y, window, poly_order, deriv=order).y
end

function derivative(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                    order::Int=1, window::Int=11, poly_order::Int=3)
    dx = median(diff(collect(x)))
    rate = 1.0 / abs(dx)
    return _sg_filter(y, window, poly_order, deriv=order, rate=rate).y
end

"""
    derivative(s::Spectrum; order=1, window=11, poly_order=3) -> Spectrum

Savitzky-Golay derivative of a [`Spectrum`](@ref), scaled by the x-spacing.
Returns a new `Spectrum` with shallow-copied metadata.
"""
function derivative(s::Spectrum; order::Int=1, window::Int=11, poly_order::Int=3)
    return Spectrum(s.x, derivative(s.x, s.y; order=order, window=window, poly_order=poly_order),
                    copy(s.metadata))
end

"""
    band_area(x, y, x_min, x_max)
    band_area(spec, x_min, x_max)

Compute the integrated area under a spectrum within a given range using
trapezoidal integration.

The `spec` form accepts any 1D `AbstractSpectroscopyData` (uses `xdata`/`ydata`).

# Arguments
- `x::AbstractVector{<:Real}`: x-axis values (e.g., wavenumber, wavelength).
- `y::AbstractVector{<:Real}`: y-axis values (e.g., intensity).
- `x_min::Real`: Lower bound of integration range.
- `x_max::Real`: Upper bound of integration range.

# Returns
- `Float64`: Integrated area.

# Examples
```julia
x = 400.0:0.5:800.0
y = @. 100 * exp(-(x - 520)^2 / (2 * 10^2))
area = band_area(x, y, 480.0, 560.0)
```
"""
function band_area(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                   x_min::Real, x_max::Real)
    x_lo, x_hi = minmax(x_min, x_max)
    mask = findall(xi -> x_lo <= xi <= x_hi, x)
    length(mask) >= 2 || throw(ArgumentError(
        "Fewer than 2 points in range [$x_lo, $x_hi]"))
    xr = x[mask]
    yr = y[mask]
    area = 0.0
    for i in 2:length(xr)
        area += 0.5 * (yr[i] + yr[i-1]) * abs(xr[i] - xr[i-1])
    end
    return area
end

# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function band_area(spec::AbstractSpectroscopyData, x_min::Real, x_max::Real)
    _check_1d(spec, "band_area")
    return band_area(xdata(spec), ydata(spec), x_min, x_max)
end

"""
    normalize_area(x, y)

Normalize a spectrum so its total integrated area equals 1.

# Arguments
- `x::AbstractVector{<:Real}`: x-axis values.
- `y::AbstractVector{<:Real}`: y-axis values.

# Returns
- `Vector{Float64}`: Area-normalized y-values.

# Examples
```julia
x = 1.0:0.1:10.0
y = ones(length(x))
y_norm = normalize_area(x, y)  # Total area ≈ 1.0
```
"""
function normalize_area(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    total = band_area(x, y, x[1], x[end])
    abs(total) < eps(Float64) && throw(ArgumentError(
        "Total area is zero; cannot normalize"))
    return Float64.(y ./ total)
end

"""
    normalize_area(s::Spectrum) -> Spectrum

Normalize a [`Spectrum`](@ref) to unit integrated area. Returns a new
`Spectrum` with shallow-copied metadata.
"""
function normalize_area(s::Spectrum)
    return Spectrum(s.x, normalize_area(s.x, s.y), copy(s.metadata))
end

"""
    normalize_to_peak(x, y, position; tolerance=5.0)

Normalize a spectrum by dividing by the intensity at a specified peak position.

Finds the data point nearest to `position` within `tolerance` and divides
the entire spectrum by that point's intensity.

# Arguments
- `x::AbstractVector{<:Real}`: x-axis values.
- `y::AbstractVector{<:Real}`: y-axis values.
- `position::Real`: Target x-position for normalization (e.g., 520 cm-1 for Si).

# Keywords
- `tolerance::Real=5.0`: Maximum allowed distance from `position` to nearest data point.

# Returns
- `Vector{Float64}`: Peak-normalized y-values.

# Examples
```julia
x = 400.0:1.0:800.0
y = @. 50 * exp(-(x - 520)^2 / (2 * 10^2))
y_norm = normalize_to_peak(x, y, 520.0)  # y at x≈520 becomes 1.0
```
"""
function normalize_to_peak(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                           position::Real; tolerance::Real=5.0)
    idx = _find_nearest_idx(collect(x), position)
    dist = abs(x[idx] - position)
    dist <= tolerance || throw(ArgumentError(
        "No data point within tolerance=$tolerance of position=$position " *
        "(nearest at x=$(x[idx]), distance=$dist)"))
    val = y[idx]
    abs(val) < eps(Float64) && throw(ArgumentError(
        "Intensity at position=$position is zero; cannot normalize"))
    return Float64.(y ./ val)
end

"""
    normalize_to_peak(s::Spectrum, position; tolerance=5.0) -> Spectrum

Normalize a [`Spectrum`](@ref) to the intensity at `position`. Returns a new
`Spectrum` with shallow-copied metadata.
"""
function normalize_to_peak(s::Spectrum, position::Real; tolerance::Real=5.0)
    return Spectrum(s.x, normalize_to_peak(s.x, s.y, position; tolerance=tolerance),
                    copy(s.metadata))
end

"""
    estimate_snr(y)
    estimate_snr(spec)

Estimate the signal-to-noise ratio using the DER-SNR method.

The `spec` form accepts any 1D `AbstractSpectroscopyData` (signal comes from `ydata`).

Uses the second-order finite difference of adjacent pixels to estimate noise
(Stoehr et al., 2008, "DER_SNR: A Simple & General Spectroscopic Signal-to-Noise
Measurement Algorithm").

# Arguments
- `y::AbstractVector{<:Real}`: Spectral intensity values.

# Returns
- `NamedTuple{(:snr, :signal, :noise)}`: Estimated SNR, signal level, and noise level.

# Examples
```julia
y_clean = 100 * ones(100)
y_noisy = y_clean .+ 2 * randn(100)
result = estimate_snr(y_noisy)
result.snr  # ≈ 50
```
"""
function estimate_snr(y::AbstractVector{<:Real})
    n = length(y)
    n >= 4 || throw(ArgumentError("Need at least 4 points to estimate SNR"))
    noise_arr = similar(y, n - 2)
    for i in 2:(n - 1)
        noise_arr[i - 1] = abs(2 * y[i] - y[i - 1] - y[i + 1])
    end
    noise = median(noise_arr) / 0.6744897501960817
    noise = noise / sqrt(6.0)
    signal = median(y)
    snr = noise > 0 ? signal / noise : Inf
    return (snr=snr, signal=signal, noise=noise)
end

# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function estimate_snr(spec::AbstractSpectroscopyData)
    _check_1d(spec, "estimate_snr")
    return estimate_snr(ydata(spec))
end

# ============================================================================
# SPECTRAL ARITHMETIC
# ============================================================================

"""
    _align_spectra(a, b; interpolate=false)

Internal helper: return `(a_y, b_y)` on a common x-grid.
If `interpolate=true`, resamples `b` onto `a.x` using linear interpolation.
Otherwise, validates that grids match.
"""
function _align_spectra(a, b; interpolate=false)
    if interpolate
        xs = issorted(b.x) ? collect(b.x) : reverse(collect(b.x))
        ys = issorted(b.x) ? collect(b.y) : reverse(collect(b.y))
        itp = Interpolations.linear_interpolation(xs, ys, extrapolation_bc=Interpolations.Flat())
        return (collect(a.y), itp.(a.x))
    end
    length(a.x) == length(b.x) || error(
        "Grid mismatch: $(length(a.x)) vs $(length(b.x)) points. Use interpolate=true.")
    max_diff = maximum(abs.(collect(a.x) .- collect(b.x)))
    max_diff <= 0.01 || error(
        "Grid mismatch: x-values differ by up to $(round(max_diff, digits=3)). Use interpolate=true.")
    return (collect(a.y), collect(b.y))
end

"""
    add_spectra(a, b; interpolate=false)

Add two spectra element-wise. Returns `(x=a.x, y=a.y + b.y)`.
"""
function add_spectra(a, b; interpolate=false)
    a_y, b_y = _align_spectra(a, b; interpolate)
    return (x=collect(a.x), y=a_y .+ b_y)
end

"""
    divide_spectra(a, b; interpolate=false)

Divide spectrum `a` by spectrum `b` element-wise. Returns `(x=a.x, y=a.y / b.y)`.
"""
function divide_spectra(a, b; interpolate=false)
    a_y, b_y = _align_spectra(a, b; interpolate)
    return (x=collect(a.x), y=a_y ./ b_y)
end

"""
    multiply_spectrum(spec, factor::Real)

Scale a spectrum by a constant factor. Returns `(x=spec.x, y=spec.y * factor)`.
"""
function multiply_spectrum(spec, factor::Real)
    return (x=collect(spec.x), y=collect(spec.y) .* factor)
end

"""
    average_spectra(specs...; interpolate=false)

Compute the point-wise average of multiple spectra. All spectra are
aligned to the x-grid of the first spectrum.
"""
function average_spectra(specs...; interpolate=false)
    first_spec = specs[1]
    n = length(specs)
    sum_y = collect(Float64.(first_spec.y))
    for i in 2:n
        _, b_y = _align_spectra(first_spec, specs[i]; interpolate)
        sum_y .+= b_y
    end
    return (x=collect(first_spec.x), y=sum_y ./ n)
end

"""
    interpolate_spectrum(x, y, new_x)

Resample a spectrum onto a new x-grid using linear interpolation.
"""
function interpolate_spectrum(x, y, new_x)
    xs = issorted(x) ? collect(x) : reverse(collect(x))
    ys = issorted(x) ? collect(y) : reverse(collect(y))
    itp = Interpolations.linear_interpolation(xs, ys, extrapolation_bc=Interpolations.Flat())
    return itp.(new_x)
end

# Typed dispatches for AbstractSpectroscopyData
function add_spectra(a::AbstractSpectroscopyData, b::AbstractSpectroscopyData; kwargs...)
    _check_1d(a, "add_spectra"); _check_1d(b, "add_spectra")
    add_spectra((x=xdata(a), y=ydata(a)), (x=xdata(b), y=ydata(b)); kwargs...)
end

function divide_spectra(a::AbstractSpectroscopyData, b::AbstractSpectroscopyData; kwargs...)
    _check_1d(a, "divide_spectra"); _check_1d(b, "divide_spectra")
    divide_spectra((x=xdata(a), y=ydata(a)), (x=xdata(b), y=ydata(b)); kwargs...)
end

function multiply_spectrum(spec::AbstractSpectroscopyData, factor::Real)
    _check_1d(spec, "multiply_spectrum")
    multiply_spectrum((x=xdata(spec), y=ydata(spec)), factor)
end
