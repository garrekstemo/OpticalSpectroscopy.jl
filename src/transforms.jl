# Spectral transforms: Kramers-Kronig, Kubelka-Munk, Tauc, SNV, Beer-Lambert, etc.

"""
    kramers_kronig(omega, chi; type=:imag_to_real)

Kramers-Kronig relation: compute the real part of a response function
from its imaginary part (or vice versa).

Uses the Maclaurin formula (trapezoidal Cauchy principal value integration)
for numerical stability without requiring FFT.

# Arguments
- `omega`: Frequency/energy/wavenumber axis (should be uniformly spaced)
- `chi`: Input spectrum (imaginary or real part)
- `type`: `:imag_to_real` (default) or `:real_to_imag`

Returns the computed conjugate part.
"""
function kramers_kronig(omega, chi; type=:imag_to_real)
    n = length(chi)
    n == length(omega) || throw(ArgumentError("omega and chi must have the same length"))
    w = Float64.(omega)
    f = Float64.(chi)
    result = zeros(n)

    # Maclaurin formula for Cauchy principal value:
    # chi_real(w_i) = (2/pi) * sum over odd j-i of: w_j * chi_imag(w_j) * dw / (w_j^2 - w_i^2)
    dw = length(w) > 1 ? abs(w[2] - w[1]) : 1.0
    sign_factor = type == :imag_to_real ? 1.0 : -1.0

    for i in eachindex(w)
        s = 0.0
        for j in eachindex(w)
            if j == i
                continue
            end
            # Only sum over points where (j-i) is odd (Maclaurin rule for principal value)
            if isodd(j - i)
                denom = w[j]^2 - w[i]^2
                if abs(denom) > eps(Float64)
                    s += w[j] * f[j] / denom
                end
            end
        end
        result[i] = sign_factor * (2.0 / pi) * dw * s
    end
    return result
end

"""
    kubelka_munk(R)

Convert diffuse reflectance R to the Kubelka-Munk function F(R).

    F(R) = (1 - R)^2 / (2R)

`R` should be fractional (0 to 1). Use `R/100` for percent reflectance.
Scalar function; use broadcast for vectors: `kubelka_munk.(R_vec)`.
"""
function kubelka_munk(R)
    R > 0 || throw(ArgumentError("Reflectance must be positive (got $R)"))
    return (1 - R)^2 / (2R)
end

"""
    tauc_plot(energy, absorption; gap_type=:direct, fit_range=nothing)

Construct a Tauc plot for optical bandgap determination.

Returns `(hv, tauc_y, bandgap, fit_x, fit_y)`.

# Arguments
- `energy`: Photon energy in eV
- `absorption`: Absorption coefficient, absorbance, or Kubelka-Munk F(R)
- `gap_type`: `:direct` (n=2), `:indirect` (n=1/2), `:direct_forbidden` (n=2/3),
  `:indirect_forbidden` (n=1/3)
- `fit_range`: Energy range `(min, max)` for linear fit (auto-detected if `nothing`)
"""
function tauc_plot(energy, absorption; gap_type=:direct, fit_range=nothing)
    exponents = Dict(
        :direct => 2,
        :indirect => 1//2,
        :direct_forbidden => 2//3,
        :indirect_forbidden => 1//3
    )
    haskey(exponents, gap_type) || throw(ArgumentError(
        "Unknown gap_type: $gap_type. Use :direct, :indirect, :direct_forbidden, :indirect_forbidden"))
    n = exponents[gap_type]

    hv = Float64.(energy)
    alpha = Float64.(absorption)
    tauc_y = @. (alpha * hv) ^ n

    # Determine fit range
    if isnothing(fit_range)
        # Auto-detect: find the steepest region via derivative
        dy = diff(tauc_y)
        dx = diff(hv)
        slopes = dy ./ dx
        # Smooth the slopes to find the steepest region
        win = min(11, length(slopes))
        if win >= 3
            slopes_smooth = _sg_filter(slopes, win, 2).y
        else
            slopes_smooth = slopes
        end
        # Find the peak slope region
        idx_max = argmax(slopes_smooth)
        # Expand around peak to capture linear region (20% of spectrum width)
        margin = max(5, length(hv) ÷ 10)
        i_start = max(1, idx_max - margin)
        i_end = min(length(hv), idx_max + margin)
    else
        i_start = searchsortedfirst(hv, fit_range[1])
        i_end = searchsortedlast(hv, fit_range[2])
    end

    # Linear fit in the selected region
    x_fit = hv[i_start:i_end]
    y_fit = tauc_y[i_start:i_end]
    n_pts = length(x_fit)
    n_pts >= 2 || error("Not enough points in fit range for linear regression")
    x_mean = mean(x_fit)
    y_mean = mean(y_fit)
    slope = sum((x_fit .- x_mean) .* (y_fit .- y_mean)) / sum((x_fit .- x_mean).^2)
    intercept = y_mean - slope * x_mean

    # Bandgap = x-intercept of the linear fit
    bandgap = -intercept / slope

    # Generate fit line for plotting
    fit_x = [bandgap, x_fit[end]]
    fit_y = [0.0, slope * x_fit[end] + intercept]

    return (hv=hv, tauc_y=tauc_y, bandgap=bandgap, fit_x=fit_x, fit_y=fit_y)
end

"""
    reflectance_to_absorbance(R)

Convert specular reflectance to pseudo-absorbance: A = -log10(R).
Works on scalars and vectors via broadcasting.
"""
reflectance_to_absorbance(R) = -log10.(R)

"""
    transmittance_to_reflectance(T; percent=false)

Convert transmittance to reflectance for a non-absorbing, non-scattering
sample: R = 1 - T. Input `T` is fractional (0 to 1); use `percent=true` for
percent transmittance. Works on scalars and vectors.
"""
transmittance_to_reflectance(T::Real; percent::Bool=false) = 1.0 - (percent ? T / 100.0 : T)
transmittance_to_reflectance(T::AbstractVector; percent::Bool=false) =
    transmittance_to_reflectance.(T; percent=percent)

"""
    absorbance_to_reflectance(A)

Convert absorbance to reflectance via transmittance: R = 1 - 10^(-A).
Works on scalars and vectors.
"""
absorbance_to_reflectance(A::Real) = 1.0 - 10.0^(-A)
absorbance_to_reflectance(A::AbstractVector) = absorbance_to_reflectance.(A)

"""
    snv(y::AbstractVector)

Standard Normal Variate: `(y - mean(y)) / std(y)`.

Removes multiplicative scatter effects in diffuse reflectance spectra.
"""
function snv(y::AbstractVector)
    m = mean(y)
    s = std(y)
    s < eps(Float64) && error("Cannot SNV-normalize: standard deviation is near zero")
    return (y .- m) ./ s
end

"""
    beer_lambert(A, l)
    beer_lambert(A, l, c)

Compute absorption-related quantities from Beer-Lambert law.
- `beer_lambert(A, l)` returns A/l (absorbance per unit path length)
- `beer_lambert(A, l, c)` returns molar extinction coefficient epsilon = A/(l*c)
"""
beer_lambert(A, l) = A / l
beer_lambert(A, l, c) = A / (l * c)

"""
    urbach_tail(energy, absorption; fit_range=nothing)

Fit the Urbach tail to determine the Urbach energy Eu.

Below the bandgap, absorption follows: alpha(E) = alpha_0 * exp((E - E0) / Eu)

Returns `(Eu, alpha0, E0, fit_x, fit_y)`.

# Arguments
- `energy`: Photon energy (eV)
- `absorption`: Absorption coefficient or absorbance
- `fit_range`: Energy range `(min, max)` for the exponential fit region.
  If `nothing`, auto-selects the sub-gap region.
"""
function urbach_tail(energy, absorption; fit_range=nothing)
    hv = Float64.(energy)
    alpha = Float64.(absorption)

    # Work in log space: ln(alpha) = ln(alpha0) + (E - E0) / Eu
    mask = alpha .> 0
    any(mask) || error("All absorption values are zero or negative")

    if isnothing(fit_range)
        # Auto-detect: use the lower 30% of the absorption range (sub-gap region)
        log_alpha = log.(alpha[mask])
        threshold = quantile(log_alpha, 0.3)
        fit_mask = log.(alpha) .< threshold .&& mask
        any(fit_mask) || (fit_mask = mask)
    else
        fit_mask = (hv .>= fit_range[1]) .&& (hv .<= fit_range[2]) .&& mask
    end

    hv_fit = hv[fit_mask]
    log_alpha_fit = log.(alpha[fit_mask])
    length(hv_fit) >= 2 || error("Not enough points for Urbach fit")

    # Linear regression in log space
    x_mean = mean(hv_fit)
    y_mean = mean(log_alpha_fit)
    slope = sum((hv_fit .- x_mean) .* (log_alpha_fit .- y_mean)) / sum((hv_fit .- x_mean).^2)
    intercept = y_mean - slope * x_mean

    Eu = 1.0 / slope
    E0 = hv_fit[1]
    alpha0 = exp(intercept + E0 / Eu)

    # Generate fit curve
    fit_x = hv_fit
    fit_y = exp.(slope .* hv_fit .+ intercept)

    return (Eu=Eu, alpha0=alpha0, E0=E0, fit_x=fit_x, fit_y=fit_y)
end

"""
    thickness_from_fringes(wavenumber, spectrum; n)

Estimate thin film thickness from interference fringe spacing.

    d = 1 / (2n * delta_nu)

where delta_nu is the fringe spacing in cm^-1 and n is the refractive index.

Returns `(thickness, fringe_positions)` where thickness is in cm.
"""
function thickness_from_fringes(wavenumber, spectrum; n::Real)
    n > 0 || throw(ArgumentError("Refractive index must be positive"))
    wn = Float64.(wavenumber)
    y = Float64.(spectrum)

    # Find local maxima (fringe peaks)
    peak_indices = Int[]
    for i in 2:(length(y)-1)
        if y[i] > y[i-1] && y[i] > y[i+1]
            push!(peak_indices, i)
        end
    end
    length(peak_indices) >= 2 || error("Need at least 2 fringes to estimate thickness")

    fringe_positions = wn[peak_indices]

    # Average fringe spacing
    spacings = abs.(diff(fringe_positions))
    delta_nu = mean(spacings)

    # d = 1 / (2 * n * delta_nu)
    thickness = 1.0 / (2 * n * delta_nu)

    return (thickness=thickness, fringe_positions=fringe_positions)
end
