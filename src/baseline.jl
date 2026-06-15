"""
Baseline correction algorithms for spectroscopy.

Provides:
- `arpls_baseline` — Asymmetrically Reweighted PLS (Baek et al., 2015)
- `snip_baseline` — Statistics-sensitive Non-linear Iterative Peak-clipping (Ryan et al., 1988)
- `rubberband_baseline` — Lower convex hull (rubber band) baseline
- `imodpoly_baseline` — Improved Modified Polynomial (Lieber & Mahadevan-Jansen, 2003)
- `rolling_ball_baseline` — Morphological erosion-dilation envelope
"""

# =============================================================================
# Asymmetrically Reweighted Penalized Least Squares (arPLS)
# =============================================================================

"""
    arpls_baseline(y; λ=1e5, maxiter=100, tol=1e-6) -> Vector

Asymmetrically Reweighted Penalized Least Squares baseline correction.
"""
function arpls_baseline(y::AbstractVector{<:Real};
                        λ::Real=1e5,
                        maxiter::Int=100,
                        tol::Real=1e-6)
    n = length(y)

    λ > 0 || throw(ArgumentError("λ must be positive, got $λ"))
    n ≥ 3 || throw(ArgumentError("Need at least 3 points, got $n"))

    D = _diff_matrix(n, 2)
    H = λ * (D' * D)                   # constant penalty term — compute once

    w   = ones(n)
    z   = similar(y, Float64)
    d   = similar(y, Float64)           # residual vector (reused)
    rhs = similar(y, Float64)           # right-hand side (reused)

    # Build initial system matrix and symbolic Cholesky factorization
    A = copy(H)
    for i in eachindex(w)
        A[i, i] += w[i]
    end
    F = cholesky(A)

    for iter in 1:maxiter
        w_norm_prev = norm(w)

        # Rebuild system matrix: A = H + diag(w)
        # Copy the constant part and add current weights to diagonal
        copy!(A, H)
        for i in eachindex(w)
            A[i, i] += w[i]
        end

        # Reuse symbolic factorization, only redo numeric factorization
        cholesky!(F, A)

        # Solve (H + diag(w)) * z = w .* y
        for i in eachindex(rhs)
            rhs[i] = w[i] * y[i]
        end
        z .= F \ rhs

        # Residuals: d = y - z
        for i in eachindex(d)
            d[i] = y[i] - z[i]
        end

        # Single-pass mean and variance of negative residuals (no allocation)
        neg_sum  = 0.0
        neg_sum2 = 0.0
        neg_count = 0
        for i in eachindex(d)
            if d[i] < 0
                neg_count += 1
                neg_sum  += d[i]
                neg_sum2 += d[i] * d[i]
            end
        end

        neg_count == 0 && break

        m = neg_sum / neg_count
        σ = sqrt(neg_sum2 / neg_count - m * m)

        σ < eps() && break

        threshold = -m + 2σ
        for i in eachindex(w)
            w[i] = 1 / (1 + exp(2 * (d[i] - threshold) / σ))
        end

        w_norm = norm(w)
        δ = abs(w_norm - w_norm_prev) / (w_norm + eps())
        δ < tol && break
    end

    return z
end

# =============================================================================
# SNIP
# =============================================================================

"""
    snip_baseline(y; iterations=40, decreasing=true) -> Vector

SNIP baseline correction using iterative peak clipping.
"""
function snip_baseline(y::AbstractVector{<:Real};
                       iterations::Int=40,
                       decreasing::Bool=true)
    n = length(y)

    iterations > 0 || throw(ArgumentError("iterations must be positive"))
    n ≥ 3 || throw(ArgumentError("Need at least 3 points, got $n"))

    z = convert(Vector{Float64}, copy(y))

    _lls_transform!(z)

    window_sizes = decreasing ? (iterations:-1:1) : (1:iterations)

    for k in window_sizes
        for i in (k+1):(n-k)
            neighbor_avg = (z[i-k] + z[i+k]) / 2
            z[i] = min(z[i], neighbor_avg)
        end
    end

    _lls_inverse!(z)

    return z
end

function _lls_transform!(z::Vector{Float64})
    for i in eachindex(z)
        v = max(z[i], 0.0)
        z[i] = log(log(sqrt(v + 1) + 1) + 1)
    end
end

function _lls_inverse!(z::Vector{Float64})
    for i in eachindex(z)
        v = z[i]
        z[i] = (exp(exp(v) - 1) - 1)^2 - 1
    end
end

# =============================================================================
# Rubber band baseline
# =============================================================================

"""
    rubberband_baseline(x, y)

Rubber band baseline correction using the lower convex hull.

Equivalent to stretching a rubber band under the spectrum -- the baseline
follows the lower envelope. Used in OPUS for FTIR baseline correction.

Returns the baseline y-values (same length as input).
"""
function rubberband_baseline(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || throw(ArgumentError("x and y must have the same length"))
    n >= 2 || throw(ArgumentError("Need at least 2 points"))

    xv = Float64.(x)
    yv = Float64.(y)

    # Compute lower convex hull indices using Andrew's monotone chain
    # We want the lower hull of points (x[i], y[i])
    # Sort by x, then build lower hull
    order = sortperm(xv)
    hull = Int[]  # indices into the original arrays (via order)

    for idx in order
        while length(hull) >= 2
            h1 = hull[end-1]
            h2 = hull[end]
            # Cross product to check left turn (lower hull wants right turns)
            cross = (xv[h2] - xv[h1]) * (yv[idx] - yv[h1]) -
                    (yv[h2] - yv[h1]) * (xv[idx] - xv[h1])
            if cross <= 0
                pop!(hull)
            else
                break
            end
        end
        push!(hull, idx)
    end

    # Interpolate hull back to original x-grid
    hull_x = xv[hull]
    hull_y = yv[hull]
    itp = Interpolations.linear_interpolation(hull_x, hull_y, extrapolation_bc=Interpolations.Flat())
    return itp.(xv)
end

# =============================================================================
# Improved Modified Polynomial (iModPoly)
# =============================================================================

"""
    imodpoly_baseline(x, y; poly_order=4, maxiter=100, tol=1e-3) -> Vector

Improved Modified Polynomial baseline (Lieber & Mahadevan-Jansen, 2003).

Iteratively fits a polynomial to the spectrum, removing points that are
above the fit by more than one standard deviation of the residuals.
Converges when the fit changes less than `tol` between iterations.
"""
function imodpoly_baseline(x::AbstractVector, y::AbstractVector;
                           poly_order::Int=4,
                           maxiter::Int=100,
                           tol::Real=1e-3)
    n = length(x)
    n == length(y) || throw(ArgumentError("x and y must have the same length"))
    n > poly_order || throw(ArgumentError("Need more points than polynomial order"))

    xv = Float64.(x)
    yv = Float64.(y)

    # Normalize x to [-1, 1] for numerical stability
    x_min, x_max = extrema(xv)
    x_range = x_max - x_min
    xn = if x_range > 0
        @. 2 * (xv - x_min) / x_range - 1
    else
        zeros(n)
    end

    y_work = copy(yv)
    baseline = similar(yv)

    for iter in 1:maxiter
        coeffs = _polyfit(xn, y_work, poly_order)
        baseline_new = _polyeval(coeffs, xn)

        if iter > 1
            δ = norm(baseline_new - baseline) / (norm(baseline_new) + eps())
            if δ < tol
                baseline .= baseline_new
                break
            end
        end

        baseline .= baseline_new

        residuals_neg = filter(<(0), yv .- baseline)
        dev = isempty(residuals_neg) ? 0.0 : std(residuals_neg; corrected=false)
        if dev < eps()
            break
        end

        for i in eachindex(y_work)
            y_work[i] = yv[i] > baseline[i] + dev ? baseline[i] : yv[i]
        end
    end

    return baseline
end

function _polyfit(x::Vector{Float64}, y::Vector{Float64}, d::Int)
    n = length(x)
    V = zeros(n, d + 1)
    for j in 0:d
        for i in eachindex(x)
            V[i, j+1] = x[i]^j
        end
    end
    return V \ y
end

"""
Evaluate polynomial with ascending-order coefficients at a scalar using Horner's method.
"""
function _polyeval(coeffs::AbstractVector, x::Real)
    result = coeffs[end]
    for i in (length(coeffs) - 1):-1:1
        result = muladd(result, x, coeffs[i])
    end
    return result
end

"""
Evaluate polynomial with ascending-order coefficients at each element of a vector.
"""
_polyeval(coeffs::AbstractVector, x::AbstractVector) = [_polyeval(coeffs, xj) for xj in x]

# =============================================================================
# Rolling ball baseline
# =============================================================================

"""
    rolling_ball_baseline(y; half_window=50, smooth_half_window=nothing) -> Vector

Rolling ball baseline estimation using morphological operations.

Applies erosion (rolling minimum) then dilation (rolling maximum) to estimate
the baseline envelope, followed by moving-average smoothing.

# Arguments
- `half_window::Int=50`: Half-window size for morphological operations.
- `smooth_half_window::Int`: Half-window for final smoothing (default: `half_window`).

# Reference
M. A. Kneen and H. J. Annegarn, "Algorithm for fitting XRF, SEM and PIXE X-ray
spectra backgrounds", Nucl. Instrum. Methods Phys. Res. B 109/110, 209 (1996).
[doi:10.1016/0168-583X(95)00908-6](https://doi.org/10.1016/0168-583X(95)00908-6)
"""
function rolling_ball_baseline(y::AbstractVector{<:Real};
                               half_window::Int=50,
                               smooth_half_window::Union{Int,Nothing}=nothing)
    n = length(y)
    half_window > 0 || throw(ArgumentError("half_window must be positive"))
    n > 2 * half_window || throw(ArgumentError("Signal too short for window size"))

    s_hw = something(smooth_half_window, half_window)
    yv = Float64.(y)

    eroded = similar(yv)
    for i in eachindex(yv)
        lo = max(1, i - half_window)
        hi = min(n, i + half_window)
        eroded[i] = minimum(@view yv[lo:hi])
    end

    dilated = similar(yv)
    for i in eachindex(eroded)
        lo = max(1, i - half_window)
        hi = min(n, i + half_window)
        dilated[i] = maximum(@view eroded[lo:hi])
    end

    baseline = similar(yv)
    for i in eachindex(dilated)
        lo = max(1, i - s_hw)
        hi = min(n, i + s_hw)
        baseline[i] = mean(@view dilated[lo:hi])
    end

    return baseline
end

# =============================================================================
# Unified API
# =============================================================================

"""
    correct_baseline(y; method=:arpls, kwargs...) -> NamedTuple

Correct baseline and return both corrected spectrum and baseline.

Returns `(y=corrected, baseline=baseline)`.
"""
function correct_baseline(y::AbstractVector{<:Real};
                          method::Symbol=:arpls,
                          kwargs...)
    baseline = if method == :arpls
        arpls_baseline(y; kwargs...)
    elseif method == :snip
        snip_baseline(y; kwargs...)
    elseif method == :rubberband
        rubberband_baseline(collect(1.0:length(y)), y)
    elseif method == :imodpoly
        imodpoly_baseline(collect(1.0:length(y)), y; kwargs...)
    elseif method == :rolling_ball
        rolling_ball_baseline(y; kwargs...)
    else
        available = (:arpls, :snip, :rubberband, :imodpoly, :rolling_ball)
        throw(ArgumentError("Unknown method :$method. Available: $available"))
    end

    corrected = y - baseline
    return (y=corrected, baseline=baseline)
end

"""
    correct_baseline(x, y; kwargs...) -> NamedTuple

Version that also returns x values for convenience.
Methods that use x-values (rubberband) receive real x, not dummy indices.
"""
function correct_baseline(x::AbstractVector, y::AbstractVector{<:Real};
                          method::Symbol=:arpls, kwargs...)
    if method == :rubberband
        baseline = rubberband_baseline(x, y)
        corrected = y - baseline
        return (x=collect(x), y=corrected, baseline=baseline)
    elseif method == :imodpoly
        baseline = imodpoly_baseline(x, y; kwargs...)
        corrected = y - baseline
        return (x=collect(x), y=corrected, baseline=baseline)
    end
    result = correct_baseline(y; method=method, kwargs...)
    return (x=collect(x), y=result.y, baseline=result.baseline)
end

"""
    correct_baseline(s::Spectrum; method=:arpls, kwargs...) -> Spectrum

Baseline-correct a [`Spectrum`](@ref). Returns a new `Spectrum` with
shallow-copied metadata. The baseline itself is not returned; to retrieve it,
call `correct_baseline(s.x, s.y; method=method, kwargs...)` directly, which
returns `(x=..., y=..., baseline=...)`.
"""
function correct_baseline(s::Spectrum; method::Symbol=:arpls, kwargs...)
    res = correct_baseline(s.x, s.y; method=method, kwargs...)
    return Spectrum(res.x, res.y, copy(s.metadata))
end

# =============================================================================
# Internal utilities
# =============================================================================

function _diff_matrix(n::Int, d::Int)
    D = sparse(1.0I, n, n)

    for _ in 1:d
        m = size(D, 1)
        D = D[2:m, :] - D[1:m-1, :]
    end

    return D
end
