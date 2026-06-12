"""
Peak fitting for spectroscopy data.

Unified multi-peak fitting that works on any spectrum type.
"""

# =============================================================================
# Helpers
# =============================================================================

function _model_name(model::Function)
    name = string(model)
    if occursin("lorentzian", lowercase(name))
        return "lorentzian"
    elseif occursin("gaussian", lowercase(name))
        return "gaussian"
    elseif occursin("pseudo_voigt", lowercase(name))
        return "pseudo_voigt"
    elseif occursin("voigt", lowercase(name))
        return "voigt"
    elseif occursin("fano", lowercase(name))
        return "fano"
    else
        return name
    end
end

_is_known_model(f::Function) = f in (lorentzian, gaussian, pseudo_voigt, voigt, fano)

function _n_peak_params(model::Function)
    if model in (lorentzian, gaussian)
        return 3
    elseif model in (pseudo_voigt, voigt, fano)
        return 4
    else
        error("Unknown model. Provide n_peak_params manually.")
    end
end

function _peak_param_names(model::Function)
    if model === lorentzian
        return [:amplitude, :center, :fwhm]
    elseif model === gaussian
        return [:amplitude, :center, :sigma]
    elseif model === pseudo_voigt
        return [:amplitude, :center, :sigma, :mixing]
    elseif model === voigt
        return [:amplitude, :center, :sigma, :gamma]
    elseif model === fano
        return [:amplitude, :center, :width, :q]
    else
        error("Unknown model. Provide param_names manually.")
    end
end

function _width_guess(fwhp::Real, model::Function)
    if model === lorentzian
        return fwhp                            # Γ is the FWHM
    elseif model === gaussian
        return fwhp / (2 * sqrt(2 * log(2)))   # σ is the std dev
    elseif model === pseudo_voigt
        return fwhp / 2                        # σ is the HWHM of both components
    elseif model === voigt
        return fwhp / (2 * sqrt(2 * log(2)))   # σ is the Gaussian std dev
    elseif model === fano
        return fwhp
    else
        return fwhp
    end
end

function _peaks_to_p0(detected::Vector{PeakInfo}, x, y, model::Function; baseline_order::Int=1)
    npp = _n_peak_params(model)
    n_peaks = length(detected)
    n_baseline = baseline_order + 1

    p0 = similar(x, n_peaks * npp + n_baseline)

    for (i, pk) in enumerate(detected)
        offset = (i - 1) * npp
        p0[offset + 1] = pk.prominence
        p0[offset + 2] = pk.position
        if npp >= 3
            p0[offset + 3] = _width_guess(pk.width, model)
        end
        if npp >= 4
            if model === pseudo_voigt
                p0[offset + 4] = 0.5
            elseif model === voigt
                p0[offset + 4] = pk.width / 4  # γ: small Lorentzian width relative to peak
            elseif model === fano
                p0[offset + 4] = 1.0  # q = 1 (moderate asymmetry, |q| -> Inf is Lorentzian)
            end
        end
    end

    if baseline_order == 0
        p0[end] = minimum(y)
    else
        p0[end - baseline_order] = minimum(y)
        for j in 1:baseline_order
            p0[end - baseline_order + j] = 0.0
        end
    end

    return p0
end

# Normalization constants for the polynomial baseline basis. These are fixed
# by the FIT region; every later evaluation (predict on a wider grid, etc.)
# must reuse them, since the fitted baseline coefficients are tied to them.
function _baseline_norm(x::AbstractVector)
    x_mid = (x[1] + x[end]) / 2
    x_range = max(x[end] - x[1], one(eltype(x)))
    return x_mid, x_range
end

function _build_multipeak_model(peak_fn::Function, n_peaks::Int, baseline_order::Int, npp::Int,
                                x_mid::Real, x_range::Real)
    function composite_model(p, x)
        y = similar(p, length(x))
        fill!(y, zero(eltype(p)))

        for i in 1:n_peaks
            offset = (i - 1) * npp
            peak_params = p[offset+1:offset+npp]
            y = y .+ peak_fn(peak_params, x)
        end

        baseline_start = n_peaks * npp + 1
        x_norm = (x .- x_mid) ./ x_range

        for j in 0:baseline_order
            c = p[baseline_start + j]
            y = y .+ c .* x_norm .^ j
        end

        return y
    end
    return composite_model
end

function _eval_baseline(coef_vec::AbstractVector, x::AbstractVector,
                        n_peaks::Int, npp::Int, baseline_order::Int,
                        x_mid::Real, x_range::Real)
    baseline_start = n_peaks * npp + 1
    x_norm = (x .- x_mid) ./ x_range

    y = zeros(eltype(coef_vec), length(x))
    for j in 0:baseline_order
        c = coef_vec[baseline_start + j]
        y .+= c .* x_norm .^ j
    end
    return y
end

# =============================================================================
# Core API
# =============================================================================

"""
    fit_peaks(x, y; kwargs...) -> MultiPeakFitResult
    fit_peaks(spec::AbstractSpectroscopyData, region; kwargs...) -> MultiPeakFitResult
    fit_peaks(spec::AbstractSpectroscopyData; kwargs...) -> MultiPeakFitResult

Fit one or more peaks in spectroscopy data.
"""
function fit_peaks(x::AbstractVector, y::AbstractVector;
                   model::Function=lorentzian,
                   n_peaks::Union{Int, Nothing}=nothing,
                   peaks::Union{Vector{PeakInfo}, Nothing}=nothing,
                   p0::Union{Vector, Nothing}=nothing,
                   baseline_order::Int=1,
                   min_prominence::Real=0.05,
                   sample_id::String="")

    length(x) == length(y) || throw(ArgumentError("x and y must have same length"))
    length(x) < 5 && throw(ArgumentError("Need at least 5 data points"))

    x_f = collect(Float64, x)
    y_f = collect(Float64, y)
    region = (Float64(x_f[1]), Float64(x_f[end]))

    npp = _is_known_model(model) ? _n_peak_params(model) : 3

    if !isnothing(p0)
        n_baseline = baseline_order + 1
        if isnothing(n_peaks)
            n_peaks_inferred = (length(p0) - n_baseline) / npp
            if n_peaks_inferred != round(Int, n_peaks_inferred)
                throw(ArgumentError("p0 length $(length(p0)) inconsistent with $npp params/peak + $n_baseline baseline params"))
            end
            n_peaks = round(Int, n_peaks_inferred)
        end
        p0_use = collect(Float64, p0)
    else
        detected = if !isnothing(peaks)
            peaks
        else
            find_peaks(x_f, y_f; min_prominence=min_prominence)
        end

        if !isnothing(n_peaks)
            if length(detected) < n_peaks
                detected = _synthesize_peak_guesses(x_f, y_f, n_peaks, detected)
            elseif length(detected) > n_peaks
                sort!(detected, by=p -> p.prominence, rev=true)
                detected = detected[1:n_peaks]
                sort!(detected, by=p -> p.position)
            end
        else
            n_peaks = max(1, length(detected))
            if isempty(detected)
                detected = _synthesize_peak_guesses(x_f, y_f, 1, detected)
            end
        end

        p0_use = _peaks_to_p0(detected, x_f, y_f, model; baseline_order=baseline_order)
    end

    x_mid, x_range = _baseline_norm(x_f)
    composite = _build_multipeak_model(model, n_peaks, baseline_order, npp, x_mid, x_range)
    prob = NonlinearCurveFitProblem(composite, p0_use, x_f, y_f)
    sol = solve(prob)

    p = coef(sol)
    p_err = stderror(sol)
    ci_vals = confint(sol)

    # Width parameters (index 3 per peak) must be non-negative.
    # Models are symmetric in width, so abs() gives the identical curve.
    for i in 1:n_peaks
        wi = (i - 1) * npp + 3
        if p[wi] < 0
            p[wi] = -p[wi]
            lo, hi = ci_vals[wi]
            ci_vals[wi] = (-hi, -lo)
        end
    end

    ss_res = rss(sol)
    ss_tot = sum((y_f .- mean(y_f)).^2)
    r_squared = ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0
    mse_val = mse(sol)

    model_name = _is_known_model(model) ? _model_name(model) : string(model)
    param_names = _is_known_model(model) ? _peak_param_names(model) : [Symbol("p$i") for i in 1:npp]

    peak_results = PeakFitResult[]
    for i in 1:n_peaks
        offset = (i - 1) * npp
        pk_vals = collect(p[offset+1:offset+npp])
        pk_errs = collect(p_err[offset+1:offset+npp])
        pk_ci = collect(ci_vals[offset+1:offset+npp])

        pk_r2 = r_squared

        push!(peak_results, PeakFitResult(
            copy(param_names),
            pk_vals, pk_errs, pk_ci,
            pk_r2, ss_res, mse_val,
            length(x_f), region, model_name, sample_id
        ))
    end

    n_baseline = baseline_order + 1
    baseline_start = n_peaks * npp + 1
    bl_params = collect(p[baseline_start:baseline_start + n_baseline - 1])

    return MultiPeakFitResult(
        peak_results, bl_params, baseline_order,
        r_squared, ss_res, mse_val,
        length(x_f), region, sample_id,
        collect(p), model, npp, x_f, y_f,
        x_mid, x_range
    )
end

# Dispatch for raw vectors with a fit region. Restricts data to `region` then
# delegates to the 2-argument core.
function fit_peaks(x::AbstractVector, y::AbstractVector, region::Tuple{Real, Real}; kwargs...)
    length(x) == length(y) || throw(ArgumentError("x and y must have same length"))

    mask = region[1] .<= x .<= region[2]
    x_r = x[mask]
    y_r = y[mask]

    length(x_r) < 10 && error("Region $(region) contains only $(length(x_r)) points. Need at least 10.")

    return fit_peaks(x_r, y_r; kwargs...)
end

# Dispatch for AbstractSpectroscopyData with region
function fit_peaks(spec::AbstractSpectroscopyData, region::Tuple{Real, Real}; kwargs...)
    _check_1d(spec, "fit_peaks")
    x_full = xdata(spec)
    y_full = ydata(spec)

    mask = region[1] .<= x_full .<= region[2]
    x = x_full[mask]
    y = y_full[mask]

    if length(x) < 10
        error("Region $(region) contains only $(length(x)) points. Need at least 10.")
    end

    sid = get(kwargs, :sample_id, "")

    return fit_peaks(x, y; sample_id=sid, kwargs...)
end

# Dispatch for AbstractSpectroscopyData without region (full range)
function fit_peaks(spec::AbstractSpectroscopyData; kwargs...)
    _check_1d(spec, "fit_peaks")
    x = xdata(spec)
    y = ydata(spec)

    sid = get(kwargs, :sample_id, "")

    return fit_peaks(x, y; sample_id=sid, kwargs...)
end

# =============================================================================
# Predict / Decomposition
# =============================================================================

function predict(r::MultiPeakFitResult)
    composite = _build_multipeak_model(r._peak_fn, length(r.peaks), r.baseline_order,
                                       r._n_peak_params, r._x_mid, r._x_range)
    return composite(r._coef, r._x)
end

function predict(r::MultiPeakFitResult, x::AbstractVector)
    composite = _build_multipeak_model(r._peak_fn, length(r.peaks), r.baseline_order,
                                       r._n_peak_params, r._x_mid, r._x_range)
    return composite(r._coef, collect(Float64, x))
end

"""
    predict_peak(result::MultiPeakFitResult, i::Int) -> Vector
    predict_peak(result::MultiPeakFitResult, i::Int, x::AbstractVector) -> Vector

Evaluate the `i`-th peak of a multi-peak fit, excluding the baseline. Without `x`,
the peak is evaluated on the original fit region. Pass a custom `x` to evaluate
elsewhere (e.g., for plotting on a wider range).
"""
function predict_peak(r::MultiPeakFitResult, i::Int)
    return predict_peak(r, i, r._x)
end

function predict_peak(r::MultiPeakFitResult, i::Int, x::AbstractVector)
    1 <= i <= length(r.peaks) || throw(BoundsError(r.peaks, i))
    npp = r._n_peak_params
    offset = (i - 1) * npp
    peak_params = r._coef[offset+1:offset+npp]
    return r._peak_fn(peak_params, collect(Float64, x))
end

"""
    predict_baseline(result::MultiPeakFitResult) -> Vector
    predict_baseline(result::MultiPeakFitResult, x::AbstractVector) -> Vector

Evaluate the polynomial baseline component of a multi-peak fit. Without `x`,
the baseline is evaluated on the original fit region. Pass a custom `x` to
evaluate on a different range.
"""
function predict_baseline(r::MultiPeakFitResult)
    return _eval_baseline(r._coef, r._x, length(r.peaks), r._n_peak_params, r.baseline_order,
                          r._x_mid, r._x_range)
end

function predict_baseline(r::MultiPeakFitResult, x::AbstractVector)
    return _eval_baseline(r._coef, collect(Float64, x), length(r.peaks), r._n_peak_params,
                          r.baseline_order, r._x_mid, r._x_range)
end

function residuals(r::MultiPeakFitResult)
    return r._y .- predict(r)
end

# =============================================================================
# Internal helpers
# =============================================================================

function _synthesize_peak_guesses(x, y, n_peaks::Int, detected::Vector{PeakInfo})
    result = copy(detected)
    n_existing = length(result)

    if n_existing >= n_peaks
        return result[1:n_peaks]
    end

    x_min, x_max = extrema(x)
    data_range = maximum(y) - minimum(y)

    n_needed = n_peaks - n_existing
    existing_positions = [p.position for p in result]

    for _ in 1:n_needed
        all_positions = sort(vcat(existing_positions, [x_min, x_max]))
        gaps = diff(all_positions)
        max_gap_idx = argmax(gaps)
        new_pos = (all_positions[max_gap_idx] + all_positions[max_gap_idx + 1]) / 2

        idx = argmin(abs.(x .- new_pos))

        est_width = gaps[max_gap_idx] / 4

        push!(result, PeakInfo(
            x[idx],
            y[idx],
            data_range * 0.1,
            est_width,
            (x[idx] - est_width/2, x[idx] + est_width/2),
            idx
        ))
        push!(existing_positions, new_pos)
    end

    sort!(result, by=p -> p.position)
    return result
end
