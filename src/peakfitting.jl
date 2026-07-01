"""
Peak fitting for spectroscopy data.

Unified multi-peak fitting that works on any spectrum type.
"""

# =============================================================================
# Helpers
# =============================================================================

# Single source of truth for the known lineshape models. Per model:
#   name            display/storage name
#   nparams         parameters per peak
#   params          parameter names, fit-vector order
#   width_scale     width guess = width_scale · detected FWHP
#     (lorentzian: Γ is the FWHM; gaussian/voigt: σ is the std dev;
#      pseudo_voigt: σ is the HWHM of both components; fano: Γ ~ width)
#   fourth_p0       initial guess for the 4th parameter, given the detected
#                   FWHP (`nothing` for 3-parameter models)
const _MODEL_INFO = IdDict{Function,NamedTuple}(
    lorentzian   => (name="lorentzian",   nparams=3, params=[:amplitude, :center, :fwhm],
                     width_scale=1.0,             fourth_p0=nothing),
    gaussian     => (name="gaussian",     nparams=3, params=[:amplitude, :center, :sigma],
                     width_scale=1 / FWHM_FACTOR, fourth_p0=nothing),
    pseudo_voigt => (name="pseudo_voigt", nparams=4, params=[:amplitude, :center, :sigma, :mixing],
                     width_scale=0.5,             fourth_p0=fwhp -> 0.5),
    voigt        => (name="voigt",        nparams=4, params=[:amplitude, :center, :sigma, :gamma],
                     width_scale=1 / FWHM_FACTOR, fourth_p0=fwhp -> fwhp / 4),
    fano         => (name="fano",         nparams=4, params=[:amplitude, :center, :width, :q],
                     width_scale=1.0,             fourth_p0=fwhp -> 1.0),
)

# Model name for storage/reporting; unknown models keep their own name (no
# string matching — a user function called `my_gaussian` is not "gaussian").
_model_name(model::Function) =
    haskey(_MODEL_INFO, model) ? _MODEL_INFO[model].name : string(model)

_is_known_model(f::Function) = haskey(_MODEL_INFO, f)

function _n_peak_params(model::Function)
    haskey(_MODEL_INFO, model) ||
        throw(ArgumentError("Unknown model. Provide n_peak_params manually."))
    return _MODEL_INFO[model].nparams
end

function _peak_param_names(model::Function)
    haskey(_MODEL_INFO, model) ||
        throw(ArgumentError("Unknown model. Provide param_names manually."))
    return _MODEL_INFO[model].params
end

_width_guess(fwhp::Real, model::Function) =
    haskey(_MODEL_INFO, model) ? _MODEL_INFO[model].width_scale * fwhp : fwhp

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
            fourth = _MODEL_INFO[model].fourth_p0
            isnothing(fourth) || (p0[offset + 4] = fourth(pk.width))
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

# `x_cache`: the fit grid, if known. Its normalized basis is precomputed once
# and reused whenever the model is evaluated on that same array (the objective
# calls it thousands of times); other grids fall back to computing x_norm.
function _build_multipeak_model(peak_fn::Function, n_peaks::Int, baseline_order::Int, npp::Int,
                                x_mid::Real, x_range::Real;
                                x_cache::Union{AbstractVector,Nothing}=nothing)
    cached_norm = isnothing(x_cache) ? nothing : (x_cache .- x_mid) ./ x_range
    function composite_model(p, x)
        y = similar(p, length(x))
        fill!(y, zero(eltype(p)))

        for i in 1:n_peaks
            offset = (i - 1) * npp
            peak_params = @view p[offset+1:offset+npp]
            y .+= peak_fn(peak_params, x)
        end

        baseline_start = n_peaks * npp + 1
        x_norm = (cached_norm !== nothing && x === x_cache) ? cached_norm :
                 (x .- x_mid) ./ x_range

        for j in 0:baseline_order
            c = p[baseline_start + j]
            y .+= c .* x_norm .^ j
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
        else
            expected = n_peaks * npp + n_baseline
            length(p0) == expected || throw(ArgumentError(
                "p0 has $(length(p0)) entries but n_peaks=$n_peaks with $npp params/peak " *
                "+ $n_baseline baseline params requires $expected"))
        end
        p0_use = collect(Float64, p0)
    else
        # Defensive copy: the trim path below sorts `detected` in place, and
        # the caller's `peaks` vector must not be reordered behind their back.
        detected = if !isnothing(peaks)
            copy(peaks)
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
    composite = _build_multipeak_model(model, n_peaks, baseline_order, npp, x_mid, x_range;
                                       x_cache=x_f)
    prob = NonlinearCurveFitProblem(composite, p0_use, x_f, y_f)
    sol = solve(prob)

    p = coef(sol)
    p_err = stderror(sol)
    ci_vals = confint(sol)

    # Width parameters (index 3 per peak) must be non-negative. For symmetric
    # lineshapes abs() gives the identical curve; fano is NOT symmetric in Γ —
    # flipping Γ mirrors the asymmetry, which is equivalent to q → −q, so q
    # (param 4) is negated along with it. Voigt is symmetric in γ (param 4)
    # separately, so a negative γ is also flipped.
    flip_q = model === fano
    flip_p4 = model === voigt
    for i in 1:n_peaks
        base = (i - 1) * npp
        wi = base + 3
        if p[wi] < 0
            p[wi] = -p[wi]
            lo, hi = ci_vals[wi]
            ci_vals[wi] = (-hi, -lo)
            if flip_q
                qi = base + 4
                p[qi] = -p[qi]
                lo, hi = ci_vals[qi]
                ci_vals[qi] = (-hi, -lo)
            end
        end
        if flip_p4 && p[base + 4] < 0
            gi = base + 4
            p[gi] = -p[gi]
            lo, hi = ci_vals[gi]
            ci_vals[gi] = (-hi, -lo)
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

    length(x_r) < 10 && throw(ArgumentError(
        "Region $(region) contains only $(length(x_r)) points. Need at least 10."))

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
        throw(ArgumentError(
            "Region $(region) contains only $(length(x)) points. Need at least 10."))
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

# data − fit, the package-wide residual convention
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
