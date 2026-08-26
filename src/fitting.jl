"""
Curve fitting routines for spectroscopy data.

Model functions come from CurveFitModels.jl — this file contains
automatic fitting routines that use those models.
"""

# Internal IRF functions

# Exponential decay ⊗ Gaussian IRF:
#   (A/2)·exp(σ²/2τ² − t′/τ)·erfc((σ/τ − t′/σ)/√2),  t′ = t − t₀.
# For τ ≪ σ the exp factor overflows while erfc underflows, so for
# arg_erfc ≥ 0 the product is evaluated in the exactly equivalent scaled form
#   (A/2)·erfcx(arg_erfc)·exp(−t′²/2σ²),   erfcx(x) = exp(x²)·erfc(x),
# which is finite and accurate for any σ/τ. Both branches are
# ForwardDiff-compatible (SpecialFunctions provides erfc/erfcx rules).
function _exp_decay_irf_conv(t, A, tau, t0, sigma)
    t_shifted = t - t0
    arg_erfc = (sigma / tau - t_shifted / sigma) / sqrt(2)
    if arg_erfc >= 0
        return (A / 2) * erfcx(arg_erfc) * exp(-t_shifted^2 / (2 * sigma^2))
    else
        arg_exp = sigma^2 / (2 * tau^2) - t_shifted / tau
        return (A / 2) * exp(arg_exp) * erfc(arg_erfc)
    end
end

# Internal helpers for fitting

# From pre-computed residual sum of squares (e.g., rss(sol) from CurveFit)
function _rsquared(y_data, ss_res::Real)
    ss_tot = sum((y_data .- mean(y_data)).^2)
    return ss_tot > 0 ? 1 - ss_res / ss_tot : 0.0
end

# From fitted values (e.g., for per-trace R² in global fitting)
_rsquared(y_data, y_fit::AbstractVector) = _rsquared(y_data, sum((y_data .- y_fit).^2))

function _detect_signal_type(signal)
    max_val = maximum(signal)
    min_val = minimum(signal)
    if abs(max_val) >= abs(min_val)
        return :esa, max_val, argmax(signal)
    else
        return :gsb, min_val, argmin(signal)
    end
end

# User-supplied initial guesses. Each fit computes its automatic guesses as
# before; a user value, where given, replaces the corresponding automatic one.

function _checked_guess(v::Float64, name::Symbol; positive::Bool=false)
    isfinite(v) || throw(ArgumentError("$name must be finite, got $v"))
    positive && v <= 0 && throw(ArgumentError("$name must be positive, got $v"))
    return v
end

# Normalize a guess spec to a length-n vector whose entries are Float64
# overrides or `nothing` (= keep the automatic guess). Accepts `nothing`
# (all automatic), a Real (n must be 1), or a vector of length n whose
# entries are Reals or `nothing`. Idempotent, so already-normalized vectors
# pass through unchanged.
function _guess_vector(val, n::Int, name::Symbol; positive::Bool=false)
    out = Vector{Union{Nothing,Float64}}(nothing, n)
    isnothing(val) && return out
    if val isa Real
        n == 1 || throw(ArgumentError(
            "$name must be a vector of length $n when n_exp = $n"))
        out[1] = _checked_guess(Float64(val), name; positive)
    elseif val isa AbstractVector
        length(val) == n || throw(ArgumentError(
            "$name has $(length(val)) entries but n_exp = $n"))
        for (i, v) in enumerate(val)
            isnothing(v) && continue
            v isa Real || throw(ArgumentError(
                "$name entries must be real numbers or nothing, got $(typeof(v))"))
            out[i] = _checked_guess(Float64(v), name; positive)
        end
    else
        throw(ArgumentError("$name must be a real number, a vector, or nothing"))
    end
    return out
end

function _scalar_guess(val, name::Symbol; positive::Bool=false)
    isnothing(val) && return nothing
    val isa Real || throw(ArgumentError("$name must be a real number or nothing"))
    return _checked_guess(Float64(val), name; positive)
end

_override(auto::Real, user::Union{Nothing,Float64}) =
    isnothing(user) ? Float64(auto) : user

"""
    fit_decay_irf(t, signal; sigma_init=5.0, tau0=nothing, amplitude0=nothing, offset0=nothing) -> ExpDecayFit

Fit an exponential decay convolved with a Gaussian IRF to pump-probe data.

`tau0`, `amplitude0`, and `offset0` override the automatic initial guesses
for τ, A, and the constant offset; any guess not supplied is estimated from
the data. The t₀ guess is always the signal peak position, and `sigma_init`
seeds the IRF width.
"""
function fit_decay_irf(t::AbstractVector{<:Real}, signal::AbstractVector{<:Real};
                       sigma_init::Real=5.0,
                       tau0=nothing, amplitude0=nothing, offset0=nothing)
    tau_user = _scalar_guess(tau0, :tau0; positive=true)
    amp_user = _scalar_guess(amplitude0, :amplitude0)
    offset_user = _scalar_guess(offset0, :offset0)

    signal_type, peak_val, peak_idx = _detect_signal_type(signal)

    t0_init = t[peak_idx]
    n_edge = max(5, length(signal) ÷ 20)
    offset_init = _override(mean(signal[1:n_edge]), offset_user)
    A_init = _override(peak_val - offset_init, amp_user)

    half_val = (peak_val + offset_init) / 2
    half_idx = peak_idx
    for i in peak_idx:length(signal)
        if signal_type == :esa && signal[i] < half_val
            half_idx = i
            break
        elseif signal_type == :gsb && signal[i] > half_val
            half_idx = i
            break
        end
    end
    tau_init = _override(max(abs(t[half_idx] - t[peak_idx]) / log(2), 1.0), tau_user)

    function model(p, t_vec)
        A, tau, t0, sigma, offset = p
        tau = abs(tau)
        sigma = abs(sigma)
        return [_exp_decay_irf_conv(ti, A, tau, t0, sigma) + offset for ti in t_vec]
    end

    p0 = [A_init, tau_init, t0_init, sigma_init, offset_init]

    prob = NonlinearCurveFitProblem(model, p0, t, signal)
    sol = solve(prob, _FIT_ALG)

    A, tau, t0, sigma, offset = coef(sol)
    tau = abs(tau)
    sigma = abs(sigma)

    # data − fit, computed explicitly: CurveFit's residuals(sol) convention
    # has flipped sign between releases, so it is never trusted here.
    return ExpDecayFit(A, tau, t0, sigma, offset, signal_type,
                       signal .- fitted(sol), _rsquared(signal, rss(sol)))
end

# Pulse width estimation (FWHM_FACTOR lives in units.jl)

"""
    irf_fwhm(sigma)

Convert IRF standard deviation σ to full-width at half-maximum (FWHM).
"""
irf_fwhm(sigma) = FWHM_FACTOR * sigma

"""
    pulse_fwhm(sigma_irf)

Estimate individual pulse FWHM from fitted IRF width, assuming identical
Gaussian pump and probe pulses.
"""
pulse_fwhm(sigma_irf) = FWHM_FACTOR * sigma_irf / sqrt(2)


# =============================================================================
# Multi-exponential IRF convolution
# =============================================================================

function _multiexp_irf_conv(t, taus, amplitudes, t0, sigma, offset)
    result = offset
    for i in eachindex(taus)
        result += _exp_decay_irf_conv(t, amplitudes[i], taus[i], t0, sigma)
    end
    return result
end

# =============================================================================
# Unified TA API: fit_exp_decay
# =============================================================================

"""
    fit_exp_decay(trace::KineticTrace; n_exp=1, irf=false, irf_width=0.15, t_start=0.0, t_range=nothing, model=:exponential,
                  tau0=nothing, amplitude0=nothing, offset0=nothing, beta0=nothing)

Fit an exponential decay model to a `KineticTrace`.

# Arguments
- `trace`: KineticTrace
- `n_exp`: Number of exponential components (default 1)
- `irf`: Include IRF convolution (default false)
- `irf_width`: Initial guess for IRF σ in ps (default 0.15)
- `t_start`: Start time for fitting when irf=false (default 0.0)
- `t_range`: Optional (t_min, t_max) to restrict fit region
- `model`: `:exponential` (default) or `:stretched` — Kohlrausch–Williams–Watts

# Initial guesses
Initial parameter guesses are estimated from the data. The keywords below
override individual automatic guesses; any guess not supplied keeps its
automatic value.
- `tau0`, `amplitude0`: For `n_exp = 1` a number; for `n_exp > 1` a vector of
  length `n_exp` whose entries are numbers or `nothing` (`nothing` = keep the
  automatic guess for that component). `tau0` values must be positive.
- `offset0`: Constant-offset guess.
- `beta0`: Stretching-exponent guess in (0, 1] (`model=:stretched` only).

# Returns
- `n_exp=1`: `ExpDecayFit`
- `n_exp>1`: `MultiexpDecayFit`
- `model=:stretched`: `StretchedDecayFit`
"""
function fit_exp_decay(trace::KineticTrace; n_exp::Int=1, irf::Bool=false, irf_width::Float64=0.15,
                       t_start::Float64=0.0, t_range=nothing, model::Symbol=:exponential,
                       tau0=nothing, amplitude0=nothing, offset0=nothing, beta0=nothing)
    @assert n_exp >= 1 "n_exp must be at least 1"

    isnothing(beta0) || model === :stretched || throw(ArgumentError(
        "beta0 only applies to model=:stretched"))

    if model === :stretched
        n_exp == 1 || throw(ArgumentError(
            "model=:stretched fits a single stretched component; n_exp must be 1"))
        irf && throw(ArgumentError("model=:stretched does not support IRF convolution"))
        beta_user = _scalar_guess(beta0, :beta0; positive=true)
        isnothing(beta_user) || beta_user <= 1.0 || throw(ArgumentError(
            "beta0 must be in (0, 1], got $beta_user"))
        return _fit_stretched_decay(trace; t_start=t_start, t_range=t_range,
                                    tau0=_guess_vector(tau0, 1, :tau0; positive=true)[1],
                                    amplitude0=_guess_vector(amplitude0, 1, :amplitude0)[1],
                                    offset0=_scalar_guess(offset0, :offset0),
                                    beta0=beta_user)
    end
    model === :exponential || throw(ArgumentError(
        "unknown model: $model (expected :exponential or :stretched)"))

    tau_user = _guess_vector(tau0, n_exp, :tau0; positive=true)
    amp_user = _guess_vector(amplitude0, n_exp, :amplitude0)
    offset_user = _scalar_guess(offset0, :offset0)

    if n_exp > 1
        return _fit_multiexp_decay(trace; n_exp=n_exp, irf=irf, irf_width=irf_width,
                                   t_start=t_start, t_range=t_range,
                                   tau0=tau_user, amplitude0=amp_user, offset0=offset_user)
    end

    t = trace.time
    signal = trace.signal

    if !isnothing(t_range)
        t_min, t_max = t_range
        mask = (t .>= t_min) .& (t .<= t_max)
        t = t[mask]
        signal = signal[mask]
    end

    if irf
        return fit_decay_irf(t, signal; sigma_init=irf_width,
                             tau0=tau_user[1], amplitude0=amp_user[1], offset0=offset_user)
    else
        mask = t .>= t_start
        count(mask) >= 5 || throw(ArgumentError(
            "fit region contains $(count(mask)) points; need at least 5"))
        t_fit = t[mask]
        signal_fit = signal[mask]

        signal_type = first(_detect_signal_type(signal))
        peak_val = signal_type == :esa ? maximum(signal_fit) : minimum(signal_fit)

        n_end = max(1, min(10, length(signal_fit) ÷ 4))
        offset_init = _override(mean(signal_fit[end-n_end+1:end]), offset_user)

        A_init = _override(peak_val - offset_init, amp_user[1])
        tau_init = _override((t_fit[end] - t_fit[1]) / 3.0, tau_user[1])

        # CurveFitModels single_exponential on the t_start-shifted axis,
        # mirroring the t-shift + n_exponentials approach of the multi-exp path
        t_shifted = t_fit .- t_start
        prob = NonlinearCurveFitProblem(single_exponential, [A_init, tau_init, offset_init],
                                        t_shifted, signal_fit)
        sol = solve(prob, _FIT_ALG)

        A, tau, offset = coef(sol)
        tau = abs(tau)

        return ExpDecayFit(
            A, tau, t_start, NaN, offset,
            signal_type, signal_fit .- fitted(sol), _rsquared(signal_fit, rss(sol))
        )
    end
end

# =============================================================================
# Stretched-exponential fitting (internal)
# =============================================================================

function _fit_stretched_decay(trace::KineticTrace; t_start::Float64=0.0, t_range=nothing,
                              tau0=nothing, amplitude0=nothing, offset0=nothing, beta0=nothing)
    t = trace.time
    sig = trace.signal

    if !isnothing(t_range)
        mask = (t .>= t_range[1]) .& (t .<= t_range[2])
        origin = float(t_range[1])
    else
        mask = t .>= t_start
        origin = t_start
    end
    count(mask) >= 5 || throw(ArgumentError(
        "fit region contains $(count(mask)) points; need at least 5"))

    t_fit = t[mask] .- origin
    t_fit = max.(t_fit, eps(Float64))  # avoid NaN gradient of (t/τ)^β at t = 0
    signal_fit = sig[mask]

    signal_type = first(_detect_signal_type(signal_fit))
    peak_val = signal_type == :esa ? maximum(signal_fit) : minimum(signal_fit)
    n_end = max(1, min(10, length(signal_fit) ÷ 4))
    offset_init = _override(mean(signal_fit[end-n_end+1:end]), offset0)
    A_init = _override(peak_val - offset_init, amplitude0)
    tau_init = _override((t_fit[end] - t_fit[1]) / 3.0, tau0)
    beta_init = _override(0.8, beta0)

    function model(p, tv)
        A, tau, beta, offset = p
        return stretched_exponential([A, abs(tau), clamp(beta, 0.05, 1.0), offset], tv)
    end

    prob = NonlinearCurveFitProblem(model, [A_init, tau_init, beta_init, offset_init], t_fit, signal_fit)
    sol = solve(prob, _FIT_ALG)
    A, tau, beta, offset = coef(sol)

    beta_c = clamp(beta, 0.05, 1.0)
    beta_c <= 0.051 && @warn "stretched fit: β converged to the lower clamp (0.05); fit may be unreliable or the data is non-KWW"
    beta_c >= 0.999 && @warn "stretched fit: β converged to the upper clamp (1.0); consider model=:exponential"

    return StretchedDecayFit(A, abs(tau), beta_c, origin, offset, signal_type,
                             signal_fit .- fitted(sol), _rsquared(signal_fit, rss(sol)))
end

# =============================================================================
# Multi-exponential fitting (internal)
# =============================================================================

function _fit_multiexp_decay(trace::KineticTrace; n_exp::Int, irf::Bool, irf_width::Float64,
                             t_start::Float64, t_range,
                             tau0=nothing, amplitude0=nothing, offset0=nothing)
    tau_user = _guess_vector(tau0, n_exp, :tau0; positive=true)
    amp_user = _guess_vector(amplitude0, n_exp, :amplitude0)
    offset_user = _scalar_guess(offset0, :offset0)

    t = trace.time
    signal = trace.signal

    if !isnothing(t_range)
        t_min, t_max = t_range
        mask = (t .>= t_min) .& (t .<= t_max)
        t = t[mask]
        signal = signal[mask]
    end

    if irf
        t_fit = t
        signal_fit = signal
    else
        mask = t .>= t_start
        count(mask) >= 5 || throw(ArgumentError(
            "fit region contains $(count(mask)) points; need at least 5"))
        t_fit = t[mask]
        signal_fit = signal[mask]
    end

    signal_type = first(_detect_signal_type(signal))
    peak_val = signal_type == :esa ? maximum(signal_fit) : minimum(signal_fit)

    n_end = max(1, min(10, length(signal_fit) ÷ 4))
    offset_init = _override(mean(signal_fit[end-n_end+1:end]), offset_user)

    half_val = (peak_val + offset_init) / 2
    half_idx = findfirst(i -> begin
        if signal_type == :esa
            signal_fit[i] <= half_val
        else
            signal_fit[i] >= half_val
        end
    end, eachindex(signal_fit))
    tau_est = isnothing(half_idx) ? (t_fit[end] - t_fit[1]) / 3 : t_fit[half_idx] - t_fit[1]
    tau_est = max(tau_est, 0.1)

    taus_init = Float64[]
    if n_exp == 1
        push!(taus_init, tau_est)
    elseif n_exp == 2
        push!(taus_init, tau_est / 3.0)
        push!(taus_init, tau_est * 2.0)
    else
        log_min = log10(tau_est / 10)
        log_max = log10(tau_est * 10)
        for i in 1:n_exp
            push!(taus_init, 10^(log_min + (log_max - log_min) * (i - 1) / (n_exp - 1)))
        end
    end

    total_amp = peak_val - offset_init
    amps_init = fill(total_amp / n_exp, n_exp)

    for i in 1:n_exp
        isnothing(tau_user[i]) || (taus_init[i] = tau_user[i])
        isnothing(amp_user[i]) || (amps_init[i] = amp_user[i])
    end

    if irf
        p0 = vcat(taus_init, amps_init, [0.0, irf_width, offset_init])

        function multiexp_irf_model(p, t_vec)
            # @views: parameter slices are hot-loop reads, never mutated
            taus = @view p[1:n_exp]
            amps = @view p[n_exp+1:2*n_exp]
            t0 = p[2*n_exp+1]
            sigma = abs(p[2*n_exp+2])
            offset = p[2*n_exp+3]
            y = similar(p, length(t_vec))
            for (i, ti) in enumerate(t_vec)
                acc = offset
                for j in eachindex(taus)
                    acc += _exp_decay_irf_conv(ti, amps[j], abs(taus[j]), t0, sigma)
                end
                y[i] = acc
            end
            return y
        end

        prob = NonlinearCurveFitProblem(multiexp_irf_model, p0, t_fit, signal_fit)
        sol = solve(prob, _FIT_ALG)
        p_opt = coef(sol)

        taus_fit = abs.(p_opt[1:n_exp])
        amps_fit = p_opt[n_exp+1:2*n_exp]
        t0 = p_opt[2*n_exp+1]
        sigma = abs(p_opt[2*n_exp+2])
        offset = p_opt[2*n_exp+3]
    else
        model = n_exponentials(n_exp)

        p0 = Float64[]
        for i in 1:n_exp
            push!(p0, amps_init[i])
            push!(p0, taus_init[i])
        end
        push!(p0, offset_init)

        t_shifted = t_fit .- t_fit[1]

        prob = NonlinearCurveFitProblem(model, p0, t_shifted, signal_fit)
        sol = solve(prob, _FIT_ALG)
        p_opt = coef(sol)

        amps_fit = Float64[]
        taus_fit = Float64[]
        for i in 1:n_exp
            push!(amps_fit, p_opt[2*i - 1])
            push!(taus_fit, abs(p_opt[2*i]))
        end
        offset = p_opt[end]
        t0 = t_fit[1]
        sigma = NaN
    end

    sort_idx = sortperm(taus_fit)
    taus_sorted = taus_fit[sort_idx]
    amps_sorted = amps_fit[sort_idx]

    rsquared = _rsquared(signal_fit, rss(sol))

    max_reasonable_tau = 10 * (t_fit[end] - t_fit[1])
    if any(τ -> τ > max_reasonable_tau, taus_sorted)
        @warn "Multi-exponential fit may have failed — time constants unreasonably large. Consider fewer components."
    end

    return MultiexpDecayFit(
        taus_sorted, amps_sorted,
        t0, sigma, offset,
        signal_type, signal_fit .- fitted(sol), rsquared
    )
end

# =============================================================================
# Global fitting: shared time constants across multiple traces
# =============================================================================

"""
    fit_global(traces::Vector{KineticTrace}; n_exp=1, irf_width=0.15, labels=nothing) -> GlobalFitResult

Fit multiple traces simultaneously with shared time constant(s) τ.

Supports single-exponential (`n_exp=1`) and multi-exponential (`n_exp>1`)
global analysis. All traces share the same time constants, IRF width, and
time zero, while amplitudes and offsets are fitted per-trace.

# Keywords
- `n_exp::Int=1`: Number of exponential components
- `irf_width::Float64=0.15`: Initial guess for IRF σ in ps
- `labels=nothing`: Optional trace labels (defaults to "Trace 1", "Trace 2", ...)

# Returns
`GlobalFitResult` with shared `taus` and per-trace `amplitudes` matrix.

# Examples
```julia
# Single-exponential global fit
result = fit_global([trace_esa, trace_gsb])

# Multi-exponential with 2 shared time constants
result = fit_global([trace1, trace2, trace3]; n_exp=2)
report(result)
```
"""
function fit_global(traces::Vector{KineticTrace}; n_exp::Int=1, irf_width::Float64=0.15, labels=nothing)
    n_traces = length(traces)
    @assert n_traces >= 2 "Need at least 2 traces for global fitting"
    @assert n_exp >= 1 "n_exp must be at least 1"

    if isnothing(labels)
        labels = ["Trace $i" for i in 1:n_traces]
    end

    # Bootstrap initial guesses from individual fits
    individual_fits = [fit_exp_decay(tr; n_exp=n_exp, irf=true, irf_width=irf_width) for tr in traces]

    # Extract shared parameter inits (average across traces)
    if n_exp == 1
        taus_init = [mean([f.tau for f in individual_fits])]
        sigma_init = mean([f.sigma for f in individual_fits])
        t0_init = mean([f.t0 for f in individual_fits])
        amps_init = [[f.amplitude] for f in individual_fits]
        offsets_init = [f.offset for f in individual_fits]
    else
        taus_init = [mean([f.taus[j] for f in individual_fits]) for j in 1:n_exp]
        sigma_init = mean([f.sigma for f in individual_fits])
        t0_init = mean([f.t0 for f in individual_fits])
        amps_init = [f.amplitudes for f in individual_fits]
        offsets_init = [f.offset for f in individual_fits]
    end

    # Parameter layout: [τ₁...τₙ, σ, t₀, A₁₁...A₁ₙ, off₁, A₂₁...A₂ₙ, off₂, ...]
    n_shared = n_exp + 2
    n_per_trace = n_exp + 1

    p0 = Float64[]
    append!(p0, taus_init)
    push!(p0, sigma_init)
    push!(p0, t0_init)
    for i in 1:n_traces
        append!(p0, amps_init[i])
        push!(p0, offsets_init[i])
    end

    total_len = sum(length(tr.time) for tr in traces)

    function global_model(p, dummy_x)
        # @views + inline abs: this objective is called once per Jacobian
        # column per iteration, so per-call slice copies dominate allocations
        sigma = abs(p[n_exp+1])
        t0 = p[n_exp+2]

        y_pred = similar(p, total_len)
        idx = 1
        for i in 1:n_traces
            base = n_shared + (i-1) * n_per_trace
            offset = p[base+n_exp+1]
            t_vec = traces[i].time

            for t in t_vec
                acc = offset
                for j in 1:n_exp
                    acc += _exp_decay_irf_conv(t, p[base+j], abs(p[j]), t0, sigma)
                end
                y_pred[idx] = acc
                idx += 1
            end
        end
        return y_pred
    end

    x_all = Float64[]
    y_all = Float64[]
    for tr in traces
        append!(x_all, tr.time)
        append!(y_all, tr.signal)
    end

    prob = NonlinearCurveFitProblem(global_model, p0, x_all, y_all)
    sol = solve(prob, _FIT_ALG)
    p_opt = coef(sol)

    # Extract results
    taus_fit = abs.(p_opt[1:n_exp])
    sigma = abs(p_opt[n_exp+1])
    t0 = p_opt[n_exp+2]

    amplitudes = Matrix{Float64}(undef, n_traces, n_exp)
    offsets = Vector{Float64}(undef, n_traces)
    for i in 1:n_traces
        base = n_shared + (i-1) * n_per_trace
        amplitudes[i, :] = p_opt[base+1:base+n_exp]
        offsets[i] = p_opt[base+n_exp+1]
    end

    # Sort time constants fast→slow, reorder amplitude columns to match
    sort_idx = sortperm(taus_fit)
    taus_sorted = taus_fit[sort_idx]
    amplitudes = amplitudes[:, sort_idx]

    # Compute per-trace residuals and R²
    residuals_vec = Vector{Vector{Float64}}(undef, n_traces)
    rsquared_individual = zeros(n_traces)

    # data − fit, computed explicitly (see fit_decay_irf)
    y_pred_all = fitted(sol)
    resid_all = y_all .- y_pred_all
    idx = 1
    for i in 1:n_traces
        n_pts = length(traces[i].time)
        residuals_vec[i] = resid_all[idx:idx+n_pts-1]
        rsquared_individual[i] = _rsquared(traces[i].signal, y_pred_all[idx:idx+n_pts-1])
        idx += n_pts
    end

    rsquared_global = _rsquared(y_all, rss(sol))

    return GlobalFitResult(
        taus_sorted, sigma, t0,
        amplitudes, offsets,
        labels, nothing,
        nothing, nothing,
        rsquared_global, rsquared_individual,
        residuals_vec
    )
end

"""
    fit_global(matrix::TimeResolvedMatrix; n_exp=1, irf_width=0.15, λ=nothing,
               max_wavelengths=200) -> GlobalFitResult

Global analysis of a TimeResolvedMatrix, extracting traces at each wavelength.

Returns a `GlobalFitResult` with the `wavelengths` field populated,
enabling decay-associated spectra (DAS) via `das(result)`.

# Keywords
- `n_exp::Int=1`: Number of exponential components
- `irf_width::Float64=0.15`: Initial guess for IRF σ in ps
- `λ=nothing`: Specific wavelengths to fit. If `nothing`, fits all wavelengths.
- `max_wavelengths::Int=200`: Refuse to fit more wavelengths than this.
  Every wavelength adds `n_exp + 1` free nonlinear parameters plus one
  bootstrap IRF fit, so an unrestricted fit on CCD-resolution data
  (e.g. 2048 pixels) builds a multi-gigabyte Jacobian. Pass a subset via
  `λ` (e.g. evenly spaced or SVD-selected wavelengths), or raise
  `max_wavelengths` explicitly if you accept the cost.

# Examples
```julia
result = fit_global(matrix; n_exp=2)
report(result)

# Get decay-associated spectra
spectra = das(result)  # n_exp × n_wavelengths matrix

# Broadband CCD data: fit a coarse wavelength subset
result = fit_global(matrix; n_exp=2, λ=range(450, 750, length=50))
```
"""
function fit_global(matrix::TimeResolvedMatrix; n_exp::Int=1, irf_width::Float64=0.15, λ=nothing,
                    max_wavelengths::Int=200)
    if isnothing(λ)
        wavelengths = matrix.wavelength
    else
        wavelengths = collect(Float64, λ)
    end

    n_wl = length(wavelengths)
    if n_wl > max_wavelengths
        throw(ArgumentError(
            "fit_global would fit $n_wl wavelengths, above max_wavelengths=$max_wavelengths. " *
            "Each wavelength adds $(n_exp + 1) free nonlinear parameters plus a bootstrap IRF fit " *
            "($(n_exp + 2 + n_wl * (n_exp + 1)) total parameters here), which is prohibitively " *
            "expensive at CCD resolution. Pass a wavelength subset via λ (e.g. " *
            "λ=range($(round(first(wavelengths), digits=1)), $(round(last(wavelengths), digits=1)), length=50) " *
            "or an SVD/coarse-grained selection), or raise max_wavelengths if you accept the cost."))
    end

    traces = KineticTrace[]
    actual_wavelengths = Float64[]
    for wl in wavelengths
        tr = matrix[λ=wl]
        push!(traces, tr)
        push!(actual_wavelengths, tr.wavelength)
    end

    labels = [string(round(wl, digits=1)) for wl in actual_wavelengths]

    result = fit_global(traces; n_exp=n_exp, irf_width=irf_width, labels=labels)

    # Carry the matrix's spectral-axis tokens so DAS plots derive the x-label.
    sq = get(matrix.metadata, :xquantity, nothing)
    su = get(matrix.metadata, :xunit, nothing)
    sq = isnothing(sq) ? nothing : Symbol(sq)
    su = isnothing(su) ? nothing : Symbol(su)

    # Return a new GlobalFitResult with wavelengths populated
    return GlobalFitResult(
        result.taus, result.sigma, result.t0,
        result.amplitudes, result.offsets,
        result.labels, actual_wavelengths,
        sq, su,
        result.rsquared, result.rsquared_individual,
        result.residuals
    )
end

# =============================================================================
# fit_lifetime_spectrum: per-wavelength-bin decay fitting
# =============================================================================

"""
    fit_lifetime_spectrum(m::TimeResolvedMatrix; n_exp=1, nbins=32,
                          t_range=nothing, min_signal=0.0) -> LifetimeSpectrumResult

Fit an exponential decay in each of `nbins` equal-width wavelength bins.

Each bin's columns are averaged into a kinetic trace and fitted with
[`fit_exp_decay`](@ref). Bins with no wavelength points, peak |signal| below
`min_signal`, or a failed fit are skipped (NaN entries, `fitted[i] == false`).

# Arguments
- `m`: time × wavelength matrix
- `n_exp`: exponential components per bin fit (default 1)
- `nbins`: number of equal-width wavelength bins (default 32)
- `t_range`: optional `(t_lo, t_hi)` fit window passed to `fit_exp_decay`
- `min_signal`: skip bins whose peak |signal| is below this (default 0.0 = fit all)
"""
function fit_lifetime_spectrum(m::TimeResolvedMatrix; n_exp::Int=1, nbins::Int=32,
                               t_range=nothing, min_signal::Real=0.0)
    nbins >= 1 || throw(ArgumentError("nbins must be >= 1, got $nbins"))
    n_exp >= 1 || throw(ArgumentError("n_exp must be >= 1, got $n_exp"))
    wl_lo, wl_hi = extrema(m.wavelength)
    edges = range(wl_lo, wl_hi; length=nbins + 1)

    centers = fill(NaN, nbins)
    taus = fill(NaN, nbins, n_exp)
    amplitudes = fill(NaN, nbins, n_exp)
    rsq = fill(NaN, nbins)
    fitted = falses(nbins)

    for i in 1:nbins
        cols = i == nbins ?
            findall(w -> edges[i] <= w <= edges[i+1], m.wavelength) :
            findall(w -> edges[i] <= w < edges[i+1], m.wavelength)
        isempty(cols) && continue
        centers[i] = mean(view(m.wavelength, cols))
        sig = vec(mean(view(m.data, :, cols), dims=2))
        maximum(abs, sig) < min_signal && continue

        trace = KineticTrace(copy(m.time), sig;
                             wavelength=centers[i], metadata=copy(m.metadata))
        fit = try
            fit_exp_decay(trace; n_exp=n_exp, t_range=t_range)
        catch e
            e isa InterruptException && rethrow()
            continue
        end

        fit isa AbstractDecayFit || continue
        ts = _taus(fit)
        length(ts) == n_exp || continue
        taus[i, :] .= ts
        amplitudes[i, :] .= _amplitudes(fit)
        rsq[i] = fit.rsquared
        fitted[i] = true
    end

    return LifetimeSpectrumResult(centers, taus, amplitudes, rsq, fitted, n_exp)
end

# =============================================================================
# Predict functions for fit results
# =============================================================================

function predict(fit::ExpDecayFit, time::AbstractVector)
    if isnan(fit.sigma)
        return [t >= fit.t0 ? fit.amplitude * exp(-(t - fit.t0) / fit.tau) + fit.offset : fit.offset
                for t in time]
    else
        return [_exp_decay_irf_conv(t, fit.amplitude, fit.tau, fit.t0, fit.sigma) + fit.offset
                for t in time]
    end
end

predict(fit::ExpDecayFit, trace::KineticTrace) = predict(fit, trace.time)

function predict(fit::GlobalFitResult, traces::Vector{KineticTrace})
    n = length(traces)
    curves = Vector{Vector{Float64}}(undef, n)
    for i in 1:n
        amps = fit.amplitudes[i, :]
        curves[i] = [_multiexp_irf_conv(t, fit.taus, amps, fit.t0, fit.sigma, fit.offsets[i])
                     for t in traces[i].time]
    end
    return curves
end

function predict(fit::GlobalFitResult, matrix::TimeResolvedMatrix)
    # The reconstruction lives on the FITTED wavelength axis, which may be a
    # subset of the matrix grid (fit_global(matrix; λ=subset)). Column j of
    # the output corresponds to fit.wavelengths[j] / fit.amplitudes[j, :].
    wavelengths = collect(Float64, something(fit.wavelengths, matrix.wavelength))
    reconstructed = Matrix{Float64}(undef, length(matrix.time), length(wavelengths))

    for j in eachindex(wavelengths)
        amps = fit.amplitudes[j, :]
        for (i, t) in enumerate(matrix.time)
            reconstructed[i, j] = _multiexp_irf_conv(t, fit.taus, amps, fit.t0, fit.sigma, fit.offsets[j])
        end
    end

    metadata = copy(matrix.metadata)
    metadata[:reconstructed] = true
    return TimeResolvedMatrix(matrix.time, wavelengths, reconstructed, metadata)
end


function predict(fit::MultiexpDecayFit, time::AbstractVector)
    if isnan(fit.sigma)
        return [begin
            val = fit.offset
            for i in eachindex(fit.taus)
                if t >= fit.t0
                    val += fit.amplitudes[i] * exp(-(t - fit.t0) / fit.taus[i])
                end
            end
            val
        end for t in time]
    else
        return [_multiexp_irf_conv(t, fit.taus, fit.amplitudes, fit.t0, fit.sigma, fit.offset)
                for t in time]
    end
end

predict(fit::MultiexpDecayFit, trace::KineticTrace) = predict(fit, trace.time)

# Evaluate A·exp(-((t - t₀)/τ)^β) + offset at each time point. Pre-origin
# times clamp to tt = max(t - t₀, 0) — negative^fractional would be complex —
# so the model returns amplitude + offset (the value at the decay origin)
# for t < t₀. Docstring-less like the sibling predict methods: docs CI
# (checkdocs=:exports) flags method docstrings that aren't @docs-included.
function predict(fit::StretchedDecayFit, time::AbstractVector)
    return [begin
        tt = max(t - fit.t0, zero(eltype(time)))
        fit.amplitude * exp(-(tt / fit.tau)^fit.beta) + fit.offset
    end for t in time]
end

predict(fit::StretchedDecayFit, trace::KineticTrace) = predict(fit, trace.time)

# =============================================================================
# TA Spectrum Fitting (generalized N-peak model)
# =============================================================================

const _PEAK_SIGNS = Dict(:esa => 1, :gsb => -1, :se => -1, :positive => 1, :negative => -1)

function _build_ta_model(fns, signs, npps, fit_offset)
    function model(p, x)
        y = similar(p, length(x))
        fill!(y, zero(eltype(p)))

        p_idx = 1
        for i in eachindex(fns)
            npp = npps[i]
            peak_p = @view p[p_idx:p_idx+npp-1]
            y .+= signs[i] .* fns[i](peak_p, x)
            p_idx += npp
        end

        if fit_offset
            y .+= p[p_idx]
        end

        return y
    end
    return model
end

function _ta_initial_guesses(ν, y, peak_specs, signs, npps, fit_offset)
    p0 = Float64[]

    n_pos = count(s -> s > 0, signs)
    n_neg = count(s -> s < 0, signs)

    # Detect positive peaks (ESA-type) using find_peaks on the raw signal
    pos_peaks = PeakInfo[]
    if n_pos > 0
        detected = find_peaks(ν, y; min_prominence=0.01)
        if length(detected) < n_pos
            detected = _synthesize_peak_guesses(ν, y, n_pos, detected)
        elseif length(detected) > n_pos
            sort!(detected, by=p -> p.prominence, rev=true)
            detected = detected[1:n_pos]
            sort!(detected, by=p -> p.position)
        end
        pos_peaks = detected
    end

    # Detect negative peaks (GSB/SE-type) by inverting the signal
    neg_peaks = PeakInfo[]
    if n_neg > 0
        detected = find_peaks(ν, -y; min_prominence=0.01)
        if length(detected) < n_neg
            detected = _synthesize_peak_guesses(ν, -y, n_neg, detected)
        elseif length(detected) > n_neg
            sort!(detected, by=p -> p.prominence, rev=true)
            detected = detected[1:n_neg]
            sort!(detected, by=p -> p.position)
        end
        neg_peaks = detected
    end

    pos_idx = 0
    neg_idx = 0

    for (i, (label, fn)) in enumerate(peak_specs)
        if signs[i] > 0
            pos_idx += 1
            pk = pos_peaks[pos_idx]
        else
            neg_idx += 1
            pk = neg_peaks[neg_idx]
        end

        push!(p0, pk.prominence)
        push!(p0, pk.position)
        push!(p0, _width_guess(pk.width, fn))

        # Every 4-parameter model needs its 4th guess (voigt γ, fano q,
        # pseudo-voigt mixing) or the parameter vector misaligns downstream.
        if npps[i] >= 4
            fourth = _MODEL_INFO[fn].fourth_p0
            isnothing(fourth) || push!(p0, fourth(pk.width))
        end
    end

    if fit_offset
        push!(p0, 0.0)
    end

    return p0
end

"""
    fit_ta_spectrum(spec::Spectrum; kwargs...) -> TASpectrumFit

Fit a transient absorption spectrum with N peaks of arbitrary lineshape.

Uses `find_peaks` to automatically detect initial peak positions from the data,
so multiple well-separated peaks of the same type (e.g., three GSB peaks for
W(CO)₆) are initialized correctly.

# Keywords
- `peaks=[:esa, :gsb]` — Peak types. Each element is either a `Symbol`
  (`:esa`, `:gsb`, `:se`, `:positive`, `:negative`) or a `(Symbol, Function)`
  tuple specifying label and lineshape model.
- `model=gaussian` — Default lineshape for peaks specified as symbols only.
- `region=nothing` — Optional `(x_min, x_max)` fitting region.
- `fit_offset=false` — Whether to fit a constant offset.
- `p0=nothing` — Manual initial parameter vector. Overrides automatic detection.

# Peak signs
- `:esa`, `:positive` → +1 (positive ΔA)
- `:gsb`, `:se`, `:negative` → -1 (negative ΔA)

# Examples
```julia
# Default: 1 Gaussian ESA + 1 Gaussian GSB
result = fit_ta_spectrum(spec)

# Three GSB peaks (e.g., W(CO)₆ carbonyl stretches)
result = fit_ta_spectrum(spec; peaks=[:esa, :esa, :esa, :gsb, :gsb, :gsb])

# Per-peak lineshapes
result = fit_ta_spectrum(spec; peaks=[(:esa, lorentzian), (:gsb, gaussian)])

# Access results
result[:esa].center      # first ESA peak
result[2].center         # second peak by index
anharmonicity(result)    # GSB - ESA center (only if exactly 1 of each)
predict(result, ν)       # full fitted curve
predict_peak(result, 1)  # single peak contribution
```
"""
function fit_ta_spectrum(spec::Spectrum;
                         peaks::AbstractVector=[:esa, :gsb],
                         model::Function=gaussian,
                         region=nothing,
                         fit_offset::Bool=false,
                         p0::Union{Nothing, AbstractVector}=nothing)

    xv = xdata(spec)
    yv = ydata(spec)
    if isnothing(region)
        ν = collect(Float64, xv)
        y = collect(Float64, yv)
    else
        mask = (xv .>= region[1]) .& (xv .<= region[2])
        ν = collect(Float64, xv[mask])
        y = collect(Float64, yv[mask])
    end

    peak_specs = if eltype(peaks) <: Symbol
        [(label, model) for label in peaks]
    else
        [(label, fn) for (label, fn) in peaks]
    end

    n_peaks = length(peak_specs)

    signs = Int[]
    for (label, _) in peak_specs
        haskey(_PEAK_SIGNS, label) || throw(ArgumentError(
            "Unknown peak type :$label. Use :esa, :gsb, :se, :positive, or :negative."))
        push!(signs, _PEAK_SIGNS[label])
    end

    fns = [fn for (_, fn) in peak_specs]
    npps = [_n_peak_params(fn) for fn in fns]

    p0_use = if isnothing(p0)
        _ta_initial_guesses(ν, y, peak_specs, signs, npps, fit_offset)
    else
        collect(Float64, p0)
    end

    composite = _build_ta_model(fns, signs, npps, fit_offset)
    prob = NonlinearCurveFitProblem(composite, p0_use, ν, y)
    sol = solve(prob, _FIT_ALG)
    p_opt = coef(sol)

    ta_peaks = TAPeak[]
    idx = 1
    for i in 1:n_peaks
        label, fn = peak_specs[i]
        npp = npps[i]
        amp = abs(p_opt[idx])
        center = p_opt[idx + 1]
        width = abs(p_opt[idx + 2])
        idx += npp

        push!(ta_peaks, TAPeak(label, _model_name(fn), center, width, amp))
    end

    offset_val = fit_offset ? p_opt[end] : 0.0

    return TASpectrumFit(
        ta_peaks, offset_val,
        _rsquared(y, rss(sol)), y .- fitted(sol),
        collect(p_opt), fns, signs, npps, fit_offset, ν
    )
end

function predict(fit::TASpectrumFit)
    return predict(fit, fit._x)
end

function predict(fit::TASpectrumFit, x::AbstractVector)
    x_f = collect(Float64, x)
    y = zeros(length(x_f))
    idx = 1
    for i in eachindex(fit._peak_fns)
        npp = fit._peak_npp[i]
        peak_p = fit._coef[idx:idx+npp-1]
        y .+= fit._peak_signs[i] .* fit._peak_fns[i](peak_p, x_f)
        idx += npp
    end
    if fit._fit_offset
        y .+= fit._coef[end]
    end
    return y
end

predict(fit::TASpectrumFit, spec::Spectrum) = predict(fit, xdata(spec))

function predict_peak(fit::TASpectrumFit, i::Int)
    return predict_peak(fit, i, fit._x)
end

function predict_peak(fit::TASpectrumFit, i::Int, x::AbstractVector)
    1 <= i <= length(fit.peaks) || throw(BoundsError(fit.peaks, i))
    x_f = collect(Float64, x)
    idx = 1
    for j in 1:i-1
        idx += fit._peak_npp[j]
    end
    npp = fit._peak_npp[i]
    peak_p = fit._coef[idx:idx+npp-1]
    return fit._peak_signs[i] .* fit._peak_fns[i](peak_p, x_f)
end
