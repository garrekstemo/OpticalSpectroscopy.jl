"""
Data types for spectroscopy analysis.
"""

# =============================================================================
# Abstract types for spectroscopy data
# =============================================================================

"""
    AbstractSpectroscopyData

Abstract base type for all spectroscopy data types.

Subtypes must implement the following interface:
- `xdata(d)` — Primary x-axis data (Vector{Float64})
- `ydata(d)` — Signal data (Vector{Float64}) or secondary axis for 2D
- `zdata(d)` — Matrix data for 2D types, `nothing` for 1D
- `xlabel(d)` — X-axis label string
- `ylabel(d)` — Y-axis or signal label string
- `is_matrix(d)` — Whether data is 2D (returns Bool)

Optional (have defaults):
- `source_file(d)` — Source filename (default: `""`)
- `npoints(d)` — Number of data points (default: `length(xdata(d))`, tuple for 2D)
- `title(d)` — Display title (default: `source_file(d)`)

This enables uniform handling in data viewers and plotting functions
while maintaining semantic field names in each concrete type.

# Example
```julia
# Works for any spectroscopy data type
function plot_data(data::AbstractSpectroscopyData)
    if is_matrix(data)
        heatmap(xdata(data), ydata(data), zdata(data)')
    else
        lines(xdata(data), ydata(data))
    end
end
```
"""
abstract type AbstractSpectroscopyData end

# Interface functions (default implementations raise errors)

"""
    xdata(d::AbstractSpectroscopyData) -> Vector{Float64}

Return the primary x-axis data (e.g. time for [`KineticTrace`](@ref), the spectral
axis for a [`Spectrum`](@ref) slice, wavelength for [`TimeResolvedMatrix`](@ref)).
Every concrete subtype implements this.
"""
xdata(::AbstractSpectroscopyData) = error("xdata not implemented for this type")

"""
    ydata(d::AbstractSpectroscopyData) -> Vector{Float64}

Return the signal data for 1D types, or the secondary axis for 2D types
(e.g. the time axis of a [`TimeResolvedMatrix`](@ref)).
"""
ydata(::AbstractSpectroscopyData) = error("ydata not implemented for this type")

"""
    zdata(d::AbstractSpectroscopyData) -> Union{Matrix{Float64}, Nothing}

Return the signal matrix for 2D types (e.g. the signal matrix of a
[`TimeResolvedMatrix`](@ref), the intensity map of a [`PLMap`](@ref)).
Returns `nothing` for 1D data.
"""
zdata(::AbstractSpectroscopyData) = nothing  # Default: not a matrix

"""
    xlabel(d::AbstractSpectroscopyData) -> String

Return a display label for the x axis (e.g. `"Time (ps)"`, `"Wavenumber (cm⁻¹)"`).
"""
xlabel(::AbstractSpectroscopyData) = "X"

"""
    ylabel(d::AbstractSpectroscopyData) -> String

Return a display label for the y axis or signal (e.g. `"ΔA"`).
"""
ylabel(::AbstractSpectroscopyData) = "Y"

"""
    zlabel(d::AbstractSpectroscopyData) -> String

Return a display label for the signal of 2D data (e.g. `"ΔA"`, `"PL Intensity"`).
"""
zlabel(::AbstractSpectroscopyData) = "Signal"

"""
    is_matrix(d::AbstractSpectroscopyData) -> Bool

Whether the data is 2D (`true` for [`TimeResolvedMatrix`](@ref) and [`PLMap`](@ref);
`false` for 1D types). When `true`, [`zdata`](@ref) returns a matrix.
"""
is_matrix(::AbstractSpectroscopyData) = false  # Default: 1D data

"""
    source_file(d::AbstractSpectroscopyData) -> String

Return the source filename for the data.
"""
source_file(::AbstractSpectroscopyData) = ""

"""
    npoints(d::AbstractSpectroscopyData) -> Int
    npoints(d::TimeResolvedMatrix) -> Tuple{Int,Int}

Return the number of data points. For 1D data, returns an `Int`.
For 2D data (`TimeResolvedMatrix`), returns `(n_time, n_wavelength)`.
"""
npoints(d::AbstractSpectroscopyData) = length(xdata(d))

"""
    title(d::AbstractSpectroscopyData) -> String

Return a display title for the data. Defaults to `source_file(d)`.
"""
title(d::AbstractSpectroscopyData) = source_file(d)

# Guard for generic 1D dispatches: matrix types must be sliced first.
function _check_1d(d::AbstractSpectroscopyData, fname::AbstractString)
    is_matrix(d) && throw(ArgumentError(
        "$fname requires 1D data; got 2D $(nameof(typeof(d))). " *
        "Extract a 1D slice first (e.g. spectral_slice or matrix[t=1.0])."))
    return nothing
end

# =============================================================================
# Transient Absorption types (unified API)
# =============================================================================

"""
    KineticTrace <: AbstractSpectroscopyData

Single-wavelength kinetic trace: signal versus time.

Covers transient-absorption kinetics (ΔA) and time-resolved PL decays
(counts). Signal/axis semantics live in token metadata (:yquantity/:yunit
for the signal; :xunit for the time axis).

# Fields
- `time::Vector{Float64}`: Time axis
- `signal::Vector{Float64}`: Signal at each time point
- `wavelength::Float64`: Probe/emission wavelength, NaN if unknown
- `metadata::Dict{Symbol,Any}`: Additional info
"""
struct KineticTrace <: AbstractSpectroscopyData
    time::Vector{Float64}
    signal::Vector{Float64}
    wavelength::Float64
    metadata::Dict{Symbol,Any}

    function KineticTrace(time, signal, wavelength, metadata)
        length(time) == length(signal) || throw(ArgumentError(
            "KineticTrace: time and signal must have equal length; " *
            "got $(length(time)) time points and $(length(signal)) signal points"))
        new(time, signal, wavelength, metadata)
    end
end

# Constructor with default wavelength
KineticTrace(time, signal; wavelength=NaN, metadata=Dict{Symbol,Any}()) =
    KineticTrace(time, signal, wavelength, metadata)

# AbstractSpectroscopyData interface
xdata(t::KineticTrace) = t.time
ydata(t::KineticTrace) = t.signal
xlabel(t::KineticTrace) = _time_label(t.metadata, :xunit)
ylabel(t::KineticTrace) = _signal_label(t.metadata)
source_file(t::KineticTrace) = get(t.metadata, :filename, get(t.metadata, :source, ""))

# Semantic accessors
"""
    delay(t::KineticTrace) -> Vector{Float64}

Return the time axis.
"""
delay(t::KineticTrace) = t.time

"""
    signal(t::KineticTrace) -> Vector{Float64}

Return the signal values (ΔA, counts, or other units depending on data source).
"""
signal(t::KineticTrace) = t.signal

# Pretty printing
function Base.show(io::IO, t::KineticTrace)
    n = length(t.time)
    tu = (u = Symbol(get(t.metadata, :xunit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    t_range = "$(round(minimum(t.time), digits=2)) to $(round(maximum(t.time), digits=2)) $tu"
    src = source_file(t)
    print(io, "KineticTrace: $n points, $t_range")
    isempty(src) || print(io, " ($(src))")
end

function Base.show(io::IO, ::MIME"text/plain", t::KineticTrace)
    tu = (u = Symbol(get(t.metadata, :xunit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    println(io, "KineticTrace")
    println(io, "  Time points: $(length(t.time))")
    println(io, "  Time range:  $(round(minimum(t.time), digits=2)) to $(round(maximum(t.time), digits=2)) $tu")
    println(io, "  Wavelength:  $(isnan(t.wavelength) ? "unknown" : t.wavelength)")
    src = source_file(t)
    isempty(src) || println(io, "  File:        $src")
    haskey(t.metadata, :mode) && println(io, "  Mode:        $(t.metadata[:mode])")
end

"""
    SweepData(X, Y, DC)

Per-sweep multi-channel lock-in detector data. Each matrix is
`n_points × n_sweeps`. `NaN` marks an unmeasured point (e.g. from an
aborted partial sweep). This is the un-averaged counterpart to
[`KineticTrace`](@ref) — sweep-resolved raw output before collapsing across
sweeps.

# Fields
- `X::Matrix{Float64}`: in-phase (signal) component per sweep
- `Y::Matrix{Float64}`: quadrature component per sweep
- `DC::Matrix{Float64}`: probe DC level per sweep (`0.0` if unavailable)

# Examples

```julia
n_points, n_sweeps = 100, 10
X  = zeros(n_points, n_sweeps)
Y  = zeros(n_points, n_sweeps)
DC = zeros(n_points, n_sweeps)
sweeps = SweepData(X, Y, DC)
```
"""
struct SweepData
    X::Matrix{Float64}
    Y::Matrix{Float64}
    DC::Matrix{Float64}
end

function Base.show(io::IO, s::SweepData)
    n_pts, n_sweeps = size(s.X)
    print(io, "SweepData: $(n_pts) points × $(n_sweeps) sweeps")
end

function Base.show(io::IO, ::MIME"text/plain", s::SweepData)
    n_pts, n_sweeps = size(s.X)
    n_nan = count(isnan, s.X)
    println(io, "SweepData")
    println(io, "  Points:  $n_pts")
    println(io, "  Sweeps:  $n_sweeps")
    println(io, "  NaN X:   $n_nan")
end

"""
    TAPeak

Information about a single peak in a TA spectrum fit.

# Fields
- `label::Symbol` — Peak type (`:esa`, `:gsb`, `:se`, `:positive`, `:negative`)
- `model::String` — Lineshape model name (`"gaussian"`, `"lorentzian"`, `"pseudo_voigt"`)
- `center::Float64` — Peak center position
- `width::Float64` — Width parameter (sigma for gaussian/voigt, fwhm for lorentzian)
- `amplitude::Float64` — Peak amplitude (always positive; sign determined by label)
"""
struct TAPeak
    label::Symbol
    model::String
    center::Float64
    width::Float64
    amplitude::Float64
end

function Base.show(io::IO, pk::TAPeak)
    print(io, "TAPeak(:$(pk.label), $(round(pk.center, digits=1)), $(pk.model))")
end

function Base.show(io::IO, ::MIME"text/plain", pk::TAPeak)
    label = uppercase(string(pk.label))
    println(io, "TAPeak ($label)")
    println(io, "  Model:     $(pk.model)")
    println(io, "  Center:    $(round(pk.center, digits=2))")
    println(io, "  Width:     $(round(pk.width, digits=2))")
    print(io, "  Amplitude: $(round(pk.amplitude, sigdigits=4))")
end

"""
    TASpectrumFit

Result of TA spectrum fitting with N peaks of arbitrary lineshape.

Supports any combination of ESA, GSB, and SE peaks, each with its own
lineshape model (Gaussian, Lorentzian, pseudo-Voigt).

# Access peaks
- `result.peaks` — Vector of `TAPeak`
- `result[i]` — i-th peak
- `result[:esa]` — first peak with label `:esa`
- `anharmonicity(result)` — GSB center minus ESA center (NaN if not applicable)

# Fields
- `peaks::Vector{TAPeak}` — Fitted peak parameters
- `offset`, `rsquared`, `residuals` — Fit metadata
"""
struct TASpectrumFit
    peaks::Vector{TAPeak}
    offset::Float64
    rsquared::Float64
    residuals::Vector{Float64}
    _coef::Vector{Float64}
    _peak_fns::Vector{Function}
    _peak_signs::Vector{Int}
    _peak_npp::Vector{Int}
    _fit_offset::Bool
    _x::Vector{Float64}
end

Base.getindex(r::TASpectrumFit, i::Int) = r.peaks[i]
Base.length(r::TASpectrumFit) = length(r.peaks)

xdata(r::TASpectrumFit) = r._x
wavenumber(r::TASpectrumFit) = r._x

function Base.getindex(r::TASpectrumFit, label::Symbol)
    idx = findfirst(p -> p.label == label, r.peaks)
    isnothing(idx) && error("No peak with label :$label. Available: $(unique([p.label for p in r.peaks]))")
    return r.peaks[idx]
end

"""
    anharmonicity(fit::TASpectrumFit) -> Float64

Compute the anharmonicity (GSB center - ESA center) from a TA spectrum fit.
Returns `NaN` if there is not exactly one ESA and one GSB peak.
"""
function anharmonicity(fit::TASpectrumFit)
    esa = filter(p -> p.label == :esa, fit.peaks)
    gsb = filter(p -> p.label == :gsb, fit.peaks)
    if length(esa) == 1 && length(gsb) == 1
        return gsb[1].center - esa[1].center
    else
        return NaN
    end
end

function Base.show(io::IO, fit::TASpectrumFit)
    labels = join([uppercase(string(p.label)) for p in fit.peaks], "+")
    print(io, "TASpectrumFit: $labels, R² = $(round(fit.rsquared, digits=4))")
end

function Base.show(io::IO, ::MIME"text/plain", fit::TASpectrumFit)
    n = length(fit.peaks)

    println(io, "TASpectrumFit ($n peak$(n == 1 ? "" : "s"))")
    println(io)
    println(io, "  Peak   Type   Model          Center        Width      Amplitude")
    println(io, "  ─────────────────────────────────────────────────────────────────")
    for (i, pk) in enumerate(fit.peaks)
        label = rpad(uppercase(string(pk.label)), 6)
        model = rpad(pk.model, 14)
        center = lpad(round(pk.center, digits=1), 10)
        width = lpad(round(pk.width, digits=2), 10)
        amp = lpad(round(pk.amplitude, sigdigits=3), 10)
        println(io, "  $(lpad(i, 4))   $label $model $center $width $amp")
    end

    anharm = anharmonicity(fit)
    if !isnan(anharm)
        println(io)
        println(io, "  Anharmonicity: $(round(anharm, digits=1))")
    end

    if fit.offset != 0.0
        println(io, "  Offset:        $(round(fit.offset, sigdigits=3))")
    end

    println(io)
    print(io, "  R² = $(round(fit.rsquared, digits=5))")
end

# =============================================================================
# Exponential decay fit types
# =============================================================================

"""
    ExpDecayFit

Result of exponential decay fitting with instrument response function convolution.

# Fields
- `amplitude`, `tau`, `t0`, `sigma`, `offset`, `signal_type`, `residuals`, `rsquared`
"""
struct ExpDecayFit
    amplitude::Float64
    tau::Float64
    t0::Float64
    sigma::Float64
    offset::Float64
    signal_type::Symbol
    residuals::Vector{Float64}
    rsquared::Float64
end

function Base.show(io::IO, fit::ExpDecayFit)
    signal_str = fit.signal_type == :esa ? "ESA" : "GSB"
    print(io, "ExpDecayFit: τ = $(round(fit.tau, digits=2)) ps, R² = $(round(fit.rsquared, digits=4)) ($signal_str)")
end

function Base.show(io::IO, ::MIME"text/plain", fit::ExpDecayFit)
    signal_str = fit.signal_type == :esa ? "ESA (positive)" : "GSB (negative)"
    has_irf = !isnan(fit.sigma)

    println(io, "ExpDecayFit ($signal_str)")
    println(io)
    println(io, "  Parameter       Value")
    println(io, "  ─────────────────────────────")
    println(io, "  τ          $(lpad(round(fit.tau, digits=3), 10)) ps")
    if has_irf
        println(io, "  σ_IRF      $(lpad(round(fit.sigma, digits=3), 10)) ps")
    end
    println(io, "  t₀         $(lpad(round(fit.t0, digits=3), 10)) ps")
    println(io, "  Amplitude  $(lpad(round(fit.amplitude, sigdigits=4), 10))")
    println(io, "  Offset     $(lpad(round(fit.offset, sigdigits=4), 10))")
    println(io)
    print(io, "  R² = $(round(fit.rsquared, digits=5))")
end


"""
    MultiexpDecayFit

Result of multi-exponential decay fitting (n >= 1 components).

# Fields
- `taus::Vector{Float64}`: Time constants (sorted fast->slow)
- `amplitudes::Vector{Float64}`: Corresponding amplitudes
- `t0`, `sigma`, `offset`, `signal_type`, `residuals`, `rsquared`

# Derived properties
- `n_exp(fit)`: Number of exponential components
- `weights(fit)`: Relative amplitude weights (normalized to 100%)
"""
struct MultiexpDecayFit
    taus::Vector{Float64}
    amplitudes::Vector{Float64}
    t0::Float64
    sigma::Float64
    offset::Float64
    signal_type::Symbol
    residuals::Vector{Float64}
    rsquared::Float64
end

n_exp(fit::MultiexpDecayFit) = length(fit.taus)
function weights(fit::MultiexpDecayFit)
    total = sum(abs, fit.amplitudes)
    return total > 0 ? abs.(fit.amplitudes) ./ total : fill(1.0 / length(fit.amplitudes), length(fit.amplitudes))
end

function Base.show(io::IO, fit::MultiexpDecayFit)
    n = length(fit.taus)
    signal_str = fit.signal_type == :esa ? "ESA" : "GSB"
    tau_str = join(["$(round(τ, digits=2))" for τ in fit.taus], ", ")
    print(io, "MultiexpDecayFit: n=$n, τ = [$tau_str] ps, R² = $(round(fit.rsquared, digits=4)) ($signal_str)")
end

function Base.show(io::IO, ::MIME"text/plain", fit::MultiexpDecayFit)
    n = length(fit.taus)
    signal_str = fit.signal_type == :esa ? "ESA (positive)" : "GSB (negative)"
    has_irf = !isnan(fit.sigma)
    w = weights(fit)

    println(io, "MultiexpDecayFit ($signal_str, $n components)")
    println(io)
    println(io, "  Component     τ (ps)        Amplitude      Weight")
    println(io, "  ────────────────────────────────────────────────────")
    for i in 1:n
        println(io, "  $(lpad(i, 5))       $(lpad(round(fit.taus[i], digits=3), 8))    $(lpad(round(fit.amplitudes[i], sigdigits=4), 12))    $(lpad(round(100*w[i], digits=1), 5))%")
    end
    println(io)
    if has_irf
        println(io, "  σ_IRF      $(lpad(round(fit.sigma, digits=3), 10)) ps")
    end
    println(io, "  t₀         $(lpad(round(fit.t0, digits=3), 10)) ps")
    println(io, "  Offset     $(lpad(round(fit.offset, sigdigits=4), 10))")
    println(io)
    print(io, "  R² = $(round(fit.rsquared, digits=5))")
end

# =============================================================================
# Stretched-exponential (KWW) decay fit type
# =============================================================================

"""
    StretchedDecayFit

Result of stretched-exponential (Kohlrausch–Williams–Watts) decay fitting:

`signal(t) = A·exp(-((t - t₀)/τ)^β) + offset`

# Fields
- `amplitude::Float64`
- `tau::Float64`: Characteristic time constant
- `beta::Float64`: Stretching exponent (0 < β ≤ 1)
- `t0::Float64`: Fit-region time origin
- `offset::Float64`
- `signal_type::Symbol`: `:esa` (positive) or `:gsb` (negative)
- `residuals::Vector{Float64}`
- `rsquared::Float64`

See [`mean_lifetime`](@ref) for ⟨τ⟩ = (τ/β)·Γ(1/β).
"""
struct StretchedDecayFit
    amplitude::Float64
    tau::Float64
    beta::Float64
    t0::Float64
    offset::Float64
    signal_type::Symbol
    residuals::Vector{Float64}
    rsquared::Float64
end

"""
    mean_lifetime(fit::StretchedDecayFit) -> Float64

Mean lifetime of a stretched-exponential decay: ⟨τ⟩ = (τ/β)·Γ(1/β).
"""
mean_lifetime(fit::StretchedDecayFit) = (fit.tau / fit.beta) * gamma(1 / fit.beta)

function Base.show(io::IO, fit::StretchedDecayFit)
    print(io, "StretchedDecayFit: τ = $(round(fit.tau, digits=2)), β = $(round(fit.beta, digits=3)), R² = $(round(fit.rsquared, digits=4))")
end

function Base.show(io::IO, ::MIME"text/plain", fit::StretchedDecayFit)
    println(io, "StretchedDecayFit ($(fit.signal_type == :esa ? "ESA (positive)" : "GSB (negative)"))")
    println(io)
    println(io, "  Parameter       Value")
    println(io, "  ─────────────────────────────")
    println(io, "  τ          $(lpad(round(fit.tau, digits=3), 10))")
    println(io, "  β          $(lpad(round(fit.beta, digits=3), 10))")
    println(io, "  ⟨τ⟩        $(lpad(round(mean_lifetime(fit), digits=3), 10))")
    println(io, "  t₀         $(lpad(round(fit.t0, digits=3), 10))")
    println(io, "  Amplitude  $(lpad(round(fit.amplitude, sigdigits=4), 10))")
    println(io, "  Offset     $(lpad(round(fit.offset, sigdigits=4), 10))")
    println(io)
    print(io, "  R² = $(round(fit.rsquared, digits=5))")
end

# =============================================================================
# LifetimeSpectrumResult: per-wavelength-bin decay fits
# =============================================================================

"""
    LifetimeSpectrumResult

Result of [`fit_lifetime_spectrum`](@ref): per-wavelength-bin decay fits.

# Fields
- `wavelength::Vector{Float64}`: Bin centers (mean wavelength per bin); NaN for empty bins
- `taus::Matrix{Float64}`: Time constants, size (nbins, n_exp); NaN where skipped
- `amplitudes::Matrix{Float64}`: Amplitudes, size (nbins, n_exp); NaN where skipped
- `rsquared::Vector{Float64}`: R² per bin; NaN where skipped
- `fitted::BitVector`: Whether each bin was fitted
- `n_exp::Int`: Exponential components per fit
"""
struct LifetimeSpectrumResult
    wavelength::Vector{Float64}
    taus::Matrix{Float64}
    amplitudes::Matrix{Float64}
    rsquared::Vector{Float64}
    fitted::BitVector
    n_exp::Int
end

function Base.show(io::IO, r::LifetimeSpectrumResult)
    print(io, "LifetimeSpectrumResult: $(count(r.fitted))/$(length(r.fitted)) bins fitted, n_exp = $(r.n_exp)")
end

# =============================================================================
# Peak fitting result types
# =============================================================================

"""
    PeakFitResult

Result of peak fitting with any lineshape model.

Access parameters by name: `result[:center].value`, `result[:fwhm].err`
"""
struct PeakFitResult
    params::Vector{Symbol}
    values::Vector{Float64}
    errors::Vector{Float64}
    ci::Vector{Tuple{Float64,Float64}}
    r_squared::Float64
    rss::Float64
    mse::Float64
    npoints::Int
    region::Tuple{Float64,Float64}
    model::String
    sample_id::String
end

# Index by parameter name
function Base.getindex(r::PeakFitResult, name::Symbol)
    idx = findfirst(==(name), r.params)
    isnothing(idx) && error("Parameter :$name not found. Available: $(r.params)")
    return (value=r.values[idx], err=r.errors[idx], ci=r.ci[idx])
end

Base.haskey(r::PeakFitResult, name::Symbol) = name in r.params

function Base.show(io::IO, r::PeakFitResult)
    println(io, "PeakFitResult: $(r.sample_id)")
    println(io, "  Region: $(round(Int, r.region[1])) - $(round(Int, r.region[2])) cm⁻¹ ($(r.npoints) points)")
    println(io, "  Model:  $(r.model)")
    println(io)

    println(io, "  Parameter       Value       Std Err     95% CI")
    println(io, "  ─────────────────────────────────────────────────────────")

    function fmt_value(x)
        if abs(x) < 0.01 || abs(x) >= 10000
            return rpad(string(round(x, sigdigits=4)), 10)
        else
            return rpad(string(round(x, digits=2)), 10)
        end
    end
    fmt_ci(ci) = "($(round(ci[1], digits=2)), $(round(ci[2], digits=2)))"

    for (i, name) in enumerate(r.params)
        label = rpad(string(name), 14)
        println(io, "  $(label)  $(fmt_value(r.values[i]))  $(fmt_value(r.errors[i]))  $(fmt_ci(r.ci[i]))")
    end

    println(io)
    print(io, "  R² = $(round(r.r_squared, digits=5))    MSE = $(round(r.mse, sigdigits=4))")
end

# =============================================================================
# Multi-peak fitting result
# =============================================================================

"""
    MultiPeakFitResult

Result of multi-peak fitting (1 to N peaks with polynomial baseline).

Supports indexing by peak number: `result[1]` returns first peak's `PeakFitResult`.
"""
struct MultiPeakFitResult
    peaks::Vector{PeakFitResult}
    baseline_params::Vector{Float64}
    baseline_order::Int
    r_squared::Float64
    rss::Float64
    mse::Float64
    npoints::Int
    region::Tuple{Float64,Float64}
    sample_id::String
    _coef::Vector{Float64}
    _peak_fn::Function
    _n_peak_params::Int
    _x::Vector{Float64}
    _y::Vector{Float64}
    # Baseline-basis normalization fixed by the fit region; reused for every
    # later evaluation (predict / predict_baseline on arbitrary grids).
    _x_mid::Float64
    _x_range::Float64
end

Base.getindex(r::MultiPeakFitResult, i::Int) = r.peaks[i]
Base.length(r::MultiPeakFitResult) = length(r.peaks)
Base.iterate(r::MultiPeakFitResult) = iterate(r.peaks)
Base.iterate(r::MultiPeakFitResult, state) = iterate(r.peaks, state)
Base.firstindex(r::MultiPeakFitResult) = 1
Base.lastindex(r::MultiPeakFitResult) = length(r.peaks)

xdata(r::MultiPeakFitResult) = r._x
ydata(r::MultiPeakFitResult) = r._y

function Base.show(io::IO, r::MultiPeakFitResult)
    n = length(r.peaks)
    model = n > 0 ? r.peaks[1].model : "unknown"
    print(io, "MultiPeakFitResult: $n peak$(n == 1 ? "" : "s") ($model), R² = $(round(r.r_squared, digits=4))")
end

function Base.show(io::IO, ::MIME"text/plain", r::MultiPeakFitResult)
    n = length(r.peaks)
    model = n > 0 ? r.peaks[1].model : "unknown"

    println(io, "MultiPeakFitResult ($n peak$(n == 1 ? "" : "s"), $model)")
    println(io, "  Region: $(round(Int, r.region[1])) - $(round(Int, r.region[2])) ($(r.npoints) points)")
    if !isempty(r.sample_id)
        println(io, "  Sample: $(r.sample_id)")
    end
    println(io)

    if n > 0
        param_names = r.peaks[1].params
        header = "  Peak   " * join([rpad(string(p), 14) for p in param_names], "")
        println(io, header)
        println(io, "  " * "─"^(length(header) - 2))

        for (i, pk) in enumerate(r.peaks)
            row = "  $(lpad(i, 4))   "
            for (j, _) in enumerate(pk.params)
                v = round(pk.values[j], digits=2)
                row *= rpad(string(v), 14)
            end
            println(io, row)
        end
    end

    println(io)
    if r.baseline_order == 0
        println(io, "  Baseline: constant = $(round(r.baseline_params[1], sigdigits=4))")
    elseif r.baseline_order == 1
        println(io, "  Baseline: linear, c₀ = $(round(r.baseline_params[1], sigdigits=4)), c₁ = $(round(r.baseline_params[2], sigdigits=4))")
    else
        println(io, "  Baseline: order $(r.baseline_order), params = $(round.(r.baseline_params, sigdigits=4))")
    end

    println(io)
    print(io, "  R² = $(round(r.r_squared, digits=5))    MSE = $(round(r.mse, sigdigits=4))")
end

# =============================================================================
# FitMapResult: Batch fitting results for spatial maps
# =============================================================================

"""
    FitMapResult

Result of batch peak fitting across a spatial map (e.g., PLMap).

Contains per-pixel fit results and pre-extracted summary arrays suitable
for heatmap visualization. Pixels that were skipped (masked) or failed
to converge have `nothing` in the results matrix and `NaN` in summary arrays.

Summary arrays are 3D `(nx, ny, n_peaks)` for per-peak quantities (centers,
fwhms, amplitudes) and 2D `(nx, ny)` for per-pixel quantities (r_squareds).

Created by [`fit_map`](@ref).

# Fields
- `results::Matrix{Any}` — Per-pixel `MultiPeakFitResult` or `nothing`
- `mask::Union{BitMatrix, Nothing}` — Fit mask used (if any)
- `centers::Array{Float64, 3}` — Peak center per peak `(nx, ny, n_peaks)`
- `fwhms::Array{Float64, 3}` — FWHM per peak `(nx, ny, n_peaks)`
- `amplitudes::Array{Float64, 3}` — Amplitude per peak `(nx, ny, n_peaks)`
- `r_squareds::Matrix{Float64}` — Per-pixel R²
- `n_peaks::Int` — Number of peaks fitted
- `n_converged::Int` — Number of successfully fitted pixels
- `n_failed::Int` — Number of pixels where fitting threw an error
- `n_skipped::Int` — Number of pixels excluded by the mask
- `median_r_squared::Float64` — Median R² across converged pixels
"""
struct FitMapResult
    results::Matrix{Any}
    mask::Union{BitMatrix, Nothing}
    centers::Array{Float64, 3}
    fwhms::Array{Float64, 3}
    amplitudes::Array{Float64, 3}
    r_squareds::Matrix{Float64}
    n_peaks::Int
    n_converged::Int
    n_failed::Int
    n_skipped::Int
    median_r_squared::Float64
end

Base.getindex(r::FitMapResult, ix::Int, iy::Int) = r.results[ix, iy]
Base.size(r::FitMapResult) = size(r.results)

function Base.show(io::IO, r::FitMapResult)
    nx, ny = size(r.results)
    print(io, "FitMapResult($(nx)×$(ny), $(r.n_converged) converged, median R² = $(round(r.median_r_squared, digits=4)))")
end

function Base.show(io::IO, ::MIME"text/plain", r::FitMapResult)
    nx, ny = size(r.results)
    println(io, "FitMapResult")
    println(io, "  Grid:       $(nx) × $(ny)")
    println(io, "  Converged:  $(r.n_converged)")
    println(io, "  Failed:     $(r.n_failed)")
    println(io, "  Skipped:    $(r.n_skipped)")
    print(io, "  Median R²:  $(round(r.median_r_squared, digits=5))")
end

# =============================================================================
# TimeResolvedMatrix: 2D time-resolved spectroscopy data
# =============================================================================

"""
    TimeResolvedMatrix <: AbstractSpectroscopyData

Two-dimensional time-resolved spectroscopy data (time × wavelength).

Covers transient absorption (ΔA) and streak-camera PL (counts). Spectral
axis tokens :xquantity/:xunit; signal tokens :yquantity/:yunit; time-axis
unit :time_unit. matrix[t=…] returns a Spectrum; matrix[λ=…] returns a
KineticTrace.

# Fields
- `time::Vector{Float64}`: Time axis
- `wavelength::Vector{Float64}`: Wavelength axis (nm) or wavenumber (cm⁻¹)
- `data::Matrix{Float64}`: Signal matrix, size (n_time, n_wavelength)
- `metadata::Dict{Symbol,Any}`: Additional info

# Indexing
```julia
matrix[λ=800]     # Extract KineticTrace at λ ≈ 800 nm
matrix[t=1.0]     # Extract Spectrum at t ≈ 1.0 ps (tagged with :time_delay)
```

`matrix[t=...]` returns a `Spectrum` whose axis/signal labels derive from the
matrix's tokens (honest `"x"`/`"Signal"` floor if absent), tagged with the slice's
`:time_delay`. For a spectrum gated/integrated over a time window, use
[`spectral_slice`](@ref) or [`integrate_time`](@ref).
"""
struct TimeResolvedMatrix <: AbstractSpectroscopyData
    time::Vector{Float64}
    wavelength::Vector{Float64}
    data::Matrix{Float64}
    metadata::Dict{Symbol,Any}

    function TimeResolvedMatrix(time, wavelength, data, metadata)
        expected = (length(time), length(wavelength))
        if size(data) != expected
            if size(data) == reverse(expected)
                throw(ArgumentError(
                    "TimeResolvedMatrix: data is $(size(data)) but expected (n_time, n_wavelength) = " *
                    "$expected — data appears transposed; pass permutedims(data)"))
            else
                throw(ArgumentError(
                    "TimeResolvedMatrix: data is $(size(data)) but expected (n_time, n_wavelength) = $expected"))
            end
        end
        new(time, wavelength, data, metadata)
    end
end

TimeResolvedMatrix(time, wavelength, data; metadata=Dict{Symbol,Any}()) =
    TimeResolvedMatrix(time, wavelength, data, metadata)

xdata(m::TimeResolvedMatrix) = m.wavelength
ydata(m::TimeResolvedMatrix) = m.time
zdata(m::TimeResolvedMatrix) = m.data
is_matrix(::TimeResolvedMatrix) = true
source_file(m::TimeResolvedMatrix) = get(m.metadata, :source, "")
npoints(m::TimeResolvedMatrix) = size(m.data)

# Semantic accessors
"""
    wavelength(m::TimeResolvedMatrix) -> Vector{Float64}

Return the wavelength axis (nm) or wavenumber axis (cm⁻¹).
"""
wavelength(m::TimeResolvedMatrix) = m.wavelength

"""
    delay(m::TimeResolvedMatrix) -> Vector{Float64}

Return the time axis.
"""
delay(m::TimeResolvedMatrix) = m.time

"""
    signal(m::TimeResolvedMatrix) -> Matrix{Float64}

Return the signal matrix (n_time × n_wavelength).
"""
signal(m::TimeResolvedMatrix) = m.data

xlabel(m::TimeResolvedMatrix) = _spectral_xlabel(m.metadata)
ylabel(m::TimeResolvedMatrix) = _time_label(m.metadata, :time_unit)
zlabel(m::TimeResolvedMatrix) = _signal_label(m.metadata)

# Spectral axis unit heuristic — opt-in only; the sole caller is guess_units! (no automatic call sites).
function _detect_spectral_unit(wavelengths::AbstractVector{<:Real})
    isempty(wavelengths) && return "nm"
    minval, maxval = extrema(wavelengths)
    (minval > 1200 && maxval < 5000) ? "cm⁻¹" : "nm"
end

function Base.show(io::IO, m::TimeResolvedMatrix)
    n_time, n_wl = size(m.data)
    tu = (u = Symbol(get(m.metadata, :time_unit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    t_range = "$(round(minimum(m.time), digits=2)) to $(round(maximum(m.time), digits=2)) $tu"
    wl_range = "$(round(minimum(m.wavelength), digits=1)) to $(round(maximum(m.wavelength), digits=1))"
    print(io, "TimeResolvedMatrix: $n_time × $n_wl ($t_range, $wl_range)")
end

function Base.show(io::IO, ::MIME"text/plain", m::TimeResolvedMatrix)
    n_time, n_wl = size(m.data)
    tu = (u = Symbol(get(m.metadata, :time_unit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    println(io, "TimeResolvedMatrix")
    println(io, "  Time points:   $n_time ($(round(minimum(m.time), digits=2)) to $(round(maximum(m.time), digits=2)) $tu)")
    println(io, "  Wavelengths:   $n_wl ($(round(minimum(m.wavelength), digits=1)) to $(round(maximum(m.wavelength), digits=1)))")
    println(io, "  Data range:    $(round(minimum(m.data), sigdigits=3)) to $(round(maximum(m.data), sigdigits=3))")
    haskey(m.metadata, :source) && println(io, "  Source:        $(m.metadata[:source])")
end

# =============================================================================
# TimeResolvedMatrix indexing
# =============================================================================

function _find_nearest_idx(arr::AbstractVector, target::Real)
    _, idx = findmin(abs.(arr .- target))
    return idx
end

# A kinetic (time-axis) slice inherits matrix metadata but its x-axis is time:
# remap the matrix's time-axis unit (:time_unit) to the trace's :xunit and drop
# the matrix's spectral x-tokens (which describe wavelength, not the trace's time
# axis). Interim until the axisN refactor (§10 #3).
function _trace_metadata(m::TimeResolvedMatrix)
    md = copy(m.metadata)
    md[:xunit] = Symbol(get(m.metadata, :time_unit, :ps))
    delete!(md, :xquantity)
    return md
end

function Base.getindex(m::TimeResolvedMatrix; λ=nothing, t=nothing)
    if !isnothing(λ) && isnothing(t)
        idx = _find_nearest_idx(m.wavelength, λ)
        actual_λ = m.wavelength[idx]
        sig = m.data[:, idx]

        md = _trace_metadata(m)
        md[:extracted_from] = get(m.metadata, :source, "TimeResolvedMatrix")
        md[:requested_wavelength] = λ
        md[:actual_wavelength] = actual_λ
        md[:wavelength_index] = idx

        return KineticTrace(m.time, sig, actual_λ, md)

    elseif !isnothing(t) && isnothing(λ)
        idx = _find_nearest_idx(m.time, t)
        actual_t = m.time[idx]
        sig = m.data[idx, :]

        # Spectral slice: inherit the matrix's spectral (:xquantity/:xunit) and
        # signal (:yquantity/:yunit) tokens directly; record the fixed delay.
        md = copy(m.metadata)
        md[:extracted_from] = get(m.metadata, :source, "TimeResolvedMatrix")
        md[:time_index] = idx
        md[:time_delay] = actual_t
        haskey(m.metadata, :time_unit) &&
            (md[:time_delay_unit] = Symbol(m.metadata[:time_unit]))

        return Spectrum(m.wavelength, sig, md)

    elseif !isnothing(λ) && !isnothing(t)
        error("Cannot specify both λ and t. Use matrix[λ=...] or matrix[t=...]")
    else
        error("Must specify either λ or t for indexing. Use matrix[λ=...] or matrix[t=...]")
    end
end

# =============================================================================
# Spectrum: generic 1D steady-state spectrum
# =============================================================================

"""
    Spectrum <: AbstractSpectroscopyData

    Spectrum(x, y)
    Spectrum(x, y, metadata::AbstractDict)
    Spectrum(x, y; axis=:wavenumber, metadata...)
    Spectrum(M::AbstractMatrix)

Generic 1D steady-state spectrum: signal versus a spectral axis.

Covers FTIR, Raman, UV-Vis, photoluminescence, cavity transmission, transient-
absorption slices, and any other 1D data. Axis semantics live in `metadata` as
reserved `(quantity, unit)` tokens, from which labels are derived (never guessed):

- `:xquantity` / `:xunit`, `:yquantity` / `:yunit` — axis tokens (e.g.
  `:wavenumber` / `:per_cm`). See `axis_label`.
- `:xlabel` / `:ylabel` — literal label strings; override the tokens.
- `:time_delay` (+ `:time_delay_unit`) — fixed delay of a slice from a
  `TimeResolvedMatrix`; `:gate_start` / `:gate_end` (+ `:gate_unit`) — the time
  window of a gated/integrated slice.
- `:filename` / `:source` — source file for [`source_file`](@ref).
- `:cavity_length` — default cavity length for `fit_cavity_spectrum`.

With no axis metadata, labels fall back to the honest generic floor (`"x"` /
`"Signal"`) — bare row-column data loads with no units guessed.

Metadata keys are stored as `Symbol`s. The positional-dict and keyword
constructors both accept `String` keys (converted via `Symbol`), so a
`Dict{String,Any}` from the lab layer (JASCO/QPSTools sample metadata) can be
passed directly.

# Fields
- `x::Vector{Float64}`: Spectral axis
- `y::Vector{Float64}`: Signal
- `metadata::Dict{Symbol,Any}`: Additional info

# Examples
```julia
spec = Spectrum(nu, T)
spec = Spectrum(nu, T; mirror="Au", angle=10, cavity_length=12e-4)
spec.metadata[:mirror]
```
"""
struct Spectrum <: AbstractSpectroscopyData
    x::Vector{Float64}
    y::Vector{Float64}
    metadata::Dict{Symbol,Any}

    function Spectrum(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                      metadata::AbstractDict)
        length(x) == length(y) || throw(ArgumentError(
            "Spectrum: x and y must have equal length; " *
            "got $(length(x)) x points and $(length(y)) y points"))
        # Symbolize keys so String-keyed dicts (e.g. JASCO/QPSTools sample
        # metadata, which are Dict{String,Any}) are accepted directly.
        new(Float64.(x), Float64.(y), Dict{Symbol,Any}(Symbol(k) => v for (k, v) in metadata))
    end
end

function Spectrum(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                  axis::Union{Symbol,Nothing}=nothing, metadata...)
    md = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in metadata)
    if axis !== nothing
        md[:xquantity] = axis
        get!(md, :xunit, get(CANONICAL_UNIT, axis, :dimensionless))
    end
    return Spectrum(x, y, md)
end

"""
    Spectrum(M::AbstractMatrix)

Build a `Spectrum` from an `N×2` matrix (column 1 = x, column 2 = y), so a plain
row-column export goes straight in: `Spectrum(readdlm("homemade.txt"))`. Keyword
arguments (`axis=`, metadata) are forwarded to the vector constructor.
"""
function Spectrum(M::AbstractMatrix{<:Real}; kwargs...)
    size(M, 2) == 2 || throw(ArgumentError(
        "Spectrum(M): expected an N×2 matrix (column 1 = x, column 2 = y); " *
        "got size $(size(M))"))
    return Spectrum(M[:, 1], M[:, 2]; kwargs...)
end

# AbstractSpectroscopyData interface
xdata(s::Spectrum) = s.x
ydata(s::Spectrum) = s.y
xlabel(s::Spectrum) = _spectral_xlabel(s.metadata)
ylabel(s::Spectrum) = _signal_label(s.metadata)
source_file(s::Spectrum) = get(s.metadata, :filename, get(s.metadata, :source, ""))

"""
    signal(s::Spectrum) -> Vector{Float64}

Return the signal.
"""
signal(s::Spectrum) = s.y

function Base.show(io::IO, s::Spectrum)
    n = length(s.x)
    range = isempty(s.x) ? "" :
        ", $(round(minimum(s.x), digits=1)) to $(round(maximum(s.x), digits=1))"
    print(io, "Spectrum: $n points$range")
end

function Base.show(io::IO, ::MIME"text/plain", s::Spectrum)
    println(io, "Spectrum")
    println(io, "  Points:  $(length(s.x))")
    if !isempty(s.x)
        println(io, "  X:       $(xlabel(s)), $(round(minimum(s.x), digits=1)) to $(round(maximum(s.x), digits=1))")
    end
    println(io, "  Y:       $(ylabel(s))")
    for key in (:sample, :mirror, :cavity_length, :angle)
        val = get(s.metadata, key, nothing)
        !isnothing(val) && println(io, "  $key: $val")
    end
    src = source_file(s)
    !isempty(src) && println(io, "  Source:  $src")
end

# =============================================================================
# Global fitting result
# =============================================================================

"""
    GlobalFitResult

Result of global fitting multiple traces with shared time constants.

Supports single-exponential (n_exp=1) and multi-exponential (n_exp>1)
global analysis with shared τ values and per-trace amplitudes.

# Fields
- `taus::Vector{Float64}`: Shared time constants (sorted fast→slow)
- `sigma::Float64`: Shared IRF width
- `t0::Float64`: Shared time zero
- `amplitudes::Matrix{Float64}`: Per-trace amplitudes (n_traces × n_exp)
- `offsets::Vector{Float64}`: Per-trace offsets
- `labels::Vector{String}`: Trace labels
- `wavelengths::Union{Nothing, Vector{Float64}}`: Wavelength axis (from TimeResolvedMatrix input)
- `rsquared::Float64`: Global R²
- `rsquared_individual::Vector{Float64}`: Per-trace R²
- `residuals::Vector{Vector{Float64}}`: Per-trace residuals

# Derived properties
- `n_exp(fit)`: Number of exponential components
- `das(fit)`: Decay-associated spectra (requires TimeResolvedMatrix input)
"""
struct GlobalFitResult
    taus::Vector{Float64}
    sigma::Float64
    t0::Float64
    amplitudes::Matrix{Float64}
    offsets::Vector{Float64}
    labels::Vector{String}
    wavelengths::Union{Nothing, Vector{Float64}}
    rsquared::Float64
    rsquared_individual::Vector{Float64}
    residuals::Vector{Vector{Float64}}
end

n_exp(fit::GlobalFitResult) = length(fit.taus)

"""
    das(fit::GlobalFitResult) -> Matrix{Float64}

Return the decay-associated spectra (DAS) as an `n_exp × n_wavelengths` matrix.
Each row is the amplitude spectrum for one time constant.

Requires that the fit was performed on a `TimeResolvedMatrix` (wavelengths must be available).
"""
function das(fit::GlobalFitResult)
    isnothing(fit.wavelengths) && error("DAS requires wavelength axis (use TimeResolvedMatrix input)")
    return permutedims(fit.amplitudes)  # n_exp × n_wavelengths
end

function Base.show(io::IO, r::GlobalFitResult)
    n_e = length(r.taus)
    n_tr = size(r.amplitudes, 1)
    tau_str = join(["$(round(τ, digits=2))" for τ in r.taus], ", ")
    print(io, "GlobalFitResult: τ = [$tau_str] ps, R² = $(round(r.rsquared, digits=4)) ($n_tr traces, $n_e exp)")
end

function Base.show(io::IO, ::MIME"text/plain", r::GlobalFitResult)
    n_tr = size(r.amplitudes, 1)
    n_e = length(r.taus)

    println(io, "GlobalFitResult ($n_tr traces, $n_e component$(n_e == 1 ? "" : "s"))")
    println(io)

    println(io, "  Shared Parameters")
    println(io, "  ─────────────────────────────")
    for (i, τ) in enumerate(r.taus)
        label = n_e == 1 ? "τ" : "τ$i"
        println(io, "  $(rpad(label, 10))$(lpad(round(τ, digits=3), 8)) ps")
    end
    println(io, "  σ_IRF    $(lpad(round(r.sigma, digits=3), 8)) ps")
    println(io, "  t₀       $(lpad(round(r.t0, digits=3), 8)) ps")
    println(io)

    # Build header
    amp_headers = n_e == 1 ? ["Amplitude"] : ["A$i" for i in 1:n_e]
    header = "  $(rpad("Trace", 12))" * join([lpad(h, 12) for h in amp_headers], "") * "$(lpad("Offset", 12))$(lpad("R²", 10))"
    println(io, header)
    println(io, "  " * "─"^(length(header) - 2))
    for i in 1:n_tr
        label = rpad(r.labels[i], 12)
        amps = join([lpad(round(r.amplitudes[i, j], sigdigits=4), 12) for j in 1:n_e], "")
        off = lpad(round(r.offsets[i], sigdigits=4), 12)
        r2 = round(r.rsquared_individual[i], digits=4)
        println(io, "  $label$amps$off$(lpad(r2, 10))")
    end

    println(io)
    print(io, "  Global R² = $(round(r.rsquared, digits=5))")
end

# =============================================================================
# report() and format_results()
# =============================================================================

"""
    report(result)

Print a formatted summary of a fit result to the terminal.
"""
report(result) = (show(stdout, MIME("text/plain"), result); println(); nothing)

"""
    format_results(result) -> String

Return a Markdown-formatted string of fit results.
"""
function format_results end

function format_results(r::MultiPeakFitResult)
    io = IOBuffer()
    n = length(r.peaks)
    model = n > 0 ? r.peaks[1].model : "unknown"

    println(io, "## Peak Fit Results")
    println(io)

    for (i, pk) in enumerate(r.peaks)
        if n > 1
            println(io, "### Peak $i")
            println(io)
        end
        println(io, "| Parameter | Value | Uncertainty |")
        println(io, "|-----------|-------|-------------|")
        for (j, name) in enumerate(pk.params)
            val = round(pk.values[j], digits=4)
            err = round(pk.errors[j], digits=4)
            println(io, "| $(name) | $(val) | ± $(err) |")
        end
        println(io)
    end

    if r.baseline_order == 0
        println(io, "**Baseline:** constant = $(round(r.baseline_params[1], sigdigits=4))")
    elseif r.baseline_order == 1
        println(io, "**Baseline:** linear")
    else
        println(io, "**Baseline:** polynomial order $(r.baseline_order)")
    end

    println(io, "**Model:** $(model) | **R²:** $(round(r.r_squared, digits=5)) | **Region:** $(round(Int, r.region[1]))–$(round(Int, r.region[2]))")

    return String(take!(io))
end

function format_results(r::ExpDecayFit)
    io = IOBuffer()
    signal_str = r.signal_type == :esa ? "ESA" : "GSB"
    has_irf = !isnan(r.sigma)

    println(io, "## Exponential Decay Fit ($signal_str)")
    println(io)
    println(io, "| Parameter | Value |")
    println(io, "|-----------|-------|")
    println(io, "| τ | $(round(r.tau, digits=3)) ps |")
    if has_irf
        println(io, "| σ_IRF | $(round(r.sigma, digits=3)) ps |")
    end
    println(io, "| t₀ | $(round(r.t0, digits=3)) ps |")
    println(io, "| Amplitude | $(round(r.amplitude, sigdigits=4)) |")
    println(io, "| Offset | $(round(r.offset, sigdigits=4)) |")
    println(io)
    println(io, "**R²:** $(round(r.rsquared, digits=5))")

    return String(take!(io))
end


function format_results(r::MultiexpDecayFit)
    io = IOBuffer()
    n = length(r.taus)
    signal_str = r.signal_type == :esa ? "ESA" : "GSB"
    has_irf = !isnan(r.sigma)
    w = weights(r)

    println(io, "## Multi-exponential Decay Fit ($signal_str, $n components)")
    println(io)
    println(io, "| Component | τ (ps) | Amplitude | Weight |")
    println(io, "|-----------|--------|-----------|--------|")
    for i in 1:n
        println(io, "| $i | $(round(r.taus[i], digits=3)) | $(round(r.amplitudes[i], sigdigits=4)) | $(round(100*w[i], digits=1))% |")
    end
    println(io)
    if has_irf
        println(io, "**σ_IRF:** $(round(r.sigma, digits=3)) ps | **t₀:** $(round(r.t0, digits=3)) ps | **Offset:** $(round(r.offset, sigdigits=4))")
    else
        println(io, "**t₀:** $(round(r.t0, digits=3)) ps | **Offset:** $(round(r.offset, sigdigits=4))")
    end
    println(io, "**R²:** $(round(r.rsquared, digits=5))")

    return String(take!(io))
end

function format_results(r::GlobalFitResult)
    io = IOBuffer()
    n_tr = size(r.amplitudes, 1)
    n_e = length(r.taus)

    println(io, "## Global Fit Results ($n_tr traces, $n_e component$(n_e == 1 ? "" : "s"))")
    println(io)
    println(io, "### Shared Parameters")
    println(io)
    println(io, "| Parameter | Value |")
    println(io, "|-----------|-------|")
    for (i, τ) in enumerate(r.taus)
        label = n_e == 1 ? "τ" : "τ$i"
        println(io, "| $label | $(round(τ, digits=3)) ps |")
    end
    println(io, "| σ_IRF | $(round(r.sigma, digits=3)) ps |")
    println(io, "| t₀ | $(round(r.t0, digits=3)) ps |")
    println(io)
    println(io, "### Per-Trace Parameters")
    println(io)

    amp_headers = n_e == 1 ? ["Amplitude"] : ["A$i" for i in 1:n_e]
    header = "| Trace | " * join(amp_headers, " | ") * " | Offset | R² |"
    sep = "|" * join(fill("---", n_e + 3), "|") * "|"
    println(io, header)
    println(io, sep)
    for i in 1:n_tr
        label = r.labels[i]
        amps = join([string(round(r.amplitudes[i, j], sigdigits=4)) for j in 1:n_e], " | ")
        off = round(r.offsets[i], sigdigits=4)
        r2 = round(r.rsquared_individual[i], digits=4)
        println(io, "| $label | $amps | $off | $r2 |")
    end
    println(io)
    println(io, "**Global R²:** $(round(r.rsquared, digits=5))")

    return String(take!(io))
end

function format_results(r::PeakFitResult)
    io = IOBuffer()

    println(io, "## Peak Fit Result")
    if !isempty(r.sample_id)
        println(io, "**Sample:** $(r.sample_id)")
    end
    println(io)
    println(io, "| Parameter | Value | Uncertainty |")
    println(io, "|-----------|-------|-------------|")
    for (i, name) in enumerate(r.params)
        val = round(r.values[i], digits=4)
        err = round(r.errors[i], digits=4)
        println(io, "| $(name) | $(val) | ± $(err) |")
    end
    println(io)
    println(io, "**Model:** $(r.model) | **R²:** $(round(r.r_squared, digits=5)) | **Region:** $(round(Int, r.region[1]))–$(round(Int, r.region[2]))")

    return String(take!(io))
end

function format_results(r::StretchedDecayFit)
    io = IOBuffer()
    println(io, "## Stretched Exponential Decay Fit")
    println(io)
    println(io, "| Parameter | Value |")
    println(io, "|-----------|-------|")
    println(io, "| τ | $(round(r.tau, digits=3)) |")
    println(io, "| β | $(round(r.beta, digits=3)) |")
    println(io, "| ⟨τ⟩ | $(round(mean_lifetime(r), digits=3)) |")
    println(io, "| t₀ | $(round(r.t0, digits=3)) |")
    println(io, "| Amplitude | $(round(r.amplitude, sigdigits=4)) |")
    println(io, "| Offset | $(round(r.offset, sigdigits=4)) |")
    println(io)
    println(io, "**R²:** $(round(r.rsquared, digits=5))")
    return String(take!(io))
end

function format_results(r::TASpectrumFit)
    io = IOBuffer()

    println(io, "## TA Spectrum Fit")
    println(io)

    _label_names = Dict(:esa => "ESA (Excited State Absorption)",
                        :gsb => "GSB (Ground State Bleach)",
                        :se => "SE (Stimulated Emission)",
                        :positive => "Positive", :negative => "Negative")

    for (i, pk) in enumerate(r.peaks)
        label_str = get(_label_names, pk.label, uppercase(string(pk.label)))
        println(io, "### Peak $i: $label_str")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Model | $(pk.model) |")
        println(io, "| Center | $(round(pk.center, digits=1)) |")
        println(io, "| Width | $(round(pk.width, digits=2)) |")
        println(io, "| Amplitude | $(round(pk.amplitude, sigdigits=4)) |")
        println(io)
    end

    anharm = anharmonicity(r)
    if !isnan(anharm)
        println(io, "**Anharmonicity (Δν):** $(round(anharm, digits=1)) cm⁻¹")
    end
    if r.offset != 0.0
        print(io, "**Offset:** $(round(r.offset, sigdigits=4)) | ")
    end
    println(io, "**R²:** $(round(r.rsquared, digits=5))")

    return String(take!(io))
end

# =============================================================================
# Token edge accessors and opt-in unit guessing (metadata token contract §4/§6)
# =============================================================================

_metadata(s::Spectrum) = s.metadata
_metadata(t::KineticTrace) = t.metadata
_metadata(m::TimeResolvedMatrix) = m.metadata

"""
    xdata_unitful(d) ; ydata_unitful(d)

Return the x / signal data with their unit tokens (`:xunit` / `:yunit`) attached
via Unitful (§6). With no token the unit is `Unitful.NoUnits` and the plain
`Float64` vector is returned unchanged. Opt-in; core storage stays unitless.
"""
xdata_unitful(d::AbstractSpectroscopyData) =
    xdata(d) .* _unitful(Symbol(get(_metadata(d), :xunit, :dimensionless)))
ydata_unitful(d::AbstractSpectroscopyData) =
    ydata(d) .* _unitful(Symbol(get(_metadata(d), :yunit, :dimensionless)))

"""
    guess_units!(s::Spectrum) -> Spectrum

Opt-in heuristic: infer the x-axis quantity/unit from the data magnitude and
stamp `:xquantity`/`:xunit`. Nothing calls this automatically — labels never
guess on the user's behalf (the no-guess rule).
"""
function guess_units!(s::Spectrum)
    if _detect_spectral_unit(s.x) == "cm⁻¹"
        s.metadata[:xquantity] = :wavenumber
        get!(s.metadata, :xunit, :per_cm)
    else
        s.metadata[:xquantity] = :wavelength
        get!(s.metadata, :xunit, :nm)
    end
    return s
end
