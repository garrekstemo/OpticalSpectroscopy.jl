"""
Peak detection for spectroscopy.

Wraps Peaks.jl with spectroscopy-specific defaults.
"""


# =============================================================================
# Types
# =============================================================================

"""
    PeakInfo

Information about a detected peak.

# Fields
- `position::Float64` — Peak center in x-units
- `intensity::Float64` — Peak height
- `prominence::Float64` — How much the peak stands out from surrounding baseline
- `width::Float64` — Full width at half prominence (FWHP) in x-units
- `bounds::Tuple{Float64,Float64}` — Left and right edges at half prominence
- `index::Int` — Index in the original data array
"""
struct PeakInfo
    position::Float64
    intensity::Float64
    prominence::Float64
    width::Float64
    bounds::Tuple{Float64, Float64}
    index::Int
end

function Base.show(io::IO, p::PeakInfo)
    print(io, "PeakInfo($(round(p.position, digits=1)), prominence=$(round(p.prominence, digits=2)))")
end

function Base.show(io::IO, ::MIME"text/plain", p::PeakInfo)
    println(io, "PeakInfo:")
    println(io, "  position:   $(round(p.position, digits=2))")
    println(io, "  intensity:  $(round(p.intensity, digits=2))")
    println(io, "  prominence: $(round(p.prominence, digits=2))")
    println(io, "  width:      $(round(p.width, digits=2))")
    println(io, "  bounds:     ($(round(p.bounds[1], digits=2)), $(round(p.bounds[2], digits=2)))")
end

# =============================================================================
# Main API
# =============================================================================

"""
    find_peaks(y; kwargs...) -> Vector{PeakInfo}
    find_peaks(x, y; kwargs...) -> Vector{PeakInfo}

Find peaks in spectroscopic data.

# Keyword Arguments
- `min_prominence::Real=0.05` — Minimum prominence as fraction of data range
- `min_width::Real=0` — Minimum peak width in x-units
- `max_width::Real=Inf` — Maximum peak width in x-units
- `min_height::Real=-Inf` — Minimum peak height (absolute)
- `window::Int=1` — Comparison window for local maxima detection
- `mode::Symbol=:maxima` — `:maxima` for peaks, `:minima` for valleys (e.g. 2nd derivative)
- `baseline::Union{Symbol,Nothing}=nothing` — Apply baseline correction before peak detection
- `baseline_kw::NamedTuple=NamedTuple()` — Keyword arguments for baseline correction
"""
function find_peaks(x::AbstractVector, y::AbstractVector;
                    min_prominence::Real=0.05,
                    min_width::Real=0,
                    max_width::Real=Inf,
                    min_height::Real=-Inf,
                    window::Int=1,
                    mode::Symbol=:maxima,
                    baseline::Union{Symbol, Nothing}=nothing,
                    baseline_kw::NamedTuple=NamedTuple())

    length(x) == length(y) || throw(ArgumentError("x and y must have same length"))
    length(y) < 3 && return PeakInfo[]

    y_work = if !isnothing(baseline)
        result = correct_baseline(y; method=baseline, baseline_kw...)
        result.y
    else
        collect(Float64, y)
    end

    data_range = maximum(y_work) - minimum(y_work)
    abs_min_prom = min_prominence * data_range

    _find = mode === :minima ? findminima : findmaxima
    pks = _find(y_work, window)

    if min_height > -Inf
        mask = pks.heights .>= min_height
        if !any(mask)
            return PeakInfo[]
        end
        pks = (
            indices = pks.indices[mask],
            heights = pks.heights[mask],
            data = pks.data
        )
    end

    isempty(pks.indices) && return PeakInfo[]

    pks = peakproms!(pks; min=abs_min_prom)
    isempty(pks.indices) && return PeakInfo[]

    dx = mean(diff(x))

    min_width_idx = min_width / dx
    max_width_idx = max_width == Inf ? Inf : max_width / dx

    pks = peakwidths!(pks; min=min_width_idx, max=max_width_idx)
    isempty(pks.indices) && return PeakInfo[]

    peaks = PeakInfo[]
    for i in eachindex(pks.indices)
        idx = pks.indices[i]
        pos = x[idx]
        intensity = y[idx]
        prom = pks.proms[i]

        ledge_idx, redge_idx = pks.edges[i]

        left_x = _interp_x(x, ledge_idx)
        right_x = _interp_x(x, redge_idx)
        width = abs(right_x - left_x)

        push!(peaks, PeakInfo(pos, intensity, prom, width, (left_x, right_x), idx))
    end

    sort!(peaks, by=p -> p.position)

    return peaks
end

# Version without x-values (uses indices)
function find_peaks(y::AbstractVector; kwargs...)
    x = collect(1:length(y))
    return find_peaks(x, y; kwargs...)
end

# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function find_peaks(spec::AbstractSpectroscopyData; kwargs...)
    _check_1d(spec, "find_peaks")
    return find_peaks(xdata(spec), ydata(spec); kwargs...)
end

"""
    peak_bounds(y, idx) -> (left, right)

Find the extent of a peak located at index `idx` by walking outward to the
nearest local minimum on each side. Returns the bounding indices
`(left, right)` with `left <= idx <= right`. Used to delimit a peak around a
known position (e.g. a user-selected point) rather than detecting peaks.
"""
function peak_bounds(y::AbstractVector, idx::Integer)
    1 <= idx <= length(y) || throw(BoundsError(y, idx))
    left = idx
    while left > 1 && y[left - 1] <= y[left]
        left -= 1
    end
    right = idx
    while right < length(y) && y[right + 1] <= y[right]
        right += 1
    end
    return (left, right)
end

# =============================================================================
# Utilities
# =============================================================================

function _interp_x(x::AbstractVector, idx::Real)
    idx_floor = floor(Int, idx)
    idx_ceil = ceil(Int, idx)

    idx_floor = clamp(idx_floor, 1, length(x))
    idx_ceil = clamp(idx_ceil, 1, length(x))

    if idx_floor == idx_ceil
        return x[idx_floor]
    end

    frac = idx - idx_floor
    return x[idx_floor] + frac * (x[idx_ceil] - x[idx_floor])
end

"""
    peak_table(peaks::Vector{PeakInfo}) -> String

Format detected peaks as an aligned text table for terminal display.
"""
function peak_table(peaks::Vector{PeakInfo})
    if isempty(peaks)
        return "No peaks detected"
    end

    lines = String[]
    push!(lines, "  Position   Intensity   Prominence   Width")
    push!(lines, "  " * "-"^50)

    for p in peaks
        pos = lpad(string(round(p.position, digits=1)), 10)
        int = lpad(string(round(p.intensity, digits=1)), 10)
        prom = lpad(string(round(p.prominence, digits=2)), 12)
        wid = lpad(string(round(p.width, digits=1)), 8)
        push!(lines, "$pos $int $prom $wid")
    end

    return join(lines, "\n")
end
