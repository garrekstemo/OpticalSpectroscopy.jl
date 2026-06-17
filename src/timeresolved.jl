# Slice extraction and binning for TimeResolvedMatrix.
#
# kinetic_trace / spectral_slice average over the band/window (gated mean),
# so SNR improves with width without changing signal magnitude;
# integrate_time sums to preserve total counts.

"""
    kinetic_trace(m::TimeResolvedMatrix; wavelength, band=0.0) -> KineticTrace

Extract a kinetic trace at `wavelength`, averaged over `wavelength ± band/2`.

With `band == 0` (default), the single nearest wavelength column is used; if
no columns fall inside the band, the nearest column is used as fallback.
The trace inherits the matrix metadata, plus `:band` when band > 0.
"""
function kinetic_trace(m::TimeResolvedMatrix; wavelength::Real, band::Real=0.0)
    cols = _axis_window(m.wavelength, wavelength, band)
    sig = vec(mean(view(m.data, :, cols), dims=2))
    md = _trace_metadata(m)
    band > 0 && (md[:band] = float(band))
    return KineticTrace(copy(m.time), sig;
                        wavelength=mean(view(m.wavelength, cols)), metadata=md)
end

"""
    spectral_slice(m::TimeResolvedMatrix; time, window=0.0) -> Spectrum

Extract the spectrum at `time`, averaged over `time ± window/2` (gated mean).

With `window == 0` (default), the single nearest time row is used; if no rows
fall inside the window, the nearest row is used as fallback.
"""
function spectral_slice(m::TimeResolvedMatrix; time::Real, window::Real=0.0)
    rows = _axis_window(m.time, time, window)
    sig = vec(mean(view(m.data, rows, :), dims=1))
    t_lo, t_hi = extrema(view(m.time, rows))
    return Spectrum(copy(m.wavelength), sig, _gated_metadata(m, t_lo, t_hi))
end

"""
    integrate_time(m::TimeResolvedMatrix; t_range=nothing) -> Spectrum

Time-integrated spectrum: sum over the time axis (preserves total counts),
optionally restricted to `t_range = (t_lo, t_hi)`.
"""
function integrate_time(m::TimeResolvedMatrix; t_range::Union{Nothing,NTuple{2,Real}}=nothing)
    rows = if isnothing(t_range)
        eachindex(m.time)
    else
        found = findall(t -> t_range[1] <= t <= t_range[2], m.time)
        isempty(found) && throw(ArgumentError(
            "integrate_time: no time points inside t_range = $t_range"))
        found
    end
    sig = vec(sum(view(m.data, rows, :), dims=1))
    t_lo, t_hi = extrema(view(m.time, rows))
    return Spectrum(copy(m.wavelength), sig, _gated_metadata(m, t_lo, t_hi))
end

# Spectral slice inherits the matrix's spectral + signal tokens; the time window
# becomes flat :gate_start / :gate_end (+ :gate_unit) tokens (no compound values).
function _gated_metadata(m::TimeResolvedMatrix, t_lo::Real, t_hi::Real)
    md = copy(m.metadata)
    md[:gate_start] = float(t_lo)
    md[:gate_end] = float(t_hi)
    haskey(m.metadata, :time_unit) && (md[:gate_unit] = Symbol(m.metadata[:time_unit]))
    return md
end

"""
    bin_matrix(m::TimeResolvedMatrix; time=1, wavelength=1) -> TimeResolvedMatrix

Block-average the matrix by integer factors along each axis.

Axis values are averaged the same way as the data. A partial final block is
averaged over the elements it contains (no data is dropped). Factors of 1
return an identical copy. Records `:bin_time` / `:bin_wavelength` in metadata
when the corresponding factor is > 1; calling `bin_matrix` repeatedly multiplies
the stored factors so the metadata always reflects the total binning applied
(e.g. binning by 2 twice records `:bin_time => 4`).
"""
function bin_matrix(m::TimeResolvedMatrix; time::Int=1, wavelength::Int=1)
    time >= 1 || throw(ArgumentError("bin_matrix: time factor must be >= 1, got $time"))
    wavelength >= 1 || throw(ArgumentError("bin_matrix: wavelength factor must be >= 1, got $wavelength"))

    time == 1 && wavelength == 1 && return TimeResolvedMatrix(
        copy(m.time), copy(m.wavelength), copy(m.data), copy(m.metadata))

    t_blocks = _blocks(length(m.time), time)
    w_blocks = _blocks(length(m.wavelength), wavelength)

    new_time = [mean(view(m.time, b)) for b in t_blocks]
    new_wl = [mean(view(m.wavelength, b)) for b in w_blocks]
    data = Matrix{Float64}(undef, length(t_blocks), length(w_blocks))
    for (j, wb) in enumerate(w_blocks), (i, tb) in enumerate(t_blocks)
        data[i, j] = mean(view(m.data, tb, wb))
    end

    md = copy(m.metadata)
    time > 1 && (md[:bin_time] = get(md, :bin_time, 1) * time)
    wavelength > 1 && (md[:bin_wavelength] = get(md, :bin_wavelength, 1) * wavelength)
    return TimeResolvedMatrix(new_time, new_wl, data, md)
end

# Index ranges covering 1:n in blocks of k (final block may be partial).
_blocks(n::Int, k::Int) = [((i - 1) * k + 1):min(i * k, n) for i in 1:cld(n, k)]

# Indices of `axis` within `center ± width/2`; nearest single index when the
# window is empty or width <= 0.
function _axis_window(axis::AbstractVector, center::Real, width::Real)
    isnan(center) && throw(ArgumentError("axis window center is NaN"))
    if width > 0
        half = width / 2
        found = findall(a -> center - half <= a <= center + half, axis)
        !isempty(found) && return found
    end
    idx = argmin(abs.(axis .- center))
    return idx:idx
end
