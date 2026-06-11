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
    md = copy(m.metadata)
    band > 0 && (md[:band] = float(band))
    return KineticTrace(copy(m.time), sig;
                        wavelength=mean(view(m.wavelength, cols)), metadata=md)
end

"""
    spectral_slice(m::TimeResolvedMatrix; time, window=0.0) -> GatedSpectrum

Extract the spectrum at `time`, averaged over `time ± window/2` (gated mean).

With `window == 0` (default), the single nearest time row is used; if no rows
fall inside the window, the nearest row is used as fallback.
"""
function spectral_slice(m::TimeResolvedMatrix; time::Real, window::Real=0.0)
    rows = _axis_window(m.time, time, window)
    sig = vec(mean(view(m.data, rows, :), dims=1))
    t_lo, t_hi = extrema(view(m.time, rows))
    return GatedSpectrum(copy(m.wavelength), sig;
                         t_range=(t_lo, t_hi), metadata=copy(m.metadata))
end

"""
    integrate_time(m::TimeResolvedMatrix; t_range=nothing) -> GatedSpectrum

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
    return GatedSpectrum(copy(m.wavelength), sig;
                         t_range=(t_lo, t_hi), metadata=copy(m.metadata))
end

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
