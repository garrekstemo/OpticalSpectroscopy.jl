# Correct & display PL map data — mirrors the QPSLab GUI pipeline:
# load → cosmic-ray removal → ARPLS baseline → display.
# Data: 4× spatially binned PL map of a TMDC flake, 301 channels around 800 nm.
# Original scan: 86×96 grid at 2 μm step, 2000 channels (656.6–943.7 nm).

using OpticalSpectroscopy, GLMakie, DelimitedFiles

datadir = joinpath(@__DIR__, "data")
wl   = vec(readdlm(joinpath(datadir, "wavelength.txt"); skipstart=1))
data = readdlm(joinpath(datadir, "plmap.lvm"), '\t', Float64; skipstart=1)

nx, ny, step = 21, 24, 8.0
spectra   = reshape(data, nx, ny, length(wl))
intensity = dropdims(sum(spectra; dims=3); dims=3)
x = collect(range(-(nx-1)/2*step, (nx-1)/2*step, length=nx))
y = collect(range(-(ny-1)/2*step, (ny-1)/2*step, length=ny))
m_raw = PLMap(intensity, spectra, x, y, wl, Dict{String,Any}())

# 1. Cosmic ray removal
cr = detect_cosmic_rays(m_raw; threshold=20.0)
m  = remove_cosmic_rays(m_raw, cr)

# 2. ARPLS baseline per pixel spectrum
corrected = similar(m.spectra)
Threads.@threads for idx in CartesianIndices((nx, ny))
    i, j = Tuple(idx)
    corrected[i, j, :] = correct_baseline(m.pixel, vec(m.spectra[i, j, :]);
                                          method=:arpls, λ=1e7).y
end
m = PLMap(dropdims(sum(corrected; dims=3); dims=3),
          corrected, m.x, m.y, m.pixel, m.metadata)

# 3. Display: corrected intensity map + raw vs corrected spectrum at center
fig = Figure(size=(1000, 400))

ax1 = Axis(fig[1, 1]; xlabel="X (μm)", ylabel="Y (μm)",
           title="Corrected PL intensity", aspect=DataAspect())
hm = heatmap!(ax1, m.x, m.y, m.intensity; colormap=:hot)
Colorbar(fig[1, 2], hm)

ax2 = Axis(fig[1, 3]; xlabel="Wavelength (nm)", ylabel="Counts",
           title="Spectrum at (0, 0)")
raw = extract_spectrum(m_raw; x=0.0, y=0.0)
fix = extract_spectrum(m;     x=0.0, y=0.0)
lines!(ax2, raw.pixel, raw.signal; color=(:gray, 0.7), label="raw")
lines!(ax2, fix.pixel, fix.signal; color=:crimson,    label="corrected")
axislegend(ax2; position=:rt)

fig
