# Correct & display a Raman spectrum loaded from a JASCO CSV.
# load → ARPLS baseline → peak detection → display.

using OpticalSpectroscopy, JASCOFiles, GLMakie

src  = joinpath(@__DIR__, "data", "raman_mose2_synthetic.csv")
spec = JASCOSpectrum(src)

# Optional cosmic-ray removal
# crs = detect_cosmic_rays(spec.y; threshold=5.0)
# y = remove_cosmic_rays(spec.y, crs)

# 1. ARPLS baseline
y_corr = correct_baseline(spec.x, spec.y; method=:arpls, λ=1e5).y

# 2. Peak detection on the corrected spectrum
peaks = find_peaks(spec.x, y_corr; min_prominence=0.1)

# 3. Display: raw + corrected on one axis, peaks marked
fig = Figure(size=(900, 500))

ax  = Axis(fig[1, 1],
    xlabel="Raman shift (cm⁻¹)",
    ylabel="Intensity",
    title=spec.title
)

lines!(ax, spec.x, spec.y; color=(:gray, 0.7), label="raw")
lines!(ax, spec.x, y_corr; color=:crimson,    label="corrected")
scatter!(ax, [p.position for p in peaks], [p.intensity for p in peaks];
         color=:black, marker=:dtriangle, markersize=10, label="peaks")
axislegend(ax; position=:rt)

save(joinpath(@__DIR__, "raman_correction.png"), fig)
fig
