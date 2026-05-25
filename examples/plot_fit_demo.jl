using OpticalSpectroscopy, GLMakie

x = 1500.0:1.0:2500.0
y = gaussian([1.0, 1800.0, 15.0], x) .+
    gaussian([0.6, 2100.0, 20.0], x) .+
    0.02 .* randn(length(x))

fit = fit_peaks(x, y; n_peaks=2, model=gaussian)

fig = plot_fit(fit; xlabel="Wavenumber (cm⁻¹)", ylabel="Absorbance")
fig
