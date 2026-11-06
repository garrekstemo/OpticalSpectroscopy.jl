# Precompile workload (PrecompileTools): runs the package's common analysis
# paths on tiny synthetic data while the package precompiles, so the first
# call of each in a fresh session (the Afterglow server starting on a lab
# laptop) does not pay the compile time. Every step is wrapped so that a
# failing call only costs that step's coverage, never the package build.
#
# Keep the data small: the goal is to compile the methods, not to converge
# to anything in particular.

using PrecompileTools: @setup_workload, @compile_workload

function _precompile_step(f::Function)
    try
        f()
    catch
        # A step that throws still compiled the code it reached.
    end
    return nothing
end

@setup_workload begin
    # One spectrum: a Lorentzian peak on a sloped baseline with a spike.
    px = collect(range(500.0, 600.0; length = 64))
    py = 1.0 .+ 0.002 .* (px .- 500.0) .+ lorentzian([100.0, 550.0, 4.0], px)
    py[20] += 50.0
    spec = Spectrum(px, py)

    # A 3 × 3 PL map with the same peak in every pixel.
    nx, ny = 3, 3
    spectra = zeros(length(px), nx, ny)
    for ix in 1:nx, iy in 1:ny
        spectra[:, ix, iy] .= 5.0 .+ 100.0 .* gaussian([1.0, 550.0, 5.0], px)
    end
    intens = dropdims(sum(spectra; dims = 1); dims = 1)
    plmap = PLMap(intens, spectra, collect(1.0:nx), collect(1.0:ny), px)

    # A small transient-absorption matrix and one kinetic trace.
    t = collect(range(-1.0, 20.0; length = 24))
    wl = collect(range(450.0, 650.0; length = 16))
    decay = @. (t > 0) * exp(-t / 5.0)
    band = @. exp(-((wl - 550.0) / 30.0)^2)
    ta = TimeResolvedMatrix(t, wl, decay * band')
    trace = KineticTrace(t, decay .+ 0.01 .* sin.(t); wavelength = 550.0)

    # Absorption edge for the bandgap transforms.
    ev = collect(range(1.0, 3.0; length = 40))
    alpha = @. 1.0 + 1.0e3 * max(ev - 2.0, 0.0)^0.5 + 10.0 * exp((ev - 2.0) / 0.05)

    @compile_workload begin
        _precompile_step() do
            find_peaks(px, py)
            find_peaks(spec)
        end
        _precompile_step() do
            fit_peaks(px, py; n_peaks = 1)
            fit_peaks(px, py; n_peaks = 1, model = gaussian)
            fit_peaks(spec, (520.0, 580.0); n_peaks = 1)
        end
        _precompile_step() do
            correct_baseline(py; method = :arpls)
            correct_baseline(px, py; method = :snip)
            correct_baseline(spec; method = :arpls)
        end
        _precompile_step() do
            smooth_data(py; window = 3)
            savitzky_golay_smooth(py; window = 7, order = 2)
            derivative(px, py; window = 7)
            band_area(px, py, 520.0, 580.0)
            estimate_snr(py)
            estimate_snr(spec)
        end
        _precompile_step() do
            r = detect_cosmic_rays(py; threshold = 5.0)
            remove_cosmic_rays(py, r)
        end
        _precompile_step() do
            r = detect_cosmic_rays(plmap; threshold = 5.0)
            remove_cosmic_rays(plmap, r)
        end
        _precompile_step() do
            integrated_intensity(plmap)
            peak_centers(plmap)
            extract_spectrum(plmap, 1, 1)
            fit_map(plmap; model = gaussian, n_peaks = 1)
            fit_map(plmap; model = lorentzian, n_peaks = 1, region = (520.0, 580.0))
        end
        _precompile_step() do
            pca_map(plmap; n_components = 2)
        end
        _precompile_step() do
            kinetic_trace(ta; wavelength = 550.0, band = 10.0)
            spectral_slice(ta; time = 2.0, window = 1.0)
            bin_matrix(ta; time = 2, wavelength = 2)
            fit_exp_decay(trace; n_exp = 1, irf = false)
            fit_exp_decay(trace; n_exp = 1, irf = true, irf_width = 0.2)
        end
        _precompile_step() do
            tauc_plot(ev, alpha; gap_type = :direct)
            urbach_tail(ev, alpha)
        end
    end
end
