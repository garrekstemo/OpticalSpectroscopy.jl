# Tests for the June 2026 six-reviewer audit fixes.
# Each testset is labeled with the audit item number it reproduces.

using Test
using OpticalSpectroscopy
using Statistics
using LinearAlgebra
using Unitful
import Random
using Random: MersenneTwister

const OS = OpticalSpectroscopy

@testset "Audit fixes" begin

    # =========================================================================
    # Priority 1 — correctness bugs
    # =========================================================================

    @testset "#1 derivative: descending x-axis keeps dy/dx sign" begin
        x = collect(800.0:-0.5:400.0)
        y = @. 100 * exp(-(x - 520.0)^2 / (2 * 20.0^2))
        dy = derivative(x, y; order=1)
        i540 = argmin(abs.(x .- 540.0))
        # Right of the peak (x = 540 > 520) the true dy/dx is negative.
        @test dy[i540] < 0
        i500 = argmin(abs.(x .- 500.0))
        @test dy[i500] > 0

        # Descending-axis derivative must equal the reversed ascending-axis one.
        xa = reverse(x)
        ya = reverse(y)
        dya = derivative(xa, ya; order=1)
        @test isapprox(dy, reverse(dya); rtol=1e-10, atol=1e-12)

        # Even orders are unaffected by axis direction.
        d2 = derivative(x, y; order=2)
        d2a = derivative(xa, ya; order=2)
        @test isapprox(d2, reverse(d2a); rtol=1e-10, atol=1e-12)
    end

    @testset "#2 fit_ta_spectrum: 4-parameter lineshapes get full initial guesses" begin
        ν = collect(1900.0:0.5:2100.0)

        # Two voigt peaks (npps = 4 each); previously BoundsError in the model.
        y = voigt([0.5, 1980.0, 4.0, 2.0], ν) .- voigt([0.7, 2010.0, 4.0, 2.0], ν)
        spec = Spectrum(ν, y)
        fit = fit_ta_spectrum(spec; peaks=[(:esa, voigt), (:gsb, voigt)])
        @test fit.rsquared > 0.99
        @test isapprox(fit[:esa].center, 1980.0; atol=1.0)
        @test isapprox(fit[:gsb].center, 2010.0; atol=1.0)

        # Mixed 4-param (fano) + 3-param (gaussian) must not misalign parameters.
        y2 = fano([0.3, 1975.0, 6.0, 1.5], ν) .- gaussian([0.6, 2020.0, 5.0], ν)
        spec2 = Spectrum(ν, y2)
        fit2 = fit_ta_spectrum(spec2; peaks=[(:esa, fano), (:gsb, gaussian)])
        @test fit2.rsquared > 0.9
    end

    @testset "#3 _xcorr_peak: sub-sample delay recovery" begin
        t = collect(0.0:1.0:199.0)
        pulse(t0) = @. exp(-((t - t0)^2) / (2 * 8.0^2))
        ref = pulse(60.0)
        for d in (2.33, -1.7, 0.4)
            col = pulse(60.0 + d)
            lag = OS._xcorr_peak(ref, col, 30)
            @test isapprox(lag, d; atol=0.05)
        end
        # Zero delay stays zero
        @test isapprox(OS._xcorr_peak(ref, pulse(60.0), 30), 0.0; atol=1e-6)
    end

    @testset "#4 IRF convolution: numerically stable scaled form" begin
        f = OS._exp_decay_irf_conv

        # τ ≪ σ previously overflowed exp(σ²/2τ²) to Inf → NaN
        @test isfinite(f(-1.0, 1.0, 0.01, 0.0, 0.5))
        # σ/τ = 40: finite everywhere
        @test all(isfinite, [f(t, 1.0, 0.05, 0.0, 2.0) for t in -5.0:0.1:5.0])

        # High relative accuracy at σ/τ = 20 against the exact BigFloat expression
        function exact_conv(t, A, tau, t0, sigma)
            T = BigFloat
            tp = T(t) - T(t0)
            arg = (T(sigma) / T(tau) - tp / T(sigma)) / sqrt(T(2))
            return Float64((T(A) / 2) * exp(T(sigma)^2 / (2 * T(tau)^2) - tp / T(tau)) *
                           OS.erfc(arg))
        end
        for t in (-1.0, 0.0, 0.05, 0.2, 1.0, 5.0)
            v = f(t, 1.0, 0.1, 0.0, 2.0)
            ve = exact_conv(t, 1.0, 0.1, 0.0, 2.0)
            @test isapprox(v, ve; rtol=1e-9)
        end

        # ForwardDiff compatibility: the solver differentiates through the
        # erfcx path when fitting an IRF-convolved decay.
        t = collect(-2.0:0.05:8.0)
        sig = [f(ti, 1.0, 2.0, 0.0, 0.15) for ti in t]
        fit = OS.fit_decay_irf(t, sig)
        @test isapprox(fit.tau, 2.0; rtol=0.05)
        @test isapprox(fit.sigma, 0.15; rtol=0.1)
    end

    @testset "#5 PLMap cosmic-ray fraction cap: small pixel_range" begin
        rng = MersenneTwister(1)
        np, nx, ny = 64, 5, 5
        spectra = 100.0 .+ randn(rng, np, nx, ny)
        spectra[10, 3, 3] += 5000.0
        intensity = dropdims(sum(spectra; dims=1); dims=1)
        m = PLMap(intensity, spectra, collect(1.0:nx), collect(1.0:ny),
                  collect(1.0:np), Dict{Symbol,Any}())

        # 15-channel range: n_ch ÷ 20 == 0 previously cleared every flag
        cr = detect_cosmic_rays(m; pixel_range=(5, 19))
        @test cr.count >= 1
        @test cr.mask[10, 3, 3]
    end

    @testset "#6 kubelka_munk: honors :percent yunit token" begin
        x = collect(200.0:1.0:400.0)
        Rpct = fill(50.0, length(x))
        s = Spectrum(x, Rpct; yquantity=:reflectance, yunit=:percent)
        k = kubelka_munk(s)
        # F(0.5) = (1 - 0.5)^2 / (2·0.5) = 0.25
        @test all(v -> isapprox(v, 0.25; atol=1e-12), k.y)
        @test k.metadata[:yquantity] == :kubelka_munk
        @test k.metadata[:yunit] == :dimensionless
        @test !haskey(k.metadata, :ylabel)

        # Fractional input unchanged
        sf = Spectrum(x, fill(0.5, length(x)); yquantity=:reflectance, yunit=:fraction)
        @test all(v -> isapprox(v, 0.25; atol=1e-12), kubelka_munk(sf).y)

        # Wrong quantity refuses (no silent guessing)
        st = Spectrum(x, Rpct; yquantity=:transmittance, yunit=:percent)
        @test_throws ArgumentError kubelka_munk(st)
    end

    # =========================================================================
    # Priority 2 — silent-corruption class
    # =========================================================================

    @testset "#7 no axis/metadata aliasing in derived objects" begin
        time = collect(0.0:1.0:10.0)
        wl = collect(500.0:10.0:600.0)
        data = randn(MersenneTwister(2), length(time), length(wl))
        m = TimeResolvedMatrix(time, wl, data)

        tr = m[λ=520.0]
        @test tr.time !== m.time

        crm = CosmicRayMatrixResult([CartesianIndex(2, 2)], 1, 5.0)
        out = remove_cosmic_rays(m, crm)
        @test out.time !== m.time
        @test out.wavelength !== m.wavelength

        np, nx, ny = 32, 6, 6
        spectra = 50.0 .+ randn(MersenneTwister(3), np, nx, ny)
        intensity = dropdims(sum(spectra; dims=1); dims=1)
        pm = PLMap(intensity, spectra, collect(1.0:nx), collect(1.0:ny),
                   collect(1.0:np), Dict{Symbol,Any}(:stale => 1))

        bg = subtract_background(pm; positions=[(1.0, 1.0), (6.0, 1.0)])
        @test bg.x !== pm.x
        @test bg.y !== pm.y
        @test bg.pixel !== pm.pixel

        mask = falses(np, nx, ny)
        mask[5, 2, 2] = true
        crp = CosmicRayMapResult(mask, 1, 1, zeros(Int, np))
        cleaned = remove_cosmic_rays(pm, crp)
        @test cleaned.metadata !== pm.metadata
        @test cleaned.x !== pm.x
        @test cleaned.y !== pm.y
        @test cleaned.pixel !== pm.pixel
    end

    @testset "#8 _align_spectra: tolerance relative to point spacing" begin
        x1 = collect(0.0:0.001:0.1)
        y1 = ones(length(x1))
        # 5 full steps of offset previously passed the absolute 0.01 tolerance
        a = (x=x1, y=y1)
        b = (x=x1 .+ 0.005, y=y1)
        @test_throws ArgumentError add_spectra(a, b)

        # Sub-tolerance float jitter still passes
        c = (x=x1 .+ 1e-9, y=y1)
        @test add_spectra(a, c).y ≈ 2 .* y1
    end

    @testset "#9 average_spectra: typed AbstractSpectroscopyData dispatch" begin
        kt1 = KineticTrace([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
        kt2 = KineticTrace([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
        avg = average_spectra(kt1, kt2)
        @test avg.y ≈ [2.0, 3.0, 4.0]
        @test avg.x ≈ [0.0, 1.0, 2.0]

        # 2D input rejected with the standard guard
        mtx = TimeResolvedMatrix([0.0, 1.0], [500.0, 510.0], zeros(2, 2))
        @test_throws ArgumentError average_spectra(mtx, mtx)
    end

    @testset "#10 find_peaks: width filters on descending x" begin
        x = collect(100.0:-0.5:0.0)
        y = @. 10.0 * exp(-(x - 50.0)^2 / (2 * 2.0^2))
        pks = find_peaks(x, y; max_width=20.0)
        @test length(pks) == 1
        @test isapprox(pks[1].position, 50.0; atol=1.0)
    end

    @testset "#11 fit_peaks: does not mutate the caller's peaks vector" begin
        x = collect(0.0:0.5:100.0)
        # Peak at 30 is SHORTER than peak at 70, so prominence order differs
        # from position order and the trim path re-sorts.
        y = @. 3.0 * exp(-(x - 30.0)^2 / (2 * 3.0^2)) + 8.0 * exp(-(x - 70.0)^2 / (2 * 3.0^2))
        pks = find_peaks(x, y)
        @test length(pks) == 2
        positions_before = [p.position for p in pks]
        fit_peaks(x, y; peaks=pks, n_peaks=1)
        @test [p.position for p in pks] == positions_before
    end

    @testset "#12 residual convention: data − fit for every fit-result type" begin
        t = collect(0.0:0.5:50.0)

        # ExpDecayFit (no IRF)
        sig1 = @. 2.5 * exp(-t / 8.0) + 0.1
        tr1 = KineticTrace(t, sig1)
        f1 = fit_exp_decay(tr1)
        @test f1.residuals ≈ sig1 .- predict(f1, t) atol=1e-8

        # MultiexpDecayFit
        sig2 = @. 2.0 * exp(-t / 2.0) + 1.0 * exp(-t / 20.0) + 0.05
        tr2 = KineticTrace(t, sig2)
        f2 = fit_exp_decay(tr2; n_exp=2)
        @test f2.residuals ≈ sig2 .- predict(f2, t) atol=1e-6

        # StretchedDecayFit
        sig3 = @. 1.5 * exp(-(t / 5.0)^0.7) + 0.02
        tr3 = KineticTrace(t, sig3)
        f3 = fit_exp_decay(tr3; model=:stretched)
        @test f3.residuals ≈ sig3 .- predict(f3, t) atol=1e-6

        # ExpDecayFit with IRF
        tirf = collect(-2.0:0.1:20.0)
        sig4 = [OS._exp_decay_irf_conv(ti, 1.0, 3.0, 0.0, 0.2) + 0.01 for ti in tirf]
        f4 = fit_exp_decay(KineticTrace(tirf, sig4); irf=true)
        @test f4.residuals ≈ sig4 .- predict(f4, tirf) atol=1e-8

        # GlobalFitResult
        sig5a = [OS._exp_decay_irf_conv(ti, 1.0, 4.0, 0.0, 0.2) for ti in tirf]
        sig5b = [OS._exp_decay_irf_conv(ti, -0.8, 4.0, 0.0, 0.2) for ti in tirf]
        traces = [KineticTrace(tirf, sig5a), KineticTrace(tirf, sig5b)]
        g = fit_global(traces)
        curves = predict(g, traces)
        for i in 1:2
            @test g.residuals[i] ≈ traces[i].signal .- curves[i] atol=1e-8
        end

        # TASpectrumFit
        ν = collect(1900.0:0.5:2100.0)
        yta = gaussian([0.5, 1980.0, 5.0], ν) .- gaussian([0.7, 2010.0, 5.0], ν)
        fta = fit_ta_spectrum(Spectrum(ν, yta))
        @test fta.residuals ≈ yta .- predict(fta) atol=1e-8

        # MultiPeakFitResult (was already data − fit; must stay)
        xp = collect(0.0:0.5:100.0)
        yp = @. 5.0 * exp(-(xp - 50.0)^2 / (2 * 4.0^2)) + 0.5
        rp = fit_peaks(xp, yp; model=gaussian, n_peaks=1)
        @test residuals(rp) ≈ yp .- predict(rp) atol=1e-8
    end

    @testset "#13 y-changing ops retag signal tokens" begin
        x = collect(400.0:1.0:800.0)
        y = @. 1.0 + 0.5 * exp(-(x - 520.0)^2 / (2 * 10.0^2))
        md = Dict{Symbol,Any}(:yquantity => :absorbance, :yunit => :OD,
                              :ylabel => "Absorbance (OD)",
                              :xquantity => :wavelength, :xunit => :nm)
        s = Spectrum(x, y, md)

        na = normalize_area(s)
        @test na.metadata[:yquantity] == :normalized_intensity
        @test na.metadata[:yunit] == :dimensionless
        @test !haskey(na.metadata, :ylabel)
        @test ylabel(na) == "Normalized intensity"
        @test validate_tokens(na.metadata)

        npk = normalize_to_peak(s, 520.0)
        @test npk.metadata[:yquantity] == :normalized_intensity
        @test !haskey(npk.metadata, :ylabel)

        d = derivative(s)
        @test d.metadata[:yquantity] == :derivative
        @test !haskey(d.metadata, :ylabel)
        @test validate_tokens(d.metadata)

        s2 = Spectrum(x, fill(2.0, length(x)), copy(md))
        r = divide_spectra(s, s2)
        @test r.metadata[:yquantity] == :ratio
        @test r.metadata[:yunit] == :dimensionless
        @test !haskey(r.metadata, :ylabel)

        # x-axis tokens survive
        @test na.metadata[:xquantity] == :wavelength

        # normalize_intensity(PLMap) drops a stale literal :ylabel too
        np_, nx_, ny_ = 8, 3, 3
        spectra = ones(np_, nx_, ny_)
        pm = PLMap(dropdims(sum(spectra; dims=1); dims=1), spectra,
                   collect(1.0:nx_), collect(1.0:ny_), collect(1.0:np_),
                   Dict{Symbol,Any}(:ylabel => "stale", :yquantity => :intensity,
                                    :yunit => :counts))
        nrm = normalize_intensity(pm)
        @test nrm.metadata[:yunit] == :arb
        @test !haskey(nrm.metadata, :ylabel)
    end

    # =========================================================================
    # Priority 3 — baseline algorithms
    # =========================================================================

    @testset "#14 imodpoly_baseline: Zhao 2007 I-ModPoly" begin
        rng = MersenneTwister(4)
        x = collect(0.0:1.0:400.0)
        base = @. 5.0 + 0.01 * x
        peaks = @. 50.0 * exp(-(x - 100.0)^2 / (2 * 4.0^2)) +
                   30.0 * exp(-(x - 250.0)^2 / (2 * 5.0^2))
        y = base .+ peaks .+ 0.5 .* randn(rng, length(x))

        b = imodpoly_baseline(x, y; poly_order=1, maxiter=200)
        @test maximum(abs.(b .- base)) < 2.0
        # Peak-free region: baseline centered in the noise, not clipped below it
        flat = 300:400
        @test abs(mean(y[flat] .- b[flat])) < 0.5
    end

    @testset "#15 snip_baseline: negative data and scan direction" begin
        # ΔA-style data with a negative baseline must not be floored at 0
        n = 300
        x = collect(1.0:n)
        y = fill(-5.0, n) .+ [20.0 * exp(-(xi - 150.0)^2 / (2 * 8.0^2)) for xi in x]
        b = snip_baseline(y; iterations=40)
        @test all(abs.(b[1:50] .+ 5.0) .< 0.5)
        @test all(abs.(b[end-50:end] .+ 5.0) .< 0.5)

        # Result independent of scan direction
        rng = MersenneTwister(5)
        y2 = 10.0 .+ 0.02 .* x .+
             [15.0 * exp(-(xi - 80.0)^2 / (2 * 5.0^2)) +
              40.0 * exp(-(xi - 220.0)^2 / (2 * 10.0^2)) for xi in x] .+
             0.2 .* randn(rng, n)
        @test snip_baseline(reverse(y2); iterations=30) ≈
              reverse(snip_baseline(y2; iterations=30)) atol=1e-9
    end

    # =========================================================================
    # Priority 5 — remaining verified bugs
    # =========================================================================

    @testset "#22 subtract_background(PLMap): margin validation/clamping" begin
        # Tiny map: default margin=5 previously hit BoundsError
        np, nx, ny = 8, 3, 3
        spectra = 10.0 .+ zeros(np, nx, ny)
        pm = PLMap(dropdims(sum(spectra; dims=1); dims=1), spectra,
                   collect(1.0:nx), collect(1.0:ny), collect(1.0:np),
                   Dict{Symbol,Any}())
        bg = subtract_background(pm)
        @test all(abs.(bg.spectra) .< 1e-10)

        # margin ≤ nx < 2·margin previously double-weighted overlapping columns
        nx2, ny2 = 8, 8
        spectra2 = zeros(np, nx2, ny2)
        for ix in 1:nx2
            spectra2[:, ix, :] .= Float64(ix^2)
        end
        pm2 = PLMap(dropdims(sum(spectra2; dims=1); dims=1), spectra2,
                    collect(1.0:nx2), collect(1.0:ny2), collect(1.0:np),
                    Dict{Symbol,Any}())
        bg2 = subtract_background(pm2; margin=5)
        # Each edge column counted once: mean of 1²..8² = 25.5
        @test isapprox(bg2.spectra[1, 1, 1], 1.0 - 25.5; atol=1e-10)

        @test_throws ArgumentError subtract_background(pm; margin=0)
    end

    @testset "#23 nearest-index lookups skip NaN axis entries" begin
        time = collect(0.0:1.0:5.0)
        wl = [500.0, NaN, 510.0, 520.0]
        data = ones(length(time), length(wl))
        m = TimeResolvedMatrix(time, wl, data)

        tr = m[λ=505.0]
        @test !isnan(tr.wavelength)
        @test tr.wavelength in (500.0, 510.0)

        kt = kinetic_trace(m; wavelength=505.0)
        @test !isnan(kt.wavelength)
    end

    @testset "#24 ydata_unitful(TimeResolvedMatrix): no :ps guess" begin
        time = collect(0.0:1.0:5.0)
        wl = collect(500.0:10.0:550.0)
        data = zeros(length(time), length(wl))

        m_plain = TimeResolvedMatrix(time, wl, data)
        yu = ydata_unitful(m_plain)
        @test eltype(yu) == Float64        # NoUnits: plain vector back
        @test yu == time

        m_ns = TimeResolvedMatrix(time, wl, data; time_unit=:ns)
        @test string(Unitful.unit(eltype(ydata_unitful(m_ns)))) == "ns"
    end

    @testset "#25 show: empty KineticTrace / TimeResolvedMatrix" begin
        kt = KineticTrace(Float64[], Float64[])
        @test sprint(show, kt) isa String
        @test sprint(show, MIME"text/plain"(), kt) isa String

        m0 = TimeResolvedMatrix(Float64[], Float64[], zeros(0, 0))
        @test sprint(show, m0) isa String
        @test sprint(show, MIME"text/plain"(), m0) isa String

        m1 = TimeResolvedMatrix(Float64[], [500.0, 510.0], zeros(0, 2))
        @test sprint(show, m1) isa String
        @test sprint(show, MIME"text/plain"(), m1) isa String
    end

    @testset "#26 smooth_data: Int input and window validation" begin
        y = [1, 2, 10, 2, 1]
        sm = smooth_data(y; window=3)
        @test eltype(sm) <: AbstractFloat
        @test sm[2] ≈ mean([1, 2, 10])

        @test_throws ArgumentError smooth_data(collect(1.0:10.0); window=0)
        @test_throws ArgumentError smooth_data(collect(1.0:10.0); window=-3)
    end

    @testset "#27 peak_centers guard + reversed pixel_range validation" begin
        np, nx, ny = 16, 4, 4
        spectra = zeros(np, nx, ny)
        spectra[:, 1, 1] .= 1.0                       # normal pixel
        spectra[1:8, 2, 2] .= 1.0                     # signed pixel: sums to 0
        spectra[9:16, 2, 2] .= -1.0
        intensity = ones(nx, ny)                       # all pixels above cutoff
        pm = PLMap(intensity, spectra, collect(1.0:nx), collect(1.0:ny),
                   collect(1.0:np), Dict{Symbol,Any}())

        centers = peak_centers(pm; threshold=0.0)
        @test isnan(centers[2, 2])                     # not ±Inf
        @test !isinf(centers[2, 2])

        @test_throws ArgumentError integrated_intensity(pm; pixel_range=(12, 3))
        @test_throws ArgumentError peak_centers(pm; pixel_range=(12, 3))
        @test_throws ArgumentError pca_map(pm; n_components=1, pixel_range=(12, 3))
    end

    @testset "#28 nmf_map: reproducible with rng kwarg" begin
        rng = MersenneTwister(6)
        np, nx, ny = 24, 5, 5
        spectra = rand(rng, np, nx, ny) .+ 1.0
        pm = PLMap(dropdims(sum(spectra; dims=1); dims=1), spectra,
                   collect(1.0:nx), collect(1.0:ny), collect(1.0:np),
                   Dict{Symbol,Any}())
        r1 = nmf_map(pm; n_components=2, rng=MersenneTwister(7))
        r2 = nmf_map(pm; n_components=2, rng=MersenneTwister(7))
        @test r1.components ≈ r2.components
        @test r1.loadings ≈ r2.loadings
    end

    @testset "#29 decomposition pixel_range accepts float entries" begin
        np, nx, ny = 32, 4, 4
        spectra = rand(MersenneTwister(8), np, nx, ny)
        pm = PLMap(dropdims(sum(spectra; dims=1); dims=1), spectra,
                   collect(1.0:nx), collect(1.0:ny), collect(1.0:np),
                   Dict{Symbol,Any}())
        r = pca_map(pm; n_components=1, pixel_range=(10.0, 20.0))
        @test r isa DecompositionResult
        @test size(r.components, 2) == 11
    end

    @testset "#30 fit_exp_decay: t_start past the data throws cleanly" begin
        t = collect(0.0:1.0:100.0)
        sig = @. 2.0 * exp(-t / 10.0)
        tr = KineticTrace(t, sig)
        err = try
            fit_exp_decay(tr; t_start=500.0)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("at least 5", err.msg)

        err2 = try
            fit_exp_decay(tr; n_exp=2, t_start=500.0)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("at least 5", err2.msg)
    end

    @testset "#31 fit_peaks: p0 length validated against n_peaks" begin
        x = collect(0.0:0.5:100.0)
        y = @. 5.0 * exp(-(x - 50.0)^2 / (2 * 4.0^2))
        # p0 sized for ONE gaussian (3 + 2 baseline) but n_peaks = 2
        @test_throws ArgumentError fit_peaks(x, y; model=gaussian, n_peaks=2,
                                             p0=[5.0, 50.0, 4.0, 0.0, 0.0])
    end

    @testset "#32 negative-width flips: fano mirrors q, voigt abs γ" begin
        x = collect(0.0:0.25:100.0)

        # Fano: (−Γ, −q) is the exact mirror of (Γ, q). Starting there converges
        # to a negative width; the returned parameters must still reproduce y.
        yf = fano([2.0, 50.0, 5.0, 1.5], x) .+ 0.3
        rf = fit_peaks(x, yf; model=fano, baseline_order=0,
                       p0=[2.0, 50.0, -5.0, -1.5, 0.3])
        pk = rf.peaks[1]
        @test pk[:width].value >= 0
        curve = fano([pk[:amplitude].value, pk[:center].value,
                      pk[:width].value, pk[:q].value], x) .+ rf.baseline_params[1]
        @test curve ≈ yf rtol=1e-4

        # Voigt: model is symmetric in γ; the reported γ must be non-negative.
        yv = voigt([3.0, 50.0, 3.0, 1.0], x) .+ 0.1
        rv = fit_peaks(x, yv; model=voigt, baseline_order=0,
                       p0=[3.0, 50.0, 3.0, -1.0, 0.1])
        @test rv.peaks[1][:gamma].value >= 0
        pv = rv.peaks[1]
        curvev = voigt([pv[:amplitude].value, pv[:center].value,
                        pv[:sigma].value, pv[:gamma].value], x) .+ rv.baseline_params[1]
        @test curvev ≈ yv rtol=1e-4
    end

    @testset "#33 remove_cosmic_rays(PLMap): explicit pixel_range" begin
        rng = MersenneTwister(9)
        np, nx, ny = 64, 5, 5
        spectra = 100.0 .+ randn(rng, np, nx, ny)
        spectra[10, 3, 3] += 5000.0
        pm = PLMap(dropdims(sum(spectra; dims=1); dims=1), spectra,
                   collect(1.0:nx), collect(1.0:ny), collect(1.0:np),
                   Dict{Symbol,Any}())
        cr = detect_cosmic_rays(pm; pixel_range=(5, 30))
        @test cr.mask[10, 3, 3]
        cleaned = remove_cosmic_rays(pm, cr; pixel_range=(5, 30))
        @test cleaned.spectra[10, 3, 3] < 200.0
    end

    # =========================================================================
    # Priority 4 — performance changes with observable equivalence
    # =========================================================================

    @testset "#17 remove_cosmic_rays(PLMap): reuses detection MSN index" begin
        rng = MersenneTwister(17)
        np, nx, ny = 64, 5, 5
        spectra = 100.0 .+ randn(rng, np, nx, ny)
        spectra[10, 3, 3] += 5000.0
        pm = PLMap(dropdims(sum(spectra; dims=1); dims=1), spectra,
                   collect(1.0:nx), collect(1.0:ny), collect(1.0:np),
                   Dict{Symbol,Any}())

        cr = detect_cosmic_rays(pm)
        @test cr.mask[10, 3, 3]

        # (a) detection caches the MSN index; every pixel of a 5×5 map has
        # neighbors, so every entry is a valid neighbor-list index
        @test size(cr.msn_index) == (nx, ny)
        @test all(>(0), cr.msn_index)

        # (b) removal via the cached index matches the recomputed path exactly
        zeroed = CosmicRayMapResult(cr.mask, cr.count, cr.affected_spectra,
                                    cr.channel_counts, zeros(Int, nx, ny))
        cleaned_cached = remove_cosmic_rays(pm, cr)
        cleaned_recomputed = remove_cosmic_rays(pm, zeroed)
        @test cleaned_cached.spectra == cleaned_recomputed.spectra
        @test cleaned_cached.spectra[10, 3, 3] < 200.0

        # (c) backward-compatible 4-argument constructor zero-fills msn_index
        mask = falses(np, nx, ny)
        mask[5, 2, 2] = true
        compat = CosmicRayMapResult(mask, 1, 1, zeros(Int, np))
        @test compat.msn_index == zeros(Int, nx, ny)
    end

    @testset "#18 svd_filter: truncated reconstruction matches full" begin
        rng = MersenneTwister(10)
        time = collect(0.0:1.0:40.0)
        wl = collect(500.0:5.0:650.0)
        data = [exp(-t / 10.0) * exp(-(w - 570.0)^2 / 800.0) for t in time, w in wl] .+
               0.01 .* randn(rng, length(time), length(wl))
        m = TimeResolvedMatrix(time, wl, data)

        k = 2
        F = svd(data)
        Sref = copy(F.S)
        Sref[k+1:end] .= 0.0
        ref = F.U * Diagonal(Sref) * F.Vt

        @test svd_filter(m; n_components=k).data ≈ ref rtol=1e-10
        @test svd_filter(time, wl, data; n_components=k) ≈ ref rtol=1e-10
    end

    @testset "#19/#20 fitting closures and fit_map still correct" begin
        # predict on the fit grid and on a custom grid agree with the model
        x = collect(0.0:0.5:100.0)
        y = @. 5.0 * exp(-(x - 40.0)^2 / (2 * 4.0^2)) + 0.2 + 0.001 * x
        r = fit_peaks(x, y; model=gaussian, n_peaks=1, baseline_order=1)
        @test predict(r) ≈ y rtol=1e-3
        x_wide = collect(-10.0:1.0:110.0)
        pw = predict(r, x_wide)
        @test length(pw) == length(x_wide)
        @test all(isfinite, pw)

        # fit_map with a region: same result as fitting the cropped spectrum
        np, nx, ny = 128, 3, 3
        pix = collect(range(900.0, 1100.0; length=np))
        spectra = zeros(np, nx, ny)
        for ix in 1:nx, iy in 1:ny
            spectra[:, ix, iy] .= @. 100.0 * exp(-(pix - 1000.0)^2 / (2 * 15.0^2)) + 5.0
        end
        pm = PLMap(dropdims(sum(spectra; dims=1); dims=1), spectra,
                   collect(1.0:nx), collect(1.0:ny), pix, Dict{Symbol,Any}())
        fm = fit_map(pm; model=gaussian, n_peaks=1, region=(950.0, 1050.0))
        @test fm.n_converged == nx * ny
        @test all(isapprox.(fm.centers[:, :, 1], 1000.0; atol=1.0))
    end

    # =========================================================================
    # Style consolidations (item 35) with observable behavior
    # =========================================================================

    @testset "#35 consolidated model table and constants" begin
        @test OS.MAD_TO_SIGMA ≈ 1.4826 atol=2e-4
        @test OS.FWHM_FACTOR ≈ 2 * sqrt(2 * log(2))

        # No string-matching fallback for unknown model names
        my_custom_gaussian(p, x) = gaussian(p, x)
        @test OS._model_name(my_custom_gaussian) == string(my_custom_gaussian)
        @test OS._model_name(gaussian) == "gaussian"
        @test OS._model_name(pseudo_voigt) == "pseudo_voigt"

        @test OS._n_peak_params(voigt) == 4
        @test OS._peak_param_names(fano) == [:amplitude, :center, :width, :q]

        # Decay-fit supertype replaces the isa-chain
        @test ExpDecayFit <: OS.AbstractDecayFit
        @test MultiexpDecayFit <: OS.AbstractDecayFit
        @test StretchedDecayFit <: OS.AbstractDecayFit
    end

    @testset "#35 tauc_plot clamps negative αhν before fractional exponents" begin
        E = collect(1.0:0.01:3.0)
        alpha = @. 100.0 * max(E - 2.0, 0.0)^2
        alpha[1:20] .= -1.0   # noise below the gap
        res = tauc_plot(E, alpha; gap_type=:indirect)
        @test all(isfinite, res.tauc_y)
        @test all(res.tauc_y .>= 0)
    end

    @testset "#35 ArgumentError for user-input errors" begin
        time = collect(0.0:1.0:5.0)
        wl = collect(500.0:10.0:550.0)
        m = TimeResolvedMatrix(time, wl, zeros(length(time), length(wl)))
        @test_throws ArgumentError m[]
        @test_throws ArgumentError m[λ=500.0, t=1.0]

        np, nx, ny = 8, 3, 3
        pm = PLMap(zeros(nx, ny), zeros(np, nx, ny), collect(1.0:nx),
                   collect(1.0:ny), collect(1.0:np), Dict{Symbol,Any}())
        @test_throws ArgumentError extract_spectrum(pm, 0, 1)
        @test_throws ArgumentError extract_spectrum(pm, 1, 99)

        g = fit_global([KineticTrace(time, exp.(-time)),
                        KineticTrace(time, 2 .* exp.(-time))])
        @test_throws ArgumentError das(g)

        @test_throws ArgumentError snv(fill(3.0, 10))

        spec = Spectrum(collect(1900.0:0.5:2100.0),
                        gaussian([0.5, 2000.0, 5.0], collect(1900.0:0.5:2100.0)))
        @test_throws ArgumentError fit_ta_spectrum(spec; peaks=[:bogus])
    end

    # =========================================================================
    # Item 34 — zdata orientation contract stays as documented
    # =========================================================================

    @testset "#34 zdata orientation contract" begin
        time = collect(0.0:1.0:4.0)   # 5
        wl = collect(500.0:10.0:570.0) # 8
        m = TimeResolvedMatrix(time, wl, zeros(5, 8))
        # TimeResolvedMatrix: zdata is (n_time, n_wl) = (length(ydata), length(xdata))
        @test size(zdata(m)) == (length(ydata(m)), length(xdata(m)))

        pm = PLMap(zeros(3, 4), zeros(8, 3, 4), collect(1.0:3), collect(1.0:4),
                   collect(1.0:8), Dict{Symbol,Any}())
        # PLMap: zdata is (nx, ny) = (length(xdata), length(ydata))
        @test size(zdata(pm)) == (length(xdata(pm)), length(ydata(pm)))
    end
end
