# Regression tests for the November 2026 physics review of QPSLab.
# Each testset names the review section it guards.

@testset "Physics review fixes" begin

    # Exponential ⊗ Gaussian IRF, A defined at t0 (unconvolved amplitude)
    _conv(t, A, tau, t0, s) = OpticalSpectroscopy._exp_decay_irf_conv(t, A, tau, t0, s)

    @testset "2.6 stretched fit: window before time zero keeps the origin at t_start" begin
        Random.seed!(2)
        t = vcat(collect(-5.0:0.1:4.9), collect(exp10.(range(log10(5.0), log10(500.0); length=100))))
        y = [ti >= 0 ? exp(-ti / 50.0) : 0.0 for ti in t] .+ 0.01 .* randn(length(t))
        trace = KineticTrace(t, y)
        # Window starts before zero: the origin must stay at t_start = 0.
        r = fit_exp_decay(trace; model=:stretched, t_range=(-5.0, 500.0))
        @test r.t0 == 0.0
        @test r.tau ≈ 50.0 rtol=0.1
        @test r.beta > 0.95
        @test r.amplitude ≈ 1.0 atol=0.1
        @test all(r.time .>= 0.0)
        @test length(r.time) == length(r.residuals)
        # The same window starting at zero gives the same answer.
        r0 = fit_exp_decay(trace; model=:stretched, t_range=(0.0, 500.0))
        @test r0.tau ≈ r.tau rtol=1e-6
    end

    @testset "3.3 one and several exponentials define the amplitude at t_start" begin
        Random.seed!(3)
        t = collect(0.0:0.5:400.0)
        y = @. 1.0 * exp(-t / 50.0) + 0.01
        y .+= 0.002 .* randn(length(y))
        trace = KineticTrace(t, y)
        r1 = fit_exp_decay(trace; t_range=(5.0, 400.0))
        r2 = fit_exp_decay(trace; n_exp=2, t_range=(5.0, 400.0))
        @test r1.t0 == 0.0
        @test r2.t0 == 0.0
        @test r1.amplitude ≈ 1.0 atol=0.03
        # Two components of one decay: the amplitudes sum to A at t = 0,
        # not to A·exp(5/50) as they did with the origin at the window start.
        @test sum(r2.amplitudes) ≈ 1.0 atol=0.05
        @test all(r2.time .>= 5.0)
        # predict is consistent with the stored origin
        @test predict(r2, [0.0])[1] ≈ sum(r2.amplitudes) + r2.offset atol=1e-9
    end

    @testset "4.1 multi-exponential IRF guesses are taken after the peak" begin
        Random.seed!(1)
        t = vcat(collect(-5.0:0.05:4.95), collect(exp10.(range(log10(5.0), log10(1000.0); length=120))))
        for t0 in (0.3, 1.5)
            y = [_conv(ti, -0.6, 2.0, t0, 0.12) + _conv(ti, -0.4, 80.0, t0, 0.12) for ti in t]
            y .+= 0.01 .* randn(length(y))
            r = fit_exp_decay(KineticTrace(t, y); n_exp=2, irf=true, irf_width=0.15)
            @test r.taus[1] ≈ 2.0 rtol=0.15
            @test r.taus[2] ≈ 80.0 rtol=0.1
            @test r.t0 ≈ t0 atol=0.2
            rms = sqrt(mean(abs2, r.residuals))
            @test rms < 0.02
        end
    end

    @testset "2.7 non-finite samples are dropped before fitting" begin
        Random.seed!(4)
        t = collect(-2.0:0.1:60.0)
        y = [_conv(ti, 1.0, 8.0, 0.0, 0.3) + 0.02 for ti in t] .+ 0.001 .* randn(length(t))
        y[end] = NaN
        y[1] = NaN
        trace = KineticTrace(t, y)
        r = fit_exp_decay(trace; irf=true, irf_width=0.2)
        @test r.tau ≈ 8.0 atol=1.0
        @test length(r.time) == length(r.residuals) == length(t) - 2
        @test all(isfinite, r.residuals)
        r2 = fit_exp_decay(trace; n_exp=2, irf=true, irf_width=0.2)
        @test all(isfinite, r2.residuals)
        r3 = fit_exp_decay(trace)
        @test all(isfinite, r3.residuals)
        @test all(r3.time .>= 0.0)
        @test_throws ArgumentError fit_exp_decay(KineticTrace(t, fill(NaN, length(t))))
        # fit_global drops them per trace
        g = fit_global([trace, KineticTrace(t, 0.5 .* y)])
        @test g.taus[1] ≈ 8.0 atol=1.0
        @test all(all(isfinite, res) for res in g.residuals)
    end

    @testset "2.5 fit_lifetime_spectrum forwards t_start" begin
        t = collect(0.0:2.0:2000.0)
        wl = collect(600.0:10.0:700.0)
        data = [ (ti >= 200.0 ? exp(-(ti - 200.0) / 400.0) : 0.0) for ti in t, _ in wl]
        m = TimeResolvedMatrix(t, wl, data)
        res = fit_lifetime_spectrum(m; nbins=2, t_start=200.0)
        @test all(res.fitted)
        @test all(isapprox.(res.taus[:, 1], 400.0; rtol=1e-3))
        # From the sweep start the baseline and the rise are inside the fit
        res0 = fit_lifetime_spectrum(m; nbins=2)
        @test !all(abs.(res0.taus[:, 1] .- 400.0) .< 1.0)
    end

    @testset "fit results keep the 8-argument constructors" begin
        f = ExpDecayFit(0.5, 8.3, 0.1, 0.25, 0.01, :esa, zeros(10), 0.99)
        @test isempty(f.time)
        m = MultiexpDecayFit([1.0, 2.0], [0.5, 0.5], 0.0, NaN, 0.0, :esa, zeros(10), 0.99)
        @test isempty(m.time)
        s = StretchedDecayFit(1.0, 2.0, 0.8, 0.0, 0.0, :esa, zeros(10), 0.99)
        @test isempty(s.time)
    end

    @testset "4.3 pseudo-Voigt amplitude guess is an area" begin
        x = collect(400.0:1.0:1800.0)
        sigma = 5.0   # HWHM
        area = 1000.0
        y = pseudo_voigt([area, 1000.0, sigma, 0.5], x) .+ 2.0
        g = initial_peak_guesses(x, y; n_peaks=1, model=pseudo_voigt, baseline_order=0)
        height = maximum(y) - 2.0
        # The first parameter is the area: height / (unit-area height), not the height.
        @test g.p0[1] ≈ area rtol=0.3
        @test g.p0[1] > 3 * height
        fit = fit_peaks(x, y; n_peaks=1, model=pseudo_voigt, baseline_order=0)
        @test fit[1][:amplitude].value ≈ area rtol=0.02
        @test OpticalSpectroscopy._peak_height(fit[1]) ≈ height rtol=0.02
        # Height for every other model is the amplitude itself
        yl = lorentzian([3.0, 1000.0, 10.0], x) .+ 0.1
        fl = fit_peaks(x, yl; n_peaks=1, model=lorentzian, baseline_order=0)
        @test OpticalSpectroscopy._peak_height(fl[1]) ≈ fl[1][:amplitude].value
    end

    @testset "4.4 Voigt and Fano start at the detected width" begin
        # Fine grid: the measured width is read off the samples, so the grid
        # step bounds its error (0.005 on a width of 3.6 is 0.3%).
        x = collect(-30.0:0.005:30.0)
        y = voigt([1.0, 0.0, 1.0, 1.0], x)
        g = initial_peak_guesses(x, y; n_peaks=1, model=voigt, baseline_order=0)
        sigma0, gamma0 = g.p0[3], g.p0[4]
        fG = OpticalSpectroscopy.FWHM_FACTOR * sigma0
        fL = 2 * gamma0
        fV = (fG^5 + 2.69269 * fG^4 * fL + 2.42843 * fG^3 * fL^2 +
              4.47163 * fG^2 * fL^3 + 0.07842 * fG * fL^4 + fL^5)^(1 / 5)
        half = x[y .>= 0.5]
        @test fV ≈ (half[end] - half[1]) rtol=0.1
        fit = fit_peaks(x, y; n_peaks=1, model=voigt, baseline_order=0)
        @test fit[1][:sigma].value ≈ 1.0 rtol=0.05
        @test fit[1][:gamma].value ≈ 1.0 rtol=0.05
        @test OpticalSpectroscopy._peak_fwhm(fit[1]) ≈ (half[end] - half[1]) rtol=0.02

        yf = fano([1.0, 0.0, 2.0, 3.0], x)
        gf = initial_peak_guesses(x, yf; n_peaks=1, model=fano, baseline_order=0)
        @test gf.p0[3] ≈ 2.0 rtol=0.5
    end

    @testset "6.2 Kramers-Kronig and derivative on a non-uniform axis" begin
        w = collect(1.0:1.0:4000.0)
        w0, g, wp2 = 1000.0, 50.0, 1.0e6
        chi_re = @. wp2 * (w0^2 - w^2) / ((w0^2 - w^2)^2 + g^2 * w^2)
        chi_im = @. wp2 * g * w / ((w0^2 - w^2)^2 + g^2 * w^2)
        ref = kramers_kronig(w, chi_im; type=:imag_to_real)
        # A stretched (non-uniform) copy of the same curve
        wn = @. 1.0 + 3999.0 * ((w - 1.0) / 3999.0)^1.2
        chi_im_n = @. wp2 * g * wn / ((w0^2 - wn^2)^2 + g^2 * wn^2)
        res = kramers_kronig(wn, chi_im_n; type=:imag_to_real)
        peak = maximum(abs.(chi_re))
        for i in (900, 1100, 1500)
            j = argmin(abs.(wn .- w[i]))
            @test abs(res[j] - ref[i]) / peak < 0.05
        end

        xn = collect(range(400.0, 800.0; length=401)) .^ 1.5   # non-uniform
        yn = @. 0.01 * xn
        d = derivative(xn, yn)
        @test maximum(abs.(d[20:end-20] .- 0.01)) < 1e-6
    end

    @testset "6.2 DER_SNR stencil is not fooled by correlated noise" begin
        Random.seed!(5)
        n = 4000
        white = randn(n)
        corr = 0.5 .* (white .+ circshift(white, 1))   # 2-pixel oversampling
        y = 100.0 .+ corr
        r = estimate_snr(y)
        @test r.noise ≈ std(corr) rtol=0.2
        @test r.snr ≈ 100.0 / std(corr) rtol=0.25
    end
end
