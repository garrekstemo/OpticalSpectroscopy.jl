# Cavity & polariton physics and fitting (merged from CavitySpectroscopy.jl).
# Reseeded so the noisy-fit draws are deterministic (original suite's seed).
Random.seed!(20260611)

using LinearAlgebra: eigen

@testset "Cavity & polariton physics" begin

    @testset "cavity_transmittance physical properties" begin
        # Empty cavity (no absorption): periodic Airy function
        n, alpha, L, R, phi = 1.0, 0.0, 1.0, 0.9, 0.0

        # At resonance, transmittance should be maximum
        # Resonance condition: 4pi*n*L*nu = 2pi*m for integer m
        # For L=1, n=1: nu_res = m/2
        nu_res = 0.5  # First resonance
        T_res = cavity_transmittance([n, alpha, L, R, phi], [nu_res])[1]

        # Off resonance (halfway between resonances)
        nu_off = 0.25
        T_off = cavity_transmittance([n, alpha, L, R, phi], [nu_off])[1]

        # Peak transmittance > off-resonance transmittance
        @test T_res > T_off

        # Transmittance is bounded: 0 <= T <= 1
        nu_range = collect(0.0:0.01:2.0)
        T_range = cavity_transmittance([n, alpha, L, R, phi], nu_range)
        @test all(T_range .>= 0)
        @test all(T_range .<= 1)

        # For lossless cavity (alpha=0), peak transmittance approaches 1
        @test T_res ≈ 1.0 rtol=0.01

        # With absorption, peak transmittance decreases
        alpha_absorb = 0.5
        T_res_abs = cavity_transmittance([n, alpha_absorb, L, R, phi], [nu_res])[1]
        @test T_res_abs < T_res

        # Higher reflectance -> higher finesse -> sharper peaks
        R_high = 0.99
        R_low = 0.5
        T_res_high = cavity_transmittance([n, 0.0, L, R_high, phi], [nu_res])[1]
        T_off_high = cavity_transmittance([n, 0.0, L, R_high, phi], [nu_off])[1]
        T_res_low = cavity_transmittance([n, 0.0, L, R_low, phi], [nu_res])[1]
        T_off_low = cavity_transmittance([n, 0.0, L, R_low, phi], [nu_off])[1]
        contrast_high = T_res_high / T_off_high
        contrast_low = T_res_low / T_off_low
        @test contrast_high > contrast_low

        # Free spectral range: peaks separated by FSR = 1/(2nL)
        FSR = 1 / (2 * n * L)
        nu_res2 = nu_res + FSR
        T_res2 = cavity_transmittance([n, alpha, L, R, phi], [nu_res2])[1]
        @test T_res2 ≈ T_res rtol=0.01

        # Scalar x with tuple p works (fn(p, x) convention)
        @test cavity_transmittance((n, alpha, L, R, phi), nu_res) ≈ T_res
    end

    @testset "refractive_index and extinction_coeff" begin
        # For a non-absorbing medium, k should be zero
        eps1 = 2.0
        eps2 = 0.0
        n = refractive_index(eps1, eps2)
        k = extinction_coeff(eps1, eps2)
        @test n ≈ sqrt(2.0)
        @test k ≈ 0.0 atol=1e-14

        # n^2 - k^2 = eps1, 2nk = eps2
        eps1 = 1.5
        eps2 = 0.3
        n = refractive_index(eps1, eps2)
        k = extinction_coeff(eps1, eps2)
        @test n^2 - k^2 ≈ eps1 atol=1e-10
        @test 2 * n * k ≈ eps2 atol=1e-10

        # Vector dispatch
        eps1v = [1.5, 2.0, 2.5]
        eps2v = [0.3, 0.0, 0.1]
        nv = refractive_index(eps1v, eps2v)
        kv = extinction_coeff(eps1v, eps2v)
        @test nv.^2 .- kv.^2 ≈ eps1v atol=1e-10
        @test 2 .* nv .* kv ≈ eps2v atol=1e-10
    end

    @testset "compute_cavity_transmittance" begin
        # Empty cavity (no oscillators): should match bare cavity_transmittance
        nu = collect(1900.0:1.0:2100.0)
        R, L, n_bg, phi = 0.9, 12.0e-4, 1.4, 0.3

        T_full = compute_cavity_transmittance(nu, Float64[], Float64[], Float64[],
                                              R, L, n_bg, phi)
        T_bare = [cavity_transmittance((n_bg, 0.0, L, R, phi), v) for v in nu]
        @test T_full ≈ T_bare atol=1e-10

        # With oscillator: transmittance should show splitting
        T_osc = compute_cavity_transmittance(nu, [2055.0], [23.0], [3000.0],
                                             R, L, n_bg, phi)
        @test all(T_osc .>= 0)
        @test all(T_osc .<= 1)

        # With oscillator, the transmission pattern should differ from bare
        @test !isapprox(T_osc, T_bare, atol=0.01)

        # Scalar dispatch should match array dispatch
        T_scalar = compute_cavity_transmittance(2000.0, [2055.0], [23.0], [3000.0],
                                                R, L, n_bg, phi)
        @test T_scalar ≈ T_osc[nu .== 2000.0][1]
    end

    @testset "cavity_mode_energy" begin
        E0 = 2000.0
        n_eff = 1.5

        # At normal incidence, energy equals E0
        E_0deg = cavity_mode_energy([E0, n_eff], [0.0])
        @test E_0deg[1] ≈ E0

        # At non-zero angle, energy increases (blue shift)
        theta = deg2rad(10.0)
        E_10deg = cavity_mode_energy([E0, n_eff], [theta])
        @test E_10deg[1] > E0

        # Energy increases monotonically with angle
        angles = deg2rad.(collect(0.0:5.0:40.0))
        E_sweep = cavity_mode_energy([E0, n_eff], angles)
        for i in 2:length(E_sweep)
            @test E_sweep[i] > E_sweep[i-1]
        end

        # Verify analytic formula: E(theta) = E0 / sqrt(1 - (sin(theta)/n_eff)^2)
        theta_check = deg2rad(20.0)
        E_analytic = E0 / sqrt(1 - (sin(theta_check) / n_eff)^2)
        E_computed = cavity_mode_energy([E0, n_eff], [theta_check])[1]
        @test E_computed ≈ E_analytic atol=1e-10
    end

    @testset "polariton_branches" begin
        E_cav = 2050.0
        E_vib = 2050.0  # Zero detuning
        Omega = 20.0

        LP, UP = polariton_branches(E_cav, E_vib, Omega)

        # At zero detuning, splitting equals Omega
        @test UP - LP ≈ Omega atol=0.1

        # Branches are symmetric around the molecular mode at zero detuning
        @test (LP + UP) / 2 ≈ E_vib atol=0.1

        # LP < E_vib < UP
        @test LP < E_vib
        @test UP > E_vib

        # Vector dispatch
        E_cav_vec = collect(2030.0:2.0:2070.0)
        LP_vec, UP_vec = polariton_branches(E_cav_vec, E_vib, Omega)
        @test length(LP_vec) == length(E_cav_vec)
        @test all(LP_vec .< UP_vec)
    end

    @testset "polariton_branches textbook values" begin
        # Textbook scenario: cavity at 2000 cm^-1, Rabi splitting 100 cm^-1
        E_vib = 2000.0
        Omega = 100.0

        # Zero detuning: splitting exactly equals Rabi splitting
        LP_0, UP_0 = polariton_branches(E_vib, E_vib, Omega)
        @test UP_0 - LP_0 ≈ Omega atol=1e-10

        # Anti-crossing: minimum splitting occurs at zero detuning
        # Sweep cavity energy across the molecular resonance
        E_cav_sweep = collect(1800.0:5.0:2200.0)
        LP_sweep, UP_sweep = polariton_branches(E_cav_sweep, E_vib, Omega)
        splittings = UP_sweep .- LP_sweep

        # Minimum splitting should be at zero detuning (E_cav = E_vib)
        min_split_idx = argmin(splittings)
        @test abs(E_cav_sweep[min_split_idx] - E_vib) < 10.0
        @test splittings[min_split_idx] ≈ Omega atol=1.0

        # LP is always below both bare energies, UP always above
        for i in eachindex(E_cav_sweep)
            @test LP_sweep[i] < min(E_cav_sweep[i], E_vib)
            @test UP_sweep[i] > max(E_cav_sweep[i], E_vib)
        end

        # Verify analytic formula: E_pm = (E_c + E_v)/2 +/- sqrt(Omega^2 + delta^2)/2
        E_cav_test = 1950.0
        delta = E_cav_test - E_vib
        E_avg = (E_cav_test + E_vib) / 2
        half_split = sqrt(Omega^2 + delta^2) / 2
        LP_expected = E_avg - half_split
        UP_expected = E_avg + half_split

        LP_test, UP_test = polariton_branches(E_cav_test, E_vib, Omega)
        @test LP_test ≈ LP_expected atol=1e-10
        @test UP_test ≈ UP_expected atol=1e-10

        # At large detuning, branches approach bare energies
        E_cav_far = E_vib + 1000.0  # Far positive detuning
        LP_far, UP_far = polariton_branches(E_cav_far, E_vib, Omega)
        @test LP_far ≈ E_vib atol=5.0      # LP approaches molecular mode
        @test UP_far ≈ E_cav_far atol=5.0  # UP approaches cavity mode
    end

    @testset "polariton_eigenvalues" begin
        # N=1 should exactly match polariton_branches (2-level coupled oscillator)
        E_cav = 2000.0
        E_vib = 2000.0
        Omega = 100.0

        eigs = polariton_eigenvalues(E_cav, [E_vib], [Omega])
        LP, UP = polariton_branches(E_cav, E_vib, Omega)

        @test length(eigs) == 2
        @test eigs[1] ≈ LP atol=1e-10
        @test eigs[2] ≈ UP atol=1e-10

        # N=1 with detuning should also match
        E_cav_det = 1950.0
        eigs_det = polariton_eigenvalues(E_cav_det, [E_vib], [Omega])
        LP_det, UP_det = polariton_branches(E_cav_det, E_vib, Omega)
        @test eigs_det[1] ≈ LP_det atol=1e-10
        @test eigs_det[2] ≈ UP_det atol=1e-10

        # N-mode returns N+1 eigenvalues
        eigs3 = polariton_eigenvalues(E_cav, [2030.0, 2060.0, 2090.0], [15.0, 20.0, 10.0])
        @test length(eigs3) == 4
        @test issorted(eigs3)

        # N=2: two identical modes should give sqrt(2) enhancement of splitting
        # (collective coupling: Omega_eff = Omega * sqrt(N) for N identical modes)
        E_mol = 2000.0
        Omega_single = 100.0
        eigs_2mode = polariton_eigenvalues(E_mol, [E_mol, E_mol], [Omega_single, Omega_single])
        @test length(eigs_2mode) == 3

        # LP and UP should have enhanced splitting: sqrt(2) * Omega
        effective_splitting = eigs_2mode[end] - eigs_2mode[1]
        @test effective_splitting ≈ sqrt(2) * Omega_single atol=1.0

        # Middle eigenvalue should be at bare molecular energy (dark state)
        @test eigs_2mode[2] ≈ E_mol atol=1e-10

        # Integer inputs should work (eltype promotion)
        eigs_int = polariton_eigenvalues(2000, [2000], [100])
        @test eigs_int ≈ eigs atol=1e-10

        # Mismatched vector lengths should error
        @test_throws AssertionError polariton_eigenvalues(E_cav, [2000.0], [50.0, 60.0])
    end

    @testset "hopfield_coefficients" begin
        E_vib = 2050.0
        Omega = 20.0

        # At zero detuning, should give 50/50 mixing
        h = hopfield_coefficients(E_vib, E_vib, Omega)
        @test h.photon_LP ≈ 0.5 atol=0.01
        @test h.matter_LP ≈ 0.5 atol=0.01
        @test h.photon_UP ≈ 0.5 atol=0.01
        @test h.matter_UP ≈ 0.5 atol=0.01

        # Fractions sum to 1
        @test h.photon_LP + h.matter_LP ≈ 1.0 atol=1e-10
        @test h.photon_UP + h.matter_UP ≈ 1.0 atol=1e-10

        # Far positive detuning (E_cav above E_vib): LP converges to the bare
        # vibration -> matter-like; the UP takes the photon character.
        h_pos = hopfield_coefficients(E_vib + 100, E_vib, Omega)
        @test h_pos.matter_LP > 0.9
        @test h_pos.photon_UP > 0.9

        # Far negative detuning (E_cav below E_vib): LP converges to the bare
        # cavity mode -> photon-like; the UP takes the matter character.
        h_neg = hopfield_coefficients(E_vib - 100, E_vib, Omega)
        @test h_neg.photon_LP > 0.9
        @test h_neg.matter_UP > 0.9

        # Vector dispatch: fractions sum to 1 at every detuning
        E_cav_vec = collect(1800.0:10.0:2300.0)
        h_vec = hopfield_coefficients(E_cav_vec, E_vib, Omega)
        @test length(h_vec.photon_LP) == length(E_cav_vec)
        @test all(h_vec.photon_LP .+ h_vec.matter_LP .≈ 1.0)
        @test all(h_vec.photon_UP .+ h_vec.matter_UP .≈ 1.0)

        # All fractions are between 0 and 1
        @test all(0.0 .<= h_vec.photon_LP .<= 1.0)
        @test all(0.0 .<= h_vec.matter_LP .<= 1.0)
        @test all(0.0 .<= h_vec.photon_UP .<= 1.0)
        @test all(0.0 .<= h_vec.matter_UP .<= 1.0)

        # Complementarity: LP photon fraction = UP matter fraction
        @test all(h_vec.photon_LP .≈ h_vec.matter_UP)
        @test all(h_vec.matter_LP .≈ h_vec.photon_UP)

        # Monotonicity: as E_cav increases past the vibration, the LP pins to
        # the vibration, so its photon fraction decreases.
        for i in 2:length(E_cav_vec)
            @test h_vec.photon_LP[i] <= h_vec.photon_LP[i-1] + 1e-10
        end
    end

    @testset "hopfield_coefficients match Hamiltonian eigenvectors" begin
        # Reference: direct diagonalization of H = [E_cav Omega/2; Omega/2 E_vib].
        # Basis order (photon, vibration); eigen() returns ascending eigenvalues,
        # so column 1 is the lower polariton. Squared components are the fractions,
        # independent of the trig parameterization used in the implementation.
        E_vib = 2050.0
        Omega = 50.0

        for delta in (-300.0, -50.0, 0.0, 50.0, 300.0)
            E_cav = E_vib + delta
            F = eigen([E_cav Omega/2; Omega/2 E_vib])
            photon_LP_ref = F.vectors[1, 1]^2
            matter_LP_ref = F.vectors[2, 1]^2
            photon_UP_ref = F.vectors[1, 2]^2
            matter_UP_ref = F.vectors[2, 2]^2

            h = hopfield_coefficients(E_cav, E_vib, Omega)
            @test h.photon_LP ≈ photon_LP_ref atol=1e-10
            @test h.matter_LP ≈ matter_LP_ref atol=1e-10
            @test h.photon_UP ≈ photon_UP_ref atol=1e-10
            @test h.matter_UP ≈ matter_UP_ref atol=1e-10
        end

        # Closed form: photon_LP = (1 - delta/sqrt(delta^2 + Omega^2))/2
        delta = 200.0
        h = hopfield_coefficients(E_vib + delta, E_vib, Omega)
        @test h.photon_LP ≈ (1 - delta / hypot(delta, Omega)) / 2 atol=1e-12

        # Limiting case: cavity far ABOVE the vibration (positive detuning),
        # the LP converges to the bare vibration -> matter-like, photon_LP -> 0.
        h_far = hopfield_coefficients(E_vib + 1e4, E_vib, Omega)
        @test h_far.photon_LP < 0.01
        @test h_far.matter_LP > 0.99
    end

end

@testset "Cavity & dispersion fitting" begin

    @testset "Cavity transmittance reflectance fit" begin
        nu = collect(0.0:0.005:2.0)
        p_true = [1.0, 0.0, 1.0, 0.9, 0.0]
        T_true = cavity_transmittance(p_true, nu)
        noise = 0.005 * randn(length(nu))
        T_data = clamp.(T_true .+ noise, 0.0, 1.0)

        # Fit reflectance with the raw model (keeping other params fixed)
        model(p, x) = cavity_transmittance([1.0, 0.0, 1.0, p[1], 0.0], x)
        p0 = [0.85]
        prob = NonlinearCurveFitProblem(model, p0, nu, T_data)
        sol = solve(prob)
        R_fit = coef(sol)[1]

        @test isapprox(R_fit, p_true[4], atol=0.05)
    end

    @testset "Synthetic cavity spectrum round-trip" begin
        # Generate synthetic cavity spectrum with known parameters
        nu = collect(1900.0:0.5:2200.0)
        R_true = 0.92
        L_true = 12.0e-4
        n_bg_true = 1.4
        phi_true = 0.3
        A_true = 3000.0
        nu0_true = 2055.0
        Gamma_true = 23.0

        T_true = compute_cavity_transmittance(nu, [nu0_true], [Gamma_true], [A_true],
                                              R_true, L_true, n_bg_true, phi_true)

        # Add small noise
        T_noisy = clamp.(T_true .+ 0.002 .* randn(length(nu)), 0.0, 1.0)

        result = fit_cavity_spectrum(nu, T_noisy;
            oscillators=[(nu0=nu0_true, Gamma=Gamma_true)],
            L=L_true,
            n_bg=n_bg_true,
            R_init=0.9,
            phi_init=0.2,
            A_init=2500.0)

        @test result isa CavityFitResult
        @test result.rsquared > 0.95
        @test isapprox(result.R, R_true, atol=0.05)
        @test length(result.oscillators) == 1
        @test result.oscillators[1].nu0 ≈ nu0_true
        @test result.oscillators[1].Gamma ≈ Gamma_true

        # Strong coupling: fitted curve should expose two polariton peaks
        @test length(result.polariton_peaks) >= 2

        # predict should work
        y_fit = predict(result)
        @test length(y_fit) == length(nu)
        @test all(isfinite, y_fit)

        y_fit_custom = predict(result, nu[1:10])
        @test length(y_fit_custom) == 10

        # residuals
        res = residuals(result)
        @test length(res) == length(nu)
    end

    @testset "Region parameter" begin
        nu = collect(1900.0:0.5:2200.0)
        T = compute_cavity_transmittance(nu, [2055.0], [23.0], [3000.0],
                                         0.92, 12.0e-4, 1.4, 0.3)

        result = fit_cavity_spectrum(nu, T;
            oscillators=[(nu0=2055.0, Gamma=23.0)],
            L=12.0e-4, n_bg=1.4,
            region=(1950, 2150))

        @test result.rsquared > 0.99
        @test length(result._nu) < length(nu)
        @test all(1950 .<= result._nu .<= 2150)
    end

    @testset "fit_nu0 and fit_Gamma flags" begin
        nu = collect(1900.0:0.5:2200.0)
        T = compute_cavity_transmittance(nu, [2055.0], [23.0], [3000.0],
                                         0.92, 12.0e-4, 1.4, 0.3)

        # Start the oscillator slightly off and let the fit recover it
        result = fit_cavity_spectrum(nu, T;
            oscillators=[(nu0=2050.0, Gamma=20.0)],
            L=12.0e-4, n_bg=1.4,
            fit_nu0=true, fit_Gamma=true)

        @test result.rsquared > 0.99
        @test isapprox(result.oscillators[1].nu0, 2055.0, atol=1.0)
        @test isapprox(result.oscillators[1].Gamma, 23.0, atol=2.0)
    end

    @testset "Dispersion round-trip" begin
        # Generate synthetic dispersion data
        E_vib = 2055.0
        Omega_true = 25.0
        E0_true = 2040.0
        n_eff_true = 1.5

        angles = collect(0.0:2.0:30.0) .* (pi / 180)

        E_cav = cavity_mode_energy([E0_true, n_eff_true], angles)
        lp_true, up_true = polariton_branches(E_cav, E_vib, Omega_true)

        # Add small noise
        lp_noisy = lp_true .+ 0.5 .* randn(length(angles))
        up_noisy = up_true .+ 0.5 .* randn(length(angles))

        result = fit_dispersion(angles, lp_noisy, up_noisy;
            molecular_modes=E_vib,
            E0_init=2035.0,
            n_eff_init=1.4,
            Omega_init=20.0)

        @test result isa DispersionFitResult
        @test result.rsquared > 0.95
        @test isapprox(result.rabi_splitting, Omega_true, atol=3.0)
        @test isapprox(result.E0, E0_true, atol=5.0)
        @test isapprox(result.n_eff, n_eff_true, atol=0.2)
        @test length(result.molecular_modes) == 1

        # Hopfield at zero detuning should be ~50/50
        @test result.hopfield_zero.photon_LP ≈ 0.5 atol=0.05

        # Stored data should match input
        @test length(result.lp_angles) == length(angles)
        @test length(result.up_angles) == length(angles)
        @test result.lp_positions ≈ lp_noisy atol=1e-10
        @test result.up_positions ≈ up_noisy atol=1e-10

        # Uncertainties should be finite and positive
        @test result.rabi_err > 0 && isfinite(result.rabi_err)
        @test result.E0_err > 0 && isfinite(result.E0_err)
        @test result.n_eff_err > 0 && isfinite(result.n_eff_err)
    end

    @testset "Dispersion textbook values (Omega=100)" begin
        # Textbook: cavity at 2000 cm^-1, Rabi splitting 100 cm^-1
        E_vib = 2000.0
        Omega_true = 100.0
        E0_true = 1950.0
        n_eff_true = 1.5

        angles = collect(0.0:2.0:40.0) .* (pi / 180)

        E_cav = cavity_mode_energy([E0_true, n_eff_true], angles)
        lp_true, up_true = polariton_branches(E_cav, E_vib, Omega_true)

        # Noiseless round-trip: should recover exact parameters
        result = fit_dispersion(angles, lp_true, up_true;
            molecular_modes=E_vib,
            E0_init=1940.0,
            n_eff_init=1.4,
            Omega_init=80.0)

        @test result isa DispersionFitResult
        @test result.rsquared > 0.999
        @test isapprox(result.rabi_splitting, Omega_true, atol=0.5)
        @test isapprox(result.E0, E0_true, atol=1.0)
        @test isapprox(result.n_eff, n_eff_true, atol=0.01)
    end

    @testset "Dispersion with different LP/UP angles" begin
        # In experiments, LP and UP may be measured at different angles
        E_vib = 2000.0
        Omega_true = 80.0
        E0_true = 1970.0
        n_eff_true = 1.5

        lp_angles = collect(0.0:3.0:25.0) .* (pi / 180)
        up_angles = collect(5.0:3.0:35.0) .* (pi / 180)

        E_cav_lp = cavity_mode_energy([E0_true, n_eff_true], lp_angles)
        E_cav_up = cavity_mode_energy([E0_true, n_eff_true], up_angles)
        lp_true, _ = polariton_branches(E_cav_lp, E_vib, Omega_true)
        _, up_true = polariton_branches(E_cav_up, E_vib, Omega_true)

        result = fit_dispersion(lp_angles, lp_true, up_angles, up_true;
            molecular_modes=E_vib,
            Omega_init=60.0)

        @test result isa DispersionFitResult
        @test result.rsquared > 0.999
        @test isapprox(result.rabi_splitting, Omega_true, atol=1.0)
        @test length(result.lp_angles) == length(lp_angles)
        @test length(result.up_angles) == length(up_angles)
    end

    @testset "Multi-mode dispersion fit" begin
        # Two molecular modes coupled to one cavity mode; LP/UP positions
        # generated from the (N+1)-level Hamiltonian eigenvalues.
        mol_modes = [2000.0, 2060.0]
        Omega_true = 60.0
        E0_true = 1960.0
        n_eff_true = 1.5

        angles = collect(0.0:2.0:35.0) .* (pi / 180)
        E_cav = cavity_mode_energy([E0_true, n_eff_true], angles)

        lp = [polariton_eigenvalues(e, mol_modes, [Omega_true, Omega_true])[1] for e in E_cav]
        up = [polariton_eigenvalues(e, mol_modes, [Omega_true, Omega_true])[end] for e in E_cav]

        result = fit_dispersion(angles, lp, up;
            molecular_modes=mol_modes,
            E0_init=1950.0,
            n_eff_init=1.4,
            Omega_init=50.0)

        @test result isa DispersionFitResult
        @test result.rsquared > 0.999
        @test isapprox(result.rabi_splitting, Omega_true, atol=1.0)
        @test isapprox(result.E0, E0_true, atol=2.0)
        @test length(result.molecular_modes) == 2
        # Uncertainties require ForwardDiff through the eigenvalue solver
        @test isfinite(result.rabi_err)
    end

    @testset "fit_dispersion from CavityFitResult vector" begin
        # Build synthetic CavityFitResults whose polariton_peaks follow a
        # known dispersion; the angle-resolved peak extraction should
        # recover the Rabi splitting.
        E_vib = 2055.0
        Omega_true = 25.0
        E0_true = 2040.0
        n_eff_true = 1.5

        angles = collect(0.0:5.0:30.0) .* (pi / 180)
        E_cav = cavity_mode_energy([E0_true, n_eff_true], angles)
        lp, up = polariton_branches(E_cav, E_vib, Omega_true)

        nu = collect(1900.0:1.0:2200.0)
        T = zeros(length(nu))
        results = [CavityFitResult(0.92, 12.0e-4, 1.4, 0.3, 1.0, 0.0,
                                   [(nu0=E_vib, Gamma=23.0, A=3000.0)],
                                   [lp[i], up[i]], 0.999, nu, T, nothing)
                   for i in eachindex(angles)]

        result = fit_dispersion(results; molecular_modes=E_vib, angles=angles)

        @test result isa DispersionFitResult
        @test isapprox(result.rabi_splitting, Omega_true, atol=1.0)
        @test isapprox(result.E0, E0_true, atol=2.0)

        # Mismatched lengths should error
        @test_throws AssertionError fit_dispersion(results;
            molecular_modes=E_vib, angles=angles[1:2])

        # Too few valid LP/UP pairs should error
        few = results[1:2]
        @test_throws ErrorException fit_dispersion(few;
            molecular_modes=E_vib, angles=angles[1:2])
    end

end
