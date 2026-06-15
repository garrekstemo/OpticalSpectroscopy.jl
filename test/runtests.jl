using Test
using OpticalSpectroscopy
using Unitful
using Statistics
using LinearAlgebra
using JSON
using Aqua
import Random
Random.seed!(42)

@testset "OpticalSpectroscopy.jl" begin

    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(OpticalSpectroscopy;
            deps_compat=(check_extras=false, ignore=[:LinearAlgebra, :SparseArrays, :Statistics],),
            persistent_tasks=false)
    end

    @testset "Type hierarchy" begin
        @test KineticTrace <: AbstractSpectroscopyData
        @test TASpectrum <: AbstractSpectroscopyData
        @test TimeResolvedMatrix <: AbstractSpectroscopyData
    end

    @testset "Constructor shape validation" begin
        # KineticTrace: time and signal must have equal length
        @test_throws ArgumentError KineticTrace([1.0, 2.0, 3.0], [1.0, 2.0])
        err = try KineticTrace([1.0, 2.0, 3.0], [1.0, 2.0]) catch e; e end
        @test occursin("3", err.msg) && occursin("2", err.msg)
        # positional (inner) path validates too
        @test_throws ArgumentError KineticTrace([1.0, 2.0, 3.0], [1.0, 2.0], NaN, Dict{Symbol,Any}())
        # valid construction unaffected
        @test KineticTrace([1.0, 2.0], [0.1, 0.2]) isa KineticTrace

        # TASpectrum: wavenumber and signal must have equal length
        @test_throws ArgumentError TASpectrum([2000.0, 2050.0], [0.1, 0.2, 0.3])
        @test_throws ArgumentError TASpectrum([2000.0, 2050.0], [0.1, 0.2, 0.3], NaN, Dict{Symbol,Any}())
        @test TASpectrum([2000.0, 2050.0], [0.1, 0.2]) isa TASpectrum

        # TimeResolvedMatrix: data must be (n_time, n_wavelength)
        time = collect(0.0:1.0:10.0)        # 11 points
        wl = collect(500.0:10.0:650.0)      # 16 points
        good = zeros(length(time), length(wl))
        @test TimeResolvedMatrix(time, wl, good) isa TimeResolvedMatrix

        # transposed data: error must name the expected shape and suggest permutedims
        bad_t = zeros(length(wl), length(time))
        @test_throws ArgumentError TimeResolvedMatrix(time, wl, bad_t)
        err_t = try TimeResolvedMatrix(time, wl, bad_t) catch e; e end
        @test occursin("(16, 11)", err_t.msg)
        @test occursin("(11, 16)", err_t.msg)
        @test occursin("permutedims", err_t.msg)

        # arbitrary wrong shape also throws (positional path)
        @test_throws ArgumentError TimeResolvedMatrix(time, wl, zeros(3, 4), Dict{Symbol,Any}())
    end

    @testset "AbstractSpectroscopyData interface - KineticTrace" begin
        trace = KineticTrace([0.0, 1.0, 2.0], [0.1, 0.5, 0.3])

        @test xdata(trace) == [0.0, 1.0, 2.0]
        @test ydata(trace) == [0.1, 0.5, 0.3]
        @test zdata(trace) === nothing
        @test xlabel(trace) == "Time (ps)"
        @test ylabel(trace) == "ΔA"
        @test is_matrix(trace) == false
    end

    @testset "AbstractSpectroscopyData interface - TASpectrum" begin
        spec = TASpectrum([2000.0, 2050.0, 2100.0], [0.1, 0.5, 0.3])

        @test xdata(spec) == [2000.0, 2050.0, 2100.0]
        @test ydata(spec) == [0.1, 0.5, 0.3]
        @test zdata(spec) === nothing
        @test xlabel(spec) == "Wavenumber (cm⁻¹)"
        @test ylabel(spec) == "ΔA"
        @test is_matrix(spec) == false
    end

    @testset "SweepData" begin
        n_points, n_sweeps = 5, 3
        X = reshape(collect(1.0:15.0), n_points, n_sweeps)
        Y = reshape(collect(101.0:115.0), n_points, n_sweeps)
        DC = zeros(n_points, n_sweeps)
        sd = SweepData(X, Y, DC)

        @test sd.X === X
        @test sd.Y === Y
        @test sd.DC === DC
        @test size(sd.X) == (n_points, n_sweeps)

        # NaN-tolerant: matches partial-sweep aborts
        X_with_nan = copy(X); X_with_nan[end, end] = NaN
        sd_nan = SweepData(X_with_nan, Y, DC)
        @test isnan(sd_nan.X[end, end])

        # SweepData is NOT a subtype of AbstractSpectroscopyData (it's raw lock-in
        # output, not a finished spectroscopy product like KineticTrace).
        @test !(SweepData <: AbstractSpectroscopyData)
    end

    @testset "AbstractSpectroscopyData interface - TimeResolvedMatrix" begin
        time = [0.0, 1.0, 2.0]
        wavelength = [800.0, 850.0, 900.0]
        data = rand(3, 3)
        matrix = TimeResolvedMatrix(time, wavelength, data)

        @test xdata(matrix) == wavelength
        @test ydata(matrix) == time
        @test zdata(matrix) === data
        @test xlabel(matrix) == "Wavelength (nm)"
        @test ylabel(matrix) == "Time (ps)"
        @test zlabel(matrix) == "ΔA"
        @test is_matrix(matrix) == true

        # Test wavenumber detection
        matrix_wn = TimeResolvedMatrix(time, [1900.0, 2000.0, 2100.0], data)
        @test xlabel(matrix_wn) == "Wavenumber (cm⁻¹)"
    end

    @testset "Extended interface - source_file, npoints, title" begin
        # KineticTrace
        trace = KineticTrace([0.0, 1.0, 2.0], [0.1, 0.5, 0.3];
                        metadata=Dict{Symbol,Any}(:filename => "test.lvm"))
        @test source_file(trace) == "test.lvm"
        @test npoints(trace) == 3
        @test title(trace) == "test.lvm"

        # KineticTrace without filename
        trace_empty = KineticTrace([0.0, 1.0], [0.1, 0.2])
        @test source_file(trace_empty) == ""
        @test npoints(trace_empty) == 2
        @test title(trace_empty) == ""

        # TASpectrum
        spec = TASpectrum([2000.0, 2050.0, 2100.0], [0.1, 0.5, 0.3];
                          metadata=Dict{Symbol,Any}(:filename => "spec.lvm"))
        @test source_file(spec) == "spec.lvm"
        @test npoints(spec) == 3
        @test title(spec) == "spec.lvm"

        # TimeResolvedMatrix
        time = [0.0, 1.0, 2.0]
        wavelength = [800.0, 850.0, 900.0, 950.0]
        data = rand(3, 4)
        matrix = TimeResolvedMatrix(time, wavelength, data;
                          metadata=Dict{Symbol,Any}(:source => "broadband-TA/"))
        @test source_file(matrix) == "broadband-TA/"
        @test npoints(matrix) == (3, 4)
        @test title(matrix) == "broadband-TA/"
    end

    @testset "Semantic accessors - KineticTrace" begin
        trace = KineticTrace([0.0, 1.0, 2.0], [0.1, 0.5, 0.3])
        @test delay(trace) === trace.time
        @test signal(trace) === trace.signal
    end

    @testset "Semantic accessors - TASpectrum" begin
        spec = TASpectrum([2000.0, 2050.0, 2100.0], [0.1, 0.5, 0.3])
        @test wavenumber(spec) === spec.wavenumber
        @test signal(spec) === spec.signal
    end

    @testset "Semantic accessors - TimeResolvedMatrix" begin
        time = [0.0, 1.0, 2.0]
        wl = [800.0, 850.0, 900.0]
        data = rand(3, 3)
        matrix = TimeResolvedMatrix(time, wl, data)
        @test wavelength(matrix) === matrix.wavelength
        @test delay(matrix) === matrix.time
        @test signal(matrix) === matrix.data
    end

    @testset "TimeResolvedMatrix indexing" begin
        time = [0.0, 1.0, 2.0, 3.0, 4.0]
        wavelength = [700.0, 750.0, 800.0, 850.0]
        data = rand(5, 4)
        matrix = TimeResolvedMatrix(time, wavelength, data)

        # Extract KineticTrace at wavelength
        trace = matrix[λ=800]
        @test trace isa KineticTrace
        @test length(trace.time) == 5
        @test trace.wavelength ≈ 800.0

        # Extract TASpectrum at time
        spec = matrix[t=2.0]
        @test spec isa TASpectrum
        @test length(spec.wavenumber) == 4
        @test spec.time_delay ≈ 2.0

        # Error cases
        @test_throws ErrorException matrix[]
        @test_throws ErrorException matrix[λ=800, t=1.0]
    end

    @testset "Spectrum - construction" begin
        @test Spectrum <: AbstractSpectroscopyData

        # Plain construction, empty metadata
        s = Spectrum([1500.0, 1501.0], [0.1, 0.2])
        @test s.x == [1500.0, 1501.0]
        @test s.y == [0.1, 0.2]
        @test isempty(s.metadata)

        # Inputs convert to Vector{Float64}
        s_range = Spectrum(1500.0:1.0:1504.0, [1, 2, 3, 2, 1])
        @test s_range.x isa Vector{Float64}
        @test s_range.y isa Vector{Float64}

        # kwargs constructor splats into metadata
        s_kw = Spectrum([1.0, 2.0], [3.0, 4.0]; cavity_length=12e-4, mirror="Au")
        @test s_kw.metadata[:cavity_length] == 12e-4
        @test s_kw.metadata[:mirror] == "Au"

        # Positional dict constructor
        md = Dict{Symbol,Any}(:sample => "NH4SCN")
        s_md = Spectrum([1.0, 2.0], [3.0, 4.0], md)
        @test s_md.metadata[:sample] == "NH4SCN"

        # String-keyed dict (lab layer: JASCO/QPSTools Dict{String,Any}) is
        # accepted and symbolized
        s_str = Spectrum([1.0, 2.0], [3.0, 4.0],
                         Dict{String,Any}("sample" => "NH4SCN", "cavity_length" => 12e-4))
        @test s_str.metadata[:sample] == "NH4SCN"
        @test s_str.metadata[:cavity_length] == 12e-4
        @test keytype(s_str.metadata) == Symbol

        # Int x data converts to Vector{Float64}
        s_int_x = Spectrum([1500, 1501], [0.1, 0.2])
        @test s_int_x.x isa Vector{Float64}

        # Length mismatch throws
        @test_throws ArgumentError Spectrum([1.0, 2.0], [1.0])
    end

    @testset "Spectrum - AbstractSpectroscopyData interface" begin
        s = Spectrum(collect(1500.0:1.0:1504.0), [1.0, 2.0, 3.0, 2.0, 1.0])
        @test xdata(s) == s.x
        @test ydata(s) == s.y
        @test zdata(s) === nothing
        @test is_matrix(s) == false
        @test npoints(s) == 5
        @test signal(s) == s.y

        # Label fallbacks: 1500-1504 detected as wavenumber
        @test xlabel(s) == "Wavenumber (cm⁻¹)"
        @test ylabel(s) == "Signal"

        # nm-range x detected as wavelength
        s_nm = Spectrum([500.0, 600.0], [1.0, 2.0])
        @test xlabel(s_nm) == "Wavelength (nm)"

        # Metadata labels win over detection
        s_lbl = Spectrum([1.0, 2.0], [3.0, 4.0]; xlabel="Energy (eV)", ylabel="Counts")
        @test xlabel(s_lbl) == "Energy (eV)"
        @test ylabel(s_lbl) == "Counts"

        # source_file / title from metadata
        @test source_file(s) == ""
        s_src = Spectrum([1.0, 2.0], [3.0, 4.0]; filename="a.csv")
        @test source_file(s_src) == "a.csv"
        @test title(s_src) == "a.csv"
        s_src2 = Spectrum([1.0, 2.0], [3.0, 4.0]; source="b.csv")
        @test source_file(s_src2) == "b.csv"
    end

    @testset "Spectrum - show" begin
        s = Spectrum(collect(1500.0:1.0:1504.0), [1.0, 2.0, 3.0, 2.0, 1.0];
                     sample="test", cavity_length=12e-4)
        @test occursin("Spectrum", sprint(show, s))
        @test occursin("5 points", sprint(show, s))
        @test occursin("cm⁻¹", sprint(show, s))
        s_xlbl = Spectrum([1.0, 2.0], [3.0, 4.0]; xlabel="Energy (eV)")
        @test !occursin("nm", sprint(show, s_xlbl))
        long = sprint(show, MIME("text/plain"), s)
        @test occursin("Points", long)
        @test occursin("sample", long)
        # Empty spectrum must not error
        s_empty = Spectrum(Float64[], Float64[])
        @test occursin("0 points", sprint(show, s_empty))
        @test sprint(show, MIME("text/plain"), s_empty) isa String
    end

    @testset "Generic dispatches reject 2D data" begin
        m = TimeResolvedMatrix([0.0, 1.0, 2.0], [700.0, 750.0], rand(3, 2))
        @test_throws ArgumentError fit_peaks(m)
        @test_throws ArgumentError fit_peaks(m, (700.0, 750.0))
        @test_throws ArgumentError subtract_spectrum(m, m)
        @test_throws ArgumentError add_spectra(m, m)
        @test_throws ArgumentError divide_spectra(m, m)
        @test_throws ArgumentError multiply_spectrum(m, 2.0)
        p = PLMap(rand(2, 2), rand(2, 2, 3), [0.0, 1.0], [0.0, 1.0], [1.0, 2.0, 3.0])
        @test_throws ArgumentError multiply_spectrum(p, 2.0)
    end

    @testset "Generic analysis dispatches - Spectrum and family" begin
        x = collect(400.0:0.5:800.0)
        y = @. 100 * exp(-(x - 520)^2 / (2 * 10^2)) + 5
        s = Spectrum(x, y)

        # find_peaks
        pks_typed = find_peaks(s)
        pks_vec = find_peaks(x, y)
        @test length(pks_typed) == length(pks_vec) == 1
        @test pks_typed[1].position == pks_vec[1].position

        # band_area
        @test band_area(s, 480.0, 560.0) == band_area(x, y, 480.0, 560.0)

        # calc_fwhm
        @test calc_fwhm(s) == calc_fwhm(x, y)

        # estimate_snr
        @test estimate_snr(s) == estimate_snr(y)

        # Works for other 1D family members too
        g = GatedSpectrum(x, y)
        @test find_peaks(g)[1].position == pks_vec[1].position
        @test band_area(g, 480.0, 560.0) == band_area(x, y, 480.0, 560.0)

        # KineticTrace and TASpectrum flow through the same generic methods
        kt = KineticTrace(x, y)
        @test estimate_snr(kt) == estimate_snr(y)
        ta = TASpectrum(x, y)
        @test band_area(ta, 480.0, 560.0) == band_area(x, y, 480.0, 560.0)

        # 2D guard
        m = TimeResolvedMatrix([0.0, 1.0], [700.0, 750.0], rand(2, 2))
        @test_throws ArgumentError find_peaks(m)
        @test_throws ArgumentError band_area(m, 1.0, 2.0)
        @test_throws ArgumentError calc_fwhm(m)
        @test_throws ArgumentError estimate_snr(m)
    end

    @testset "Spectrum transformations - smoothing and normalization" begin
        x = collect(400.0:1.0:800.0)
        y = @. 50 * exp(-(x - 520)^2 / (2 * 10^2)) + 0.5
        s = Spectrum(x, y; sample="test")

        sm = smooth_data(s; window=5)
        @test sm isa Spectrum
        @test sm.x == s.x
        @test sm.y ≈ smooth_data(y; window=5)
        @test sm.metadata == s.metadata

        # Metadata is copied, not aliased
        sm.metadata[:extra] = 1
        @test !haskey(s.metadata, :extra)

        sg = savitzky_golay_smooth(s; window=11, order=3)
        @test sg isa Spectrum
        @test sg.y ≈ savitzky_golay_smooth(y; window=11, order=3)
        sg.metadata[:extra2] = 1
        @test !haskey(s.metadata, :extra2)

        dv = derivative(s; order=1)
        @test dv isa Spectrum
        @test dv.y ≈ derivative(x, y; order=1)

        na = normalize_area(s)
        @test na isa Spectrum
        @test na.y ≈ normalize_area(x, y)
        @test band_area(na, 400.0, 800.0) ≈ 1.0

        np = normalize_to_peak(s, 520.0)
        @test np isa Spectrum
        @test np.y ≈ normalize_to_peak(x, y, 520.0)
    end

    @testset "Spectrum transformations - transmittance/absorbance" begin
        x = [1500.0, 1600.0]
        s_t = Spectrum(x, [0.5, 0.1]; sample="test")

        a = transmittance_to_absorbance(s_t)
        @test a isa Spectrum
        @test a.y ≈ transmittance_to_absorbance([0.5, 0.1])
        @test a.metadata[:ylabel] == "Absorbance"
        @test a.metadata[:sample] == "test"   # other keys preserved
        @test !haskey(s_t.metadata, :ylabel)  # input untouched

        t = absorbance_to_transmittance(a)
        @test t isa Spectrum
        @test t.y ≈ [0.5, 0.1]
        @test t.metadata[:ylabel] == "Transmittance"

        t_pct = absorbance_to_transmittance(a; percent=true)
        @test t_pct.y ≈ [50.0, 10.0]
        @test t_pct.metadata[:ylabel] == "Transmittance (%)"
    end

    @testset "Spectrum arithmetic returns Spectrum" begin
        x = collect(1500.0:1.0:1599.0)
        ya = @. 1.0 + 0.01 * (x - 1500)
        yb = fill(0.5, length(x))
        a = Spectrum(x, ya; sample="A")
        b = Spectrum(x, yb; sample="B")

        d = subtract_spectrum(a, b)
        @test d isa Spectrum
        @test d.y ≈ ya .- yb
        @test d.metadata[:sample] == "A"   # first argument's metadata wins

        sc = subtract_spectrum(a, b; scale=2.0)
        @test sc.y ≈ ya .- 2.0 .* yb

        su = add_spectra(a, b)
        @test su isa Spectrum
        @test su.y ≈ ya .+ yb
        @test su.metadata[:sample] == "A"

        q = divide_spectra(a, b)
        @test q isa Spectrum
        @test q.y ≈ ya ./ yb

        m2 = multiply_spectrum(a, 2.0)
        @test m2 isa Spectrum
        @test m2.y ≈ 2.0 .* ya

        av = average_spectra(a, b)
        @test av isa Spectrum
        @test av.y ≈ (ya .+ yb) ./ 2
        @test av.metadata[:sample] == "A"

        @test_throws ArgumentError average_spectra()

        b_shift = Spectrum(x .+ 0.25, yb)
        d_itp = subtract_spectrum(a, b_shift; interpolate=true)
        @test d_itp isa Spectrum
        @test d_itp.x == a.x

        new_x = collect(1500.0:0.5:1599.0)
        it = interpolate_spectrum(a, new_x)
        @test it isa Spectrum
        @test it.x == new_x
        @test it.y ≈ interpolate_spectrum(x, ya, new_x)
        @test it.metadata[:sample] == "A"

        # Mixed family types still return NamedTuples (generic path)
        g = GatedSpectrum(x, yb)
        nt = subtract_spectrum(a, g)
        @test nt isa NamedTuple
        @test nt.y ≈ ya .- yb
    end

    @testset "Spectrum baseline correction and transforms" begin
        x = collect(1000.0:1.0:1399.0)
        y = @. 20 * exp(-(x - 1200)^2 / (2 * 15^2)) + 0.01 * (x - 1000) + 2
        s = Spectrum(x, y; sample="test")

        c = correct_baseline(s; method=:arpls)
        @test c isa Spectrum
        ref = correct_baseline(x, y; method=:arpls)
        @test c.y ≈ ref.y
        @test c.x == x
        @test c.metadata[:sample] == "test"

        # rubberband goes through the (x, y) branch
        c_rb = correct_baseline(s; method=:rubberband)
        @test c_rb isa Spectrum
        @test c_rb.y ≈ correct_baseline(x, y; method=:rubberband).y

        sv = snv(s)
        @test sv isa Spectrum
        @test sv.y ≈ snv(y)
        @test sv.metadata == s.metadata

        r = Spectrum([500.0, 600.0], [0.3, 0.6])
        km = kubelka_munk(r)
        @test km isa Spectrum
        @test km.y ≈ kubelka_munk.([0.3, 0.6])
        @test km.metadata[:ylabel] == "F(R)"
        @test_throws ArgumentError kubelka_munk(Spectrum([500.0, 600.0], [0.3, 0.0]))
    end

    @testset "fit_peaks with raw vectors" begin
        # Synthetic single lorentzian peak
        x = collect(1900.0:0.5:2200.0)
        y = @. 0.5 / (1 + ((x - 2050.0) / 10.0)^2) + 0.01

        result = fit_peaks(x, y; n_peaks=1)
        @test result isa MultiPeakFitResult
        @test length(result) == 1
        @test result[1][:center].value ≈ 2050.0 atol=1.0
        @test result[1][:fwhm].value ≈ 20.0 atol=2.0
        @test result.r_squared > 0.99
    end

    @testset "fit_peaks with Fano model" begin
        # Synthetic asymmetric Fano peak: A=1.0, x0=2050, Gamma=10, q=3
        x = collect(1900.0:0.5:2200.0)
        A, x0, Gamma, q = 1.0, 2050.0, 10.0, 3.0
        y = @. A * (q + (x - x0) / Gamma)^2 / (1 + ((x - x0) / Gamma)^2) + 0.01

        result = fit_peaks(x, y; n_peaks=1, model=fano)
        @test result isa MultiPeakFitResult
        @test length(result) == 1
        @test result[1][:center].value ≈ x0 atol=1.0
        @test result[1][:width].value ≈ Gamma atol=2.0
        @test result[1][:q].value ≈ q atol=1.0
        @test haskey(result[1], :amplitude)
        @test result.r_squared > 0.99
    end

    @testset "fit_peaks multi-peak" begin
        # Synthetic two-peak spectrum
        x = collect(1900.0:0.5:2200.0)
        y = @. 0.5 / (1 + ((x - 2020.0) / 8.0)^2) + 0.3 / (1 + ((x - 2080.0) / 6.0)^2) + 0.01

        result = fit_peaks(x, y; n_peaks=2)
        @test result isa MultiPeakFitResult
        @test length(result) == 2

        centers = sort([result[1][:center].value, result[2][:center].value])
        @test centers[1] ≈ 2020.0 atol=5.0
        @test centers[2] ≈ 2080.0 atol=5.0

        y1 = predict_peak(result, 1)
        y2 = predict_peak(result, 2)
        @test length(y1) == length(x)
        @test length(y2) == length(x)

        @test result.r_squared > 0.99
    end

    @testset "MultiPeakFitResult indexing and iteration" begin
        x = collect(1900.0:0.5:2200.0)
        y = @. 0.5 / (1 + ((x - 2050.0) / 10.0)^2) + 0.01

        result = fit_peaks(x, y; n_peaks=1)
        @test result[1] isa PeakFitResult
        @test firstindex(result) == 1
        @test lastindex(result) == 1

        count = 0
        for pk in result
            @test pk isa PeakFitResult
            count += 1
        end
        @test count == length(result)
    end

    @testset "MultiPeakFitResult predict and residuals" begin
        x = collect(1900.0:0.5:2200.0)
        y = @. 0.5 / (1 + ((x - 2050.0) / 10.0)^2) + 0.01

        result = fit_peaks(x, y; n_peaks=1)

        y_fit = predict(result)
        @test length(y_fit) == result.npoints

        y_bl = predict_baseline(result)
        @test length(y_bl) == result.npoints

        res = residuals(result)
        @test length(res) == result.npoints
    end

    @testset "predict/predict_baseline use the fit-region baseline basis" begin
        # The polynomial baseline basis is normalized by the FIT region's
        # midpoint and span. Evaluating on a different x grid must reuse that
        # normalization, otherwise the baseline silently changes.
        x_full = collect(1900.0:0.5:2200.0)
        y_full = @. 0.5 / (1 + ((x_full - 2050.0) / 10.0)^2) +
                    0.02 + 0.0005 * (x_full - 2000.0)   # sloped baseline

        region = (2000.0, 2100.0)
        r = fit_peaks(x_full, y_full, region; n_peaks=1)
        @test r.r_squared > 0.999

        in_region = region[1] .<= x_full .<= region[2]
        x_fit = x_full[in_region]

        # Reference: evaluation on the original fit grid
        y_ref = predict(r)
        bl_ref = predict_baseline(r)

        # Asymmetric wider grid (different midpoint AND span) containing the
        # fit-region points exactly
        x_wide = collect(1950.0:0.5:2180.0)
        match = [findfirst(==(xv), x_wide) for xv in x_fit]
        @test !any(isnothing, match)

        y_wide = predict(r, x_wide)
        bl_wide = predict_baseline(r, x_wide)

        @test y_wide[match] ≈ y_ref
        @test bl_wide[match] ≈ bl_ref

        # The fitted linear baseline extrapolates the true baseline
        bl_true_wide = @. 0.02 + 0.0005 * (x_wide - 2000.0)
        @test maximum(abs.(bl_wide .- bl_true_wide)) < 0.01

        # Same-grid evaluation must equal the no-argument form
        @test predict(r, x_fit) ≈ y_ref
        @test predict_baseline(r, x_fit) ≈ bl_ref
    end

    @testset "Semantic accessors - MultiPeakFitResult" begin
        x = collect(1900.0:0.5:2200.0)
        y = @. 0.5 / (1 + ((x - 2050.0) / 10.0)^2) + 0.01
        result = fit_peaks(x, y; n_peaks=1)
        @test xdata(result) === result._x
        @test ydata(result) === result._y
    end

    @testset "Baseline correction - arPLS" begin
        x = collect(1.0:200.0)
        baseline_true = 0.1 .+ 0.001 .* x
        peaks = 2.0 .* exp.(-((x .- 50.0) ./ 10.0).^2)
        y = baseline_true .+ peaks

        bl = arpls_baseline(y; λ=1e5)
        @test length(bl) == length(y)
        @test maximum(bl) < maximum(y)
    end

    @testset "Baseline correction - SNIP" begin
        x = collect(1.0:200.0)
        baseline_true = 0.5 .* ones(200)
        peaks = 3.0 .* exp.(-((x .- 100.0) ./ 15.0).^2)
        y = baseline_true .+ peaks

        bl = snip_baseline(y; iterations=40)
        @test length(bl) == length(y)
        @test maximum(bl) < maximum(y)
    end

    @testset "Baseline correction - correct_baseline API" begin
        y = randn(100) .+ 10.0
        result = correct_baseline(y; method=:arpls)
        @test haskey(result, :y)
        @test haskey(result, :baseline)
        @test length(result.y) == length(y)
        @test length(result.baseline) == length(y)

        # With x values
        x = collect(1.0:100.0)
        result2 = correct_baseline(x, y; method=:arpls)
        @test haskey(result2, :x)
        @test haskey(result2, :y)
        @test haskey(result2, :baseline)

        # Invalid method
        @test_throws ArgumentError correct_baseline(y; method=:invalid)
    end

    @testset "Exponential fitting - synthetic decay" begin
        # Create synthetic decay: A * exp(-(t-t0)/tau) convolved with Gaussian IRF
        t = collect(-5.0:0.1:50.0)
        tau_true = 8.0
        A_true = 1.0
        sigma_true = 0.3
        offset_true = 0.02

        # Use the internal IRF convolution to generate test data
        signal = [OpticalSpectroscopy._exp_decay_irf_conv(ti, A_true, tau_true, 0.0, sigma_true) + offset_true
                  for ti in t]
        # Add small noise
        signal .+= 0.001 .* randn(length(signal))

        trace = KineticTrace(t, signal)
        result = fit_exp_decay(trace; irf=true, irf_width=0.2)

        @test result isa ExpDecayFit
        @test result.tau ≈ tau_true atol=1.0
        @test !isnan(result.sigma)
        @test result.rsquared > 0.99
    end

    @testset "Exponential fitting - no IRF" begin
        t = collect(0.0:0.1:50.0)
        tau_true = 10.0
        A_true = 0.8
        offset_true = 0.01

        signal = @. A_true * exp(-t / tau_true) + offset_true
        signal .+= 0.001 .* randn(length(signal))

        trace = KineticTrace(t, signal)
        result = fit_exp_decay(trace; irf=false)

        @test result isa ExpDecayFit
        @test result.tau ≈ tau_true atol=1.5
        @test isnan(result.sigma)
        @test result.rsquared > 0.99
    end

    @testset "Biexponential fitting via n_exp=2" begin
        t = collect(-2.0:0.1:50.0)
        tau1_true = 2.0
        tau2_true = 20.0
        A1_true = 0.4
        A2_true = 0.6
        sigma_true = 0.3

        signal = [OpticalSpectroscopy._exp_decay_irf_conv(ti, A1_true, tau1_true, 0.0, sigma_true) +
                  OpticalSpectroscopy._exp_decay_irf_conv(ti, A2_true, tau2_true, 0.0, sigma_true) + 0.01
                  for ti in t]
        signal .+= 0.002 .* randn(length(signal))

        trace = KineticTrace(t, signal)
        result = fit_exp_decay(trace; n_exp=2, irf=true, irf_width=0.2)

        @test result isa MultiexpDecayFit
        @test length(result.taus) == 2
        @test all(result.taus .> 0)
        @test result.taus[1] < result.taus[2]
        @test result.rsquared > 0.95
    end

    @testset "Multi-exponential fitting (n_exp parameter)" begin
        t = collect(-2.0:0.1:50.0)
        signal = [OpticalSpectroscopy._exp_decay_irf_conv(ti, 0.5, 3.0, 0.0, 0.3) +
                  OpticalSpectroscopy._exp_decay_irf_conv(ti, 0.3, 15.0, 0.0, 0.3) + 0.01
                  for ti in t]
        signal .+= 0.002 .* randn(length(signal))

        trace = KineticTrace(t, signal)

        # n_exp=2
        result2 = fit_exp_decay(trace; n_exp=2, irf=true, irf_width=0.2)
        @test result2 isa MultiexpDecayFit
        @test OpticalSpectroscopy.n_exp(result2) == 2
        @test length(result2.taus) == 2
        @test length(result2.amplitudes) == 2
        @test all(result2.taus .> 0)
        @test result2.taus[1] <= result2.taus[2]
        @test result2.rsquared > 0.95

        w = OpticalSpectroscopy.weights(result2)
        @test length(w) == 2
        @test sum(w) ≈ 1.0 atol=1e-10

        # predict should work
        curve = predict(result2, trace)
        @test length(curve) == length(trace.time)
        @test all(isfinite, curve)
    end

    @testset "Global fitting - synthetic (n_exp=1)" begin
        t = collect(-2.0:0.1:30.0)
        tau_true = 5.0
        sigma_true = 0.3

        signal_esa = [OpticalSpectroscopy._exp_decay_irf_conv(ti, 0.8, tau_true, 0.0, sigma_true) + 0.01
                      for ti in t]
        signal_gsb = [OpticalSpectroscopy._exp_decay_irf_conv(ti, -0.5, tau_true, 0.0, sigma_true) - 0.005
                      for ti in t]

        trace_esa = KineticTrace(t, signal_esa)
        trace_gsb = KineticTrace(t, signal_gsb)

        result = fit_global([trace_esa, trace_gsb]; n_exp=1, labels=["ESA", "GSB"], irf_width=0.2)

        @test result isa GlobalFitResult
        @test length(result.taus) == 1
        @test result.taus[1] > 0
        @test result.taus[1] ≈ tau_true atol=2.0
        @test size(result.amplitudes) == (2, 1)
        @test result.labels == ["ESA", "GSB"]
        @test result.rsquared > 0.95
        @test isnothing(result.wavelengths)
        @test OpticalSpectroscopy.n_exp(result) == 1

        # predict
        curves = predict(result, [trace_esa, trace_gsb])
        @test length(curves) == 2
        @test length(curves[1]) == length(t)
    end

    @testset "Global fitting - multi-exp (n_exp=2)" begin
        t = collect(-2.0:0.1:50.0)
        tau1_true = 2.0
        tau2_true = 15.0
        sigma_true = 0.3

        # Trace 1: both components positive (ESA-like)
        signal1 = [OpticalSpectroscopy._exp_decay_irf_conv(ti, 0.6, tau1_true, 0.0, sigma_true) +
                   OpticalSpectroscopy._exp_decay_irf_conv(ti, 0.4, tau2_true, 0.0, sigma_true) + 0.01
                   for ti in t]
        # Trace 2: both components negative (GSB-like)
        signal2 = [OpticalSpectroscopy._exp_decay_irf_conv(ti, -0.3, tau1_true, 0.0, sigma_true) +
                   OpticalSpectroscopy._exp_decay_irf_conv(ti, -0.5, tau2_true, 0.0, sigma_true) - 0.005
                   for ti in t]

        trace1 = KineticTrace(t, signal1)
        trace2 = KineticTrace(t, signal2)

        result = fit_global([trace1, trace2]; n_exp=2, irf_width=0.2, labels=["ESA", "GSB"])

        @test result isa GlobalFitResult
        @test length(result.taus) == 2
        @test result.taus[1] < result.taus[2]  # sorted fast→slow
        @test result.taus[1] ≈ tau1_true atol=2.0
        @test result.taus[2] ≈ tau2_true atol=5.0
        @test size(result.amplitudes) == (2, 2)
        @test result.rsquared > 0.95
        @test OpticalSpectroscopy.n_exp(result) == 2
    end

    @testset "Global fitting - TimeResolvedMatrix dispatch" begin
        t = collect(-2.0:0.2:30.0)
        wavelength = collect(500.0:20.0:700.0)
        tau_true = 5.0
        sigma_true = 0.3

        data = zeros(length(t), length(wavelength))
        for (j, wl) in enumerate(wavelength)
            amp = 0.5 * sin((wl - 500) / 200 * pi)
            for (i, ti) in enumerate(t)
                data[i, j] = OpticalSpectroscopy._exp_decay_irf_conv(ti, amp, tau_true, 0.0, sigma_true) + 0.01
            end
        end

        matrix = TimeResolvedMatrix(t, wavelength, data)
        result = fit_global(matrix; n_exp=1, irf_width=0.2)

        @test result isa GlobalFitResult
        @test !isnothing(result.wavelengths)
        @test length(result.wavelengths) == length(wavelength)
        @test result.taus[1] ≈ tau_true atol=2.0
        @test size(result.amplitudes, 1) == length(wavelength)
        @test result.rsquared > 0.95

        # DAS accessor
        d = das(result)
        @test size(d) == (1, length(wavelength))
    end

    @testset "fit_global guards against intractable all-wavelength fits" begin
        t = collect(-2.0:0.5:20.0)

        # 300-wavelength matrix: default (fit every column) must refuse with
        # a helpful error instead of building a multi-thousand-parameter fit.
        wl_big = collect(range(500.0, 700.0, length=300))
        data_big = zeros(length(t), length(wl_big))
        for (j, wl) in enumerate(wl_big)
            amp = 0.5 + 0.4 * sin((wl - 500) / 200 * pi)
            for (i, ti) in enumerate(t)
                data_big[i, j] = OpticalSpectroscopy._exp_decay_irf_conv(ti, amp, 5.0, 0.0, 0.3) + 0.01
            end
        end
        matrix_big = TimeResolvedMatrix(t, wl_big, data_big)

        @test_throws ArgumentError fit_global(matrix_big; n_exp=1)
        err = try
            fit_global(matrix_big; n_exp=1)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("300", err.msg)
        @test occursin("max_wavelengths", err.msg)
        @test occursin("λ", err.msg)

        # λ=subset on the same matrix still works
        fit_sub = fit_global(matrix_big; n_exp=1, irf_width=0.2,
                             λ=[550.0, 600.0, 650.0])
        @test fit_sub isa GlobalFitResult
        @test length(fit_sub.wavelengths) == 3
        @test fit_sub.taus[1] ≈ 5.0 atol=1.0

        # max_wavelengths override is respected (lowering the threshold)
        wl_small = collect(range(500.0, 700.0, length=5))
        data_small = data_big[:, [1, 75, 150, 225, 300]]
        matrix_small = TimeResolvedMatrix(t, wl_small, data_small)
        @test_throws ArgumentError fit_global(matrix_small; n_exp=1, max_wavelengths=3)
        # ... and raising it allows the fit through
        fit_small = fit_global(matrix_small; n_exp=1, irf_width=0.2, max_wavelengths=10)
        @test fit_small isa GlobalFitResult
        @test length(fit_small.wavelengths) == 5
    end

    @testset "predict(GlobalFitResult, TimeResolvedMatrix) with wavelength subset" begin
        t = collect(-2.0:0.2:30.0)
        wavelength = collect(500.0:20.0:700.0)  # 11 wavelengths
        tau_true = 5.0
        sigma_true = 0.3

        data = zeros(length(t), length(wavelength))
        for (j, wl) in enumerate(wavelength)
            amp = 0.5 * sin((wl - 500) / 200 * pi)
            for (i, ti) in enumerate(t)
                data[i, j] = OpticalSpectroscopy._exp_decay_irf_conv(ti, amp, tau_true, 0.0, sigma_true) + 0.01
            end
        end
        matrix = TimeResolvedMatrix(t, wavelength, data)

        # Fit a subset at the HIGH end of the grid: full-grid indices (7, 9, 11)
        # exceed the subset column count, so any full-grid indexing bug
        # surfaces as a BoundsError or wrong columns.
        λ_subset = [620.0, 660.0, 700.0]
        fit = fit_global(matrix; n_exp=1, irf_width=0.2, λ=λ_subset)
        @test fit.wavelengths ≈ λ_subset

        recon = predict(fit, matrix)
        @test recon isa TimeResolvedMatrix
        @test size(recon.data) == (length(t), length(λ_subset))
        @test recon.wavelength ≈ λ_subset
        @test recon.time == t
        @test all(isfinite, recon.data)

        # Each column must be the model curve for the matching fitted trace
        for (j, _) in enumerate(λ_subset)
            expected = [OpticalSpectroscopy._multiexp_irf_conv(
                            ti, fit.taus, fit.amplitudes[j, :], fit.t0,
                            fit.sigma, fit.offsets[j]) for ti in t]
            @test recon.data[:, j] ≈ expected
        end

        # Reconstruction should track the noiseless input closely
        full_idx = [OpticalSpectroscopy._find_nearest_idx(wavelength, wl) for wl in λ_subset]
        for (j, fi) in enumerate(full_idx)
            @test maximum(abs.(recon.data[:, j] .- data[:, fi])) < 0.02
        end
    end

    @testset "das accessor - error without wavelengths" begin
        r = GlobalFitResult(
            [5.0], 0.25, 0.1,
            reshape([0.5, -0.3], 2, 1), [0.01, -0.005],
            ["ESA", "GSB"], nothing,
            0.9945, [0.9950, 0.9940],
            [zeros(10), zeros(10)]
        )
        @test_throws ErrorException das(r)
    end

    @testset "predict - ExpDecayFit" begin
        t = collect(-2.0:0.1:30.0)
        signal = [OpticalSpectroscopy._exp_decay_irf_conv(ti, 1.0, 5.0, 0.0, 0.3) + 0.01
                  for ti in t]

        trace = KineticTrace(t, signal)
        result = fit_exp_decay(trace; irf=true, irf_width=0.2)

        curve = predict(result, trace)
        @test length(curve) == length(t)
        @test all(isfinite, curve)

        # Also test with vector
        curve2 = predict(result, t)
        @test curve2 == curve
    end

    @testset "predict - MultiexpDecayFit (n_exp=2)" begin
        t = collect(-2.0:0.1:30.0)
        signal = [OpticalSpectroscopy._exp_decay_irf_conv(ti, 0.5, 2.0, 0.0, 0.3) +
                  OpticalSpectroscopy._exp_decay_irf_conv(ti, 0.5, 15.0, 0.0, 0.3) + 0.01
                  for ti in t]

        trace = KineticTrace(t, signal)
        result = fit_exp_decay(trace; n_exp=2, irf=true, irf_width=0.2)

        curve = predict(result, trace)
        @test length(curve) == length(t)
        @test all(isfinite, curve)
    end

    @testset "TA spectrum fitting - synthetic" begin
        ν = collect(1950.0:1.0:2150.0)
        # ESA at 2040, GSB at 2060
        esa = @. 0.005 * exp(-4 * log(2) * ((ν - 2040.0) / 15.0)^2)
        gsb = @. 0.008 * exp(-4 * log(2) * ((ν - 2060.0) / 18.0)^2)
        signal = esa .- gsb

        spec = TASpectrum(ν, signal)
        result = fit_ta_spectrum(spec; region=(1980, 2120))

        @test result isa TASpectrumFit
        @test length(result.peaks) == 2
        @test result[:esa].center ≈ 2040.0 atol=5.0
        @test result[:gsb].center ≈ 2060.0 atol=5.0
        @test result[:esa].label == :esa
        @test result[:gsb].label == :gsb
        @test anharmonicity(result) > 0
        @test result.rsquared > 0.95

        # Semantic accessors
        @test xdata(result) === result._x
        @test wavenumber(result) === result._x

        y_fit = predict(result, ν)
        @test length(y_fit) == length(ν)

        # predict_peak decomposes into individual contributions
        esa_contrib = predict_peak(result, 1, ν)
        gsb_contrib = predict_peak(result, 2, ν)
        @test all(esa_contrib .>= 0)  # ESA is positive
        @test all(gsb_contrib .<= 0)  # GSB is negative
    end

    @testset "TA spectrum fitting - three peaks" begin
        ν = collect(1900.0:1.0:2200.0)
        esa = @. 0.004 * exp(-4 * log(2) * ((ν - 2030.0) / 20.0)^2)
        gsb = @. 0.008 * exp(-4 * log(2) * ((ν - 2060.0) / 15.0)^2)
        se = @. 0.003 * exp(-4 * log(2) * ((ν - 2100.0) / 25.0)^2)
        signal = esa .- gsb .- se

        spec = TASpectrum(ν, signal)
        result = fit_ta_spectrum(spec; peaks=[:esa, :gsb, :se])

        @test length(result.peaks) == 3
        @test result[:esa].label == :esa
        @test result[:gsb].label == :gsb
        @test result[:se].label == :se
        @test result.rsquared > 0.9
    end

    @testset "TA spectrum fitting - per-peak models" begin
        ν = collect(1950.0:1.0:2150.0)
        esa = @. 0.005 * exp(-4 * log(2) * ((ν - 2040.0) / 15.0)^2)
        gsb = @. 0.008 * exp(-4 * log(2) * ((ν - 2060.0) / 18.0)^2)
        signal = esa .- gsb

        spec = TASpectrum(ν, signal)
        result = fit_ta_spectrum(spec; peaks=[(:esa, lorentzian), (:gsb, gaussian)],
                                 region=(1980, 2120))

        @test result[:esa].model == "lorentzian"
        @test result[:gsb].model == "gaussian"
        @test result.rsquared > 0.9
    end

    @testset "TA spectrum fitting - multiple peaks same type" begin
        # W(CO)6-like: 3 ESA + 3 GSB at well-separated positions
        ν = collect(1850.0:0.5:2150.0)
        esa1 = @. 0.003 * exp(-4 * log(2) * ((ν - 1960.0) / 10.0)^2)
        esa2 = @. 0.005 * exp(-4 * log(2) * ((ν - 1990.0) / 12.0)^2)
        esa3 = @. 0.002 * exp(-4 * log(2) * ((ν - 2060.0) / 8.0)^2)
        gsb1 = @. 0.004 * exp(-4 * log(2) * ((ν - 1970.0) / 10.0)^2)
        gsb2 = @. 0.007 * exp(-4 * log(2) * ((ν - 2000.0) / 12.0)^2)
        gsb3 = @. 0.003 * exp(-4 * log(2) * ((ν - 2070.0) / 8.0)^2)
        signal = (esa1 .+ esa2 .+ esa3) .- (gsb1 .+ gsb2 .+ gsb3)

        spec = TASpectrum(ν, signal)
        result = fit_ta_spectrum(spec; peaks=[:esa, :esa, :esa, :gsb, :gsb, :gsb])

        @test length(result.peaks) == 6
        @test count(p -> p.label == :esa, result.peaks) == 3
        @test count(p -> p.label == :gsb, result.peaks) == 3
        @test result.rsquared > 0.9

        # anharmonicity is NaN with multiple ESA/GSB
        @test isnan(anharmonicity(result))

        # Individual peak contributions have correct signs
        for i in 1:3
            @test all(predict_peak(result, i) .>= -1e-10)  # ESA peaks positive
        end
        for i in 4:6
            @test all(predict_peak(result, i) .<= 1e-10)   # GSB peaks negative
        end
    end

    @testset "Spectroscopy utilities - normalize" begin
        x = [1.0, 2.0, -3.0, 0.5]
        xn = OpticalSpectroscopy.normalize(x)
        @test maximum(abs.(xn)) ≈ 1.0
        @test xn[3] ≈ -1.0

        # Zero vector
        z = zeros(5)
        @test OpticalSpectroscopy.normalize(z) == zeros(5)
    end

    @testset "No export collision with LinearAlgebra.normalize" begin
        # `normalize` must NOT be exported: it collides with
        # LinearAlgebra.normalize in any session that loads both packages.
        @test :normalize ∉ names(OpticalSpectroscopy)

        # With `using LinearAlgebra` at the top of this file, the bare name
        # must resolve to LinearAlgebra.normalize (unit 2-norm), proving the
        # two packages coexist.
        v = normalize([3.0, 4.0])
        @test v ≈ [0.6, 0.8]

        # The package function remains accessible fully qualified.
        @test OpticalSpectroscopy.normalize([2.0, -4.0]) == [0.5, -1.0]
    end

    @testset "Spectroscopy utilities - time_index" begin
        times = [0.0, 1.0, 2.0, 5.0, 10.0]
        @test OpticalSpectroscopy.time_index(times, 2.1) == 3
        @test OpticalSpectroscopy.time_index(times, 0.0) == 1
        @test OpticalSpectroscopy.time_index(times, 7.0) == 4
    end

    @testset "Spectroscopy utilities - transmittance/absorbance" begin
        # Scalar
        @test transmittance_to_absorbance(0.1) ≈ 1.0
        @test transmittance_to_absorbance(50.0, percent=true) ≈ log10(2) atol=1e-6
        @test absorbance_to_transmittance(1.0) ≈ 0.1
        @test absorbance_to_transmittance(1.0, percent=true) ≈ 10.0

        # Vector
        T = [0.9, 0.5, 0.1]
        A = transmittance_to_absorbance(T)
        @test length(A) == 3
        T_back = absorbance_to_transmittance(A)
        @test T_back ≈ T atol=1e-10

        # Error
        @test_throws ArgumentError transmittance_to_absorbance(0.0)
        @test_throws ArgumentError transmittance_to_absorbance(-1.0)
    end

    @testset "Spectroscopy utilities - reflectance conversions" begin
        # Scalar: R = 1 - T
        @test transmittance_to_reflectance(0.2) ≈ 0.8
        @test transmittance_to_reflectance(20.0; percent=true) ≈ 0.8
        # Scalar: R = 1 - 10^(-A)
        @test absorbance_to_reflectance(0.0) ≈ 0.0
        @test absorbance_to_reflectance(1.0) ≈ 0.9

        # Vector
        T = [0.1, 0.5, 0.9]
        R = transmittance_to_reflectance(T)
        @test R ≈ [0.9, 0.5, 0.1]
        A = [0.0, 1.0, 2.0]
        @test absorbance_to_reflectance(A) ≈ [0.0, 0.9, 0.99]

        # Consistency: absorbance route equals transmittance route
        @test absorbance_to_reflectance(1.0) ≈ transmittance_to_reflectance(absorbance_to_transmittance(1.0))
    end

    @testset "Spectroscopy utilities - smooth_data" begin
        y = [0.0, 0.0, 1.0, 0.0, 0.0]
        ys = smooth_data(y; window=3)
        @test length(ys) == length(y)
        @test ys[3] < 1.0  # Smoothed peak should be lower
        @test ys[3] > 0.0
    end

    @testset "Spectroscopy utilities - calc_fwhm" begin
        x = collect(1.0:0.1:100.0)
        center = 50.0
        sigma = 5.0
        y = @. exp(-((x - center) / sigma)^2)

        result = calc_fwhm(x, y; smooth_window=1)
        @test result.peak_position ≈ center atol=0.2
        fwhm_expected = 2 * sigma * sqrt(log(2))
        @test result.fwhm ≈ fwhm_expected atol=1.0
    end

    @testset "Spectroscopy utilities - subtract_spectrum" begin
        ν = collect(1900.0:1.0:2100.0)
        y1 = @. 0.5 * exp(-((ν - 2000) / 20)^2)
        y2 = @. 0.1 * exp(-((ν - 2000) / 30)^2)

        # NamedTuple interface (raw vectors)
        result = subtract_spectrum((x=ν, y=y1), (x=ν, y=y2))
        @test result.x == ν
        @test result.y ≈ y1 .- y2

        # Typed interface (TASpectrum)
        spec1 = TASpectrum(ν, y1)
        spec2 = TASpectrum(ν, y2)
        result_typed = subtract_spectrum(spec1, spec2)
        @test result_typed.x == ν
        @test result_typed.y ≈ y1 .- y2

        # scale parameter
        result_scaled = subtract_spectrum(spec1, spec2; scale=0.5)
        @test result_scaled.y ≈ y1 .- 0.5 .* y2

        # Mismatched lengths → error with interpolate hint
        ν_short = collect(1900.0:2.0:2100.0)
        y_short = @. 0.1 * exp(-((ν_short - 2000) / 30)^2)
        @test_throws ErrorException subtract_spectrum((x=ν, y=y1), (x=ν_short, y=y_short))

        # Misaligned x-values (same length, different grids) → error with hint
        ν_shifted = ν .+ 0.5
        y_shifted = @. 0.1 * exp(-((ν_shifted - 2000) / 30)^2)
        @test_throws ErrorException subtract_spectrum((x=ν, y=y1), (x=ν_shifted, y=y_shifted))

        # Small misalignment (< 0.01) passes without error
        ν_tiny_shift = ν .+ 0.005
        y_tiny = @. 0.1 * exp(-((ν_tiny_shift - 2000) / 30)^2)
        result_close = subtract_spectrum((x=ν, y=y1), (x=ν_tiny_shift, y=y_tiny))
        @test result_close.y ≈ y1 .- y_tiny

        # interpolate=true with mismatched lengths → works
        result_interp = subtract_spectrum((x=ν, y=y1), (x=ν_short, y=y_short); interpolate=true)
        @test length(result_interp.x) == length(ν)
        @test length(result_interp.y) == length(ν)

        # interpolate=true with misaligned grids → works
        result_interp2 = subtract_spectrum((x=ν, y=y1), (x=ν_shifted, y=y_shifted); interpolate=true)
        @test result_interp2.x == ν
        @test length(result_interp2.y) == length(ν)

        # interpolate=true with scale
        result_interp_scaled = subtract_spectrum((x=ν, y=y1), (x=ν_short, y=y_short); interpolate=true, scale=0.5)
        @test length(result_interp_scaled.y) == length(ν)
    end

    @testset "Spectroscopy conversions - wavenumber/wavelength" begin
        @test wavenumber_to_wavelength(2000) ≈ 5000.0u"nm" rtol=1e-6
        @test wavenumber_to_wavelength(1000) ≈ 10000.0u"nm" rtol=1e-6
        @test wavenumber_to_wavelength(2000, output_unit=u"μm") ≈ 5.0u"μm" rtol=1e-6

        @test wavenumber_to_wavelength(2000u"cm^-1") ≈ 5000.0u"nm" rtol=1e-6

        @test wavelength_to_wavenumber(500) ≈ 20000.0u"cm^-1" rtol=1e-6
        @test wavelength_to_wavenumber(5000) ≈ 2000.0u"cm^-1" rtol=1e-6

        @test wavelength_to_wavenumber(500u"nm") ≈ 20000.0u"cm^-1" rtol=1e-6
        @test wavelength_to_wavenumber(5u"μm") ≈ 2000.0u"cm^-1" rtol=1e-6

        # Round-trip
        λ = wavenumber_to_wavelength(2000)
        @test wavelength_to_wavenumber(λ) ≈ 2000.0u"cm^-1" rtol=1e-6

        ν̃ = wavelength_to_wavenumber(800)
        @test wavenumber_to_wavelength(ν̃) ≈ 800.0u"nm" rtol=1e-6

        # Physical sanity
        @test wavelength_to_wavenumber(10.6u"μm") ≈ 943.4u"cm^-1" rtol=1e-3
        @test wavelength_to_wavenumber(400) ≈ 25000.0u"cm^-1" rtol=1e-6

        # Errors
        @test_throws ArgumentError wavenumber_to_wavelength(-100)
        @test_throws ArgumentError wavenumber_to_wavelength(0)
        @test_throws ArgumentError wavelength_to_wavenumber(-100)
        @test_throws ArgumentError wavelength_to_wavenumber(0)
    end

    @testset "Spectroscopy conversions - energy" begin
        @test wavenumber_to_energy(8065.54) ≈ 1.0u"eV" rtol=1e-4
        @test wavenumber_to_energy(2000) ≈ 0.248u"eV" rtol=1e-2
        @test wavenumber_to_energy(2000, output_unit=u"meV") ≈ 248.0u"meV" rtol=1e-2

        @test wavenumber_to_energy(2000u"cm^-1") ≈ 0.248u"eV" rtol=1e-2

        @test wavelength_to_energy(1239.84) ≈ 1.0u"eV" rtol=1e-4
        @test wavelength_to_energy(500) ≈ 2.48u"eV" rtol=1e-2

        @test wavelength_to_energy(500u"nm") ≈ 2.48u"eV" rtol=1e-2
        @test wavelength_to_energy(5u"μm") ≈ 0.248u"eV" rtol=1e-2

        @test energy_to_wavelength(1.0) ≈ 1239.84u"nm" rtol=1e-4
        @test energy_to_wavelength(2.0u"eV") ≈ 619.92u"nm" rtol=1e-4
        @test energy_to_wavelength(100u"meV") ≈ 12398.4u"nm" rtol=1e-4
        @test energy_to_wavelength(1.0, output_unit=u"μm") ≈ 1.24u"μm" rtol=1e-2

        # Round-trip
        E = wavelength_to_energy(800)
        @test energy_to_wavelength(E) ≈ 800.0u"nm" rtol=1e-6

        E2 = wavenumber_to_energy(2000)
        λ = energy_to_wavelength(E2)
        @test wavelength_to_wavenumber(λ) ≈ 2000.0u"cm^-1" rtol=1e-6

        # Physical sanity
        @test wavelength_to_energy(800) ≈ 1.55u"eV" rtol=1e-2
        @test wavelength_to_energy(532) ≈ 2.33u"eV" rtol=1e-2
        @test wavenumber_to_energy(1700, output_unit=u"meV") ≈ 211.0u"meV" rtol=1e-2

        # Errors
        @test_throws ArgumentError wavenumber_to_energy(-100)
        @test_throws ArgumentError wavenumber_to_energy(0)
        @test_throws ArgumentError wavelength_to_energy(-100)
        @test_throws ArgumentError wavelength_to_energy(0)
        @test_throws ArgumentError energy_to_wavelength(-1)
        @test_throws ArgumentError energy_to_wavelength(0)
    end

    @testset "Linewidth ↔ Decay time conversions" begin
        @test decay_time_to_linewidth(1.0) ≈ 0.658u"meV" rtol=1e-2
        @test decay_time_to_linewidth(1u"ps") ≈ 0.658u"meV" rtol=1e-2

        @test decay_time_to_linewidth(1000u"fs") ≈ 0.658u"meV" rtol=1e-2
        @test decay_time_to_linewidth(0.001u"ns") ≈ 0.658u"meV" rtol=1e-2
        @test decay_time_to_linewidth(100u"fs") ≈ 6.58u"meV" rtol=1e-2

        @test decay_time_to_linewidth(1u"ps", output_unit=u"eV") ≈ 0.000658u"eV" rtol=1e-2
        @test decay_time_to_linewidth(1u"ps", output_unit=u"cm^-1") ≈ 5.31u"cm^-1" rtol=1e-2
        @test decay_time_to_linewidth(1u"ps", output_unit=u"THz") ≈ 0.159u"THz" rtol=1e-2

        @test linewidth_to_decay_time(0.658) ≈ 1.0u"ps" rtol=1e-2
        @test linewidth_to_decay_time(0.658u"meV") ≈ 1.0u"ps" rtol=1e-2
        @test linewidth_to_decay_time(6.58u"meV") ≈ 100.0u"fs" rtol=1e-2

        @test linewidth_to_decay_time(5.31u"cm^-1") ≈ 1.0u"ps" rtol=1e-2
        @test linewidth_to_decay_time(0.159u"THz") ≈ 1.0u"ps" rtol=1e-2

        @test linewidth_to_decay_time(0.658u"meV", output_unit=u"fs") ≈ 1000.0u"fs" rtol=1e-2
        @test linewidth_to_decay_time(0.658u"meV", output_unit=u"ns") ≈ 0.001u"ns" rtol=1e-2

        # Round-trip
        τ = 2.5u"ps"
        Γ = decay_time_to_linewidth(τ)
        @test linewidth_to_decay_time(Γ) ≈ τ rtol=1e-10

        Γ2 = 10u"meV"
        τ2 = linewidth_to_decay_time(Γ2)
        @test decay_time_to_linewidth(τ2) ≈ Γ2 rtol=1e-10

        # Physical sanity
        @test decay_time_to_linewidth(100u"fs") ≈ 6.58u"meV" rtol=1e-2
        @test decay_time_to_linewidth(1u"ns") ≈ 0.658u"μeV" rtol=1e-2
        @test decay_time_to_linewidth(10u"fs") ≈ 65.8u"meV" rtol=1e-2

        # Errors
        @test_throws ArgumentError decay_time_to_linewidth(-1.0)
        @test_throws ArgumentError decay_time_to_linewidth(0.0)
        @test_throws ArgumentError linewidth_to_decay_time(-1.0)
        @test_throws ArgumentError linewidth_to_decay_time(0.0)
    end

    @testset "format_results - MultiPeakFitResult" begin
        pk = PeakFitResult(
            [:amplitude, :center, :fwhm],
            [0.452, 2062.3, 24.7],
            [0.002, 0.12, 0.31],
            [(0.448, 0.456), (2062.06, 2062.54), (24.08, 25.32)],
            0.9985, 0.001, 0.0001, 100,
            (2000.0, 2100.0), "lorentzian", "NH4SCN_DMF_1M"
        )
        result = MultiPeakFitResult(
            [pk], [0.01], 0, 0.9985, 0.001, 0.0001, 100,
            (2000.0, 2100.0), "NH4SCN_DMF_1M",
            [0.452, 2062.3, 24.7, 0.01], lorentzian, 3,
            collect(2000.0:1.0:2099.0), zeros(100),
            2049.5, 99.0   # fit-region x_mid, x_range
        )
        md = OpticalSpectroscopy.format_results(result)
        @test occursin("Peak Fit Results", md)
        @test occursin("2062.3", md)
        @test occursin("24.7", md)
        @test occursin("0.452", md)
        @test occursin("lorentzian", md)
        @test occursin("0.9985", md)
        @test occursin("2000", md)
        @test occursin("2100", md)
    end

    @testset "format_results - ExpDecayFit" begin
        result = ExpDecayFit(0.5, 8.3, 0.1, 0.25, 0.01, :esa, zeros(10), 0.9923)
        md = OpticalSpectroscopy.format_results(result)
        @test occursin("Exponential Decay Fit", md)
        @test occursin("ESA", md)
        @test occursin("8.3", md)
        @test occursin("0.25", md)
        @test occursin("0.9923", md)
    end

    @testset "format_results - MultiexpDecayFit" begin
        result = MultiexpDecayFit(
            [0.5, 5.0, 50.0], [0.3, 0.4, 0.1],
            0.1, 0.25, 0.01, :esa, zeros(10), 0.9991
        )
        md = OpticalSpectroscopy.format_results(result)
        @test occursin("Multi-exponential Decay Fit", md)
        @test occursin("3 components", md)
        @test occursin("0.5", md)
        @test occursin("5.0", md)
        @test occursin("50.0", md)
        @test occursin("0.9991", md)
    end

    @testset "format_results - GlobalFitResult" begin
        result = GlobalFitResult(
            [8.5], 0.25, 0.1,
            reshape([0.5, -0.3], 2, 1), [0.01, -0.005],
            ["ESA", "GSB"], nothing,
            0.9945,
            [0.9950, 0.9940],
            [zeros(10), zeros(10)]
        )
        md = OpticalSpectroscopy.format_results(result)
        @test occursin("Global Fit Results", md)
        @test occursin("2 traces", md)
        @test occursin("8.5", md)
        @test occursin("ESA", md)
        @test occursin("GSB", md)
        @test occursin("0.9945", md)
        @test occursin("Shared Parameters", md)
        @test occursin("Per-Trace", md)
    end

    @testset "irf_fwhm and pulse_fwhm" begin
        sigma = 0.3
        fwhm = OpticalSpectroscopy.irf_fwhm(sigma)
        @test fwhm ≈ 2 * sqrt(2 * log(2)) * sigma rtol=1e-10

        pfwhm = OpticalSpectroscopy.pulse_fwhm(sigma)
        @test pfwhm ≈ fwhm / sqrt(2) rtol=1e-10
    end

    @testset "PeakInfo struct" begin
        pi = PeakInfo(2050.0, 0.5, 0.4, 20.0, (2040.0, 2060.0), 100)
        @test pi.position == 2050.0
        @test pi.intensity == 0.5
        @test pi.prominence == 0.4
        @test pi.width == 20.0
        @test pi.bounds == (2040.0, 2060.0)
        @test pi.index == 100

        # Show methods should not error
        io = IOBuffer()
        show(io, pi)
        @test length(String(take!(io))) > 0
    end

    @testset "peak_table" begin
        peaks = [
            PeakInfo(2050.0, 0.5, 0.4, 20.0, (2040.0, 2060.0), 100),
            PeakInfo(2080.0, 0.3, 0.2, 15.0, (2072.5, 2087.5), 160)
        ]
        table = peak_table(peaks)
        @test occursin("Position", table)
        @test occursin("2050.0", table)
        @test occursin("2080.0", table)

        @test peak_table(PeakInfo[]) == "No peaks detected"
    end

    @testset "peak_bounds" begin
        # Single triangular peak at index 4; minima at the ends
        y = [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0]
        @test peak_bounds(y, 4) == (1, 7)

        # Two peaks separated by a valley at index 4
        y2 = [0.0, 2.0, 1.0, 0.5, 1.0, 3.0, 0.0]
        # From the left peak (idx 2), right edge stops at the valley (idx 4)
        @test peak_bounds(y2, 2) == (1, 4)
        # From the right peak (idx 6), left edge stops at the valley (idx 4)
        @test peak_bounds(y2, 6) == (4, 7)

        # Flat plateau is walked through (<= comparison)
        @test peak_bounds([0.0, 1.0, 1.0, 1.0, 0.0], 3) == (1, 5)

        # idx at the left edge with a rise to the right → both walks stop immediately
        @test peak_bounds([1.0, 5.0, 2.0], 1) == (1, 1)

        # Out-of-range index
        @test_throws BoundsError peak_bounds(y, 0)
        @test_throws BoundsError peak_bounds(y, 8)
    end

    @testset "Re-exports from CurveFit" begin
        # These should be accessible
        @test isdefined(OpticalSpectroscopy, :solve)
        @test isdefined(OpticalSpectroscopy, :NonlinearCurveFitProblem)
        @test isdefined(OpticalSpectroscopy, :coef)
        @test isdefined(OpticalSpectroscopy, :stderror)
    end

    @testset "Re-exports from CurveFitModels" begin
        @test isdefined(OpticalSpectroscopy, :gaussian)
        @test isdefined(OpticalSpectroscopy, :lorentzian)
        @test isdefined(OpticalSpectroscopy, :single_exponential)
        @test isdefined(OpticalSpectroscopy, :pseudo_voigt)
    end

    @testset "Chirp correction" begin

        @testset "subtract_background" begin
            n_time = 50
            n_wl = 20
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))
            background = repeat([0.01 * j for j in 1:n_wl]', n_time)

            signal = zeros(n_time, n_wl)
            for i in eachindex(time)
                if time[i] > 0
                    for j in eachindex(wavelength)
                        signal[i, j] = 0.1 * exp(-time[i] / 3.0) * sin(j * 0.5)
                    end
                end
            end

            data = signal .+ background
            metadata = Dict{Symbol,Any}(:source => "test")
            matrix = TimeResolvedMatrix(time, wavelength, data, metadata)

            corrected = subtract_background(matrix)
            @test corrected isa TimeResolvedMatrix
            @test size(corrected.data) == size(matrix.data)
            @test corrected.metadata[:background_subtracted] == true
            @test haskey(corrected.metadata, :baseline_t_range)

            pre_pump_mask = corrected.time .< -1.0
            pre_pump_data = corrected.data[pre_pump_mask, :]
            @test maximum(abs.(pre_pump_data)) < 0.01

            @test matrix.data !== corrected.data

            corrected2 = subtract_background(matrix; t_range=(-5.0, -2.0))
            @test corrected2.metadata[:baseline_t_range] == (-5.0, -2.0)
        end

        @testset "detect_chirp on synthetic data" begin
            n_time = 200
            n_wl = 100
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            ref_λ = 600.0
            chirp_fn(λ) = 0.0001 * (λ - ref_λ)^2 - 0.002 * (λ - ref_λ)

            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                λ = wavelength[j]
                t_onset = chirp_fn(λ)
                for i in eachindex(time)
                    t = time[i]
                    if t > t_onset
                        data[i, j] = 0.5 * exp(-(t - t_onset) / 3.0)
                    end
                end
            end

            metadata = Dict{Symbol,Any}(:source => "synthetic")
            matrix = TimeResolvedMatrix(time, wavelength, data, metadata)

            cal = detect_chirp(matrix; order=2, reference=ref_λ, smooth_window=7, bin_width=4)
            @test cal isa ChirpCalibration
            @test cal.poly_order == 2
            @test cal.reference_λ == ref_λ
            @test cal.r_squared > 0.9
            @test length(cal.wavelength) > 0
            @test length(cal.time_offset) == length(cal.wavelength)

            poly = polynomial(cal)
            @test abs(poly(ref_λ)) < 0.5

            @test report(cal) === nothing
        end

        @testset "correct_chirp" begin
            n_time = 100
            n_wl = 50
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            ref_λ = 600.0
            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                t_onset = 0.02 * (wavelength[j] - ref_λ)
                for i in eachindex(time)
                    if time[i] > t_onset
                        data[i, j] = exp(-(time[i] - t_onset) / 2.0)
                    end
                end
            end

            metadata = Dict{Symbol,Any}(:source => "synthetic")
            matrix = TimeResolvedMatrix(time, wavelength, data, metadata)

            cal = ChirpCalibration(
                collect(wavelength),
                [0.02 * (λ - ref_λ) for λ in wavelength],
                [-0.02 * ref_λ, 0.02],
                1,
                ref_λ,
                1.0,
                Dict{Symbol,Any}()
            )

            corrected = correct_chirp(matrix, cal)
            @test corrected isa TimeResolvedMatrix
            @test size(corrected.data) == size(matrix.data)
            @test corrected.metadata[:chirp_corrected] == true

            inner = n_wl ÷ 4 : 3 * n_wl ÷ 4
            peak_times = [time[argmax(corrected.data[:, j])] for j in inner]
            @test std(peak_times) < 1.0
        end

        @testset "save_chirp and load_chirp round-trip" begin
            cal = ChirpCalibration(
                [500.0, 600.0, 700.0],
                [-1.0, 0.0, 1.5],
                [0.1, -0.002, 0.00001],
                2,
                600.0,
                0.998,
                Dict{Symbol,Any}(:order => 2, :smooth_window => 15)
            )

            tmpfile = tempname() * ".json"
            save_chirp(tmpfile, cal)
            @test isfile(tmpfile)

            cal2 = load_chirp(tmpfile)
            @test cal2 isa ChirpCalibration
            @test cal2.wavelength ≈ cal.wavelength
            @test cal2.time_offset ≈ cal.time_offset
            @test cal2.poly_coeffs ≈ cal.poly_coeffs
            @test cal2.poly_order == cal.poly_order
            @test cal2.reference_λ ≈ cal.reference_λ
            @test cal2.r_squared ≈ cal.r_squared
            @test cal2.metadata[:order] == 2

            rm(tmpfile)
        end

        @testset "ChirpCalibration polynomial" begin
            cal = ChirpCalibration(
                Float64[], Float64[],
                [1.0, 0.5, 0.01],
                2, 0.0, 1.0,
                Dict{Symbol,Any}()
            )

            poly = polynomial(cal)
            @test poly(0.0) ≈ 1.0
            @test poly(1.0) ≈ 1.51
            @test poly(10.0) ≈ 1.0 + 5.0 + 1.0
        end

        @testset "Chirp exports available" begin
            @test isdefined(OpticalSpectroscopy, :ChirpCalibration)
            @test isdefined(OpticalSpectroscopy, :detect_chirp)
            @test isdefined(OpticalSpectroscopy, :correct_chirp)
            @test isdefined(OpticalSpectroscopy, :subtract_background)
            @test isdefined(OpticalSpectroscopy, :save_chirp)
            @test isdefined(OpticalSpectroscopy, :load_chirp)
            @test isdefined(OpticalSpectroscopy, :polynomial)
        end

        @testset "subtract_background with flat matrix" begin
            n_time = 20
            n_wl = 10
            time = collect(range(-2.0, 5.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))
            data = zeros(n_time, n_wl)
            metadata = Dict{Symbol,Any}(:source => "flat")
            matrix = TimeResolvedMatrix(time, wavelength, data, metadata)

            corrected = subtract_background(matrix)
            @test all(corrected.data .== 0.0)
        end

        @testset "detect_chirp :threshold method" begin
            n_time = 200
            n_wl = 80
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            ref_λ = 600.0
            chirp_fn(λ) = 0.005 * (λ - ref_λ)

            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                t_onset = chirp_fn(wavelength[j])
                for i in eachindex(time)
                    if time[i] > t_onset
                        data[i, j] = 0.5 * exp(-(time[i] - t_onset) / 3.0)
                    end
                end
            end

            matrix = TimeResolvedMatrix(time, wavelength, data)

            cal = detect_chirp(matrix; method=:threshold, order=1,
                               reference=ref_λ, smooth_window=7, bin_width=4)
            @test cal isa ChirpCalibration
            @test cal.r_squared > 0.8
            @test length(cal.wavelength) > 0
        end

        @testset "detect_chirp :threshold recovers known linear chirp" begin
            n_time = 200
            n_wl = 80
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            ref_λ = 600.0
            slope = 0.005
            chirp_fn(λ) = slope * (λ - ref_λ)

            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                t_onset = chirp_fn(wavelength[j])
                for i in eachindex(time)
                    if time[i] > t_onset
                        data[i, j] = 0.5 * exp(-(time[i] - t_onset) / 3.0)
                    end
                end
            end

            matrix = TimeResolvedMatrix(time, wavelength, data)
            cal = detect_chirp(matrix; method=:threshold, order=1,
                               reference=ref_λ, smooth_window=7, bin_width=4)

            poly = polynomial(cal)
            @test poly(ref_λ) ≈ 0.0 atol=0.5
            @test abs(poly(ref_λ + 50) - slope * 50) < 1.0
            @test abs(poly(ref_λ - 50) - slope * (-50)) < 1.0
            @test cal.r_squared > 0.9
        end

        @testset "detect_chirp input validation" begin
            n_time = 50
            n_wl = 20
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))
            data = rand(n_time, n_wl)
            matrix = TimeResolvedMatrix(time, wavelength, data)

            @test_throws ArgumentError detect_chirp(matrix; order=0)
            @test_throws ArgumentError detect_chirp(matrix; bin_width=0)
            @test_throws ArgumentError detect_chirp(matrix; bin_width=n_wl + 1)
            @test_throws ArgumentError detect_chirp(matrix; min_signal=0.0)
            @test_throws ArgumentError detect_chirp(matrix; min_signal=1.5)
            @test_throws ArgumentError detect_chirp(matrix; threshold=-1.0)
            @test_throws ArgumentError detect_chirp(matrix; method=:invalid)
            @test_throws ArgumentError detect_chirp(matrix; method=:threshold, onset_frac=0.0)
            @test_throws ArgumentError detect_chirp(matrix; method=:threshold, onset_frac=1.0)
        end

        @testset "detect_chirp even smooth_window accepted" begin
            n_time = 200
            n_wl = 40
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                for i in eachindex(time)
                    if time[i] > 0
                        data[i, j] = 0.5 * exp(-time[i] / 3.0)
                    end
                end
            end

            matrix = TimeResolvedMatrix(time, wavelength, data)
            cal = detect_chirp(matrix; smooth_window=10, order=1, bin_width=4)
            @test cal isa ChirpCalibration
            @test cal.metadata[:smooth_window] == 11
        end

        @testset "detect_chirp recovers known linear chirp" begin
            n_time = 200
            n_wl = 80
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            ref_λ = 600.0
            slope = 0.01
            chirp_fn(λ) = slope * (λ - ref_λ)

            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                t_onset = chirp_fn(wavelength[j])
                for i in eachindex(time)
                    if time[i] > t_onset
                        data[i, j] = 0.5 * exp(-(time[i] - t_onset) / 3.0)
                    end
                end
            end

            matrix = TimeResolvedMatrix(time, wavelength, data)
            cal = detect_chirp(matrix; order=1, reference=ref_λ, smooth_window=7, bin_width=4)

            poly = polynomial(cal)
            @test abs(poly(ref_λ)) < 0.5
            @test abs(poly(ref_λ + 50) - slope * 50) < 1.0
            @test abs(poly(ref_λ - 50) + slope * 50) < 1.0
        end

        @testset "correct_chirp tighter tolerance" begin
            n_time = 200
            n_wl = 50
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            ref_λ = 600.0
            slope = 0.02
            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                t_onset = slope * (wavelength[j] - ref_λ)
                for i in eachindex(time)
                    if time[i] > t_onset
                        data[i, j] = exp(-(time[i] - t_onset) / 2.0)
                    end
                end
            end

            matrix = TimeResolvedMatrix(time, wavelength, data)

            # Exact calibration — no detection noise
            cal = ChirpCalibration(
                collect(wavelength),
                [slope * (λ - ref_λ) for λ in wavelength],
                [-slope * ref_λ, slope],
                1, ref_λ, 1.0, Dict{Symbol,Any}()
            )

            corrected = correct_chirp(matrix, cal)
            inner = n_wl ÷ 4 : 3 * n_wl ÷ 4
            peak_times = [time[argmax(corrected.data[:, j])] for j in inner]
            @test std(peak_times) < 0.3
        end

        @testset "detect-then-correct pipeline" begin
            n_time = 200
            n_wl = 80
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            ref_λ = 600.0
            chirp_fn(λ) = 0.0001 * (λ - ref_λ)^2

            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                t_onset = chirp_fn(wavelength[j])
                for i in eachindex(time)
                    if time[i] > t_onset
                        data[i, j] = exp(-(time[i] - t_onset) / 3.0)
                    end
                end
            end

            matrix = TimeResolvedMatrix(time, wavelength, data)
            cal = detect_chirp(matrix; order=2, reference=ref_λ, bin_width=4)
            corrected = correct_chirp(matrix, cal)

            @test corrected isa TimeResolvedMatrix
            @test corrected.metadata[:chirp_corrected] == true

            # After correction, onset times should be more aligned
            inner_wl = 20:60
            peak_times_before = [time[argmax(abs.(data[:, j]))] for j in inner_wl]
            peak_times_after = [time[argmax(abs.(corrected.data[:, j]))] for j in inner_wl]
            @test std(peak_times_after) < std(peak_times_before)
        end

        @testset "ChirpCalibration show methods" begin
            cal = ChirpCalibration(
                [500.0, 600.0, 700.0], [-1.0, 0.0, 1.5],
                [0.1, -0.002, 0.00001], 2, 600.0, 0.998,
                Dict{Symbol,Any}(:order => 2)
            )

            # Compact show
            io = IOBuffer()
            show(io, cal)
            compact = String(take!(io))
            @test occursin("ChirpCalibration", compact)
            @test occursin("order 2", compact)
            @test occursin("3 points", compact)

            # MIME show
            io = IOBuffer()
            show(io, MIME("text/plain"), cal)
            full = String(take!(io))
            @test occursin("Polynomial order", full)
            @test occursin("R²", full)
            @test occursin("Coefficients", full)
        end

        @testset "load_chirp with malformed JSON" begin
            tmpfile = tempname() * ".json"

            # Missing required keys
            open(tmpfile, "w") do io
                JSON.print(io, Dict("wavelength" => [1.0], "poly_coeffs" => [0.1]))
            end
            @test_throws ArgumentError load_chirp(tmpfile)
            rm(tmpfile)
        end

        @testset "metadata key rename :mad_threshold" begin
            n_time = 200
            n_wl = 80
            time = collect(range(-5.0, 15.0, length=n_time))
            wavelength = collect(range(500.0, 700.0, length=n_wl))

            data = zeros(n_time, n_wl)
            for j in eachindex(wavelength)
                for i in eachindex(time)
                    if time[i] > 0
                        data[i, j] = 0.5 * exp(-time[i] / 3.0)
                    end
                end
            end

            matrix = TimeResolvedMatrix(time, wavelength, data)
            cal = detect_chirp(matrix; order=1, bin_width=4)

            @test haskey(cal.metadata, :mad_threshold)
            @test !haskey(cal.metadata, :threshold)
        end

    end

    @testset "SVD filtering" begin
        time = collect(range(-1.0, 10.0, length=50))
        wavelength = collect(range(400.0, 700.0, length=30))

        # Build a rank-2 signal: two spectral components with different kinetics
        signal = zeros(50, 30)
        for j in eachindex(wavelength)
            for i in eachindex(time)
                t = time[i]
                if t > 0
                    # Component 1: fast decay, peaks at 500 nm
                    signal[i, j] += 0.5 * exp(-t / 1.0) * exp(-((wavelength[j] - 500) / 30)^2)
                    # Component 2: slow decay, peaks at 600 nm
                    signal[i, j] += 0.3 * exp(-t / 5.0) * exp(-((wavelength[j] - 600) / 40)^2)
                end
            end
        end

        noise = 0.02 * randn(50, 30)
        noisy_data = signal .+ noise

        @testset "singular_values - TimeResolvedMatrix" begin
            matrix = TimeResolvedMatrix(time, wavelength, noisy_data)
            sv = singular_values(matrix)
            @test length(sv) == min(50, 30)
            @test issorted(sv, rev=true)
            @test sv[1] > sv[end]
        end

        @testset "singular_values - raw matrix" begin
            sv = singular_values(noisy_data)
            @test length(sv) == min(50, 30)
            @test issorted(sv, rev=true)
        end

        @testset "svd_filter - TimeResolvedMatrix denoising" begin
            matrix = TimeResolvedMatrix(time, wavelength, noisy_data)
            filtered = svd_filter(matrix; n_components=2)

            @test filtered isa TimeResolvedMatrix
            @test size(filtered.data) == size(matrix.data)
            @test filtered.time == matrix.time
            @test filtered.wavelength == matrix.wavelength
            @test filtered.metadata[:svd_filtered] == true
            @test filtered.metadata[:svd_n_components] == 2

            # Filtered should be closer to true signal than noisy data
            error_noisy = sum((noisy_data .- signal).^2)
            error_filtered = sum((filtered.data .- signal).^2)
            @test error_filtered < error_noisy
        end

        @testset "svd_filter - raw matrix" begin
            filtered = svd_filter(time, wavelength, noisy_data; n_components=2)
            @test size(filtered) == size(noisy_data)

            error_noisy = sum((noisy_data .- signal).^2)
            error_filtered = sum((filtered .- signal).^2)
            @test error_filtered < error_noisy
        end

        @testset "svd_filter - n_components=1 keeps dominant component" begin
            matrix = TimeResolvedMatrix(time, wavelength, noisy_data)
            filtered = svd_filter(matrix; n_components=1)

            # Rank-1 approximation
            @test rank(filtered.data) == 1
        end

        @testset "svd_filter - full rank preserves data" begin
            matrix = TimeResolvedMatrix(time, wavelength, noisy_data)
            max_comp = min(size(noisy_data)...)
            filtered = svd_filter(matrix; n_components=max_comp)

            @test filtered.data ≈ noisy_data atol=1e-10
        end

        @testset "svd_filter - input validation" begin
            matrix = TimeResolvedMatrix(time, wavelength, noisy_data)

            @test_throws ArgumentError svd_filter(matrix; n_components=0)
            @test_throws ArgumentError svd_filter(matrix; n_components=100)
            @test_throws DimensionMismatch svd_filter(time, wavelength[1:5], noisy_data; n_components=2)
        end

        @testset "estimate_n_components - elbow heuristic" begin
            # Clear gap after the third component
            @test estimate_n_components([100.0, 50.0, 20.0, 1.0, 0.5]) == 3
            # No gap exceeding the ratio → all components
            @test estimate_n_components([10.0, 9.0, 8.0, 7.0]) == 4
            # Custom ratio: a 2× drop now counts
            @test estimate_n_components([10.0, 4.0, 3.0]; ratio=2.0) == 1
            # Zero next value is not treated as a gap (no divide-by-zero) → all components
            @test estimate_n_components([5.0, 0.0]) == 2
            # Single value
            @test estimate_n_components([3.0]) == 1
            # Integrates with singular_values
            n = estimate_n_components(singular_values(noisy_data))
            @test 1 <= n <= length(singular_values(noisy_data))
        end
    end

    @testset "PLMap type and analysis" begin
        # Build synthetic PLMap: 5×5 grid, 20 pixels per spectrum
        nx, ny, np = 5, 5, 20
        spectra = rand(nx, ny, np)
        # Add a Gaussian peak at pixel 10 for center points
        for ix in 2:4, iy in 2:4
            for k in 1:np
                spectra[ix, iy, k] += 5.0 * exp(-((k - 10)^2) / 4)
            end
        end
        pixel = collect(1.0:np)
        x = collect(range(-2.0, 2.0, length=nx))
        y = collect(range(-2.0, 2.0, length=ny))
        int_matrix = dropdims(sum(spectra; dims=3); dims=3)
        meta = Dict{String,Any}("source_file" => "synthetic.lvm",
                                 "nx" => nx, "ny" => ny, "pixel_range" => nothing)
        m = PLMap(int_matrix, spectra, x, y, pixel, meta)

        # Type and interface
        @test m isa AbstractSpectroscopyData
        @test is_matrix(m) == true
        @test xdata(m) === m.x
        @test ydata(m) === m.y
        @test zdata(m) === m.intensity
        @test intensity(m) === m.intensity
        @test npoints(m) == (5, 5)
        @test xlabel(m) == "X (μm)"
        @test ylabel(m) == "Y (μm)"
        @test zlabel(m) == "PL Intensity"
        @test source_file(m) == "synthetic.lvm"

        # extract_spectrum by index
        spec = extract_spectrum(m, 3, 3)
        @test length(spec.signal) == np
        @test spec.x ≈ x[3]

        # extract_spectrum bounds check
        @test_throws ErrorException extract_spectrum(m, 0, 1)
        @test_throws ErrorException extract_spectrum(m, 1, 6)

        # extract_spectrum by position
        spec2 = extract_spectrum(m; x=0.0, y=0.0)
        @test haskey(spec2, :ix)
        @test haskey(spec2, :iy)

        # normalize_intensity
        mn = normalize_intensity(m)
        @test mn isa PLMap
        @test minimum(mn.intensity) ≈ 0.0
        @test maximum(mn.intensity) ≈ 1.0
        @test mn.spectra === m.spectra  # normalize leaves spectra untouched

        # subtract_background (auto)
        mb = subtract_background(m)
        @test mb isa PLMap
        @test size(mb.spectra) == size(m.spectra)
        @test size(mb.intensity) == size(m.intensity)
        @test mb.x === m.x && mb.y === m.y && mb.pixel === m.pixel
        @test mb.spectra != m.spectra

        # subtract_background with explicit margin
        mb_margin = subtract_background(m; margin=1)
        @test mb_margin isa PLMap
        @test mb_margin.spectra != mb.spectra  # different margin → different result

        # subtract_background (explicit positions) — bg positions see reduced signal
        bg_positions = [(x[1], y[1]), (x[end], y[1]), (x[1], y[end])]
        mb2 = subtract_background(m; positions=bg_positions)
        @test mb2 isa PLMap
        bg_before = extract_spectrum(m; x=bg_positions[1][1], y=bg_positions[1][2])
        bg_after = extract_spectrum(mb2; x=bg_positions[1][1], y=bg_positions[1][2])
        @test abs(sum(bg_after.signal)) < abs(sum(bg_before.signal))

        # normalize chains cleanly with subtract_background
        mn_bg = normalize_intensity(mb)
        @test minimum(mn_bg.intensity) ≈ 0.0
        @test maximum(mn_bg.intensity) ≈ 1.0

        # peak_centers
        centers = peak_centers(m)
        @test size(centers) == (nx, ny)
        valid = filter(!isnan, centers)
        @test !isempty(valid)

        # peak_centers with threshold=0 (no masking)
        centers0 = peak_centers(m; threshold=0)
        @test count(isnan, centers0) <= count(isnan, centers)

        # peak_centers: more cells mask to NaN after bg subtraction (corners drop near 0)
        centers_bg = peak_centers(mb)
        @test count(isnan, centers_bg) >= count(isnan, centers)

        # peak_centers with explicit pixel_range kwarg — centroids stay in range
        centers_range = peak_centers(m; pixel_range=(5, 15))
        valid_range = filter(!isnan, centers_range)
        @test all(c -> 5 <= c <= 15, valid_range)

        # show methods
        buf = IOBuffer()
        show(buf, m)
        @test occursin("5×5", String(take!(buf)))

        buf2 = IOBuffer()
        show(buf2, MIME("text/plain"), m)
        @test occursin("PLMap", String(take!(buf2)))
    end

    @testset "fit_map FWHM conversion per model" begin
        # Each lineshape parameterizes width differently; the FWHM summary
        # array must convert correctly for the model actually used.
        # Reference FWHM is measured numerically from the noiseless profile.
        nx, ny, np = 3, 3, 200
        pixel = collect(1.0:np)
        xs = collect(1.0:nx)
        ys = collect(1.0:ny)

        function make_map(profile::Vector{Float64})
            spectra = zeros(nx, ny, np)
            for ix in 1:nx, iy in 1:ny
                spectra[ix, iy, :] .= profile .+ 0.05
            end
            intens = dropdims(sum(spectra; dims=3); dims=3)
            return PLMap(intens, spectra, xs, ys, pixel)
        end

        # pseudo_voigt: σ is the HWHM of BOTH components -> FWHM = 2σ exactly
        # (both components share the same FWHM, so their mix does too).
        σ_pv = 10.0
        m_pv = make_map(pseudo_voigt([100.0, 100.0, σ_pv, 0.5], pixel))
        res_pv = fit_map(m_pv; model=pseudo_voigt)
        @test res_pv.n_converged == nx * ny
        @test all(isapprox.(res_pv.centers[:, :, 1], 100.0; atol=0.1))
        @test all(isapprox.(res_pv.fwhms[:, :, 1], 2 * σ_pv; rtol=0.02))

        # gaussian: σ is the standard deviation -> FWHM = 2√(2ln2)·σ
        σ_g = 4.0
        m_g = make_map(gaussian([5.0, 100.0, σ_g], pixel))
        res_g = fit_map(m_g; model=gaussian)
        @test res_g.n_converged == nx * ny
        @test all(isapprox.(res_g.fwhms[:, :, 1], 2 * sqrt(2 * log(2)) * σ_g; rtol=0.02))

        # lorentzian: Γ is already the FWHM
        Γ = 12.0
        m_l = make_map(lorentzian([5.0, 100.0, Γ], pixel))
        res_l = fit_map(m_l; model=lorentzian)
        @test res_l.n_converged == nx * ny
        @test all(isapprox.(res_l.fwhms[:, :, 1], Γ; rtol=0.02))

        # voigt: σ Gaussian std dev + γ Lorentzian HWHM -> Thompson approximation.
        # Compare against the numerically measured FWHM of the input profile.
        profile_v = voigt([5.0, 100.0, 4.0, 3.0], pixel)
        fwhm_v_true = calc_fwhm(pixel, profile_v; smooth_window=0).fwhm
        m_v = make_map(profile_v)
        res_v = fit_map(m_v; model=voigt)
        @test res_v.n_converged == nx * ny
        @test all(isapprox.(res_v.fwhms[:, :, 1], fwhm_v_true; rtol=0.02))
    end

    @testset "PLMap pixel_range metadata propagation" begin
        # Build a PLMap where metadata records a specific pixel_range — this mimics
        # what a file loader does when called with `pixel_range=(p1, p2)`. Operations
        # like subtract_background should honor this range when recomputing intensity.
        nx, ny, np = 5, 5, 30
        spectra = rand(nx, ny, np)
        # Put a Gaussian peak around pixel 15, centered spatially
        for ix in 2:4, iy in 2:4, k in 1:np
            spectra[ix, iy, k] += 5.0 * exp(-((k - 15)^2) / 4)
        end
        pixel = collect(1.0:np)
        x = collect(range(-2.0, 2.0, length=nx))
        y = collect(range(-2.0, 2.0, length=ny))

        # Intensity integrated over a restricted pixel range
        p1, p2 = 10, 20
        int_matrix = dropdims(sum(spectra[:, :, p1:p2]; dims=3); dims=3)
        meta = Dict{String,Any}("source_file" => "synthetic.lvm",
                                 "nx" => nx, "ny" => ny,
                                 "pixel_range" => (p1, p2))
        m = PLMap(int_matrix, spectra, x, y, pixel, meta)

        # subtract_background should recompute intensity using the same pixel_range
        mb = subtract_background(m)
        expected = dropdims(sum(mb.spectra[:, :, p1:p2]; dims=3); dims=3)
        @test mb.intensity ≈ expected
    end

    @testset "Cosmic ray detection" begin

        @testset "1D detection — synthetic spikes" begin
            # Smooth Gaussian signal with 3 injected spikes
            x = collect(1.0:100.0)
            signal = @. 5.0 * exp(-((x - 50.0) / 15.0)^2) + 0.1

            spike_positions = [20, 55, 80]
            spiked = copy(signal)
            for pos in spike_positions
                spiked[pos] += 10.0 * maximum(signal)
            end

            result = detect_cosmic_rays(spiked; threshold=5.0)
            @test result isa CosmicRayResult
            @test result.count == 3
            @test sort(result.indices) == spike_positions
        end

        @testset "1D detection — captures spike shoulders" begin
            # Spike with CCD charge spread: peak plus elevated shoulders
            # Noise is needed for a realistic MAD estimate (flat signal → MAD ≈ 0)
            signal = fill(100.0, 100) .+ 0.5 .* randn(100)
            signal[48] += 20.0   # left shoulder
            signal[49] += 100.0  # left flank
            signal[50] += 400.0  # peak
            signal[51] += 200.0  # right flank
            signal[52] += 15.0   # right shoulder

            result = detect_cosmic_rays(signal; threshold=5.0)
            # Should catch the peak and shoulders, not just the peak
            @test result.count >= 4
            @test 50 in result.indices  # peak
            @test 49 in result.indices  # left flank
            @test 51 in result.indices  # right flank
            @test 48 in result.indices  # left shoulder
        end

        @testset "1D detection — clean signal" begin
            # Broad, smooth Gaussian — no features sharp enough to trigger detection
            signal = [5.0 * exp(-((x - 100.0) / 40.0)^2) + 0.1 for x in 1.0:200.0]
            result = detect_cosmic_rays(signal; threshold=5.0)
            @test result.count == 0
            @test isempty(result.indices)
        end

        @testset "1D detection — edge cases" begin
            # Short signal (length < 3)
            @test detect_cosmic_rays([1.0, 2.0]).count == 0

            # All-zero signal (MAD ≈ 0)
            @test detect_cosmic_rays(zeros(50)).count == 0

            # Constant signal
            @test detect_cosmic_rays(fill(42.0, 50)).count == 0
        end

        @testset "1D removal — interpolation" begin
            x = collect(1.0:100.0)
            original = @. 5.0 * exp(-((x - 50.0) / 15.0)^2) + 0.1

            spiked = copy(original)
            spiked[30] += 50.0
            spiked[70] += 50.0

            result = detect_cosmic_rays(spiked; threshold=5.0)
            cleaned = remove_cosmic_rays(spiked, result)

            @test length(cleaned) == length(original)
            # Cleaned values should be close to original at spike positions
            @test abs(cleaned[30] - original[30]) < 1.0
            @test abs(cleaned[70] - original[70]) < 1.0
            # Non-spike positions should be unchanged
            @test cleaned[50] ≈ spiked[50]
        end

        @testset "1D removal — no spikes" begin
            signal = collect(1.0:10.0)
            result = CosmicRayResult(Int[], 0)
            cleaned = remove_cosmic_rays(signal, result)
            @test cleaned ≈ signal
        end

        @testset "PLMap detection — isolated spikes detected" begin
            nx, ny, np = 5, 5, 100
            spectra = zeros(nx, ny, np)
            for ix in 1:nx, iy in 1:ny, k in 1:np
                spectra[ix, iy, k] = 2.0 * exp(-((k - 50)^2) / 200.0) + 0.5 + 0.1 * randn()
            end

            # Inject isolated spike at (3, 3, 25)
            spectra[3, 3, 25] += 100.0

            pixel = collect(1.0:np)
            x = collect(1.0:nx)
            y = collect(1.0:ny)
            int_matrix = dropdims(sum(spectra; dims=3); dims=3)
            meta = Dict{String,Any}("source_file" => "test", "pixel_range" => nothing)
            m = PLMap(int_matrix, spectra, x, y, pixel, meta)

            cr = detect_cosmic_rays(m; threshold=5.0)
            @test cr isa CosmicRayMapResult
            @test cr.mask[3, 3, 25] == true
            @test cr.count >= 1
            @test cr.affected_spectra >= 1
        end

        @testset "PLMap detection — shared features not flagged" begin
            nx, ny, np = 5, 5, 100
            spectra = zeros(nx, ny, np)
            for ix in 1:nx, iy in 1:ny, k in 1:np
                spectra[ix, iy, k] = 2.0 * exp(-((k - 50)^2) / 200.0) + 0.5 + 0.1 * randn()
            end

            # Inject a sharp feature at channel 60 across a 3×3 block
            # All 4 neighbors of (3,3) share the feature → residual ≈ 0 → not flagged
            for ix in 2:4, iy in 2:4
                spectra[ix, iy, 60] += 50.0
            end

            pixel = collect(1.0:np)
            x = collect(1.0:nx)
            y = collect(1.0:ny)
            int_matrix = dropdims(sum(spectra; dims=3); dims=3)
            meta = Dict{String,Any}("source_file" => "test", "pixel_range" => nothing)
            m = PLMap(int_matrix, spectra, x, y, pixel, meta)

            cr = detect_cosmic_rays(m; threshold=5.0)
            # All 4 neighbors of (3,3) have the feature → median reference ≈ feature value
            # → residual at channel 60 ≈ 0 → not flagged
            @test cr.mask[3, 3, 60] == false
        end

        @testset "PLMap detection — wide spike with shoulders" begin
            nx, ny, np = 5, 5, 100
            spectra = zeros(nx, ny, np)
            for ix in 1:nx, iy in 1:ny, k in 1:np
                spectra[ix, iy, k] = 2.0 * exp(-((k - 50)^2) / 200.0) + 0.5 + 0.1 * randn()
            end

            # Inject a wide spike with shoulders at (3, 3)
            spectra[3, 3, 48] += 15.0   # left shoulder
            spectra[3, 3, 49] += 40.0   # left flank
            spectra[3, 3, 50] += 100.0  # peak
            spectra[3, 3, 51] += 60.0   # right flank
            spectra[3, 3, 52] += 10.0   # right shoulder

            pixel = collect(1.0:np)
            x = collect(1.0:nx)
            y = collect(1.0:ny)
            int_matrix = dropdims(sum(spectra; dims=3); dims=3)
            meta = Dict{String,Any}("source_file" => "test", "pixel_range" => nothing)
            m = PLMap(int_matrix, spectra, x, y, pixel, meta)

            cr = detect_cosmic_rays(m; threshold=5.0)
            # All channels of the wide spike should be caught
            @test cr.mask[3, 3, 50] == true  # peak
            @test cr.mask[3, 3, 49] == true  # left flank
            @test cr.mask[3, 3, 51] == true  # right flank
            @test cr.mask[3, 3, 48] == true  # left shoulder
            @test cr.mask[3, 3, 52] == true  # right shoulder
        end

        @testset "PLMap detection — edge pixel" begin
            nx, ny, np = 5, 5, 100
            spectra = zeros(nx, ny, np)
            for ix in 1:nx, iy in 1:ny, k in 1:np
                spectra[ix, iy, k] = 2.0 * exp(-((k - 50)^2) / 200.0) + 0.5 + 0.1 * randn()
            end

            # Inject spike at corner pixel (1, 1) — only 2 neighbors
            spectra[1, 1, 30] += 100.0

            pixel = collect(1.0:np)
            x = collect(1.0:nx)
            y = collect(1.0:ny)
            int_matrix = dropdims(sum(spectra; dims=3); dims=3)
            meta = Dict{String,Any}("source_file" => "test", "pixel_range" => nothing)
            m = PLMap(int_matrix, spectra, x, y, pixel, meta)

            cr = detect_cosmic_rays(m; threshold=5.0)
            @test cr.mask[1, 1, 30] == true
            @test cr.count >= 1
        end

        @testset "PLMap removal — cleaned spectra match originals" begin
            nx, ny, np = 5, 5, 100
            spectra = zeros(nx, ny, np)
            for ix in 1:nx, iy in 1:ny, k in 1:np
                spectra[ix, iy, k] = 2.0 * exp(-((k - 50)^2) / 200.0) + 0.5 + 0.1 * randn()
            end
            original_spectra = copy(spectra)

            # Inject isolated spike
            spectra[2, 2, 30] += 100.0

            pixel = collect(1.0:np)
            x = collect(1.0:nx)
            y = collect(1.0:ny)
            int_matrix = dropdims(sum(spectra; dims=3); dims=3)
            meta = Dict{String,Any}("source_file" => "test", "pixel_range" => nothing)
            m = PLMap(int_matrix, spectra, x, y, pixel, meta)

            cr = detect_cosmic_rays(m; threshold=5.0)
            cleaned = remove_cosmic_rays(m, cr)

            @test cleaned isa PLMap
            # Cleaned value at spike should be close to original (within noise + interpolation)
            @test abs(cleaned.spectra[2, 2, 30] - original_spectra[2, 2, 30]) < 2.0
            # Non-spike values should be unchanged
            @test cleaned.spectra[1, 1, 50] ≈ spectra[1, 1, 50]
        end

    end

    @testset "Spectral Math" begin

        @testset "savitzky_golay_smooth" begin
            # Smoothing reduces noise on a sine wave
            x_sg = collect(0:0.1:2π)
            y_clean = sin.(x_sg)
            y_noisy = y_clean .+ 0.1 * randn(length(x_sg))
            y_smooth = savitzky_golay_smooth(y_noisy; window=11, order=3)
            @test length(y_smooth) == length(y_noisy)
            # Smoothed MSE should be lower than noisy MSE
            mse_noisy = sum((y_noisy .- y_clean).^2) / length(y_clean)
            mse_smooth = sum((y_smooth .- y_clean).^2) / length(y_clean)
            @test mse_smooth < mse_noisy
            # Default parameters work
            y_default = savitzky_golay_smooth(y_noisy)
            @test length(y_default) == length(y_noisy)
        end

        @testset "derivative" begin
            # 1st derivative of Gaussian: zero-crossing near peak center
            x_d = collect(400.0:0.5:800.0)
            center = 520.0
            sigma = 20.0
            y_gauss = @. 100 * exp(-(x_d - center)^2 / (2 * sigma^2))

            # y-only derivative (unit spacing)
            dy_unit = derivative(y_gauss; order=1, window=11, poly_order=3)
            @test length(dy_unit) == length(y_gauss)

            # x,y derivative (correctly scaled)
            dy = derivative(x_d, y_gauss; order=1, window=11, poly_order=3)
            @test length(dy) == length(y_gauss)

            # Zero-crossing should be near the peak center
            # Find where derivative crosses zero (sign change)
            center_idx = argmin(abs.(x_d .- center))
            # derivative should be positive before peak, negative after
            @test dy[center_idx - 20] > 0  # before peak
            @test dy[center_idx + 20] < 0  # after peak
            # Near zero at peak
            @test abs(dy[center_idx]) < maximum(abs.(dy)) * 0.1

            # 2nd derivative: negative at peak center (concave down)
            d2y = derivative(x_d, y_gauss; order=2, window=11, poly_order=3)
            @test d2y[center_idx] < 0
        end

        @testset "band_area" begin
            # Gaussian with known analytical area
            x_ba = collect(400.0:0.5:800.0)
            A = 100.0
            sigma = 10.0
            center_ba = 520.0
            y_ba = @. A * exp(-(x_ba - center_ba)^2 / (2 * sigma^2))

            # Analytical area of full Gaussian = A * σ * √(2π)
            expected_area = A * sigma * sqrt(2π)
            computed_area = band_area(x_ba, y_ba, 400.0, 800.0)
            @test abs(computed_area - expected_area) / expected_area < 0.01  # Within 1%

            # Partial range
            partial_area = band_area(x_ba, y_ba, 490.0, 550.0)
            @test partial_area > 0
            @test partial_area < computed_area

            # Swapped bounds should also work (minmax)
            swapped = band_area(x_ba, y_ba, 800.0, 400.0)
            @test abs(swapped - computed_area) < 1e-10

            # Edge case: fewer than 2 points
            @test_throws ArgumentError band_area(x_ba, y_ba, 399.0, 399.5)
        end

        @testset "normalize_area" begin
            # Uniform signal: area of y=1 on [1,10] is 9
            x_na = collect(1.0:0.1:10.0)
            y_na = ones(length(x_na))
            y_norm = normalize_area(x_na, y_na)
            @test length(y_norm) == length(y_na)
            # Total area after normalization should be ≈ 1.0
            total_after = band_area(x_na, y_norm, x_na[1], x_na[end])
            @test abs(total_after - 1.0) < 1e-10

            # Ratios preserved
            x_peak = collect(400.0:1.0:800.0)
            y_peak = @. 50 * exp(-(x_peak - 520)^2 / (2 * 10^2)) + 10 * exp(-(x_peak - 620)^2 / (2 * 8^2))
            y_pn = normalize_area(x_peak, y_peak)
            # Check ratio of two points is preserved
            i1, i2 = 50, 150
            @test abs(y_pn[i1] / y_pn[i2] - y_peak[i1] / y_peak[i2]) < 1e-10
        end

        @testset "normalize_to_peak" begin
            # Peak at x=520 with amplitude 50
            x_np = collect(400.0:1.0:800.0)
            y_np = @. 50 * exp(-(x_np - 520)^2 / (2 * 10^2))
            y_norm = normalize_to_peak(x_np, y_np, 520.0)

            # Value at peak position should be 1.0
            peak_idx = argmin(abs.(x_np .- 520.0))
            @test abs(y_norm[peak_idx] - 1.0) < 1e-10

            # Ratios preserved
            @test abs(y_norm[1] / y_norm[peak_idx] - y_np[1] / y_np[peak_idx]) < 1e-10

            # Within tolerance
            y_tol = normalize_to_peak(x_np, y_np, 520.3; tolerance=1.0)
            @test abs(y_tol[peak_idx] - 1.0) < 1e-10

            # Outside tolerance throws error
            @test_throws ArgumentError normalize_to_peak(x_np, y_np, 999.0; tolerance=5.0)

            # Zero intensity at peak throws error
            y_zero = zeros(length(x_np))
            @test_throws ArgumentError normalize_to_peak(x_np, y_zero, 520.0)
        end

        @testset "estimate_snr" begin
            # High SNR: clean signal + small noise
            using Random
            Random.seed!(42)
            y_signal = 100.0 * ones(200)
            y_noisy_snr = y_signal .+ 2.0 * randn(200)
            result = estimate_snr(y_noisy_snr)
            @test result isa NamedTuple
            @test haskey(result, :snr)
            @test haskey(result, :signal)
            @test haskey(result, :noise)
            @test result.snr > 10  # SNR should be high
            @test result.signal > 0
            @test result.noise > 0

            # Very noisy signal should have lower SNR
            y_very_noisy = y_signal .+ 50.0 * randn(200)
            result_noisy = estimate_snr(y_very_noisy)
            @test result_noisy.snr < result.snr

            # Edge case: too few points
            @test_throws ArgumentError estimate_snr([1.0, 2.0, 3.0])

            # Exactly 4 points should work
            result_4 = estimate_snr([1.0, 2.0, 3.0, 4.0])
            @test result_4 isa NamedTuple
        end

    end  # Spectral Math

    @testset "Spectral Arithmetic" begin
        x = collect(1.0:0.5:10.0)
        y1 = @. sin(x)
        y2 = @. cos(x)
        a = (x=x, y=y1)
        b = (x=x, y=y2)

        @testset "add_spectra" begin
            result = add_spectra(a, b)
            @test result.x == x
            @test result.y ≈ y1 .+ y2
        end

        @testset "divide_spectra" begin
            c = (x=x, y=ones(length(x)) .* 2.0)
            result = divide_spectra(a, c)
            @test result.y ≈ y1 ./ 2.0
        end

        @testset "multiply_spectrum" begin
            result = multiply_spectrum(a, 3.0)
            @test result.y ≈ y1 .* 3.0
        end

        @testset "average_spectra" begin
            result = average_spectra(a, b)
            @test result.y ≈ (y1 .+ y2) ./ 2
        end

        @testset "interpolate_spectrum" begin
            new_x = [2.0, 5.0, 8.0]
            result = interpolate_spectrum(x, y1, new_x)
            @test length(result) == 3
            # At grid points, should be exact
            @test result[1] ≈ sin(2.0) atol=0.01
        end

        @testset "arithmetic with interpolation" begin
            x2 = collect(1.0:0.7:10.0)  # Different grid
            y2_alt = @. cos(x2)
            b_alt = (x=x2, y=y2_alt)
            result = add_spectra(a, b_alt; interpolate=true)
            @test length(result.y) == length(x)
        end

        @testset "grid mismatch error" begin
            x_short = collect(1.0:0.5:5.0)
            b_short = (x=x_short, y=ones(length(x_short)))
            @test_throws ErrorException add_spectra(a, b_short)
        end
    end

    @testset "Transforms" begin
        @testset "kubelka_munk" begin
            @test kubelka_munk(1.0) == 0.0
            @test kubelka_munk(0.5) == 0.25
            @test_throws ArgumentError kubelka_munk(0.0)
            @test_throws ArgumentError kubelka_munk(-0.1)
        end

        @testset "reflectance_to_absorbance" begin
            @test reflectance_to_absorbance(0.1) ≈ 1.0
            @test reflectance_to_absorbance(1.0) ≈ 0.0
            R = [0.1, 0.5, 1.0]
            A = reflectance_to_absorbance(R)
            @test length(A) == 3
        end

        @testset "snv" begin
            y = [1.0, 2.0, 3.0, 4.0, 5.0]
            result = snv(y)
            @test mean(result) ≈ 0.0 atol=1e-10
            @test std(result) ≈ 1.0 atol=1e-10
        end

        @testset "beer_lambert" begin
            @test beer_lambert(1.0, 0.1) ≈ 10.0
            @test beer_lambert(1.0, 0.1, 0.01) ≈ 1000.0
        end

        @testset "tauc_plot" begin
            # Create synthetic data with known bandgap
            energy = collect(1.0:0.01:4.0)
            Eg = 2.5
            alpha = @. sqrt(max(0.0, energy - Eg))  # Direct gap: alpha*hv ~ (hv-Eg)^0.5
            result = tauc_plot(energy, alpha; gap_type=:direct)
            @test haskey(result, :bandgap)
            @test haskey(result, :hv)
            @test haskey(result, :tauc_y)
            @test result.bandgap > 0
        end

        @testset "kramers_kronig" begin
            # Lorentzian absorption -> dispersive real part
            omega = collect(0.1:0.1:10.0)
            chi_imag = @. 1.0 / (1.0 + (omega - 5.0)^2)
            result = kramers_kronig(omega, chi_imag; type=:imag_to_real)
            @test length(result) == length(omega)
            # Real part should change sign near the resonance
            # (dispersive lineshape)
            @test any(result .> 0) && any(result .< 0)

            # Quantitative regression against the exact Lorentz-oscillator KK pair.
            # Catches quadrature-weight errors: the Maclaurin rule sums alternate
            # points only, so the weight is 2*dw (Ohta & Ishida 1988), and the
            # real->imag direction carries w_i outside the integral, not w_j inside.
            #   chi'(w)  = wp2 (w0^2 - w^2) / [(w0^2 - w^2)^2 + g^2 w^2]
            #   chi''(w) = wp2 g w / [(w0^2 - w^2)^2 + g^2 w^2]
            w0, g, wp2 = 1000.0, 50.0, 1.0e6
            w = collect(1.0:1.0:8000.0)
            chi_re_exact = @. wp2 * (w0^2 - w^2) / ((w0^2 - w^2)^2 + g^2 * w^2)
            chi_im_exact = @. wp2 * g * w / ((w0^2 - w^2)^2 + g^2 * w^2)

            re_num = kramers_kronig(w, chi_im_exact; type=:imag_to_real)
            for i in (900, 950, 1050, 1100, 1500)
                @test re_num[i] ≈ chi_re_exact[i] rtol=1e-3
            end
            re_peak = maximum(abs.(chi_re_exact))
            @test maximum(abs.(re_num[500:2000] .- chi_re_exact[500:2000])) / re_peak < 1e-3

            im_num = kramers_kronig(w, chi_re_exact; type=:real_to_imag)
            for i in (900, 950, 1000, 1050, 1100)
                @test im_num[i] ≈ chi_im_exact[i] rtol=1e-2
            end
            im_peak = maximum(abs.(chi_im_exact))
            @test maximum(abs.(im_num[500:2000] .- chi_im_exact[500:2000])) / im_peak < 1e-3
        end

        @testset "urbach_tail" begin
            energy = collect(1.0:0.01:3.0)
            Eu = 0.05  # 50 meV
            alpha = @. exp((energy - 2.0) / Eu)
            result = urbach_tail(energy, alpha; fit_range=(1.0, 1.8))
            @test haskey(result, :Eu)
            @test result.Eu > 0
            @test isapprox(result.Eu, Eu, rtol=0.1)
        end

        @testset "thickness_from_fringes" begin
            # Create synthetic fringe pattern
            wn = collect(1000.0:1.0:4000.0)
            n_ref = 1.5
            d_true = 0.001  # 10 micron in cm
            y = @. cos(2π * 2 * n_ref * d_true * wn)
            result = thickness_from_fringes(wn, y; n=n_ref)
            @test haskey(result, :thickness)
            @test result.thickness > 0
            @test isapprox(result.thickness, d_true, rtol=0.05)
        end
    end

    @testset "Rubber band baseline" begin
        x = collect(1.0:100.0)
        # Create a spectrum with peaks on top of a linear baseline
        y = @. 0.01 * x + sin(x / 10) * 5.0
        bl = rubberband_baseline(x, y)
        @test length(bl) == length(x)
        # Baseline should be <= spectrum everywhere (lower envelope)
        @test all(bl .<= y .+ 1e-10)
        # Baseline should touch the spectrum at its minima
        min_idx = argmin(y)
        @test isapprox(bl[min_idx], y[min_idx], atol=0.1)
    end

    @testset "correct_baseline rubberband method" begin
        y = @. sin(1:0.1:10) + 2.0
        result = correct_baseline(y; method=:rubberband)
        @test haskey(result, :y)
        @test haskey(result, :baseline)
        @test length(result.y) == length(y)
    end

    @testset "rubber band uses real x-values" begin
        # Non-uniform x with a valley at x=5
        x = [1.0, 2.0, 5.0, 8.0, 10.0]
        y = [1.0, 0.5, 0.2, 0.5, 1.0]
        result = correct_baseline(x, y; method=:rubberband)
        @test length(result.baseline) == 5
        baseline_at_5 = result.baseline[3]
        @test baseline_at_5 ≈ 0.2 atol=0.01
    end

    @testset "imodpoly_baseline" begin
        # Synthetic: polynomial background + Gaussian peak
        x = range(0, 10, length=500)
        bg = @. 0.5 * x^2 - 2x + 10
        peak = @. 5.0 * exp(-((x - 5)^2) / (2 * 0.3^2))
        y = bg + peak

        baseline = imodpoly_baseline(collect(x), collect(y); poly_order=2)
        @test length(baseline) == 500
        @test maximum(abs.(baseline - bg)) < 1.0

        # Flat spectrum
        y_flat = ones(100)
        bl_flat = imodpoly_baseline(collect(range(0, 1, length=100)), y_flat)
        @test all(isapprox.(bl_flat, 1.0, atol=0.01))

        # Via unified API
        result = correct_baseline(collect(x), collect(y); method=:imodpoly, poly_order=2)
        @test haskey(result, :baseline)
        @test haskey(result, :y)
    end

    @testset "rolling_ball_baseline" begin
        # Synthetic: sloped baseline + sharp peaks
        x = collect(1.0:500.0)
        bg = @. 0.01 * x
        peaks = zeros(500)
        peaks[100] = 5.0; peaks[250] = 8.0; peaks[400] = 3.0
        y = bg + peaks

        baseline = rolling_ball_baseline(y; half_window=20)
        @test length(baseline) == 500
        @test baseline[100] < y[100]
        @test baseline[250] < y[250]
        @test abs(baseline[50] - bg[50]) < 1.0

        # Via unified API
        result = correct_baseline(y; method=:rolling_ball, half_window=20)
        @test haskey(result, :baseline)
        @test length(result.y) == 500
    end

    @testset "Internal polynomial utilities" begin
        @testset "_polyeval scalar — Horner's method" begin
            # Linear: 2 + 3x at x=4 -> 14
            @test OpticalSpectroscopy._polyeval([2.0, 3.0], 4.0) ≈ 14.0
            # Quadratic: 1 + 0.5x + 0.01x^2 at x=10 -> 1 + 5 + 1 = 7
            @test OpticalSpectroscopy._polyeval([1.0, 0.5, 0.01], 10.0) ≈ 7.0
            # Constant polynomial
            @test OpticalSpectroscopy._polyeval([42.0], 999.0) ≈ 42.0
            # Zero polynomial
            @test OpticalSpectroscopy._polyeval([0.0, 0.0, 0.0], 5.0) ≈ 0.0
            # Higher order: 1 + x + x^2 + x^3 + x^4 at x=2 -> 1+2+4+8+16 = 31
            @test OpticalSpectroscopy._polyeval([1.0, 1.0, 1.0, 1.0, 1.0], 2.0) ≈ 31.0
        end

        @testset "_polyeval vector — element-wise" begin
            coeffs = [2.0, 3.0]  # 2 + 3x
            xs = [0.0, 1.0, 2.0, 10.0]
            result = OpticalSpectroscopy._polyeval(coeffs, xs)
            @test result ≈ [2.0, 5.0, 8.0, 32.0]
        end

        @testset "_polyfit recovers known coefficients" begin
            # Exact quadratic: y = 2 + x + 0.5x^2
            x = collect(range(0.0, 4.0, length=20))
            y = @. 2.0 + x + 0.5 * x^2
            coeffs = OpticalSpectroscopy._polyfit(x, y, 2)
            @test coeffs ≈ [2.0, 1.0, 0.5] atol=1e-10

            # Exact cubic
            x2 = collect(range(-2.0, 2.0, length=30))
            y2 = @. 1.0 - 0.5 * x2 + 0.1 * x2^2 + 0.05 * x2^3
            coeffs2 = OpticalSpectroscopy._polyfit(x2, y2, 3)
            @test coeffs2 ≈ [1.0, -0.5, 0.1, 0.05] atol=1e-10
        end

        @testset "_polyfit and _polyeval round-trip" begin
            # Fit data, evaluate, should recover original
            x = collect(range(-1.0, 1.0, length=50))
            true_coeffs = [3.0, -1.0, 2.0]
            y = @. true_coeffs[1] + true_coeffs[2] * x + true_coeffs[3] * x^2
            fit_coeffs = OpticalSpectroscopy._polyfit(x, y, 2)
            y_eval = OpticalSpectroscopy._polyeval(fit_coeffs, x)
            @test y_eval ≈ y atol=1e-10
        end
    end

    @testset "fit_map full surface (mask, abort, progress, accessors)" begin
        Random.seed!(7)
        # 6×6 map, Gaussian peak whose center varies linearly across the map.
        nx, ny, np = 6, 6, 200
        pixel = collect(1.0:np)
        xs = collect(range(-5.0, 5.0, length=nx))
        ys = collect(range(-5.0, 5.0, length=ny))
        true_center(ix, iy) = 90.0 + 2.0 * ix + 1.0 * iy
        σ_true = 5.0
        amp_true(ix, iy) = 3.0 + 0.1 * ix
        spectra = zeros(nx, ny, np)
        for ix in 1:nx, iy in 1:ny
            spectra[ix, iy, :] .= gaussian([amp_true(ix, iy), true_center(ix, iy), σ_true], pixel) .+ 0.05
        end
        # Dark column ix=1: kill the signal so threshold masking excludes it
        spectra[1, :, :] .= 0.001
        intens = dropdims(sum(spectra; dims=3); dims=3)
        m = PLMap(intens, spectra, xs, ys, pixel)

        progress_calls = Threads.Atomic{Int}(0)
        progress_total = Threads.Atomic{Int}(0)
        res = fit_map(m; model=gaussian, threshold=0.2,
                      progress=(done, total) -> begin
                          Threads.atomic_add!(progress_calls, 1)
                          Threads.atomic_max!(progress_total, total)
                      end)

        @test res isa FitMapResult
        @test size(res) == (nx, ny)
        @test res.n_peaks == 1
        @test res.n_skipped == ny              # dark column excluded
        @test res.n_converged == (nx - 1) * ny
        @test res.n_failed == 0
        @test res.median_r_squared > 0.999
        @test progress_calls[] >= res.n_converged
        @test progress_total[] == (nx - 1) * ny

        # Quantitative recovery on every fitted pixel
        for ix in 2:nx, iy in 1:ny
            @test res.centers[ix, iy, 1] ≈ true_center(ix, iy) atol=0.2
            @test res.fwhms[ix, iy, 1] ≈ 2 * sqrt(2 * log(2)) * σ_true rtol=0.02
            @test res.amplitudes[ix, iy, 1] ≈ amp_true(ix, iy) rtol=0.05
            @test res.r_squareds[ix, iy] > 0.999
            @test res[ix, iy] isa MultiPeakFitResult
        end
        # Masked pixels: nothing results, NaN summaries
        @test all(isnothing, res.results[1, :])
        @test all(isnan, res.centers[1, :, 1])
        @test all(isnan, res.fwhms[1, :, 1])
        @test res.mask isa BitMatrix
        @test !any(res.mask[1, :])

        # show methods
        buf = IOBuffer()
        show(buf, res)
        @test occursin("FitMapResult", String(take!(buf)))
        show(buf, MIME("text/plain"), res)
        @test occursin("Converged", String(take!(buf)))

        # Pre-computed mask is honored verbatim
        mask = trues(nx, ny)
        mask[:, 1] .= false
        res_mask = fit_map(m; model=gaussian, mask=mask)
        @test res_mask.n_skipped == nx
        @test all(isnothing, res_mask.results[:, 1])

        # Abort flag set up-front: only the reference pixel is fitted
        res_abort = fit_map(m; model=gaussian, abort=Threads.Atomic{Bool}(true))
        @test res_abort.n_converged <= 1
        @test count(!isnothing, res_abort.results) <= 1

        # region kwarg restricts the spectral window but still converges
        res_region = fit_map(m; model=gaussian, threshold=0.2, region=(60, 140))
        @test res_region.n_converged == (nx - 1) * ny
        @test res_region.centers[3, 3, 1] ≈ true_center(3, 3) atol=0.3
    end

    @testset "integrated_intensity and intensity_mask" begin
        nx, ny, np = 5, 4, 30
        pixel = collect(1.0:np)
        xs = collect(0.0:4.0)
        ys = collect(0.0:3.0)
        spectra = zeros(nx, ny, np)
        for ix in 1:nx, iy in 1:ny
            spectra[ix, iy, :] .= Float64(ix)   # flat spectra, value = ix
        end
        intens = dropdims(sum(spectra; dims=3); dims=3)  # = ix * np
        m = PLMap(intens, spectra, xs, ys, pixel)

        # No pixel_range and no metadata: passthrough of m.intensity
        @test integrated_intensity(m) === m.intensity

        # Explicit pixel_range: sum over the window
        ii = integrated_intensity(m; pixel_range=(11, 20))
        @test size(ii) == (nx, ny)
        @test ii[3, 2] ≈ 3.0 * 10
        @test ii[5, 1] ≈ 5.0 * 10

        # metadata fallback
        meta = Dict{String,Any}("pixel_range" => (1, 15))
        m_meta = PLMap(intens, spectra, xs, ys, pixel, meta)
        @test integrated_intensity(m_meta)[2, 2] ≈ 2.0 * 15

        # intensity_mask: intensity ranges ix*np = 30..150, cutoff at 50%
        r = intensity_mask(m; threshold=0.5)
        @test r.intensity_min ≈ 30.0
        @test r.intensity_max ≈ 150.0
        @test r.cutoff ≈ 30.0 + 0.5 * 120.0
        @test r.n_total == nx * ny
        # included: ix in (3,4,5) -> 3 columns × ny
        @test r.n_included == 3 * ny
        @test all(r.mask[3:5, :])
        @test !any(r.mask[1:2, :])

        # exclusion region: knock out the ix=5 column by spatial coordinates
        r_ex = intensity_mask(m; threshold=0.5, exclude=((3.5, 4.5), (-Inf, Inf)))
        @test !any(r_ex.mask[5, :])
        @test r_ex.n_included == 2 * ny

        # vector of regions
        r_ex2 = intensity_mask(m; threshold=0.5,
                               exclude=[((3.5, 4.5), (-Inf, Inf)),
                                        ((1.5, 2.5), (-Inf, Inf))])
        @test r_ex2.n_included == 1 * ny
    end

    @testset "pca_map and nmf_map recover planted components" begin
        Random.seed!(11)
        nx, ny, np = 8, 8, 120
        pixel = collect(1.0:np)
        spec_a = gaussian([1.0, 40.0, 6.0], pixel)    # species A
        spec_b = gaussian([1.0, 85.0, 9.0], pixel)    # species B
        spectra = zeros(nx, ny, np)
        for ix in 1:nx, iy in 1:ny
            w_a = ix <= nx ÷ 2 ? 5.0 + 0.2 * iy : 0.5
            w_b = ix > nx ÷ 2 ? 4.0 + 0.3 * iy : 0.4
            spectra[ix, iy, :] .= w_a .* spec_a .+ w_b .* spec_b .+ 0.005 .* rand(np)
        end
        intens = dropdims(sum(spectra; dims=3); dims=3)
        m = PLMap(intens, spectra, collect(1.0:nx), collect(1.0:ny), pixel)

        # --- PCA ---
        p = pca_map(m; n_components=3)
        @test p isa DecompositionResult
        @test size(p.loadings) == (nx, ny, 3)
        @test size(p.components) == (3, np)
        @test length(p.explained_variance) == 3
        @test all(0 .<= p.explained_variance .<= 1)
        @test issorted(p.explained_variance, rev=true)
        # Two-species data after centering: first PC dominates and is the
        # A-vs-B contrast direction
        @test p.explained_variance[1] > 0.9
        contrast = spec_a .- spec_b
        c1 = p.components[1, :]
        @test abs(cor(c1, contrast)) > 0.95

        # First PC loading map separates the two halves spatially
        pc1 = p.loadings[:, :, 1]
        @test sign(mean(pc1[1:nx÷2, :])) != sign(mean(pc1[nx÷2+1:end, :]))

        # n_components validation
        @test_throws ArgumentError pca_map(m; n_components=0)
        @test_throws ArgumentError pca_map(m; n_components=1000)

        # --- NMF ---
        Random.seed!(13)   # nmf_map uses the global RNG for initialization
        q = nmf_map(m; n_components=2, max_iter=500)
        @test q isa DecompositionResult
        @test size(q.loadings) == (nx, ny, 2)
        @test size(q.components) == (2, np)
        @test all(q.loadings .>= 0)
        @test all(q.components .>= 0)

        # Each planted spectrum is recovered by one NMF component (up to
        # permutation and scale)
        cors_a = [cor(q.components[k, :], spec_a) for k in 1:2]
        cors_b = [cor(q.components[k, :], spec_b) for k in 1:2]
        k_a = argmax(cors_a)
        k_b = argmax(cors_b)
        @test k_a != k_b
        @test cors_a[k_a] > 0.95
        @test cors_b[k_b] > 0.95
        # ... with the matching spatial distribution
        @test mean(q.loadings[1:nx÷2, :, k_a]) > mean(q.loadings[nx÷2+1:end, :, k_a])
        @test mean(q.loadings[nx÷2+1:end, :, k_b]) > mean(q.loadings[1:nx÷2, :, k_b])

        @test_throws ArgumentError nmf_map(m; n_components=0)

        # show methods
        buf = IOBuffer()
        show(buf, p)
        @test occursin("DecompositionResult", String(take!(buf)))
        show(buf, MIME("text/plain"), p)
        @test occursin("Components", String(take!(buf)))
    end

    @testset "find_peaks direct coverage" begin
        x = collect(400.0:0.5:700.0)
        y = gaussian([1.0, 480.0, 8.0], x) .+
            gaussian([0.6, 560.0, 6.0], x) .+
            gaussian([0.15, 640.0, 5.0], x)

        peaks = find_peaks(x, y; min_prominence=0.05)
        @test length(peaks) == 3
        @test issorted([p.position for p in peaks])
        @test peaks[1].position ≈ 480.0 atol=1.0
        @test peaks[2].position ≈ 560.0 atol=1.0
        @test peaks[3].position ≈ 640.0 atol=1.0
        @test peaks[1].intensity ≈ 1.0 atol=0.05
        @test peaks[1].prominence > peaks[2].prominence > peaks[3].prominence
        # FWHP of an isolated Gaussian ≈ FWHM = 2√(2ln2)σ
        @test peaks[1].width ≈ 2 * sqrt(2 * log(2)) * 8.0 rtol=0.15
        @test peaks[1].bounds[1] < 480.0 < peaks[1].bounds[2]
        @test x[peaks[1].index] ≈ peaks[1].position

        # min_prominence filtering drops the small peak
        peaks_strict = find_peaks(x, y; min_prominence=0.3)
        @test length(peaks_strict) == 2
        @test all(p -> p.position < 600, peaks_strict)

        # min_height filter
        peaks_tall = find_peaks(x, y; min_height=0.9)
        @test length(peaks_tall) == 1
        @test peaks_tall[1].position ≈ 480.0 atol=1.0

        # width filters
        peaks_wide = find_peaks(x, y; min_width=15.0)
        @test all(p -> p.width >= 15.0, peaks_wide)
        @test length(peaks_wide) < 3

        # minima mode finds dips
        dips = find_peaks(x, -y; mode=:minima, min_prominence=0.05)
        @test length(dips) == 3
        @test dips[1].position ≈ 480.0 atol=1.0

        # index-based variant (no x)
        peaks_idx = find_peaks(y; min_prominence=0.05)
        @test length(peaks_idx) == 3
        @test peaks_idx[1].position ≈ findfirst(==(maximum(y)), y) atol=2.0

        # degenerate input
        @test find_peaks([1.0, 2.0]) == PeakInfo[]
        @test_throws ArgumentError find_peaks([1.0, 2.0], [1.0])
    end

    @testset "voigt and log_normal model passthroughs" begin
        x = collect(range(50.0, 150.0, length=2001))

        # γ -> 0: voigt reduces exactly to the Gaussian component
        @test voigt([2.0, 100.0, 5.0, 0.0], x) ≈ gaussian([2.0, 100.0, 5.0], x) rtol=1e-8

        # σ -> 0: voigt reduces exactly to a Lorentzian with FWHM = 2γ
        @test voigt([2.0, 100.0, 0.0, 4.0], x) ≈ lorentzian([2.0, 100.0, 8.0], x) rtol=1e-6

        # peak position and amplitude at center
        v = voigt([2.0, 100.0, 3.0, 2.0], x)
        @test x[argmax(v)] ≈ 100.0 atol=0.1
        @test maximum(v) ≈ 2.0 rtol=0.02

        # offset parameter
        v_off = voigt([2.0, 100.0, 3.0, 2.0, 0.5], x)
        @test v_off ≈ v .+ 0.5

        # log_normal: maximum at exp(μ - σ²)
        μ, σ_ln = log(100.0), 0.25
        ln_y = log_normal([1.0, μ, σ_ln], x)
        @test x[argmax(ln_y)] ≈ exp(μ - σ_ln^2) rtol=0.01
        @test all(isfinite, ln_y)
        # offset parameter
        ln_off = log_normal([1.0, μ, σ_ln, 0.2], x)
        @test ln_off ≈ ln_y .+ 0.2
    end

    @testset "confint re-export brackets true parameters" begin
        Random.seed!(21)
        x = collect(range(0.0, 20.0, length=200))
        p_true = [2.0, 5.0, 0.1]   # single_exponential [A, τ, y₀]
        y = single_exponential(p_true, x) .+ 0.01 .* randn(length(x))

        prob = NonlinearCurveFitProblem(single_exponential, [1.5, 3.0, 0.0], x, y)
        sol = solve(prob)
        @test isconverged(sol)

        p_fit = coef(sol)
        ci = confint(sol)
        @test length(ci) == 3
        for i in 1:3
            lo, hi = ci[i]
            @test lo < hi
            @test lo <= p_fit[i] <= hi      # interval centered on the estimate
            @test lo <= p_true[i] <= hi     # and bracketing the truth
        end
        # interval half-width is a fixed multiple of stderror across params.
        # NOTE: CurveFit v1 computes the t-quantile with TDist(dof) where
        # dof = n_params (here 3 -> t ≈ 3.18) rather than dof_residual
        # (197 -> t ≈ 1.97); the loose upper bound below tolerates both,
        # so this keeps passing if upstream fixes margin_error.
        # Tracked upstream: https://github.com/SciML/CurveFit.jl/issues/107
        # Once a fixed CurveFit release is out, tighten to ~1.97:
        #   @test ratios[1] ≈ quantile(TDist(197), 0.975) rtol=1e-3
        se = stderror(sol)
        ratios = [(ci[i][2] - ci[i][1]) / (2 * se[i]) for i in 1:3]
        @test all(r -> isapprox(r, ratios[1]; rtol=1e-6), ratios)
        @test 1.9 < ratios[1] < 3.5
    end

    @testset "Makie extension" begin
        using CairoMakie

        x = collect(1500.0:1.0:2500.0)
        y = gaussian([1.0, 1800.0, 15.0], x) .+ gaussian([0.6, 2100.0, 20.0], x) .+
            0.02 .* randn(length(x))
        fit = fit_peaks(x, y; n_peaks=2, model=gaussian)

        spec = TASpectrum(x, y)
        @test lines(spec) isa Makie.FigureAxisPlot
        steady = Spectrum(x, y; ylabel="Transmittance")
        @test lines(steady) isa Makie.FigureAxisPlot

        fig1 = plot_fit(fit)
        @test fig1 isa Makie.Figure
        @test count(x -> x isa Makie.Axis, fig1.content) == 2  # main + residuals

        fig2 = plot_fit(fit; residuals=false)
        @test fig2 isa Makie.Figure
        @test count(x -> x isa Makie.Axis, fig2.content) == 1

        fig3 = plot_fit(fit; components=true, baseline=true)
        @test fig3 isa Makie.Figure
    end

    @testset "Metadata-driven labels" begin
        md = Dict{Symbol,Any}(:time_unit => "ns", :signal_label => "Counts")
        tr = KineticTrace([0.0, 1.0], [1.0, 0.5]; metadata=md)
        @test xlabel(tr) == "Time (ns)"
        @test ylabel(tr) == "Counts"
        @test occursin("ns", sprint(show, tr))

        tr_default = KineticTrace([0.0, 1.0], [1.0, 0.5])
        @test xlabel(tr_default) == "Time (ps)"
        @test ylabel(tr_default) == "ΔA"

        m = TimeResolvedMatrix([0.0, 1.0], [500.0, 510.0], [1.0 2.0; 3.0 4.0]; metadata=md)
        @test ylabel(m) == "Time (ns)"
        @test zlabel(m) == "Counts"
        @test occursin("ns", sprint(show, m))
        @test occursin("ns", sprint(show, MIME("text/plain"), tr))
        @test occursin("ns", sprint(show, MIME("text/plain"), m))
        @test xlabel(m[λ=505.0]) == "Time (ns)"
    end

    @testset "GatedSpectrum" begin
        g = GatedSpectrum([500.0, 510.0, 520.0], [1.0, 2.0, 1.5];
                          t_range=(0.0, 5.0),
                          metadata=Dict{Symbol,Any}(:time_unit => "ns"))
        @test g isa AbstractSpectroscopyData
        @test xdata(g) == [500.0, 510.0, 520.0]
        @test ydata(g) == [1.0, 2.0, 1.5]
        @test wavelength(g) == g.wavelength
        @test signal(g) == g.signal
        @test g.t_range == (0.0, 5.0)
        @test xlabel(g) == "Wavelength (nm)"
        @test occursin("0.0 to 5.0 ns", sprint(show, g))
        @test_throws ArgumentError GatedSpectrum([1.0, 2.0], [1.0])
        g2 = GatedSpectrum([500.0], [1.0])
        @test all(isnan, g2.t_range)
        @test ylabel(g) == "ΔA"
        g_pl = GatedSpectrum([500.0], [1.0]; metadata=Dict{Symbol,Any}(:signal_label => "Counts"))
        @test ylabel(g_pl) == "Counts"
        g_wn = GatedSpectrum([1500.0, 1600.0], [1.0, 2.0])
        @test xlabel(g_wn) == "Wavenumber (cm⁻¹)"
        @test occursin("Time gate", sprint(show, MIME("text/plain"), g))
        g_empty = GatedSpectrum(Float64[], Float64[])
        @test xlabel(g_empty) == "Wavelength (nm)"
        @test occursin("0 points", sprint(show, g_empty))
        @test sprint(show, MIME("text/plain"), g_empty) isa String
        @test occursin("500.0 to 520.0 nm", sprint(show, g))
        g_src = GatedSpectrum([500.0], [1.0]; metadata=Dict{Symbol,Any}(:source => "demo.img"))
        @test occursin("demo.img", sprint(show, MIME("text/plain"), g_src))
    end

    @testset "TimeResolvedMatrix slices" begin
        t = [0.0, 1.0, 2.0]
        wl = [500.0, 510.0, 520.0]
        data = [1.0 2.0 3.0;
                4.0 5.0 6.0;
                7.0 8.0 9.0]   # rows = time, cols = wavelength
        md = Dict{Symbol,Any}(:signal_label => "Counts", :time_unit => "ns")
        m = TimeResolvedMatrix(t, wl, data; metadata=md)

        # nearest single column
        tr = kinetic_trace(m; wavelength=511.0)
        @test tr isa KineticTrace
        @test tr.signal == [2.0, 5.0, 8.0]
        @test tr.wavelength == 510.0
        @test tr.metadata[:signal_label] == "Counts"

        # band mean over all three columns (510 ± 10 → [500, 520])
        tr_band = kinetic_trace(m; wavelength=510.0, band=20.0)
        @test tr_band.signal == [2.0, 5.0, 8.0]
        @test tr_band.wavelength == 510.0
        @test tr_band.metadata[:band] == 20.0
        @test !haskey(tr.metadata, :band)
        @test integrate_time(m).metadata[:signal_label] == "Counts"
        @test_throws ArgumentError kinetic_trace(m; wavelength=NaN)

        # empty band falls back to nearest column (deterministic: argmin ties break to first)
        tr_fb = kinetic_trace(m; wavelength=505.0, band=2.0)
        @test tr_fb.signal == [1.0, 4.0, 7.0]
        @test tr_fb.wavelength == 500.0

        # nearest single row
        sp = spectral_slice(m; time=1.2)
        @test sp isa GatedSpectrum
        @test sp.signal == [4.0, 5.0, 6.0]
        @test sp.t_range == (1.0, 1.0)
        @test sp.metadata[:time_unit] == "ns"

        # gated mean over rows 2:3 (1.5 ± 1 → [0.5, 2.5])
        sp_win = spectral_slice(m; time=1.5, window=2.0)
        @test sp_win.signal == [5.5, 6.5, 7.5]
        @test sp_win.t_range == (1.0, 2.0)

        # time-integrated spectrum (sum)
        g = integrate_time(m)
        @test g isa GatedSpectrum
        @test g.signal == [12.0, 15.0, 18.0]
        @test g.t_range == (0.0, 2.0)

        g2 = integrate_time(m; t_range=(1.0, 2.0))
        @test g2.signal == [11.0, 13.0, 15.0]
        @test g2.t_range == (1.0, 2.0)
        @test_throws ArgumentError integrate_time(m; t_range=(10.0, 20.0))

        # KineticTrace extracted from a matrix carries :source, not :filename;
        # source_file must fall back to :source
        m_src = TimeResolvedMatrix(t, wl, data; metadata=Dict{Symbol,Any}(:source => "x.img"))
        @test source_file(kinetic_trace(m_src; wavelength=510.0)) == "x.img"
    end

    @testset "bin_matrix" begin
        t = [0.0, 1.0, 2.0, 3.0]
        wl = [500.0, 510.0, 520.0, 530.0, 540.0, 550.0]
        data = reshape(collect(1.0:24.0), 4, 6)
        m = TimeResolvedMatrix(t, wl, data)

        b = bin_matrix(m; time=2, wavelength=3)
        @test size(b.data) == (2, 2)
        @test b.time == [0.5, 2.5]
        @test b.wavelength == [510.0, 540.0]
        # block (rows 1:2, cols 1:3) = mean of [1 5 9; 2 6 10] = 5.5
        @test b.data[1, 1] == 5.5
        # block (rows 3:4, cols 4:6) = mean of [15 19 23; 16 20 24] = 19.5
        @test b.data[2, 2] == 19.5
        @test b.metadata[:bin_time] == 2
        @test b.metadata[:bin_wavelength] == 3

        # partial final block: 4 rows binned by 3 → blocks 1:3 and 4:4
        b2 = bin_matrix(m; time=3)
        @test size(b2.data) == (2, 6)
        @test b2.time == [1.0, 3.0]
        @test b2.data[2, 1] == 4.0

        # identity
        b3 = bin_matrix(m)
        @test b3.data == m.data
        @test_throws ArgumentError bin_matrix(m; time=0)
        @test_throws ArgumentError bin_matrix(m; wavelength=0)

        # repeated binning accumulates the factor: bin by 2 twice → :bin_time == 4
        @test bin_matrix(b; time=2).metadata[:bin_time] == 4

        # both axes partial: 4 rows by 3 → 2 blocks; 6 cols by 4 → 2 blocks (last has 2)
        b4 = bin_matrix(m; time=3, wavelength=4)
        @test size(b4.data) == (2, 2)
        @test b4.wavelength == [515.0, 545.0]
        # block (rows 4:4, cols 5:6) = mean of [20, 24] = 22.0
        @test b4.data[2, 2] == 22.0
    end

    @testset "Matrix cosmic ray detection" begin
        t = collect(0.0:1.0:19.0)
        wl = collect(500.0:5.0:595.0)
        # smooth decay + small deterministic ripple (no RNG)
        base = [100.0 * exp(-ti / 8.0) + 10.0 for ti in t, _ in wl]
        ripple = [0.5 * sin(3.1 * i + 1.7 * j) for i in 1:20, j in 1:20]
        clean = base .+ ripple
        spikes = [CartesianIndex(3, 4), CartesianIndex(10, 15), CartesianIndex(18, 2)]
        data = copy(clean)
        for I in spikes
            data[I] += 500.0
        end
        m = TimeResolvedMatrix(t, wl, data)

        result = detect_cosmic_rays(m; threshold=8.0)
        @test result isa CosmicRayMatrixResult
        @test issubset(Set(spikes), Set(result.indices))
        @test result.count <= 6          # no mass false positives
        @test result.threshold == 8.0

        cleaned = remove_cosmic_rays(m, result)
        @test cleaned isa TimeResolvedMatrix
        for I in spikes
            @test abs(cleaned.data[I] - clean[I]) < 5.0
        end
        # untouched pixel unchanged
        @test cleaned.data[1, 1] == data[1, 1] || CartesianIndex(1, 1) in result.indices
        @test cleaned.metadata[:cosmic_rays_removed] == result.count

        # constant image → no detections
        flat = TimeResolvedMatrix(t, wl, fill(7.0, 20, 20))
        @test detect_cosmic_rays(flat).count == 0

        # sharp t0-like transient: a full time-row jumps; must not be flagged as CRs
        t0data = copy(clean)
        t0data[7, :] .+= 400.0
        t0data[CartesianIndex(15, 9)] += 500.0
        m_t0 = TimeResolvedMatrix(t, wl, t0data)
        r_t0 = detect_cosmic_rays(m_t0; threshold=8.0)
        @test CartesianIndex(15, 9) in r_t0.indices
        @test all(I -> I[1] != 7, r_t0.indices)

        # noisy but clean data: spikes found, false positives bounded
        rng = Random.MersenneTwister(7)
        noisy = base .+ 3.0 .* randn(rng, 20, 20)
        spiked = copy(noisy)
        for I in spikes
            spiked[I] += 500.0
        end
        m_n = TimeResolvedMatrix(t, wl, spiked)
        r_n = detect_cosmic_rays(m_n; threshold=8.0)
        @test issubset(Set(spikes), Set(r_n.indices))
        @test r_n.count <= 10

        # Poisson shot noise on a bright band must not be over-flagged
        rng2 = Random.MersenneTwister(21)
        bright = [1000.0 * exp(-ti / 8.0) * exp(-((w - 550.0) / 20.0)^2) + 20.0 for ti in t, w in wl]
        shot = bright .+ sqrt.(bright) .* randn(rng2, 20, 20)
        r_shot = detect_cosmic_rays(TimeResolvedMatrix(t, wl, shot); threshold=8.0)
        @test r_shot.count <= 2
        # and a real CR on top of the bright band is still caught
        shot_cr = copy(shot)
        shot_cr[CartesianIndex(2, 11)] += 5000.0
        r_shot_cr = detect_cosmic_rays(TimeResolvedMatrix(t, wl, shot_cr); threshold=8.0)
        @test CartesianIndex(2, 11) in r_shot_cr.indices
    end

    @testset "Stretched exponential decay fit" begin
        t = collect(0.0:0.1:50.0)
        sig = 1.0 .* exp.(-(t ./ 5.0) .^ 0.7) .+ 0.05
        trace = KineticTrace(t, sig; wavelength=550.0)

        fit = fit_exp_decay(trace; model=:stretched)
        @test fit isa StretchedDecayFit
        @test isapprox(fit.tau, 5.0; rtol=0.05)
        @test isapprox(fit.beta, 0.7; rtol=0.05)
        @test isapprox(fit.offset, 0.05; atol=0.01)
        @test fit.rsquared > 0.999

        # ⟨τ⟩ = (τ/β)Γ(1/β) must equal ∫₀^∞ exp(-(t/τ)^β) dt (trapezoidal check)
        ts = range(0.0, 400.0; length=400_000)
        ys = exp.(-(ts ./ fit.tau) .^ fit.beta)
        integral = sum((ys[1:end-1] .+ ys[2:end]) ./ 2 .* step(ts))
        @test isapprox(mean_lifetime(fit), integral; rtol=1e-3)

        @test occursin("β", sprint(show, MIME("text/plain"), fit))
        @test occursin("Stretched", format_results(fit))

        pred = predict(fit, t)
        @test length(pred) == length(t)
        @test maximum(abs.(pred .- sig)) < 1e-3

        @test_throws ArgumentError fit_exp_decay(trace; model=:stretched, n_exp=2)
        @test_throws ArgumentError fit_exp_decay(trace; model=:stretched, irf=true)
        @test_throws ArgumentError fit_exp_decay(trace; model=:linear)

        # noisy data (2% of amplitude)
        rng = Random.MersenneTwister(11)
        sig_n = sig .+ 0.02 .* randn(rng, length(t))
        fit_n = fit_exp_decay(KineticTrace(t, sig_n; wavelength=550.0); model=:stretched)
        @test isapprox(fit_n.tau, 5.0; rtol=0.15)
        @test isapprox(fit_n.beta, 0.7; rtol=0.15)

        # non-zero fit origin via t_range
        sig_shift = [ti < 10.0 ? 1.05 : exp(-((ti - 10.0) / 5.0)^0.7) + 0.05 for ti in t]
        fit_s = fit_exp_decay(KineticTrace(t, sig_shift; wavelength=550.0);
                              model=:stretched, t_range=(10.0, 50.0))
        @test fit_s.t0 == 10.0
        @test isapprox(fit_s.tau, 5.0; rtol=0.05)
        @test isapprox(fit_s.beta, 0.7; rtol=0.05)
    end

    @testset "fit_lifetime_spectrum" begin
        t = collect(0.0:0.2:40.0)
        wl = collect(500.0:2.0:600.0)
        τλ(w) = w < 550 ? 2.0 : 8.0
        data = [exp(-ti / τλ(w)) for ti in t, w in wl]
        m = TimeResolvedMatrix(t, wl, data)

        result = fit_lifetime_spectrum(m; nbins=10)
        @test result isa LifetimeSpectrumResult
        @test result.n_exp == 1
        @test length(result.wavelength) == 10
        @test size(result.taus) == (10, 1)
        @test all(result.fitted)
        @test isapprox(result.taus[1, 1], 2.0; rtol=0.1)
        @test isapprox(result.taus[end, 1], 8.0; rtol=0.1)
        @test all(result.rsquared[result.fitted] .> 0.99)
        @test occursin("10/10", sprint(show, result))

        # intensity floor skips weak bins
        weak = copy(data)
        weak[:, 1:10] .*= 1e-6
        m2 = TimeResolvedMatrix(t, wl, weak)
        r2 = fit_lifetime_spectrum(m2; nbins=10, min_signal=0.01)
        @test !r2.fitted[1]
        @test isnan(r2.taus[1, 1])
        @test r2.fitted[end]

        @test_throws ArgumentError fit_lifetime_spectrum(m; nbins=0)

        # n_exp=2 path (MultiexpDecayFit branch)
        data2 = [0.6 * exp(-ti / 3.0) + 0.4 * exp(-ti / 12.0) for ti in t, w in wl]
        m3 = TimeResolvedMatrix(t, wl, data2)
        r3 = fit_lifetime_spectrum(m3; nbins=4, n_exp=2)
        @test r3.n_exp == 2
        @test size(r3.taus) == (4, 2)
        @test all(r3.fitted)
        @test all(r3.rsquared[r3.fitted] .> 0.99)
        @test_throws ArgumentError fit_lifetime_spectrum(m; n_exp=0)
    end

end
