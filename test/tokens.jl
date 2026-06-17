@testset "Metadata token contract" begin

    @testset "axis_label" begin
        @test axis_label(:wavenumber, :per_cm) == "Wavenumber (cm⁻¹)"
        @test axis_label(:wavelength, :nm) == "Wavelength (nm)"
        @test axis_label(:time, :ps) == "Time (ps)"
        @test axis_label(:time, :ns) == "Time (ns)"
        @test axis_label(:absorbance, :OD) == "Absorbance (OD)"
        @test axis_label(:intensity, :counts) == "Intensity (counts)"
        @test axis_label(:delta_absorbance, :dimensionless) == "ΔA"
        @test axis_label(:transmittance, :fraction) == "Transmittance"
        @test axis_label(:foo_bar, :baz) == "Foo bar (baz)"
    end

    @testset "label resolvers (dict helpers)" begin
        @test OpticalSpectroscopy._spectral_xlabel(Dict(:xlabel => "Energy (eV)")) == "Energy (eV)"
        @test OpticalSpectroscopy._signal_label(Dict(:ylabel => "Counts")) == "Counts"
        @test OpticalSpectroscopy._spectral_xlabel(Dict(:xquantity => :wavenumber, :xunit => :per_cm)) == "Wavenumber (cm⁻¹)"
        @test OpticalSpectroscopy._signal_label(Dict(:yquantity => :absorbance, :yunit => :OD)) == "Absorbance (OD)"
        @test OpticalSpectroscopy._spectral_xlabel(Dict{Symbol,Any}()) == "x"
        @test OpticalSpectroscopy._signal_label(Dict{Symbol,Any}()) == "Signal"
        @test OpticalSpectroscopy._time_label(Dict(:xunit => :ns), :xunit) == "Time (ns)"
        @test OpticalSpectroscopy._time_label(Dict{Symbol,Any}(), :time_unit) == "Time (ps)"
    end

    @testset "is_canonical / validate_tokens" begin
        @test is_canonical(:technique, :ftir)
        @test !is_canonical(:technique, :nmr)
        @test is_canonical(:quantity, :wavenumber)
        @test is_canonical(:quantity, :absorbance)
        @test !is_canonical(:quantity, :bogus)
        @test is_canonical(:unit, :per_cm)
        @test !is_canonical(:unit, :furlong)
        @test !is_canonical(:nonsense_slot, :anything)

        good = Dict{Symbol,Any}(:technique => :ftir, :xquantity => :wavenumber,
                                :xunit => :per_cm, :yquantity => :absorbance, :yunit => :OD)
        @test validate_tokens(good)
        bad = Dict{Symbol,Any}(:xunit => :furlong)
        @test (@test_logs (:warn,) validate_tokens(bad)) == false
        @test validate_tokens(Dict{Symbol,Any}())
    end

    @testset "normalize_unit / normalize_quantity" begin
        @test normalize_unit("NANOMETERS") == :nm
        @test normalize_unit("1/cm") == :per_cm
        @test normalize_unit("cm-1") == :per_cm
        @test normalize_unit("%T") == :percent
        @test normalize_unit("") == :dimensionless
        @test normalize_unit("Bananas Per Furlong") == :bananas_per_furlong

        @test normalize_quantity("ABSORBANCE") == :absorbance
        @test normalize_quantity("%T") == :transmittance
        @test normalize_quantity("Raman Shift") == :raman_shift
        @test normalize_quantity("interferrogram") == :interferogram   # real JASCO misspelling
        @test normalize_quantity("Mystery") == :mystery
    end

    @testset "_unitful bridge / CANONICAL_UNIT" begin
        @test OpticalSpectroscopy._unitful(:per_cm) === u"cm^-1"
        @test OpticalSpectroscopy._unitful(:nm) === u"nm"
        @test OpticalSpectroscopy._unitful(:ps) === u"ps"
        @test OpticalSpectroscopy._unitful(:OD) === Unitful.NoUnits
        @test OpticalSpectroscopy._unitful(:dimensionless) === Unitful.NoUnits
        @test [1.0, 2.0] .* OpticalSpectroscopy._unitful(:counts) == [1.0, 2.0]
        @test OpticalSpectroscopy.CANONICAL_UNIT[:wavenumber] == :per_cm
        @test OpticalSpectroscopy.CANONICAL_UNIT[:wavelength] == :nm
        @test OpticalSpectroscopy.CANONICAL_UNIT[:delta_absorbance] == :mOD
    end

end
