# `Spectrum` Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a generic 1D steady-state spectrum type (`Spectrum`) to the `AbstractSpectroscopyData` family, with typed dispatches for analysis and transformation functions per `docs/superpowers/specs/2026-06-12-spectrum-type-design.md`.

**Architecture:** `Spectrum(x, y, metadata::Dict{Symbol,Any}) <: AbstractSpectroscopyData` lives in `src/types.jl`. Hybrid dispatch: read-only analysis goes generic on `AbstractSpectroscopyData` (guarded by a new `_check_1d` helper); spectrum-producing transformations get concrete `::Spectrum` methods returning new `Spectrum` objects with copied metadata. The Makie extension's existing `convert_arguments` already covers plotting.

**Tech Stack:** Julia 1.10+, existing package deps only (no new dependencies). Tests in `test/runtests.jl` (single-file suite, synthetic data only).

**Environment notes:**
- `julia` may not be on PATH in tool shells — prefix every command with `PATH="$HOME/.juliaup/bin:$PATH"`.
- Run everything from the repo root `/Users/garrek/Developer/OpticalSpectroscopy.jl`.
- Work happens on the existing branch `feat/spectrum-type` (spec already committed there).
- The full test suite takes a few minutes. Each task uses a fast inline snippet for the TDD fail/pass check, then the full suite runs once in the final task (plus after any task if you're suspicious). Quick-check snippets load the package via `using OpticalSpectroscopy`, which recompiles after edits — expect ~30 s per invocation.
- Version stays **0.1.0** — do not bump `Project.toml`.

## File Structure

| File | Change |
|------|--------|
| `src/types.jl` | Add `_check_1d` helper (after interface defaults); add `Spectrum` struct + interface + `show` (new section before "Global fitting result") |
| `src/OpticalSpectroscopy.jl` | Add `Spectrum` to exports line 47 |
| `src/peakfitting.jl` | Add `_check_1d` guards to the two existing `fit_peaks(::AbstractSpectroscopyData...)` methods |
| `src/peakdetection.jl` | New generic `find_peaks(::AbstractSpectroscopyData)` |
| `src/spectroscopy.jl` | Guards on existing generic arithmetic; new generic `band_area`/`calc_fwhm`/`estimate_snr`; concrete `::Spectrum` transformations |
| `src/baseline.jl` | `correct_baseline(::Spectrum)` |
| `src/transforms.jl` | `snv(::Spectrum)`, `kubelka_munk(::Spectrum)` |
| `test/runtests.jl` | New testsets inserted after the `"TimeResolvedMatrix indexing"` testset; one line in the `"Makie extension"` testset |
| `docs/src/api.md` | Add `Spectrum` to the data-types `@docs` block; retitle section |
| `docs/src/index.md` | Mention `Spectrum` in the steady-state feature description |
| `CLAUDE.md` | Add `Spectrum` to the Type Hierarchy diagram |

---

### Task 1: `Spectrum` type, interface, exports

**Files:**
- Modify: `src/types.jl` (new section immediately before the `# Global fitting result` section header, i.e. after the `GatedSpectrum` `show` methods)
- Modify: `src/OpticalSpectroscopy.jl:47`
- Test: `test/runtests.jl` (insert after the closing `end` of `@testset "TimeResolvedMatrix indexing"`, before `@testset "fit_peaks with raw vectors"`)

- [ ] **Step 1: Write the failing tests**

Insert into `test/runtests.jl` after the `"TimeResolvedMatrix indexing"` testset's closing `end` (currently followed by `@testset "fit_peaks with raw vectors"`):

```julia
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
        long = sprint(show, MIME("text/plain"), s)
        @test occursin("Points", long)
        @test occursin("sample", long)
        # Empty spectrum must not error
        s_empty = Spectrum(Float64[], Float64[])
        @test occursin("0 points", sprint(show, s_empty))
        @test sprint(show, MIME("text/plain"), s_empty) isa String
    end
```

- [ ] **Step 2: Verify failure**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e 'using OpticalSpectroscopy; Spectrum([1.0],[1.0])'
```
Expected: `UndefVarError: Spectrum not defined` (or `not exported`).

- [ ] **Step 3: Implement the type**

In `src/types.jl`, insert immediately BEFORE the section header comment
`# Global fitting result` (the `# ===…` block following the `GatedSpectrum` show methods):

```julia
# =============================================================================
# Spectrum: generic 1D steady-state spectrum
# =============================================================================

"""
    Spectrum <: AbstractSpectroscopyData

    Spectrum(x, y)
    Spectrum(x, y, metadata::Dict{Symbol,Any})
    Spectrum(x, y; metadata...)

Generic 1D steady-state spectrum: signal versus a spectral axis.

Covers FTIR, Raman, UV-Vis, photoluminescence, cavity transmission, and any
other steady-state 1D data. Axis semantics live in `metadata`:

- `:xlabel` — x-axis display label (default: detected from the x range,
  `"Wavenumber (cm⁻¹)"` or `"Wavelength (nm)"`)
- `:ylabel` — y-axis display label (default: `"Signal"`)
- `:filename` / `:source` — source file for [`source_file`](@ref)
- `:cavity_length` — picked up as the default cavity length by
  `fit_cavity_spectrum` (cavity-fitting tools; convention reserved here)

# Fields
- `x::Vector{Float64}`: Spectral axis
- `y::Vector{Float64}`: Signal
- `metadata::Dict{Symbol,Any}`: Additional info

# Examples
```julia
spec = Spectrum(nu, T)
spec = Spectrum(nu, T; mirror="Au", angle=10, cavity_length=12e-4)
spec.metadata[:mirror]
```
"""
struct Spectrum <: AbstractSpectroscopyData
    x::Vector{Float64}
    y::Vector{Float64}
    metadata::Dict{Symbol,Any}

    function Spectrum(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                      metadata::AbstractDict)
        length(x) == length(y) || throw(ArgumentError(
            "Spectrum: x and y must have equal length; " *
            "got $(length(x)) x points and $(length(y)) y points"))
        new(Float64.(x), Float64.(y), Dict{Symbol,Any}(metadata))
    end
end

function Spectrum(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; metadata...)
    md = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in metadata)
    return Spectrum(x, y, md)
end

# AbstractSpectroscopyData interface
xdata(s::Spectrum) = s.x
ydata(s::Spectrum) = s.y
function xlabel(s::Spectrum)
    lbl = get(s.metadata, :xlabel, nothing)
    isnothing(lbl) || return String(lbl)
    return _detect_spectral_unit(s.x) == "cm⁻¹" ? "Wavenumber (cm⁻¹)" : "Wavelength (nm)"
end
ylabel(s::Spectrum) = String(get(s.metadata, :ylabel, "Signal"))
source_file(s::Spectrum) = get(s.metadata, :filename, get(s.metadata, :source, ""))

"""
    signal(s::Spectrum) -> Vector{Float64}

Return the signal.
"""
signal(s::Spectrum) = s.y

function Base.show(io::IO, s::Spectrum)
    n = length(s.x)
    range = isempty(s.x) ? "" :
        ", $(round(minimum(s.x), digits=1)) to $(round(maximum(s.x), digits=1))"
    print(io, "Spectrum: $n points$range")
end

function Base.show(io::IO, ::MIME"text/plain", s::Spectrum)
    println(io, "Spectrum")
    println(io, "  Points:  $(length(s.x))")
    if !isempty(s.x)
        println(io, "  X:       $(xlabel(s)), $(round(minimum(s.x), digits=1)) to $(round(maximum(s.x), digits=1))")
    end
    println(io, "  Y:       $(ylabel(s))")
    for key in (:sample, :mirror, :cavity_length, :angle)
        val = get(s.metadata, key, nothing)
        !isnothing(val) && println(io, "  $key: $val")
    end
    src = source_file(s)
    !isempty(src) && println(io, "  Source:  $src")
end
```

In `src/OpticalSpectroscopy.jl`, change line 47 from:

```julia
export KineticTrace, TASpectrum, TimeResolvedMatrix, GatedSpectrum, SweepData
```

to:

```julia
export Spectrum, KineticTrace, TASpectrum, TimeResolvedMatrix, GatedSpectrum, SweepData
```

- [ ] **Step 4: Verify the quick check passes**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
s = Spectrum(collect(1500.0:1.0:1504.0), [1.0, 2.0, 3.0, 2.0, 1.0]; cavity_length=12e-4)
@assert s isa AbstractSpectroscopyData
@assert xlabel(s) == "Wavenumber (cm⁻¹)"
@assert ylabel(s) == "Signal"
@assert s.metadata[:cavity_length] == 12e-4
println(sprint(show, MIME("text/plain"), s))
println("OK")'
```
Expected: pretty-print output ending with `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl src/OpticalSpectroscopy.jl test/runtests.jl
git commit -m "feat: add Spectrum, a generic 1D steady-state spectrum type"
```

---

### Task 2: `_check_1d` guards on existing generic dispatches

Today `fit_peaks(::TimeResolvedMatrix)` silently fits the time axis (its
`ydata` is the time vector). Add a shared guard and apply it to all existing
generic `AbstractSpectroscopyData` methods.

**Files:**
- Modify: `src/types.jl` (helper, right after the `title` default near line 122)
- Modify: `src/peakfitting.jl:294,312` (the two `fit_peaks(spec::AbstractSpectroscopyData...)` methods)
- Modify: `src/spectroscopy.jl:75` (`subtract_spectrum` ASD method) and `:441-451` (generic `add_spectra`, `divide_spectra`, `multiply_spectrum`)
- Test: `test/runtests.jl` (insert after the `"Spectrum - show"` testset added in Task 1)

- [ ] **Step 1: Write the failing tests**

```julia
    @testset "Generic dispatches reject 2D data" begin
        m = TimeResolvedMatrix([0.0, 1.0, 2.0], [700.0, 750.0], rand(3, 2))
        @test_throws ArgumentError fit_peaks(m)
        @test_throws ArgumentError fit_peaks(m, (700.0, 750.0))
        @test_throws ArgumentError subtract_spectrum(m, m)
        @test_throws ArgumentError add_spectra(m, m)
        @test_throws ArgumentError divide_spectra(m, m)
        @test_throws ArgumentError multiply_spectrum(m, 2.0)
    end
```

- [ ] **Step 2: Verify failure**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
m = TimeResolvedMatrix([0.0, 1.0, 2.0], [700.0, 750.0], rand(3, 2))
try
    multiply_spectrum(m, 2.0)
    println("NO ERROR - guard missing (expected before implementation)")
catch e
    println("threw: ", typeof(e))
end'
```
Expected: `NO ERROR - guard missing` (the NamedTuple path currently succeeds nonsensically).

- [ ] **Step 3: Implement**

In `src/types.jl`, after the `title(d::AbstractSpectroscopyData) = source_file(d)` definition (before the `# Transient Absorption types` section header):

```julia
# Guard for generic 1D dispatches: matrix types must be sliced first.
function _check_1d(d::AbstractSpectroscopyData, fname::AbstractString)
    is_matrix(d) && throw(ArgumentError(
        "$fname requires 1D data; got 2D $(nameof(typeof(d))). " *
        "Extract a 1D slice first (e.g. spectral_slice or matrix[t=...])."))
    return nothing
end
```

In `src/peakfitting.jl`, add as the first line of BOTH generic methods
(`fit_peaks(spec::AbstractSpectroscopyData, region::Tuple{Real, Real}; kwargs...)`
at line 294 and `fit_peaks(spec::AbstractSpectroscopyData; kwargs...)` at line 312):

```julia
    _check_1d(spec, "fit_peaks")
```

In `src/spectroscopy.jl`, add as the first line(s) of the generic methods:

```julia
# in subtract_spectrum(sample::AbstractSpectroscopyData, reference::AbstractSpectroscopyData; kwargs...):
    _check_1d(sample, "subtract_spectrum"); _check_1d(reference, "subtract_spectrum")
# in add_spectra(a::AbstractSpectroscopyData, b::AbstractSpectroscopyData; kwargs...):
    _check_1d(a, "add_spectra"); _check_1d(b, "add_spectra")
# in divide_spectra(a::AbstractSpectroscopyData, b::AbstractSpectroscopyData; kwargs...):
    _check_1d(a, "divide_spectra"); _check_1d(b, "divide_spectra")
# in multiply_spectrum(spec::AbstractSpectroscopyData, factor::Real):
    _check_1d(spec, "multiply_spectrum")
```

(Those comments locate the insertion points — don't paste the comments themselves.)

- [ ] **Step 4: Verify the quick check passes**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
m = TimeResolvedMatrix([0.0, 1.0, 2.0], [700.0, 750.0], rand(3, 2))
for (f, args) in [(fit_peaks, (m,)), (subtract_spectrum, (m, m)), (add_spectra, (m, m)),
                  (divide_spectra, (m, m)), (multiply_spectrum, (m, 2.0))]
    try
        f(args...)
        error("$(f) did not throw")
    catch e
        e isa ArgumentError || rethrow()
    end
end
# 1D types still work
t = KineticTrace([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
@assert multiply_spectrum(t, 2.0).y == [2.0, 4.0, 2.0]
println("OK")'
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl src/peakfitting.jl src/spectroscopy.jl test/runtests.jl
git commit -m "fix: guard generic 1D dispatches against 2D matrix data"
```

---

### Task 3: New generic analysis dispatches

**Files:**
- Modify: `src/peakdetection.jl` (after `find_peaks(y::AbstractVector; kwargs...)`, currently ending near line 144)
- Modify: `src/spectroscopy.jl` (generic `band_area`, `calc_fwhm`, `estimate_snr` — place each right after its vector form)
- Test: `test/runtests.jl` (after the `"Generic dispatches reject 2D data"` testset)

- [ ] **Step 1: Write the failing tests**

```julia
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

        # 2D guard
        m = TimeResolvedMatrix([0.0, 1.0], [700.0, 750.0], rand(2, 2))
        @test_throws ArgumentError find_peaks(m)
        @test_throws ArgumentError band_area(m, 1.0, 2.0)
        @test_throws ArgumentError calc_fwhm(m)
        @test_throws ArgumentError estimate_snr(m)
    end
```

- [ ] **Step 2: Verify failure**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
s = Spectrum([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 1.0, 0.5])
find_peaks(s)'
```
Expected: `MethodError: no method matching find_peaks(::Spectrum...)`.

- [ ] **Step 3: Implement**

In `src/peakdetection.jl`, after the `find_peaks(y::AbstractVector; kwargs...)` method:

```julia
# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function find_peaks(spec::AbstractSpectroscopyData; kwargs...)
    _check_1d(spec, "find_peaks")
    return find_peaks(xdata(spec), ydata(spec); kwargs...)
end
```

In `src/spectroscopy.jl`, after the corresponding vector forms:

```julia
# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function band_area(spec::AbstractSpectroscopyData, x_min::Real, x_max::Real)
    _check_1d(spec, "band_area")
    return band_area(xdata(spec), ydata(spec), x_min, x_max)
end
```

```julia
# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function calc_fwhm(spec::AbstractSpectroscopyData; kwargs...)
    _check_1d(spec, "calc_fwhm")
    return calc_fwhm(xdata(spec), ydata(spec); kwargs...)
end
```

```julia
# Generic dispatch: any 1D AbstractSpectroscopyData via the xdata/ydata interface
function estimate_snr(spec::AbstractSpectroscopyData)
    _check_1d(spec, "estimate_snr")
    return estimate_snr(ydata(spec))
end
```

- [ ] **Step 4: Verify the quick check passes**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
x = collect(400.0:0.5:800.0)
y = @. 100 * exp(-(x - 520)^2 / (2 * 10^2)) + 5
s = Spectrum(x, y)
@assert find_peaks(s)[1].position == find_peaks(x, y)[1].position
@assert band_area(s, 480.0, 560.0) == band_area(x, y, 480.0, 560.0)
@assert calc_fwhm(s) == calc_fwhm(x, y)
@assert estimate_snr(s) == estimate_snr(y)
println("OK")'
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/peakdetection.jl src/spectroscopy.jl test/runtests.jl
git commit -m "feat: generic find_peaks/band_area/calc_fwhm/estimate_snr for 1D spectroscopy data"
```

---

### Task 4: `Spectrum` smoothing, normalization, and T↔A transformations

**Files:**
- Modify: `src/spectroscopy.jl` (methods placed after the corresponding vector forms)
- Test: `test/runtests.jl` (after the `"Generic analysis dispatches"` testset)

- [ ] **Step 1: Write the failing tests**

```julia
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
```

- [ ] **Step 2: Verify failure**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
s = Spectrum([1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
smooth_data(s; window=3)'
```
Expected: `MethodError` (the untyped `smooth_data(y; window)` hits `similar(::Spectrum)`).
Note: if this *doesn't* error, it found the untyped fallback — the typed method below
is still required so the result is a `Spectrum`, not a raw vector.

- [ ] **Step 3: Implement**

In `src/spectroscopy.jl`. Each method goes after its vector counterpart.
Pattern: unpack, call the vector API, rewrap with `copy(s.metadata)`.

```julia
"""
    smooth_data(s::Spectrum; window=3) -> Spectrum

Moving-average smoothing of a [`Spectrum`](@ref). Returns a new `Spectrum`
with copied metadata.
"""
function smooth_data(s::Spectrum; window=3)
    return Spectrum(s.x, smooth_data(s.y; window=window), copy(s.metadata))
end
```

```julia
"""
    savitzky_golay_smooth(s::Spectrum; window=11, order=3) -> Spectrum

Savitzky-Golay smoothing of a [`Spectrum`](@ref). Returns a new `Spectrum`
with copied metadata.
"""
function savitzky_golay_smooth(s::Spectrum; window::Int=11, order::Int=3)
    return Spectrum(s.x, savitzky_golay_smooth(s.y; window=window, order=order), copy(s.metadata))
end
```

```julia
"""
    derivative(s::Spectrum; order=1, window=11, poly_order=3) -> Spectrum

Savitzky-Golay derivative of a [`Spectrum`](@ref), scaled by the x-spacing.
Returns a new `Spectrum` with copied metadata.
"""
function derivative(s::Spectrum; order::Int=1, window::Int=11, poly_order::Int=3)
    return Spectrum(s.x, derivative(s.x, s.y; order=order, window=window, poly_order=poly_order),
                    copy(s.metadata))
end
```

```julia
"""
    normalize_area(s::Spectrum) -> Spectrum

Normalize a [`Spectrum`](@ref) to unit integrated area. Returns a new
`Spectrum` with copied metadata.
"""
function normalize_area(s::Spectrum)
    return Spectrum(s.x, normalize_area(s.x, s.y), copy(s.metadata))
end
```

```julia
"""
    normalize_to_peak(s::Spectrum, position; tolerance=5.0) -> Spectrum

Normalize a [`Spectrum`](@ref) to the intensity at `position`. Returns a new
`Spectrum` with copied metadata.
"""
function normalize_to_peak(s::Spectrum, position::Real; tolerance::Real=5.0)
    return Spectrum(s.x, normalize_to_peak(s.x, s.y, position; tolerance=tolerance),
                    copy(s.metadata))
end
```

```julia
"""
    transmittance_to_absorbance(s::Spectrum; percent=false) -> Spectrum

Convert a transmittance [`Spectrum`](@ref) to absorbance. Sets
`metadata[:ylabel] = "Absorbance"` on the result.
"""
function transmittance_to_absorbance(s::Spectrum; percent::Bool=false)
    md = copy(s.metadata)
    md[:ylabel] = "Absorbance"
    return Spectrum(s.x, transmittance_to_absorbance(s.y; percent=percent), md)
end
```

```julia
"""
    absorbance_to_transmittance(s::Spectrum; percent=false) -> Spectrum

Convert an absorbance [`Spectrum`](@ref) to transmittance. Sets
`metadata[:ylabel]` to `"Transmittance"` (or `"Transmittance (%)"` when
`percent=true`) on the result.
"""
function absorbance_to_transmittance(s::Spectrum; percent::Bool=false)
    md = copy(s.metadata)
    md[:ylabel] = percent ? "Transmittance (%)" : "Transmittance"
    return Spectrum(s.x, absorbance_to_transmittance(s.y; percent=percent), md)
end
```

- [ ] **Step 4: Verify the quick check passes**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
x = collect(400.0:1.0:800.0)
y = @. 50 * exp(-(x - 520)^2 / (2 * 10^2)) + 0.5
s = Spectrum(x, y; sample="test")
@assert smooth_data(s; window=5) isa Spectrum
@assert derivative(s).y ≈ derivative(x, y)
a = transmittance_to_absorbance(Spectrum([1.0, 2.0], [0.5, 0.1]))
@assert a.metadata[:ylabel] == "Absorbance"
println("OK")'
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/spectroscopy.jl test/runtests.jl
git commit -m "feat: Spectrum-in/Spectrum-out smoothing, normalization, and T<->A methods"
```

---

### Task 5: `Spectrum` arithmetic and interpolation

**Files:**
- Modify: `src/spectroscopy.jl` (after the generic `AbstractSpectroscopyData` arithmetic dispatches at the bottom of the file)
- Test: `test/runtests.jl` (after the `"Spectrum transformations - transmittance/absorbance"` testset)

- [ ] **Step 1: Write the failing tests**

```julia
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
```

- [ ] **Step 2: Verify failure**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
a = Spectrum([1.0, 2.0], [1.0, 2.0])
r = subtract_spectrum(a, a)
println(typeof(r))'
```
Expected: prints a `NamedTuple` type (the generic path), NOT `Spectrum`.

- [ ] **Step 3: Implement**

In `src/spectroscopy.jl`, after the existing generic `AbstractSpectroscopyData`
arithmetic dispatches (bottom of file). The `(x=..., y=...)` NamedTuple calls
route to the untyped vector machinery directly — do NOT call the same function
on the `Spectrum` arguments again (infinite recursion).

```julia
# Spectrum-in → Spectrum-out arithmetic. The result keeps the first
# argument's metadata (copied).

"""
    subtract_spectrum(s::Spectrum, ref::Spectrum; scale=1.0, interpolate=false) -> Spectrum

Subtract `ref` from `s`. Returns a new `Spectrum` carrying `s`'s metadata.
"""
function subtract_spectrum(s::Spectrum, ref::Spectrum; scale::Real=1.0, interpolate=false)
    res = subtract_spectrum((x=s.x, y=s.y), (x=ref.x, y=ref.y); scale=scale, interpolate=interpolate)
    return Spectrum(res.x, res.y, copy(s.metadata))
end

"""
    add_spectra(a::Spectrum, b::Spectrum; interpolate=false) -> Spectrum

Add two spectra. Returns a new `Spectrum` carrying `a`'s metadata.
"""
function add_spectra(a::Spectrum, b::Spectrum; interpolate=false)
    res = add_spectra((x=a.x, y=a.y), (x=b.x, y=b.y); interpolate=interpolate)
    return Spectrum(res.x, res.y, copy(a.metadata))
end

"""
    divide_spectra(a::Spectrum, b::Spectrum; interpolate=false) -> Spectrum

Divide `a` by `b` element-wise. Returns a new `Spectrum` carrying `a`'s metadata.
"""
function divide_spectra(a::Spectrum, b::Spectrum; interpolate=false)
    res = divide_spectra((x=a.x, y=a.y), (x=b.x, y=b.y); interpolate=interpolate)
    return Spectrum(res.x, res.y, copy(a.metadata))
end

"""
    multiply_spectrum(s::Spectrum, factor::Real) -> Spectrum

Scale a spectrum by a constant. Returns a new `Spectrum` with copied metadata.
"""
function multiply_spectrum(s::Spectrum, factor::Real)
    return Spectrum(s.x, s.y .* factor, copy(s.metadata))
end

"""
    average_spectra(specs::Spectrum...; interpolate=false) -> Spectrum

Point-wise average. Returns a new `Spectrum` carrying the first spectrum's
metadata.
"""
function average_spectra(specs::Spectrum...; interpolate=false)
    res = average_spectra(map(s -> (x=s.x, y=s.y), specs)...; interpolate=interpolate)
    return Spectrum(res.x, res.y, copy(specs[1].metadata))
end

"""
    interpolate_spectrum(s::Spectrum, new_x) -> Spectrum

Resample a spectrum onto `new_x` by linear interpolation. Returns a new
`Spectrum` with copied metadata.
"""
function interpolate_spectrum(s::Spectrum, new_x::AbstractVector{<:Real})
    return Spectrum(collect(Float64.(new_x)), interpolate_spectrum(s.x, s.y, new_x),
                    copy(s.metadata))
end
```

- [ ] **Step 4: Verify the quick check passes**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
x = collect(1500.0:1.0:1599.0)
a = Spectrum(x, fill(2.0, 100); sample="A")
b = Spectrum(x, fill(0.5, 100))
@assert subtract_spectrum(a, b) isa Spectrum
@assert add_spectra(a, b).y ≈ fill(2.5, 100)
@assert divide_spectra(a, b).y ≈ fill(4.0, 100)
@assert multiply_spectrum(a, 3.0).y ≈ fill(6.0, 100)
@assert average_spectra(a, b).metadata[:sample] == "A"
@assert interpolate_spectrum(a, collect(1500.0:0.5:1599.0)) isa Spectrum
println("OK")'
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/spectroscopy.jl test/runtests.jl
git commit -m "feat: Spectrum-in/Spectrum-out spectral arithmetic and interpolation"
```

---

### Task 6: `correct_baseline`, `snv`, `kubelka_munk` for `Spectrum`

**Files:**
- Modify: `src/baseline.jl` (after `correct_baseline(x, y; ...)` at line 386)
- Modify: `src/transforms.jl` (after `snv(y)` at line 181 and `kubelka_munk(R)` at line 68)
- Test: `test/runtests.jl` (after the `"Spectrum arithmetic returns Spectrum"` testset)

- [ ] **Step 1: Write the failing tests**

```julia
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
        @test km.y ≈ kubelka_munk([0.3, 0.6])
        @test km.metadata[:ylabel] == "F(R)"
    end
```

- [ ] **Step 2: Verify failure**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
s = Spectrum([1.0, 2.0, 3.0], [1.0, 2.0, 1.0])
snv(s)'
```
Expected: `MethodError: no method matching snv(::Spectrum)`.

- [ ] **Step 3: Implement**

In `src/baseline.jl`, after the `correct_baseline(x::AbstractVector, y::AbstractVector{<:Real}; ...)` method:

```julia
"""
    correct_baseline(s::Spectrum; method=:arpls, kwargs...) -> Spectrum

Baseline-correct a [`Spectrum`](@ref). Returns a new `Spectrum` with copied
metadata. To inspect the baseline itself, use the vector form
`correct_baseline(s.x, s.y; ...)`, which returns `(x, y, baseline)`.
"""
function correct_baseline(s::Spectrum; method::Symbol=:arpls, kwargs...)
    res = correct_baseline(s.x, s.y; method=method, kwargs...)
    return Spectrum(res.x, res.y, copy(s.metadata))
end
```

In `src/transforms.jl`, after `snv(y::AbstractVector)`:

```julia
"""
    snv(s::Spectrum) -> Spectrum

Standard normal variate transform of a [`Spectrum`](@ref). Returns a new
`Spectrum` with copied metadata.
"""
snv(s::Spectrum) = Spectrum(s.x, snv(s.y), copy(s.metadata))
```

In `src/transforms.jl`, after `kubelka_munk(R)`:

```julia
"""
    kubelka_munk(s::Spectrum) -> Spectrum

Kubelka-Munk transform of a reflectance [`Spectrum`](@ref). Sets
`metadata[:ylabel] = "F(R)"` on the result.
"""
function kubelka_munk(s::Spectrum)
    md = copy(s.metadata)
    md[:ylabel] = "F(R)"
    return Spectrum(s.x, kubelka_munk(s.y), md)
end
```

- [ ] **Step 4: Verify the quick check passes**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using OpticalSpectroscopy
x = collect(1000.0:1.0:1399.0)
y = @. 20 * exp(-(x - 1200)^2 / (2 * 15^2)) + 0.01 * (x - 1000) + 2
s = Spectrum(x, y)
@assert correct_baseline(s; method=:arpls) isa Spectrum
@assert snv(s) isa Spectrum
@assert kubelka_munk(Spectrum([1.0], [0.5])).metadata[:ylabel] == "F(R)"
println("OK")'
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/baseline.jl src/transforms.jl test/runtests.jl
git commit -m "feat: Spectrum methods for correct_baseline, snv, and kubelka_munk"
```

---

### Task 7: Makie extension test

The extension's `convert_arguments(::PointBased, ::AbstractSpectroscopyData)`
already covers `Spectrum` — only a test is needed.

**Files:**
- Test: `test/runtests.jl` (inside `@testset "Makie extension"`, after the line `@test lines(spec) isa Makie.FigureAxisPlot`)

- [ ] **Step 1: Add the test**

After `@test lines(spec) isa Makie.FigureAxisPlot` insert:

```julia
        steady = Spectrum(x, y; ylabel="Transmittance")
        @test lines(steady) isa Makie.FigureAxisPlot
```

(`x`, `y` already exist in that testset's scope. If `Spectrum` clashes with a
name exported by CairoMakie — unlikely — qualify as `OpticalSpectroscopy.Spectrum`.)

- [ ] **Step 2: Verify it passes**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e '
using Pkg; Pkg.activate("."); Pkg.instantiate()
using OpticalSpectroscopy, CairoMakie
s = Spectrum(collect(1.0:10.0), rand(10))
@assert lines(s) isa Makie.FigureAxisPlot
println("OK")' 2>/dev/null || echo "CairoMakie not in main env — rely on Pkg.test() in Task 8"
```
Expected: `OK`, or the fallback message (CairoMakie is a test-only dependency;
the full suite in Task 8 covers it either way).

- [ ] **Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "test: lines() works for Spectrum via the Makie extension"
```

---

### Task 8: Documentation, full verification, PR

**Files:**
- Modify: `docs/src/api.md:11-24`
- Modify: `docs/src/index.md:5`
- Modify: `CLAUDE.md` (Type Hierarchy section)

- [ ] **Step 1: Update docs**

In `docs/src/api.md`, change the section heading `## Time-Resolved Data Types`
to `## Data Types` and add `Spectrum` after `AbstractSpectroscopyData`:

```markdown
## Data Types

```@docs
AbstractSpectroscopyData
Spectrum
KineticTrace
TASpectrum
TimeResolvedMatrix
GatedSpectrum
SweepData
TASpectrumFit
TAPeak
fit_ta_spectrum
anharmonicity
```
```

In `docs/src/index.md` line 5, change:

```
For steady-state work, OpticalSpectroscopy provides peak fitting (Gaussian, Lorentzian, Voigt, Fano), baseline correction, peak detection, spectral transforms (Kramers-Kronig, Tauc, Kubelka-Munk), and unit conversions for FTIR, Raman, and UV-vis data.
```

to:

```
For steady-state work, OpticalSpectroscopy provides a generic `Spectrum` type plus peak fitting (Gaussian, Lorentzian, Voigt, Fano), baseline correction, peak detection, spectral transforms (Kramers-Kronig, Tauc, Kubelka-Munk), and unit conversions for FTIR, Raman, and UV-vis data.
```

In `CLAUDE.md`, update the Type Hierarchy diagram to:

```
AbstractSpectroscopyData (root interface)
├── Spectrum            (generic 1D steady-state spectrum: signal vs spectral axis)
├── KineticTrace        (kinetics: signal vs time at fixed wavelength)
├── TASpectrum          (spectrum: signal vs wavenumber at fixed time)
├── GatedSpectrum       (spectrum extracted from a time window)
├── TimeResolvedMatrix  (2D: time × wavelength heatmap)
└── PLMap               (2D: spatial PL/Raman map)
```

- [ ] **Step 2: Run the full test suite**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=. -e 'using Pkg; Pkg.test()'
```
Expected: all tests pass, including Aqua (no new ambiguities/piracy) and the
Makie extension testset. If Aqua flags an ambiguity from the new methods, fix
the offending signature (most likely candidate: varargs `average_spectra`) and re-run.

- [ ] **Step 3: Build docs locally (optional but recommended — repo enforces missing_docs)**

```bash
PATH="$HOME/.juliaup/bin:$PATH" julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate(); include("docs/make.jl")'
```
Expected: build completes; no `missing_docs` errors (the new `Spectrum` name is
documented and listed in api.md). If the docs environment fails to resolve
locally, note it and rely on the docs CI job.

- [ ] **Step 4: Commit docs**

```bash
git add docs/src/api.md docs/src/index.md CLAUDE.md
git commit -m "docs: document Spectrum type and update type hierarchy"
```

- [ ] **Step 5: Push and open PR**

```bash
git push -u origin feat/spectrum-type
gh pr create --title "Add Spectrum: generic 1D steady-state spectrum type" --body "$(cat <<'EOF'
## Summary
- New `Spectrum <: AbstractSpectroscopyData`: generic 1D steady-state spectrum (x, y, Dict{Symbol,Any} metadata), per docs/superpowers/specs/2026-06-12-spectrum-type-design.md
- Generic 1D analysis dispatches (`find_peaks`, `band_area`, `calc_fwhm`, `estimate_snr`) for the whole family via xdata/ydata
- `is_matrix` guards on pre-existing generic dispatches (`fit_peaks`, spectral arithmetic) — previously fit nonsense on 2D types
- Spectrum-in → Spectrum-out transformations: baseline, smoothing, derivative, normalization, arithmetic, interpolation, T↔A, SNV, Kubelka-Munk
- `lines(::Spectrum)` works via the existing Makie extension convert_arguments

## Notes
- Establishes the `:cavity_length` metadata convention for the upcoming CavitySpectroscopy merge's `fit_cavity_spectrum(::Spectrum)`
- Version stays 0.1.0 (pre-registration)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Checklist (run after all tasks)

- Every spec section maps to a task: type (1), guards (2), generic analysis (3), transformations (4-6), plotting (7), docs/exports/version (1, 8).
- Functions deliberately excluded (spec "not given Spectrum methods now"): `kramers_kronig`, `tauc_plot`, `urbach_tail`, raw `*_baseline` functions, `fit_ta_spectrum`, `normalize_intensity` (PLMap-only).
- All metadata copies use `copy(...)` — aliasing test in Task 4 enforces this.
