# Metadata Token Contract + Type Collapse — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace magnitude-guessing axis labels with a reserved `(quantity, unit)` token vocabulary from which labels are *derived*; make bare row-column data load with no units provided and no units guessed; and collapse `TASpectrum` and `GatedSpectrum` into `Spectrum` + tokens.

**Architecture:** A new pure-vocabulary module (`src/tokens.jl`, included before `types.jl`) owns the controlled token sets, `axis_label`, validation, instrument-string normalization, and the Symbol↔Unitful bridge. Labels resolve by a fixed three-tier precedence — literal override → token-derived → honest generic floor (`"x"` / `"Signal"`) — with **no legacy bridge and no magnitude heuristic** on any automatic path (the heuristic survives only behind an opt-in `guess_units!`). `TASpectrum` and `GatedSpectrum` are deleted; their producers (`matrix[t=…]`, `spectral_slice`, `integrate_time`) and the TA fitting path return / accept `Spectrum`, carrying the former struct fields (`time_delay`, `t_range`) as flat metadata tokens.

**Tech Stack:** Julia 1.10+, Unitful.jl (already a dependency), Test.jl + Aqua.jl.

---

## Scope & non-goals

Source of truth for the design: [`docs/superpowers/2026-06-15-metadata-token-contract.md`](../2026-06-15-metadata-token-contract.md). Greenfield — nothing has shipped except file readers, so breaking changes are fine. This plan implements **clean-break, no legacy bridge** (ratified 2026-06-16) and **folds in the TASpectrum/GatedSpectrum collapse** (ratified 2026-06-16).

**Kept:** `Spectrum`, `KineticTrace` (the time-axis "firewall"), `TimeResolvedMatrix` (rank-2 container), `PLMap`, and all fit-**result** types — including `TASpectrumFit`, `TAPeak`, `anharmonicity`, and the `wavenumber(::TASpectrumFit)` accessor (these are fit machinery, not data types).

**Out of scope (separate follow-on plans):**
- **`TimeResolvedMatrix` / `PLMap` structural `:axisN_*` field refactor** (§10 #3, unratified). This plan keeps the matrix's `time`/`wavelength` fields and uses an **interim token mapping** on the matrix: `:xquantity`/`:xunit` = spectral axis, `:yquantity`/`:yunit` = signal, `:time_unit` (a `Symbol`) = time axis. Documented in code as interim-pending-#3.
- **HDF5 typed codec** (`encode_metadata`/`decode_metadata`) — QPSScanFormat.jl (§7).
- **Loader stamping** (calling `normalize_unit`, technique disambiguation) — JASCOFiles.jl / QPSTools.jl (§8). The vocabulary functions are defined here; their callers are not.

**Invariant after every task:** the full suite (`Pkg.test()`, Aqua first) is green. New exports are added in the same task that defines their methods. The standard test command (from the project root) is `julia --project=. -e 'using Pkg; Pkg.test()'`; the first run precompiles (allow a few minutes).

---

## Interim token scheme (reference — used throughout)

| Type | x-axis | signal | extra |
|---|---|---|---|
| `Spectrum` | `:xquantity`/`:xunit` | `:yquantity`/`:yunit` | `:time_delay`(+`_unit`), `:gate_start`/`:gate_end`(+`:gate_unit`) when sliced |
| `KineticTrace` | time → `:xunit` (quantity is `:time` by definition) | `:yquantity`/`:yunit` | — |
| `TimeResolvedMatrix` | spectral → `:xquantity`/`:xunit` | `:yquantity`/`:yunit` (the data/z) | time axis → `:time_unit` (`Symbol`) |

`KineticTrace` uses `:xunit` for its time axis (time *is* x); `TimeResolvedMatrix` uses `:time_unit` for its time axis (time is the stacking axis, x is spectral). A spectral slice of a matrix therefore inherits `(x, y)` tokens by a plain metadata copy; a kinetic (time-axis) slice remaps `:time_unit → :xunit` and drops the stale spectral `:xquantity`.

---

## Phase 1 — Vocabulary core (`src/tokens.jl`)

Purely additive. After Phase 1 the suite is unchanged-green.

### Task 1: Create `tokens.jl` with vocabularies, `axis_label`, and label resolvers

**Files:**
- Create: `src/tokens.jl`
- Modify: `src/OpticalSpectroscopy.jl` (add include before `types.jl`; export `axis_label`)
- Create: `test/tokens.jl`
- Modify: `test/runtests.jl` (add `include("tokens.jl")` before the final top-level `end`)

- [ ] **Step 1: Write `src/tokens.jl`**

```julia
"""
Metadata token contract: controlled vocabularies, label derivation, instrument-
string normalization, and the Symbol↔Unitful bridge.

See docs/superpowers/2026-06-15-metadata-token-contract.md. Every axis is a
`(quantity, unit)` pair of `Symbol` tokens; display labels are *derived* from
those tokens, never stored as prose and never guessed from data magnitudes.
"""

# Controlled vocabularies (§3). Open for extension, closed for validation:
# an unknown value warns and falls back — it never silently passes.
const TECHNIQUES = Set{Symbol}([:ta, :ftir, :raman, :uvvis, :pl, :xrd])

const AXIS_QUANTITIES = Set{Symbol}([
    :wavelength, :wavenumber, :raman_shift, :opd, :two_theta,
    :time, :energy, :position])

const SIGNAL_QUANTITIES = Set{Symbol}([
    :absorbance, :transmittance, :reflectance, :single_beam,
    :interferogram, :delta_absorbance, :intensity])

const UNITS = Set{Symbol}([
    :nm, :um, :angstrom, :per_cm, :fs, :ps, :ns, :degree, :eV, :meV,
    :points, :mm, :counts, :arb, :OD, :mOD, :percent, :fraction, :dimensionless])

const _QUANTITY_DISPLAY = Dict{Symbol,String}(
    :wavelength       => "Wavelength",
    :wavenumber       => "Wavenumber",
    :raman_shift      => "Raman shift",
    :opd              => "Optical path difference",
    :two_theta        => "2θ",
    :time             => "Time",
    :energy           => "Energy",
    :position         => "Position",
    :absorbance       => "Absorbance",
    :transmittance    => "Transmittance",
    :reflectance      => "Reflectance",
    :single_beam      => "Single-beam intensity",
    :interferogram    => "Interferogram",
    :delta_absorbance => "ΔA",
    :intensity        => "Intensity",
)

const _UNIT_DISPLAY = Dict{Symbol,String}(
    :nm => "nm", :um => "µm", :angstrom => "Å", :per_cm => "cm⁻¹",
    :fs => "fs", :ps => "ps", :ns => "ns", :degree => "°",
    :eV => "eV", :meV => "meV", :points => "points", :mm => "mm",
    :counts => "counts", :arb => "arb. units", :OD => "OD", :mOD => "mOD",
    :percent => "%", :fraction => "", :dimensionless => "",
)

_token_display(s::Symbol) = uppercasefirst(replace(String(s), '_' => ' '))

"""
    axis_label(quantity::Symbol, unit::Symbol) -> String

Compose a display label from a quantity token and a unit token:
`axis_label(:wavenumber, :per_cm) == "Wavenumber (cm⁻¹)"`. A dimensionless or
empty unit yields just the quantity name:
`axis_label(:delta_absorbance, :dimensionless) == "ΔA"`. Unknown tokens fall
back to their humanized `String` form so nothing is ever lost.
"""
function axis_label(quantity::Symbol, unit::Symbol)
    name = get(_QUANTITY_DISPLAY, quantity, _token_display(quantity))
    usym = get(_UNIT_DISPLAY, unit, String(unit))
    return isempty(usym) ? name : "$name ($usym)"
end

# Dict-based label resolvers (§4 precedence). Pure: they take metadata, never
# the data values. No legacy bridge, no magnitude guess.

"""
    _spectral_xlabel(metadata) -> String

X-axis label by §4 precedence: (1) literal `:xlabel` → (2) token-derived from
`:xquantity`/`:xunit` → (3) honest generic floor `"x"`.
"""
function _spectral_xlabel(metadata::AbstractDict)
    lbl = get(metadata, :xlabel, nothing)
    isnothing(lbl) || return String(lbl)
    q = get(metadata, :xquantity, nothing)
    if !isnothing(q)
        u = Symbol(get(metadata, :xunit, :dimensionless))
        return axis_label(Symbol(q), u)
    end
    return "x"
end

"""
    _signal_label(metadata) -> String

Signal-axis label by §4 precedence: (1) literal `:ylabel` → (2) token-derived
from `:yquantity`/`:yunit` → (3) honest generic floor `"Signal"`.
"""
function _signal_label(metadata::AbstractDict)
    lbl = get(metadata, :ylabel, nothing)
    isnothing(lbl) || return String(lbl)
    q = get(metadata, :yquantity, nothing)
    if !isnothing(q)
        u = Symbol(get(metadata, :yunit, :dimensionless))
        return axis_label(Symbol(q), u)
    end
    return "Signal"
end

"""
    _time_label(metadata, unit_key) -> String

Time-axis label: `axis_label(:time, unit)` where `unit` is read from `unit_key`
(`:xunit` for `KineticTrace`, `:time_unit` for `TimeResolvedMatrix`), defaulting
to `:ps`.
"""
_time_label(metadata::AbstractDict, unit_key::Symbol) =
    axis_label(:time, Symbol(get(metadata, unit_key, :ps)))
```

- [ ] **Step 2: Wire the include and export `axis_label`**

In `src/OpticalSpectroscopy.jl`, insert the include immediately above `include("types.jl")` (line 27):

```julia
# Source files (order matters: tokens (pure vocab) before types; types before
# functions that use them)
include("tokens.jl")
include("types.jl")
```

Add to the Types & Interface export block (after `export source_file, npoints, title`):

```julia
export axis_label
```

- [ ] **Step 3: Create `test/tokens.jl` and wire it in**

Create `test/tokens.jl`:

```julia
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

end
```

In `test/runtests.jl`, add the include just before the final top-level `end`:

```julia
    include("tokens.jl")
end
```

- [ ] **Step 4: Run the suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -25`
Expected: PASS, including the `Metadata token contract` testset.

- [ ] **Step 5: Commit**

```bash
git add src/tokens.jl src/OpticalSpectroscopy.jl test/tokens.jl test/runtests.jl
git commit -m "feat(tokens): add vocabulary, axis_label, and label resolvers"
```

---

### Task 2: `is_canonical` and `validate_tokens`

**Files:** Modify `src/tokens.jl` (append), `src/OpticalSpectroscopy.jl` (export), `test/tokens.jl` (append).

- [ ] **Step 1: Write the failing test** — append inside `@testset "Metadata token contract"`:

```julia
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20`
Expected: FAIL — `is_canonical`/`validate_tokens` not defined.

- [ ] **Step 3: Implement** — append to `src/tokens.jl`:

```julia
# Validation (§8). Warns, never throws — out-of-vocabulary values are allowed.

"""
    is_canonical(slot::Symbol, value::Symbol) -> Bool

Whether `value` is in the controlled vocabulary for `slot`. Slots: `:technique`,
`:quantity` (axis or signal), `:unit`. Unknown slots return `false`.
"""
function is_canonical(slot::Symbol, value::Symbol)
    slot === :technique && return value in TECHNIQUES
    slot === :quantity  && return value in AXIS_QUANTITIES || value in SIGNAL_QUANTITIES
    slot === :unit      && return value in UNITS
    return false
end

const _RESERVED_TOKEN_SLOTS = (
    (:technique, :technique),
    (:xquantity, :quantity), (:yquantity, :quantity), (:signal_quantity, :quantity),
    (:axis1_quantity, :quantity), (:axis2_quantity, :quantity), (:axis3_quantity, :quantity),
    (:xunit, :unit), (:yunit, :unit), (:signal_unit, :unit),
    (:axis1_unit, :unit), (:axis2_unit, :unit), (:axis3_unit, :unit),
    (:time_unit, :unit), (:time_delay_unit, :unit), (:gate_unit, :unit),
    (:fixed_coordinate_unit, :unit), (:position_unit, :unit),
)

"""
    validate_tokens(metadata::AbstractDict) -> Bool

Check reserved token values against the controlled vocabularies of §3. `@warn`s
for each reserved key whose value is outside its vocabulary and returns `false`;
returns `true` when all present reserved tokens are canonical. Never throws.
"""
function validate_tokens(metadata::AbstractDict)
    ok = true
    for (key, slot) in _RESERVED_TOKEN_SLOTS
        haskey(metadata, key) || continue
        value = metadata[key]
        value isa Symbol || (value = Symbol(value))
        if !is_canonical(slot, value)
            @warn "Non-canonical metadata token" key value slot
            ok = false
        end
    end
    return ok
end
```

Export: `export axis_label, is_canonical, validate_tokens`

- [ ] **Step 4: Run to verify it passes** — `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -15` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tokens.jl src/OpticalSpectroscopy.jl test/tokens.jl
git commit -m "feat(tokens): add is_canonical and validate_tokens"
```

---

### Task 3: `normalize_unit` and `normalize_quantity`

**Files:** Modify `src/tokens.jl`, `src/OpticalSpectroscopy.jl`, `test/tokens.jl`.

- [ ] **Step 1: Write the failing test** — append inside the token testset:

```julia
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
```

- [ ] **Step 2: Run to verify it fails** — `… | tail -20` — Expected: FAIL (undefined).

- [ ] **Step 3: Implement** — append to `src/tokens.jl`:

```julia
# Instrument-string normalization (§8). Owned here so every loader maps strings
# the same way.

const _UNIT_ALIASES = Dict{String,Symbol}(
    "nm" => :nm, "nanometers" => :nm, "nanometer" => :nm,
    "um" => :um, "µm" => :um, "μm" => :um, "micron" => :um, "microns" => :um, "micrometers" => :um,
    "angstrom" => :angstrom, "angstroms" => :angstrom, "å" => :angstrom,
    "cm-1" => :per_cm, "cm^-1" => :per_cm, "cm⁻¹" => :per_cm, "1/cm" => :per_cm, "per cm" => :per_cm,
    "fs" => :fs, "ps" => :ps, "ns" => :ns,
    "deg" => :degree, "degree" => :degree, "degrees" => :degree, "°" => :degree,
    "ev" => :eV, "mev" => :meV,
    "points" => :points, "point" => :points, "pts" => :points,
    "mm" => :mm,
    "counts" => :counts, "count" => :counts, "cts" => :counts,
    "arb" => :arb, "arbitrary" => :arb, "a.u." => :arb, "au" => :arb,
    "od" => :OD, "mod" => :mOD,
    "%" => :percent, "%t" => :percent, "%r" => :percent, "percent" => :percent,
    "fraction" => :fraction,
    "dimensionless" => :dimensionless, "" => :dimensionless,
)

const _QUANTITY_ALIASES = Dict{String,Symbol}(
    "absorbance" => :absorbance, "abs" => :absorbance,
    "transmittance" => :transmittance, "%t" => :transmittance, "transmission" => :transmittance,
    "reflectance" => :reflectance, "%r" => :reflectance, "reflection" => :reflectance,
    "single beam" => :single_beam, "single-beam" => :single_beam, "sb" => :single_beam,
    "interferogram" => :interferogram, "interferrogram" => :interferogram, "igram" => :interferogram,
    "delta absorbance" => :delta_absorbance, "δa" => :delta_absorbance, "deltaa" => :delta_absorbance,
    "intensity" => :intensity, "counts" => :intensity, "pl" => :intensity,
    "wavelength" => :wavelength, "wavenumber" => :wavenumber,
    "raman shift" => :raman_shift, "raman" => :raman_shift,
    "2theta" => :two_theta, "two theta" => :two_theta, "2θ" => :two_theta,
    "time" => :time, "delay" => :time, "energy" => :energy,
)

_fallback_token(key::AbstractString) = Symbol(replace(key, ' ' => '_'))

"""
    normalize_unit(s::AbstractString) -> Symbol

Map a free-form instrument unit string to a canonical unit token (`"NANOMETERS"
→ :nm`, `"1/cm" → :per_cm`). Unrecognized strings return `Symbol` of the
lowercased, underscored text so nothing is lost.
"""
normalize_unit(s::AbstractString) =
    get(_UNIT_ALIASES, lowercase(strip(s)), _fallback_token(lowercase(strip(s))))

"""
    normalize_quantity(s::AbstractString) -> Symbol

Map a free-form instrument quantity string to a canonical quantity token
(`"ABSORBANCE" → :absorbance`, `"Raman Shift" → :raman_shift`). Unrecognized
strings return `Symbol` of the lowercased, underscored text.
"""
normalize_quantity(s::AbstractString) =
    get(_QUANTITY_ALIASES, lowercase(strip(s)), _fallback_token(lowercase(strip(s))))
```

Export: `export axis_label, is_canonical, validate_tokens, normalize_unit, normalize_quantity`

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tokens.jl src/OpticalSpectroscopy.jl test/tokens.jl
git commit -m "feat(tokens): add normalize_unit and normalize_quantity"
```

---

### Task 4: `_unitful` bridge and `CANONICAL_UNIT`

**Files:** Modify `src/tokens.jl`, `test/tokens.jl`. (Internal — not exported.)

- [ ] **Step 1: Write the failing test** — append:

```julia
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
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (undefined).

- [ ] **Step 3: Implement** — append to `src/tokens.jl`:

```julia
# Symbol↔Unitful bridge (§6) and canonical unit per quantity.

const _UNITFUL_MAP = Dict{Symbol,Any}(
    :nm => u"nm", :um => u"µm", :angstrom => u"angstrom", :per_cm => u"cm^-1",
    :fs => u"fs", :ps => u"ps", :ns => u"ns", :degree => u"°",
    :eV => u"eV", :meV => u"meV", :mm => u"mm",
)

"""
    _unitful(unit::Symbol)

Map a unit token to its Unitful unit (`_unitful(:per_cm) === u"cm^-1"`). Tokens
Unitful cannot represent (`:OD`, `:counts`, `:arb`, `:points`, `:percent`, …)
map to `Unitful.NoUnits`, the identity under `*`.
"""
_unitful(unit::Symbol) = get(_UNITFUL_MAP, unit, Unitful.NoUnits)

"""
    CANONICAL_UNIT

Default unit token per quantity token, used by the `axis=` constructor sugar.
"""
const CANONICAL_UNIT = Dict{Symbol,Symbol}(
    :wavelength => :nm, :wavenumber => :per_cm, :raman_shift => :per_cm,
    :opd => :points, :two_theta => :degree, :time => :ps, :energy => :eV,
    :position => :um,
    :absorbance => :OD, :transmittance => :percent, :reflectance => :percent,
    :single_beam => :arb, :interferogram => :arb, :delta_absorbance => :mOD,
    :intensity => :counts,
)
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tokens.jl test/tokens.jl
git commit -m "feat(tokens): add Symbol-to-Unitful bridge and canonical units"
```

---

## Phase 2 — `Spectrum` rewiring

### Task 5: `Spectrum` token labels + honest floor + de-guessed show

**Files:** Modify `src/types.jl` (`xlabel`/`ylabel` @1182–1187, `Base.show` @1197–1203, docstring @1122–1156); `test/runtests.jl` (lines 247–253, 274).

- [ ] **Step 1: Adjust the failing tests** — in `test/runtests.jl`, replace the "Label fallbacks" block (lines 247–253):

```julia
        # No-guess rule: bare row-column data asserts only what it was told.
        @test xlabel(s) == "x"
        @test ylabel(s) == "Signal"

        # tokens drive the label when present
        s_tok = Spectrum([1500.0, 1501.0], [1.0, 2.0];
                         xquantity=:wavenumber, xunit=:per_cm,
                         yquantity=:absorbance, yunit=:OD)
        @test xlabel(s_tok) == "Wavenumber (cm⁻¹)"
        @test ylabel(s_tok) == "Absorbance (OD)"
```

Replace the unit assertion at line 274 (`occursin("cm⁻¹", …)`):

```julia
        @test occursin("1500.0 to 1504.0", sprint(show, s))
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`xlabel(s)` still `"Wavenumber (cm⁻¹)"`).

- [ ] **Step 3: Implement** — in `src/types.jl`, replace `xlabel`/`ylabel` for `Spectrum` (lines 1182–1187):

```julia
xlabel(s::Spectrum) = _spectral_xlabel(s.metadata)
ylabel(s::Spectrum) = _signal_label(s.metadata)
```

Replace `Base.show(io::IO, s::Spectrum)` (lines 1197–1203):

```julia
function Base.show(io::IO, s::Spectrum)
    n = length(s.x)
    range = isempty(s.x) ? "" :
        ", $(round(minimum(s.x), digits=1)) to $(round(maximum(s.x), digits=1))"
    print(io, "Spectrum: $n points$range")
end
```

Update the `Spectrum` docstring metadata section (lines 1133–1140):

```julia
Covers FTIR, Raman, UV-Vis, photoluminescence, cavity transmission, transient-
absorption slices, and any other 1D data. Axis semantics live in `metadata` as
reserved `(quantity, unit)` tokens, from which labels are derived (never guessed):

- `:xquantity` / `:xunit`, `:yquantity` / `:yunit` — axis tokens (e.g.
  `:wavenumber` / `:per_cm`). See `axis_label`.
- `:xlabel` / `:ylabel` — literal label strings; override the tokens.
- `:time_delay` (+ `:time_delay_unit`) — fixed delay of a slice from a
  `TimeResolvedMatrix`; `:gate_start` / `:gate_end` (+ `:gate_unit`) — the time
  window of a gated/integrated slice.
- `:filename` / `:source` — source file for [`source_file`](@ref).
- `:cavity_length` — default cavity length for `fit_cavity_spectrum`.

With no axis metadata, labels fall back to the honest generic floor (`"x"` /
`"Signal"`) — bare row-column data loads with no units guessed.
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl test/runtests.jl
git commit -m "feat(spectrum): derive labels from tokens, honest floor, no magnitude guess"
```

---

### Task 6: `Spectrum(M::AbstractMatrix)` and `axis=` sugar

**Files:** Modify `src/types.jl` (keyword constructor @1174–1177; add matrix constructor; docstring header @1124–1127); `test/tokens.jl` (append).

- [ ] **Step 1: Write the failing test** — append inside the token testset:

```julia
    @testset "Spectrum minimal-input path" begin
        M = [10.0 100.0; 20.0 200.0; 30.0 300.0]
        sm = Spectrum(M)
        @test xdata(sm) == [10.0, 20.0, 30.0]
        @test ydata(sm) == [100.0, 200.0, 300.0]
        @test xlabel(sm) == "x"
        @test_throws ArgumentError Spectrum(rand(3, 3))

        sa = Spectrum([1.0, 2.0], [3.0, 4.0]; axis=:wavenumber)
        @test sa.metadata[:xquantity] == :wavenumber
        @test sa.metadata[:xunit] == :per_cm
        @test xlabel(sa) == "Wavenumber (cm⁻¹)"

        sa2 = Spectrum([1.0, 2.0], [3.0, 4.0]; axis=:wavelength, xunit=:um)
        @test xlabel(sa2) == "Wavelength (µm)"

        smt = Spectrum(M; axis=:wavelength, sample="demo")
        @test xlabel(smt) == "Wavelength (nm)"
        @test smt.metadata[:sample] == "demo"
    end
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (no matrix method / `axis` swallowed).

- [ ] **Step 3: Implement** — in `src/types.jl`, replace the keyword constructor (lines 1174–1177):

```julia
function Spectrum(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                  axis::Union{Symbol,Nothing}=nothing, metadata...)
    md = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in metadata)
    if axis !== nothing
        md[:xquantity] = axis
        get!(md, :xunit, get(CANONICAL_UNIT, axis, :dimensionless))
    end
    return Spectrum(x, y, md)
end

"""
    Spectrum(M::AbstractMatrix)

Build a `Spectrum` from an `N×2` matrix (column 1 = x, column 2 = y), so a plain
row-column export goes straight in: `Spectrum(readdlm("homemade.txt"))`. Keyword
arguments (`axis=`, metadata) are forwarded to the vector constructor.
"""
function Spectrum(M::AbstractMatrix{<:Real}; kwargs...)
    size(M, 2) == 2 || throw(ArgumentError(
        "Spectrum(M): expected an N×2 matrix (column 1 = x, column 2 = y); " *
        "got size $(size(M))"))
    return Spectrum(M[:, 1], M[:, 2]; kwargs...)
end
```

Add to the docstring constructor signatures (lines 1124–1127):

```julia
    Spectrum(x, y; axis=:wavenumber, metadata...)
    Spectrum(M::AbstractMatrix)
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl test/tokens.jl
git commit -m "feat(spectrum): add N×2 matrix constructor and axis= unit sugar"
```

---

### Task 7: Opt-in `xdata_unitful` / `ydata_unitful` / `guess_units!`

**Files:** Modify `src/types.jl` (append helpers at end of file); `src/OpticalSpectroscopy.jl` (export three); `test/tokens.jl` (append).

> `_metadata` is defined only for the three surviving data types (`Spectrum`, `KineticTrace`, `TimeResolvedMatrix`). `TASpectrum`/`GatedSpectrum` are about to be deleted and no caller needs their unitful accessors, so they get no `_metadata` method.

- [ ] **Step 1: Write the failing test** — append inside the token testset:

```julia
    @testset "edge accessors and opt-in guess_units!" begin
        s = Spectrum([1500.0, 1600.0], [1.0, 2.0]; axis=:wavenumber, yquantity=:absorbance, yunit=:OD)
        @test xdata_unitful(s) == [1500.0, 1600.0] .* u"cm^-1"
        @test ydata_unitful(s) == [1.0, 2.0]          # OD -> NoUnits

        s_bare = Spectrum([1.0, 2.0], [3.0, 4.0])
        @test xdata_unitful(s_bare) == [1.0, 2.0]

        s_guess = Spectrum(collect(1500.0:1.0:1504.0), ones(5))
        @test xlabel(s_guess) == "x"
        guess_units!(s_guess)
        @test s_guess.metadata[:xquantity] == :wavenumber
        @test xlabel(s_guess) == "Wavenumber (cm⁻¹)"

        s_nm = Spectrum([500.0, 600.0], [1.0, 2.0])
        guess_units!(s_nm)
        @test s_nm.metadata[:xquantity] == :wavelength
        @test xlabel(s_nm) == "Wavelength (nm)"
    end
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (undefined).

- [ ] **Step 3: Implement** — append to the **end** of `src/types.jl`:

```julia
# =============================================================================
# Token edge accessors and opt-in unit guessing (metadata token contract §4/§6)
# =============================================================================

_metadata(s::Spectrum) = s.metadata
_metadata(t::KineticTrace) = t.metadata
_metadata(m::TimeResolvedMatrix) = m.metadata

"""
    xdata_unitful(d) ; ydata_unitful(d)

Return the x / signal data with their unit tokens (`:xunit` / `:yunit`) attached
via Unitful (§6). With no token the unit is `Unitful.NoUnits` and the plain
`Float64` vector is returned unchanged. Opt-in; core storage stays unitless.
"""
xdata_unitful(d::AbstractSpectroscopyData) =
    xdata(d) .* _unitful(Symbol(get(_metadata(d), :xunit, :dimensionless)))
ydata_unitful(d::AbstractSpectroscopyData) =
    ydata(d) .* _unitful(Symbol(get(_metadata(d), :yunit, :dimensionless)))

"""
    guess_units!(s::Spectrum) -> Spectrum

Opt-in heuristic: infer the x-axis quantity/unit from the data magnitude and
stamp `:xquantity`/`:xunit`. Nothing calls this automatically — labels never
guess on the user's behalf (the no-guess rule).
"""
function guess_units!(s::Spectrum)
    if _detect_spectral_unit(s.x) == "cm⁻¹"
        s.metadata[:xquantity] = :wavenumber
        get!(s.metadata, :xunit, :per_cm)
    else
        s.metadata[:xquantity] = :wavelength
        get!(s.metadata, :xunit, :nm)
    end
    return s
end
```

Export: `export xdata_unitful, ydata_unitful, guess_units!`

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl src/OpticalSpectroscopy.jl test/tokens.jl
git commit -m "feat(tokens): add unitful edge accessors and opt-in guess_units!"
```

---

## Phase 3 — `KineticTrace` & `TimeResolvedMatrix` labels (clean break)

### Task 8: `KineticTrace` labels + show (token `:xunit`, no legacy keys)

**Files:** Modify `src/types.jl` (`xlabel`/`ylabel` @172–173; `Base.show` two methods @192–214); `test/runtests.jl` (lines 64–65; the trace half of the "Metadata-driven labels" testset @3147–3156).

- [ ] **Step 1: Adjust the failing tests** —

In `test/runtests.jl` lines 64–65 (KineticTrace interface testset, no metadata):

```julia
        @test xlabel(trace) == "Time (ps)"
        @test ylabel(trace) == "Signal"
```

In the "Metadata-driven labels" testset, **split the shared `md`** so the trace uses tokens and the matrix keeps its own dict (the matrix half is migrated in Task 9). Replace lines 3147–3156:

```julia
    @testset "Metadata-driven labels" begin
        md_tr = Dict{Symbol,Any}(:xunit => :ns, :yquantity => :intensity, :yunit => :counts)
        tr = KineticTrace([0.0, 1.0], [1.0, 0.5]; metadata=md_tr)
        @test xlabel(tr) == "Time (ns)"
        @test ylabel(tr) == "Intensity (counts)"
        @test occursin("ns", sprint(show, tr))

        tr_default = KineticTrace([0.0, 1.0], [1.0, 0.5])
        @test xlabel(tr_default) == "Time (ps)"
        @test ylabel(tr_default) == "Signal"

        md = Dict{Symbol,Any}(:time_unit => "ns", :signal_label => "Counts")
```

(The trailing `md = …` line re-establishes the dict the matrix half at lines 3158+ still uses; Task 9 replaces it. The `occursin("ns", …)` show assertions at 3152/3162 pass after the show update below.)

Also, in the same testset, temporarily drop line 3164 to `"Time (ps)"` — once `KineticTrace` reads `:xunit`, the *old* `matrix[λ=…]` (which copies `:time_unit`, not `:xunit`) yields the `:ps` default. Task 10 flips it back to `"Time (ns)"`:

```julia
        @test xlabel(m[λ=505.0]) == "Time (ps)"
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`ylabel(trace)` still `"ΔA"`; `xlabel(tr)` still reads `:time_unit`).

- [ ] **Step 3: Implement** — in `src/types.jl`, replace the `KineticTrace` label methods (lines 172–173):

```julia
xlabel(t::KineticTrace) = _time_label(t.metadata, :xunit)
ylabel(t::KineticTrace) = _signal_label(t.metadata)
```

Replace `Base.show(io::IO, t::KineticTrace)` (lines 192–200) — read `:xunit`:

```julia
function Base.show(io::IO, t::KineticTrace)
    n = length(t.time)
    tu = (u = Symbol(get(t.metadata, :xunit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    t_range = "$(round(minimum(t.time), digits=2)) to $(round(maximum(t.time), digits=2)) $tu"
    src = source_file(t)
    print(io, "KineticTrace: $n points, $t_range")
    isempty(src) || print(io, " ($(src))")
end
```

Replace the MIME `Base.show` (lines 202–214):

```julia
function Base.show(io::IO, ::MIME"text/plain", t::KineticTrace)
    tu = (u = Symbol(get(t.metadata, :xunit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    println(io, "KineticTrace")
    println(io, "  Time points: $(length(t.time))")
    println(io, "  Time range:  $(round(minimum(t.time), digits=2)) to $(round(maximum(t.time), digits=2)) $tu")
    println(io, "  Wavelength:  $(isnan(t.wavelength) ? "unknown" : t.wavelength)")
    src = source_file(t)
    isempty(src) || println(io, "  File:        $src")
    haskey(t.metadata, :mode) && println(io, "  Mode:        $(t.metadata[:mode])")
end
```

(`_UNIT_DISPLAY[:ns] == "ns"`, so `occursin("ns", …)` holds. `_UNIT_DISPLAY` is from `tokens.jl`, available.)

Also update the `KineticTrace` docstring (lines 144–148) — replace the `:signal_label`/`:time_unit` mention with: `Signal/axis semantics live in token metadata (:yquantity/:yunit for the signal; :xunit for the time axis).`

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl test/runtests.jl
git commit -m "feat(kinetictrace): token-derived labels and show, no legacy keys"
```

---

### Task 9: `TimeResolvedMatrix` labels + show (clean break, interim time-axis token)

**Files:** Modify `src/types.jl` (`xlabel`/`ylabel`/`zlabel` @954–957; both `Base.show` @968–989; docstring @876–898); `test/runtests.jl` (lines 111–118; matrix half of "Metadata-driven labels" @3158–3164).

> Leave `_detect_spectral_unit` (@960) and `_detect_wavelength_unit` (@966) **defined** — `_detect_spectral_unit` is still used by `guess_units!`; `_detect_wavelength_unit` is still called by `fitting.jl:590` until Task 15.

- [ ] **Step 1: Adjust the failing tests** —

Lines 111–118 (TimeResolvedMatrix interface testset, no metadata → honest floor):

```julia
        @test xlabel(matrix) == "x"
        @test ylabel(matrix) == "Time (ps)"
        @test zlabel(matrix) == "Signal"
        @test is_matrix(matrix) == true

        # tokens drive the spectral + signal labels
        matrix_tok = TimeResolvedMatrix(time, wavelength, data; metadata=Dict{Symbol,Any}(
            :xquantity => :wavenumber, :xunit => :per_cm,
            :yquantity => :delta_absorbance, :yunit => :mOD, :time_unit => :ns))
        @test xlabel(matrix_tok) == "Wavenumber (cm⁻¹)"
        @test ylabel(matrix_tok) == "Time (ns)"
        @test zlabel(matrix_tok) == "ΔA (mOD)"
```

In the "Metadata-driven labels" testset, replace the matrix half (lines 3158–3164, including the `md = …` line Task 8 left in place):

```julia
        md = Dict{Symbol,Any}(:time_unit => :ns, :yquantity => :intensity, :yunit => :counts)
        m = TimeResolvedMatrix([0.0, 1.0], [500.0, 510.0], [1.0 2.0; 3.0 4.0]; metadata=md)
        @test ylabel(m) == "Time (ns)"
        @test zlabel(m) == "Intensity (counts)"
        @test occursin("ns", sprint(show, m))
        @test occursin("ns", sprint(show, MIME("text/plain"), tr))
        @test occursin("ns", sprint(show, MIME("text/plain"), m))
        @test xlabel(m[λ=505.0]) == "Time (ps)"
```

(Line 3164 stays `"Time (ps)"` — set by Task 8, flipped to `"Time (ns)"` by Task 10 once `_trace_metadata` stamps `:xunit` on the kinetic slice. The matrix half's old `md` from Task 8 is fully replaced here by the token dict.)

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`xlabel(matrix)` still `"Wavelength (nm)"`).

- [ ] **Step 3: Implement** — in `src/types.jl`, replace the matrix label methods (lines 954–957):

```julia
xlabel(m::TimeResolvedMatrix) = _spectral_xlabel(m.metadata)
ylabel(m::TimeResolvedMatrix) = _time_label(m.metadata, :time_unit)
zlabel(m::TimeResolvedMatrix) = _signal_label(m.metadata)
```

Replace `Base.show(io::IO, m::TimeResolvedMatrix)` (lines 968–975):

```julia
function Base.show(io::IO, m::TimeResolvedMatrix)
    n_time, n_wl = size(m.data)
    tu = (u = Symbol(get(m.metadata, :time_unit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    t_range = "$(round(minimum(m.time), digits=2)) to $(round(maximum(m.time), digits=2)) $tu"
    wl_range = "$(round(minimum(m.wavelength), digits=1)) to $(round(maximum(m.wavelength), digits=1))"
    print(io, "TimeResolvedMatrix: $n_time × $n_wl ($t_range, $wl_range)")
end
```

Replace the MIME `Base.show` (lines 977–989):

```julia
function Base.show(io::IO, ::MIME"text/plain", m::TimeResolvedMatrix)
    n_time, n_wl = size(m.data)
    tu = (u = Symbol(get(m.metadata, :time_unit, :ps)); get(_UNIT_DISPLAY, u, String(u)))
    println(io, "TimeResolvedMatrix")
    println(io, "  Time points:   $n_time ($(round(minimum(m.time), digits=2)) to $(round(maximum(m.time), digits=2)) $tu)")
    println(io, "  Wavelengths:   $n_wl ($(round(minimum(m.wavelength), digits=1)) to $(round(maximum(m.wavelength), digits=1)))")
    println(io, "  Data range:    $(round(minimum(m.data), sigdigits=3)) to $(round(maximum(m.data), sigdigits=3))")
    haskey(m.metadata, :source) && println(io, "  Source:        $(m.metadata[:source])")
end
```

Update the matrix docstring (lines 884–898): replace `:signal_label`, `:time_unit` mentions with the interim token note — `Spectral axis tokens :xquantity/:xunit; signal tokens :yquantity/:yunit; time-axis unit :time_unit. matrix[t=…] returns a Spectrum; matrix[λ=…] returns a KineticTrace.`

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl test/runtests.jl
git commit -m "feat(matrix): token-derived labels and show, interim time-axis token"
```

---

## Phase 4 — Collapse `TASpectrum` → `Spectrum`

### Task 10: Reroute matrix indexing (`[t=…]` → `Spectrum`, `[λ=…]` token metadata)

**Files:** Modify `src/types.jl` (indexing block @1000–1033; add `_trace_metadata` helper); `test/runtests.jl` (indexing testset @187–191; flip line 3164 to `"Time (ns)"`).

- [ ] **Step 1: Adjust the failing tests** —

Lines 187–191 (TimeResolvedMatrix indexing testset):

```julia
        # Extract Spectrum at time
        spec = matrix[t=2.0]
        @test spec isa Spectrum
        @test length(xdata(spec)) == 4
        @test spec.metadata[:time_delay] ≈ 2.0
```

Flip the deferred assertion from Task 9 (line 3164) back to its real value:

```julia
        @test xlabel(m[λ=505.0]) == "Time (ns)"
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`spec isa Spectrum` false — still `TASpectrum`; `m[λ=505.0]` label still `"Time (ps)"`).

- [ ] **Step 3: Implement** — in `src/types.jl`, add a helper just above the indexing block (before `Base.getindex(m::TimeResolvedMatrix; …)`, ~line 1000):

```julia
# A kinetic (time-axis) slice inherits matrix metadata but its x-axis is time:
# remap the matrix's time-axis unit (:time_unit) to the trace's :xunit and drop
# the matrix's spectral x-tokens (which describe wavelength, not the trace's time
# axis). Interim until the axisN refactor (§10 #3).
function _trace_metadata(m::TimeResolvedMatrix)
    md = copy(m.metadata)
    md[:xunit] = Symbol(get(m.metadata, :time_unit, :ps))
    delete!(md, :xquantity)
    return md
end
```

Replace the whole `Base.getindex(m::TimeResolvedMatrix; λ=nothing, t=nothing)` body (lines 1000–1033):

```julia
function Base.getindex(m::TimeResolvedMatrix; λ=nothing, t=nothing)
    if !isnothing(λ) && isnothing(t)
        idx = _find_nearest_idx(m.wavelength, λ)
        actual_λ = m.wavelength[idx]
        sig = m.data[:, idx]

        md = _trace_metadata(m)
        md[:extracted_from] = get(m.metadata, :source, "TimeResolvedMatrix")
        md[:requested_wavelength] = λ
        md[:actual_wavelength] = actual_λ
        md[:wavelength_index] = idx

        return KineticTrace(m.time, sig, actual_λ, md)

    elseif !isnothing(t) && isnothing(λ)
        idx = _find_nearest_idx(m.time, t)
        actual_t = m.time[idx]
        sig = m.data[idx, :]

        # Spectral slice: inherit the matrix's spectral (:xquantity/:xunit) and
        # signal (:yquantity/:yunit) tokens directly; record the fixed delay.
        md = copy(m.metadata)
        md[:extracted_from] = get(m.metadata, :source, "TimeResolvedMatrix")
        md[:time_index] = idx
        md[:time_delay] = actual_t
        haskey(m.metadata, :time_unit) &&
            (md[:time_delay_unit] = Symbol(m.metadata[:time_unit]))

        return Spectrum(m.wavelength, sig, md)

    elseif !isnothing(λ) && !isnothing(t)
        error("Cannot specify both λ and t. Use matrix[λ=...] or matrix[t=...]")
    else
        error("Must specify either λ or t for indexing. Use matrix[λ=...] or matrix[t=...]")
    end
end
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/types.jl test/runtests.jl
git commit -m "feat(matrix): matrix[t=…] returns Spectrum; token-aware slice metadata"
```

---

### Task 11: Reroute the TA fitting path to `Spectrum`

**Files:** Modify `src/fitting.jl` (`fit_ta_spectrum` @884–898; `predict(::TASpectrumFit, ::TASpectrum)` @970; docstring @844); `test/runtests.jl` (constructions at 969, 1002, 1018, 1038, 1160–1161).

- [ ] **Step 1: Adjust the failing tests** — replace each `TASpectrum(` data-construction feeding `fit_ta_spectrum`/`subtract_spectrum` with `Spectrum(` (the fit ignores tokens; `TASpectrumFit` result and its accessors are unchanged):
  - Line 969: `spec = Spectrum(ν, signal)`
  - Line 1002: `spec = Spectrum(ν, signal)`
  - Line 1018: `spec = Spectrum(ν, signal)`
  - Line 1038: `spec = Spectrum(ν, signal)`
  - Lines 1160–1161: `spec1 = Spectrum(ν, y1)` / `spec2 = Spectrum(ν, y2)`

(Lines 982–983 `xdata(result)`/`wavenumber(result)` on the `TASpectrumFit` are unchanged and still pass.)

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`fit_ta_spectrum(::Spectrum)` no method — still typed to `TASpectrum`).

- [ ] **Step 3: Implement** — in `src/fitting.jl`:

Change the signature (line 884) and the two data reads (lines 892–897) to use the generic accessors:

```julia
function fit_ta_spectrum(spec::Spectrum;
                         peaks::AbstractVector=[:esa, :gsb],
                         model::Function=gaussian,
                         region=nothing,
                         fit_offset::Bool=false,
                         p0::Union{Nothing, AbstractVector}=nothing)

    xv = xdata(spec)
    yv = ydata(spec)
    if isnothing(region)
        ν = collect(Float64, xv)
        y = collect(Float64, yv)
    else
        mask = (xv .>= region[1]) .& (xv .<= region[2])
        ν = collect(Float64, xv[mask])
        y = collect(Float64, yv[mask])
    end
```

(Leave the rest of `fit_ta_spectrum` unchanged.)

Change the docstring signature (line 844) to `fit_ta_spectrum(spec::Spectrum; kwargs...) -> TASpectrumFit`.

Change the `predict` convenience (line 970):

```julia
predict(fit::TASpectrumFit, spec::Spectrum) = predict(fit, xdata(spec))
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitting.jl test/runtests.jl
git commit -m "feat(fitting): fit_ta_spectrum accepts Spectrum"
```

---

### Task 12: Delete the `TASpectrum` data type

**Files:** Modify `src/types.jl` (delete struct + methods @260–332; `xdata` docstring ref @49–50); `src/OpticalSpectroscopy.jl` (drop `TASpectrum` export); `test/runtests.jl` (type-hierarchy @20; constructor-validation @36–38; interface testset @69–78; extended-interface @135–140; semantic-accessors testset @159–163; generic-family @326–327; Makie @3130).

- [ ] **Step 1: Remove/convert the TASpectrum-specific tests** —

  - Line 20: delete `@test TASpectrum <: AbstractSpectroscopyData`.
  - Lines 36–38 (constructor shape validation): delete the three `TASpectrum` lines (Spectrum/KineticTrace/matrix shape validation already cover this).
  - Lines 69–78: delete the entire `@testset "AbstractSpectroscopyData interface - TASpectrum"` block (Spectrum's interface testset covers it).
  - Lines 135–140 (extended interface): replace the `# TASpectrum` block with a `Spectrum` equivalent:

```julia
        # Spectrum
        spec = Spectrum([2000.0, 2050.0, 2100.0], [0.1, 0.5, 0.3]; filename="spec.lvm")
        @test source_file(spec) == "spec.lvm"
        @test npoints(spec) == 3
        @test title(spec) == "spec.lvm"
```

  - Lines 159–163: delete the entire `@testset "Semantic accessors - TASpectrum"` block (the `wavenumber`/`signal` field accessors no longer exist on a data type).
  - Lines 326–327 (generic-family block): delete the two `ta = TASpectrum(...)` / `band_area(ta, …)` lines (the `s = Spectrum(...)` lines above already exercise the same generic path).
  - Line 3130 (Makie): `spec = Spectrum(x, y)`.

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (the deleted type is still referenced by remaining src, OR — if tests compile — the struct still exists; this step confirms the test file no longer constructs `TASpectrum`). Run `grep -n 'TASpectrum[^F]' test/runtests.jl` → expect no data-construction hits (only `TASpectrumFit`).

- [ ] **Step 3: Implement** — in `src/types.jl`, delete the `TASpectrum` data type and all its methods: the docstring + struct + constructors + interface + accessors + `show` methods (lines 260–332, i.e. from the `"""\n    TASpectrum <: AbstractSpectroscopyData` docstring through the end of the MIME `show`). **Keep** everything from `TAPeak` (line 334) onward — `TAPeak`, `TASpectrumFit`, `anharmonicity`, `wavenumber(::TASpectrumFit)` all stay.

Fix the dangling `xdata` docstring reference (lines 49–50): change "wavenumber for [`TASpectrum`](@ref)" to "the spectral axis for a [`Spectrum`](@ref) slice".

In `src/OpticalSpectroscopy.jl` line 47, drop `TASpectrum`:

```julia
export Spectrum, KineticTrace, TimeResolvedMatrix, GatedSpectrum, SweepData
```

- [ ] **Step 4: Run to verify it passes** — `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20` — Expected: PASS (Aqua confirms no undefined `TASpectrum` export).

- [ ] **Step 5: Commit**

```bash
git add src/types.jl src/OpticalSpectroscopy.jl test/runtests.jl
git commit -m "refactor: remove TASpectrum data type (absorbed into Spectrum)"
```

---

## Phase 5 — Collapse `GatedSpectrum` → `Spectrum`

### Task 13: Reroute `spectral_slice` / `integrate_time` (+ `kinetic_trace` metadata)

**Files:** Modify `src/timeresolved.jl` (`kinetic_trace` @16–23, `spectral_slice` @33–39, `integrate_time` @47–60); `test/runtests.jl` (slices testset @3197–3248; generic-family `g` @319–321; spectral-math `g` @441–444).

- [ ] **Step 1: Adjust the failing tests** —

Replace the slices testset (lines 3197–3248). Note the matrix metadata now uses tokens, slices return `Spectrum`, and `t_range` becomes `:gate_start`/`:gate_end`:

```julia
    @testset "TimeResolvedMatrix slices" begin
        t = [0.0, 1.0, 2.0]
        wl = [500.0, 510.0, 520.0]
        data = [1.0 2.0 3.0;
                4.0 5.0 6.0;
                7.0 8.0 9.0]   # rows = time, cols = wavelength
        md = Dict{Symbol,Any}(:yquantity => :intensity, :yunit => :counts, :time_unit => :ns)
        m = TimeResolvedMatrix(t, wl, data; metadata=md)

        # nearest single column
        tr = kinetic_trace(m; wavelength=511.0)
        @test tr isa KineticTrace
        @test tr.signal == [2.0, 5.0, 8.0]
        @test tr.wavelength == 510.0
        @test tr.metadata[:yquantity] == :intensity      # signal token inherited
        @test xlabel(tr) == "Time (ns)"                  # :time_unit remapped to :xunit

        # band mean over all three columns (510 ± 10 → [500, 520])
        tr_band = kinetic_trace(m; wavelength=510.0, band=20.0)
        @test tr_band.signal == [2.0, 5.0, 8.0]
        @test tr_band.wavelength == 510.0
        @test tr_band.metadata[:band] == 20.0
        @test !haskey(tr.metadata, :band)
        @test integrate_time(m).metadata[:yquantity] == :intensity
        @test_throws ArgumentError kinetic_trace(m; wavelength=NaN)

        # empty band falls back to nearest column
        tr_fb = kinetic_trace(m; wavelength=505.0, band=2.0)
        @test tr_fb.signal == [1.0, 4.0, 7.0]
        @test tr_fb.wavelength == 500.0

        # nearest single row → Spectrum with gate tokens
        sp = spectral_slice(m; time=1.2)
        @test sp isa Spectrum
        @test sp.y == [4.0, 5.0, 6.0]
        @test sp.metadata[:gate_start] == 1.0
        @test sp.metadata[:gate_end] == 1.0
        @test sp.metadata[:gate_unit] == :ns
        @test zlabel(m) == "Intensity (counts)"

        # gated mean over rows 2:3
        sp_win = spectral_slice(m; time=1.5, window=2.0)
        @test sp_win.y == [5.5, 6.5, 7.5]
        @test sp_win.metadata[:gate_start] == 1.0
        @test sp_win.metadata[:gate_end] == 2.0

        # time-integrated spectrum (sum)
        g = integrate_time(m)
        @test g isa Spectrum
        @test g.y == [12.0, 15.0, 18.0]
        @test g.metadata[:gate_start] == 0.0
        @test g.metadata[:gate_end] == 2.0

        g2 = integrate_time(m; t_range=(1.0, 2.0))
        @test g2.y == [11.0, 13.0, 15.0]
        @test g2.metadata[:gate_start] == 1.0
        @test g2.metadata[:gate_end] == 2.0
        @test_throws ArgumentError integrate_time(m; t_range=(10.0, 20.0))
```

(Keep the testset's tail beyond line 3248 — the `:source`/`:filename` KineticTrace check — as-is.)

Generic-family block, line 319 (`g = GatedSpectrum(x, y)` → `Spectrum`):

```julia
        g = Spectrum(x, y)
```

Spectral-math block, line 441 (`g = GatedSpectrum(x, yb)` → `Spectrum`):

```julia
        g = Spectrum(x, yb)
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL (`sp isa Spectrum` false — still `GatedSpectrum`).

- [ ] **Step 3: Implement** — in `src/timeresolved.jl`:

`kinetic_trace` (lines 16–23) — use `_trace_metadata`:

```julia
function kinetic_trace(m::TimeResolvedMatrix; wavelength::Real, band::Real=0.0)
    cols = _axis_window(m.wavelength, wavelength, band)
    sig = vec(mean(view(m.data, :, cols), dims=2))
    md = _trace_metadata(m)
    band > 0 && (md[:band] = float(band))
    return KineticTrace(copy(m.time), sig;
                        wavelength=mean(view(m.wavelength, cols)), metadata=md)
end
```

`spectral_slice` (lines 33–39) — return `Spectrum` with gate tokens:

```julia
function spectral_slice(m::TimeResolvedMatrix; time::Real, window::Real=0.0)
    rows = _axis_window(m.time, time, window)
    sig = vec(mean(view(m.data, rows, :), dims=1))
    t_lo, t_hi = extrema(view(m.time, rows))
    return Spectrum(copy(m.wavelength), sig, _gated_metadata(m, t_lo, t_hi))
end
```

`integrate_time` (lines 47–60) — return `Spectrum`:

```julia
function integrate_time(m::TimeResolvedMatrix; t_range::Union{Nothing,NTuple{2,Real}}=nothing)
    rows = if isnothing(t_range)
        eachindex(m.time)
    else
        found = findall(t -> t_range[1] <= t <= t_range[2], m.time)
        isempty(found) && throw(ArgumentError(
            "integrate_time: no time points inside t_range = $t_range"))
        found
    end
    sig = vec(sum(view(m.data, rows, :), dims=1))
    t_lo, t_hi = extrema(view(m.time, rows))
    return Spectrum(copy(m.wavelength), sig, _gated_metadata(m, t_lo, t_hi))
end

# Spectral slice inherits the matrix's spectral + signal tokens; the time window
# becomes flat :gate_start / :gate_end (+ :gate_unit) tokens (no compound values).
function _gated_metadata(m::TimeResolvedMatrix, t_lo::Real, t_hi::Real)
    md = copy(m.metadata)
    md[:gate_start] = float(t_lo)
    md[:gate_end] = float(t_hi)
    haskey(m.metadata, :time_unit) && (md[:gate_unit] = Symbol(m.metadata[:time_unit]))
    return md
end
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/timeresolved.jl test/runtests.jl
git commit -m "feat(timeresolved): slices return Spectrum with gate tokens"
```

---

### Task 14: Delete the `GatedSpectrum` data type

**Files:** Modify `src/types.jl` (delete struct + methods @1036–1116); `src/OpticalSpectroscopy.jl` (drop `GatedSpectrum` export); `test/runtests.jl` (delete the `@testset "GatedSpectrum"` @3167–3194).

- [ ] **Step 1: Remove the GatedSpectrum-specific test** — delete the entire `@testset "GatedSpectrum"` block (lines 3167–3194). Its behavior (construction, gate window, source) is now covered by Task 13's slice tests and the `Spectrum` testsets.

- [ ] **Step 2: Run to verify it fails** — `grep -n 'GatedSpectrum' test/runtests.jl` → expect no hits; then the suite must still load. (At this point the struct still exists but is unused.)

- [ ] **Step 3: Implement** — in `src/types.jl`, delete the entire `GatedSpectrum` section: the `# ===…` banner box around `# GatedSpectrum` (originally ~line 1035) through the end of its MIME `show` method (originally ~line 1116) — docstring + struct + constructors + interface + accessors + both `show` methods, plus the banner. Stop before the next `# ===…` banner (the `# Spectrum` section). (Line numbers have drifted from earlier edits; anchor on the `# GatedSpectrum` banner and the `GatedSpectrum` struct name.)

In `src/OpticalSpectroscopy.jl` line 47, drop `GatedSpectrum`:

```julia
export Spectrum, KineticTrace, TimeResolvedMatrix, SweepData
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (Aqua confirms no undefined `GatedSpectrum` export).

- [ ] **Step 5: Commit**

```bash
git add src/types.jl src/OpticalSpectroscopy.jl test/runtests.jl
git commit -m "refactor: remove GatedSpectrum data type (absorbed into Spectrum)"
```

---

## Phase 6 — Fitting de-guess, gate, and docs

### Task 15: De-guess `fitting.jl` DAS labels; delete the dead heuristic wrapper

**Files:** Modify `src/fitting.jl` (lines 590–591); `src/types.jl` (delete `_detect_wavelength_unit` @966).

- [ ] **Step 1: Verify no test asserts the heuristic-built labels**

Run: `grep -nE '"[0-9.]+ (nm|cm)' test/runtests.jl` — Expected: no matches near the global-fit testset.

- [ ] **Step 2: Implement** — two edits.

In `src/fitting.jl`, replace lines 590–591:

```julia
    labels = [string(round(wl, digits=1)) for wl in actual_wavelengths]
```

In `src/types.jl`, delete the now-dead wrapper (its only caller is the line just changed):

```julia
_detect_wavelength_unit(m::TimeResolvedMatrix) = _detect_spectral_unit(m.wavelength)
```

- [ ] **Step 3: Confirm the heuristic is fully demoted**

Run: `grep -rn '_detect_spectral_unit\|_detect_wavelength_unit' src/`
Expected: `_detect_wavelength_unit` gone; `_detect_spectral_unit` appears only at its definition (`types.jl`) and inside `guess_units!`.

- [ ] **Step 4: Run the full suite** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fitting.jl src/types.jl
git commit -m "feat(fitting): drop magnitude-guessed trace labels; remove dead wrapper"
```

---

### Task 16: Full-suite gate + docs sync

**Files:** Modify `CLAUDE.md` (Type Hierarchy + "All types implement"); `docs/superpowers/2026-06-15-metadata-token-contract.md` (status).

- [ ] **Step 1: Full suite, clean** — `julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30` — Expected: PASS (Aqua, token contract, and all migrated testsets green).

- [ ] **Step 2: Confirm the no-guess promise end-to-end**

```bash
julia --project=. -e 'using OpticalSpectroscopy; s = Spectrum([1500.0,1501.0,1502.0],[1.0,2.0,1.5]); println(xlabel(s), " | ", ylabel(s)); println(sprint(show, s))'
```
Expected:
```
x | Signal
Spectrum: 3 points, 1500.0 to 1502.0
```

- [ ] **Step 3: Confirm the collapse end-to-end**

```bash
julia --project=. -e '
using OpticalSpectroscopy
m = TimeResolvedMatrix([0.0,1.0,2.0],[600.0,610.0,620.0], rand(3,3);
        metadata=Dict{Symbol,Any}(:xquantity=>:wavelength,:xunit=>:nm,
        :yquantity=>:delta_absorbance,:yunit=>:mOD,:time_unit=>:ps))
s = m[t=1.0]; println(typeof(s), " | ", xlabel(s), " / ", ylabel(s), " | delay=", s.metadata[:time_delay])
g = integrate_time(m); println(typeof(g), " | gate=", g.metadata[:gate_start], "-", g.metadata[:gate_end])'
```
Expected (types are `Spectrum`, labels token-derived):
```
Spectrum | Wavelength (nm) / ΔA (mOD) | delay=1.0
Spectrum | gate=0.0-2.0
```

- [ ] **Step 4: Update `CLAUDE.md`** — in the Type Hierarchy block, remove the `TASpectrum` and `GatedSpectrum` rows and note that TA spectra / gated spectra are now `Spectrum` with tokens:

```
AbstractSpectroscopyData (root interface)
├── Spectrum            (generic 1D: signal vs spectral axis; also holds TA
│                        slices and gated spectra via metadata tokens)
├── KineticTrace        (kinetics: signal vs time at fixed wavelength)
├── TimeResolvedMatrix  (2D: time × wavelength heatmap)
└── PLMap               (2D: spatial PL/Raman map)
```

Also update the "All types implement: …" line if it enumerates the removed types, and the docstring sentence in the type-hierarchy notes.

- [ ] **Step 5: Update the contract doc status** — in `docs/superpowers/2026-06-15-metadata-token-contract.md`, change the `Status:` line (line 3):

```markdown
Status: **Implemented** (OpticalSpectroscopy.jl, 2026-06-16) — token vocabulary,
clean-break no-guess labels, and the TASpectrum/GatedSpectrum collapse landed via
docs/superpowers/plans/2026-06-16-metadata-token-contract-foundation.md. Remaining
follow-on: matrix axisN refactor (§10 #3), QPSScanFormat HDF5 codec, loader stamping.
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md docs/superpowers/2026-06-15-metadata-token-contract.md
git commit -m "docs: token contract + type collapse landed; sync hierarchy notes"
```

---

## Self-review notes (for the executing engineer)

- **No legacy bridge.** Labels read only `:xlabel`/`:ylabel` literals, the `(quantity, unit)` tokens, or the honest floor. `:signal_label` is gone; `:time_unit` survives **only** as the `TimeResolvedMatrix` time-axis unit token (a `Symbol`), and `KineticTrace` uses `:xunit` for time. The magnitude heuristic lives only inside opt-in `guess_units!`.
- **The collapse is real deletion.** `TASpectrum` and `GatedSpectrum` structs are removed; only the fit-**result** machinery (`TASpectrumFit`, `TAPeak`, `anharmonicity`, `wavenumber(::TASpectrumFit)`) survives. Producers return `Spectrum`; `time_delay` → `:time_delay`(+`_unit`); `t_range` → `:gate_start`/`:gate_end`(+`:gate_unit`).
- **Interim matrix tokens, not §10 #3.** The matrix keeps its `time`/`wavelength` fields and maps `:xquantity`/`:xunit`=spectral, `:yquantity`/`:yunit`=signal, `:time_unit`=time. This makes spectral slices inherit `(x, y)` by a plain copy and kinetic slices remap `:time_unit→:xunit` via `_trace_metadata`. The structural axisN rename is a separate plan.
- **Ordering traps that keep gates green:** (1) Task 9 temporarily sets the `m[λ=505.0]` label assertion to `"Time (ps)"`; Task 10 flips it to `"Time (ns)"` once `_trace_metadata` exists. (2) `_detect_spectral_unit` is never deleted (guess_units! needs it); only `_detect_wavelength_unit` is, and only in Task 15 after `fitting.jl` stops calling it. (3) Every export is added/removed in the same task as its method, for Aqua.
- **Line numbers are hints, anchors are truth.** All line numbers are from the *original* files; each task edits incrementally, so numbers drift (especially in `types.jl` and `test/runtests.jl`, which several tasks touch). Locate each edit by its code anchor — the quoted old code, the function/struct name, or the testset header — not by absolute line number.
- **Test command:** `julia --project=. -e 'using Pkg; Pkg.test()'` from the project root; first run precompiles.
