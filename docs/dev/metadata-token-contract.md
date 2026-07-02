# Metadata token contract

Status: **Implemented** (OpticalSpectroscopy.jl, 2026-06-16) — token vocabulary,
clean-break no-guess labels, and the TASpectrum/GatedSpectrum collapse landed 2026-06-16. Remaining
follow-on: matrix axisN refactor (§10 #3), QPSScanFormat HDF5 codec, loader stamping.

This is the cross-cutting contract the whole re-architecture leans on. Once
`Spectrum` absorbs `TASpectrum`/`GatedSpectrum` and `TimeResolvedMatrix`
generalizes its axes, the *only* thing distinguishing one spectrum from another
is its reserved metadata. That metadata therefore stops being a free-form
junk-drawer and becomes a typed, validated, round-trippable contract.

Validated against real datasets in `QPSTools.jl/data` (CCD broadband TA,
MIR pump–probe, streak PL, FTIR with six measurement kinds, Raman, UV-Vis, XRD,
dense PL maps, sparse located-spectra maps).

---

## 1. Problem this replaces

Today the same semantic information is encoded five inconsistent ways, plus two
heuristics that real files defeat:

| Current mechanism | Where | Problem |
|---|---|---|
| `:xlabel`, `:ylabel` (full strings) | `Spectrum` (`types.jl:1182`, `1187`) | human strings, not semantics; can't convert or validate |
| `:signal_label` (string) | `KineticTrace` (`172`), `GatedSpectrum` (`1075`) | duplicate concept, different key |
| `:time_unit` (string `"ps"`) | `KineticTrace` (`172`), `TimeResolvedMatrix` (`956`) | only the time axis; ad-hoc |
| `:filename` vs `:source` | split across types (`174`, `294`, `929`, `1188`) | provenance key isn't canonical |
| `_detect_spectral_unit(x)` magnitude heuristic | `types.jl:960`, used in 6 places | guesses cm⁻¹ vs nm from value size — wrong for CCD TA (nm) and any overlap range |
| `xlabel(::TASpectrum) = "Wavenumber (cm⁻¹)"` hardcoded | `types.jl:292` | **`CCD/ta_matrix.lvm` has a nm axis** — this label is a live mislabel |

The contract collapses all of this into one reserved token namespace from which
labels are *derived*, never stored as prose.

---

## 2. Reserved token namespace

Reserved keys live in the `Dict{Symbol,Any}` metadata. Everything else in the
dict is free-form passthrough (sample id, operator, instrument settings, …).
Reserved keys are the only ones with defined meaning, validation, and
label-derivation behavior.

### Axis descriptors

Every axis is described by a **(quantity, unit)** pair. An axis's *quantity* is
the physical thing measured; its *unit* is how it's scaled. They are separate
because real data needs both: Raman shift and IR wavenumber are **both** `cm⁻¹`
but are different quantities with different labels.

| Key | Type | Meaning | Example |
|---|---|---|---|
| `:xquantity` | `Symbol` | physical quantity of the independent axis | `:wavenumber` |
| `:xunit` | `Symbol` | unit of the independent axis | `:per_cm` |
| `:yquantity` | `Symbol` | physical quantity of the signal | `:absorbance` |
| `:yunit` | `Symbol` | unit of the signal (may be `:dimensionless`) | `:OD` |

For matrices/cubes the same pairs apply per axis, named by role rather than x/y:

| Key | Used by | Meaning |
|---|---|---|
| `:axis1_quantity`, `:axis1_unit` | `TimeResolvedMatrix`, `PLMap` | first stored axis |
| `:axis2_quantity`, `:axis2_unit` | `TimeResolvedMatrix`, `PLMap` | second stored axis |
| `:axis3_quantity`, `:axis3_unit` | `PLMap` | third (spectral) axis of the cube |
| `:signal_quantity`, `:signal_unit` | matrices/cubes | the z value |

This is the axis-token generalization Cluster C needs: `TimeResolvedMatrix`
stops hardcoding `time`/`wavelength` fields and instead carries axis descriptors,
so a future 2D-IR or EEM matrix (excitation-λ × emission-λ) is the same type
with different tokens.

### Scalar coordinate & provenance tokens

| Key | Type | Meaning |
|---|---|---|
| `:technique` | `Symbol` | acquisition technique (see §3) |
| `:time_delay` | `Float64` | fixed delay of a spectrum sliced from a TR matrix |
| `:time_delay_unit` | `Symbol` | unit of `:time_delay` (e.g. `:ps`) |
| `:fixed_coordinate` | `Float64` | the held-constant axis value of a 1D slice (e.g. the λ of a `KineticTrace`) |
| `:fixed_coordinate_unit` | `Symbol` | its unit |
| `:position` | `Vector{Float64}` | `[x, y]` stage coordinate of a map pixel/point |
| `:position_unit` | `Symbol` | e.g. `:um` |
| `:source` | `String` | **canonical** provenance path (accessor still accepts `:filename` as alias) |

**Convention — no compound values.** Every "value + unit" datum is stored as two
flat keys (`:time_delay` + `:time_delay_unit`), never as a `Tuple` or a Unitful
`Quantity`. This keeps every reserved value a *simple* type (number, `Symbol`,
`String`, `Bool`, or a homogeneous array), which is what makes §7 round-tripping
trivial. Unitful quantities are reconstructed only at the **edge accessors**
(§6), not stored.

---

## 3. Controlled vocabularies

Each value below is grounded in a real file under `QPSTools.jl/data`. The
vocabulary is open for extension but closed for validation: an unknown value
warns and falls back, it does not silently pass.

### `:technique`

| Token | Source dataset |
|---|---|
| `:ta` | `CCD/ta_matrix.lvm`, `MIRpumpprobe/*` |
| `:ftir` | `ftir/measurement_kinds/*` |
| `:raman` | `raman/ZIF62_*.csv` |
| `:uvvis` | `uvvis/format_example.txt` |
| `:pl` | `PL/15K.img`, `PLmap/*` |
| `:xrd` | `xrd/ZIF62-*.txt` (recognized; out of optical scope) |

### Axis quantities (`:xquantity` / `:axisN_quantity`)

| Token | Canonical unit | Source dataset |
|---|---|---|
| `:wavelength` | `:nm` (or `:um`, `:angstrom`) | `uvvis/…`, `CCD/wavelength_axis.txt` (**nm**) |
| `:wavenumber` | `:per_cm` | `ftir/measurement_kinds/Abs.csv` |
| `:raman_shift` | `:per_cm` | `raman/ZIF62_crystal_1.csv` |
| `:opd` | `:points` / `:mm` | `ftir/measurement_kinds/interferrogram.csv` |
| `:two_theta` | `:degree` | `xrd/ZIF62-Zn_crystal.txt` |
| `:time` | `:fs` / `:ps` / `:ns` | `CCD/time_axis.txt`, `MIRpumpprobe/pp_kinetics_*.lvm` |
| `:energy` | `:eV` / `:meV` | (common; not in sample set) |
| `:position` | `:um` / `:mm` | `PLmap/*` (stage grid) |

### Signal quantities (`:yquantity` / `:signal_quantity`)

| Token | Typical unit | Source dataset |
|---|---|---|
| `:absorbance` | `:OD` | `ftir/.../Abs.csv`, `uvvis` (`YUNITS=ABSORBANCE`) |
| `:transmittance` | `:percent` | `ftir/.../T.csv` |
| `:reflectance` | `:percent` | `ftir/.../R.csv` |
| `:single_beam` | `:arb` | `ftir/.../sb.csv` |
| `:interferogram` | `:arb` | `ftir/.../interferrogram.csv` |
| `:delta_absorbance` | `:mOD` | `CCD/ta_matrix.lvm`, `MIRpumpprobe/*` (ΔA) |
| `:intensity` | `:counts` / `:arb` | `raman/*`, `PL/*`, `xrd/*` |

`:yquantity` is the token the **T↔A / Kubelka–Munk / Tauc** transforms dispatch
on — it is load-bearing, and it is the token missing from the original handoff.

### Units (`:xunit` / `:yunit` / `:*_unit`)

`:nm :um :angstrom :per_cm :fs :ps :ns :degree :eV :meV :points :mm :um`
`:counts :arb :OD :mOD :percent :fraction :dimensionless`

---

## 4. Label derivation and the no-guess rule

**Ratified 2026-06-15: bare row-column data loads with no units provided and no
units guessed.** A `Spectrum` asserts only what it was told.

One pure function owns all label text:

```julia
axis_label(quantity::Symbol, unit::Symbol) -> String
```

It maps `quantity` to a display name ("Wavenumber", "Raman shift", "Absorbance",
"ΔA", "2θ") and `unit` to a symbol ("cm⁻¹", "nm", "°", "fs", ""), composing
`"Name (sym)"` — or just `"Name"` when the unit is `:dimensionless`.

Labels resolve by a fixed precedence, and **never infer from the data**:

1. **Literal override** — a `:xlabel` / `:ylabel` string in metadata, used
   verbatim. The escape hatch for "I just want this label, leave the vocabulary
   alone."
2. **Token-derived** — if `:xquantity`/`:xunit` (or `:yquantity`/`:yunit`) are
   present, `axis_label(...)`.
3. **Honest generic floor** — otherwise a neutral default (`"x"` / `"Signal"`).
   Not pretty, never wrong.

There is no automatic step that guesses units from value magnitudes.
`_detect_spectral_unit` is **removed from every automatic call site** (`xlabel`,
`ylabel`, `show`, `title`, and the `TimeResolvedMatrix`/`GatedSpectrum`/`Spectrum`
paths at `types.jl:955`, `1074`, `1185`, `1199`, …). It survives only as an
explicitly-invoked opt-in, never called on the user's behalf:

```julia
guess_units!(s)   # user asks for it; nothing calls this automatically
```

### Minimal input path

The bare path takes no metadata and adds none:

```julia
Spectrum(x, y)        # two vectors → plots with axes "x" / "Signal"
Spectrum(M)           # an N×2 matrix, e.g. Spectrum(readdlm("homemade.txt"))
```

`Spectrum(M::AbstractMatrix)` splits a two-column matrix into `(x, y)` so plain
row-column exports go straight in with no intermediate unpacking and no unit
ceremony. When units *are* known, they stay one keyword away:

```julia
Spectrum(x, y; axis=:wavenumber)      # canonical unit auto-filled → "Wavenumber (cm⁻¹)"
Spectrum(x, y; xlabel="Time (s)")     # literal escape hatch, no vocabulary
```

Tokens are an on-ramp to unit-aware features (conversions, T↔A, validation),
never a gate in front of "load and look."

---

## 5. Required vs optional per type

| Type | Required tokens (for correct labels) | Optional |
|---|---|---|
| `Spectrum` | `:xquantity`, `:xunit`, `:yquantity` | `:yunit`, `:technique`, `:time_delay(+unit)`, `:source` |
| `KineticTrace` | `:yquantity` (x is time by definition) | `:xunit` (time unit, default `:ps`), `:fixed_coordinate(+unit)`, `:source` |
| `TimeResolvedMatrix` | `:axis1_*`, `:axis2_*`, `:signal_quantity` | `:technique`, `:source` |
| `PLMap` | `:axis1_*`, `:axis2_*`, `:axis3_*`, `:signal_quantity` | `:source` |

Missing required tokens → honest generic label (`"x"` / `"Signal"`), never a
guess and never an error (the raw-vector path must keep working). See §4.

---

## 6. Edge accessors (Unitful / Measurements bridge)

Core storage stays `Vector{Float64}` / `Matrix{Float64}`. Unitful appears only
in opt-in accessors that read the unit token and attach it:

```julia
xdata_unitful(s)  = xdata(s) .* _unitful(get(meta(s), :xunit, :dimensionless))
```

`_unitful(:per_cm) === u"cm^-1"`, `_unitful(:nm) === u"nm"`, … — a Symbol↔Unitful
bijection defined once. Same pattern already used in `units.jl`
(`wavenumber_to_wavelength` dual methods). Measurements.jl, if ever added,
bridges the same way (a parallel `:y_uncertainty` vector), never as the element
type.

---

## 7. HDF5 typed round-trip sub-protocol

Because §2 forbids compound values, the only Julia types a metadata dict can
hold are: `Bool`, `Int`, `Float64`, `String`, `Symbol`, `Vector{<:Real}`,
`Vector{String}`, `Vector{Symbol}`, and nested `Dict` (→ subgroup). HDF5 stores
all of these natively **except** it cannot tell a `Symbol` from a `String`, or a
`Vector{Symbol}` from a `Vector{String}`.

So the entire protocol is: store values natively, plus one type-tag registry
that records exactly which entries are symbol-typed.

```
metadata/                      (HDF5 group)
  ├─ technique        = "ta"            (String storage)
  ├─ xquantity        = "wavelength"
  ├─ xunit            = "nm"
  ├─ time_delay       = 1.0             (Float64, native)
  ├─ time_delay_unit  = "ps"
  ├─ position         = [12.0, 8.5]     (Float64[], native)
  └─ __jltypes        = "technique=Symbol;xquantity=Symbol;xunit=Symbol;
                          time_delay_unit=Symbol"   (one string attr)
```

Decode: read each entry; if its key appears in `__jltypes` with tag `Symbol`,
`Symbol(value)`; with tag `SymbolVector`, `Symbol.(value)`; otherwise leave as
the native type. Round-trip is lossless and the registry is tiny.

Nested `Dict` (e.g. a merged sample-registry block) → recurse into a subgroup
with its own `__jltypes`.

**Ownership:** this codec is vocabulary-agnostic — it round-trips Julia types,
not spectroscopy meaning — so it lives in **QPSScanFormat.jl** (which owns the
HDF5 schema and must stay analysis-dep-free). `encode_metadata`/`decode_metadata`
are duck-typed over any `AbstractDict{Symbol}`.

---

## 8. Validation facade & ownership

**OpticalSpectroscopy.jl owns the semantics:**
- the vocabulary constants of §3 (as `Set{Symbol}` per slot),
- `axis_label(quantity, unit)` (§4),
- `is_canonical(slot::Symbol, value::Symbol) -> Bool` and a `validate_tokens(meta)`
  that warns (not errors) on unknown values,
- `normalize_unit(s::AbstractString) -> Symbol` (e.g. `"NANOMETERS" → :nm`,
  `"1/CM" → :per_cm`, `"ABSORBANCE" → (:absorbance, :OD)`), shared by all loaders,
- the `_unitful` bridge (§6).

**Loaders (JASCOFiles / QPSTools / QPSScanFormat readers)** call `normalize_unit`
and stamp tokens. Disambiguation that needs technique context (e.g. `cm⁻¹` means
`:wavenumber` for `DATA TYPE=INFRARED SPECTRUM` but `:raman_shift` for
`RAMAN SPECTRUM`) happens here, since the loader knows the technique.

**QPSScanFormat.jl** owns only the generic typed codec of §7.

This keeps the three-package boundary intact: dumb codec ↓ in the schema layer,
semantic vocabulary in the analysis layer, instrument-string mapping in the
loaders.

---

## 9. Migration of existing keys

| Old | New |
|---|---|
| `:xlabel = "Wavenumber (cm⁻¹)"` | `:xquantity = :wavenumber`, `:xunit = :per_cm` |
| `:ylabel = "Signal"` / `:signal_label = "ΔA"` | `:yquantity` (+ `:yunit`) |
| `:time_unit = "ps"` | `:xunit = :ps` (trace) or `:axisN_unit = :ps` (matrix) |
| `:filename` / `:source` | `:source` canonical; accessor `get(md,:source, get(md,:filename,""))` |
| `_detect_spectral_unit(x)` call | token lookup; **no auto-guess** — heuristic removed from all automatic paths, survives only as opt-in `guess_units!` |
| `TASpectrum.wavenumber` field | `Spectrum.x` + `:xquantity`/`:xunit` tokens |

---

## 10. Open decisions to ratify

1. **Two-token axes `(quantity, unit)` vs single `:xunit`.** Proposed: two
   tokens, because Raman shift and IR wavenumber are both `cm⁻¹` and need
   different labels. Cost: more keys.
2. **Key names** `:xquantity` / `:yquantity` (proposed) vs `:quantity` /
   `:signal`. Symmetry with `:xunit` favors the former.
3. **`:axisN_*` role-naming for matrices** vs keeping named `time`/`wavelength`
   fields on `TimeResolvedMatrix` and only adding tokens. Proposed: role-naming,
   to make 2D-IR/EEM the same type.
4. **`Spectrum` hosting non-spectral 1D** (XRD 2θ, FTIR interferogram). Proposed:
   yes — structure is exact, `:xquantity` marks the truth, name "Spectrum"
   accepted as slightly loose within optical scope; XRD treated as out-of-scope
   reuse.
