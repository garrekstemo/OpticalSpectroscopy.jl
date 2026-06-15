# Type-system re-architecture — hand-off

- **Date:** 2026-06-15
- **Status:** Hand-off. Re-architecture **not started**. This is the seed document for a future session; no code in this scope has been written yet.
- **Goal (one paragraph):** Simplify and streamline the spectroscopy type system across the QPS ecosystem. The current hierarchy has grown organically into ~22 types spread over five repos with overlapping responsibilities (a generic `Spectrum` and a `GatedSpectrum` that are structurally near-identical; three lab-layer spectrum types — QPSTools `CavitySpectrum`, QPSLab `LabSpectrum`, and a soon-dropped CavitySpectroscopy `CavitySpectrum` — that all wrap a `JASCOSpectrum`), an `AbstractSpectroscopyData` interface whose contract is only 3-of-10 methods load-bearing and which Julia does not enforce, a metadata key-type split (`Dict{Symbol,Any}` internal vs `Dict{String,Any}` provenance) with three concrete violations, and axis/units/technique semantics encoded four incompatible ways with no single source of truth. The objective is a coherent, layered type system where provenance composes a generic `Spectrum` core, derived spectra have a real home, the interface contract is honest, and metadata/technique encoding is unified — all without breaking the HDF5/JSON files already on disk.

---

## HOW TO USE THIS DOC

**You (the next session) have no memory of the analysis that produced this.** Everything you need is below or in the referenced sibling docs. Do not re-derive the inventory from scratch; trust the `file:line` citations here and spot-check only what you intend to change.

**Workflow — use the `superpowers` ultracode flow, in order:**

1. **`superpowers:brainstorming`** FIRST. Resolve the OPEN QUESTIONS below with the user before any design is locked. The recommended target (Architecture 2, reached via 3→merge→2) is a *recommendation*, not a decision — the user must pick.
2. **`superpowers:writing-plans`** to turn the chosen architecture into a phased implementation plan (one plan per increment; the RECOMMENDED PLAN OF ATTACK below is the skeleton).
3. **`superpowers:executing-plans`** (or `superpowers:subagent-driven-development` for the independent per-repo tasks) to execute, with `superpowers:test-driven-development` and `superpowers:verification-before-completion` gating each phase.
4. Use `superpowers:using-git-worktrees` to isolate work — this spans five repos and will run long.

**Repos in play (all siblings under `~/Developer/`, all owned by `garrekstemo/`):**

| Repo | Absolute path | Role in this change |
|---|---|---|
| OpticalSpectroscopy.jl | `/Users/garrek/Developer/OpticalSpectroscopy.jl` | analysis layer — owns `AbstractSpectroscopyData`, `Spectrum`, all concrete analysis types |
| QPSTools.jl | `/Users/garrek/Developer/QPSTools.jl` | lab layer — owns `AnnotatedSpectrum`/`CavitySpectrum`/`StreakPL`, eLabFTW glue, loaders |
| QPSLab (server) | `/Users/garrek/Developer/QPSLab/server` | GUI server — owns `LabSpectrum`, workspace structs, all HDF5 persistence |
| CavitySpectroscopy.jl | `/Users/garrek/Developer/CavitySpectroscopy.jl` | cavity physics — mid-merge into OpticalSpectroscopy |
| Leaf readers | `/Users/garrek/Developer/{JASCOFiles,HamamatsuStreakFiles,QPSScanFormat}.jl` | format readers — analysis-dep-free; touched only via field-name/constructor contracts |

**Run tests** from each repo root: `julia --project=. -e 'using Pkg; Pkg.test()'`. **Never edit Manifest.toml.** Live-editing siblings requires temporarily switching `[sources]` to `{path = "../Foo.jl"}` and reverting before commit (see global CLAUDE.md).

---

## CURRENT STATE

Inventory of every type by layer. `file:line` citations; bare `src/...` = OpticalSpectroscopy.jl.

### Layered diagram

```
LEAF READERS (analysis-dep-free, no AbstractSpectroscopyData relationship)
  JASCOFiles.jl:      AbstractJASCOSpectrum → JASCOSpectrum            (src/types.jl:6, :41)
  HamamatsuStreak:    AbstractStreakImage  → StreakImage              (src/types.jl:6, :48)
  QPSScanFormat.jl:   LoadedScanResult / LoadedSpectralResult /
                      LoadedNoiseResult / LoadedCompositeResult       (src/types.jl:40,68,87,111)
                      + NamedTuple aliases TraceData/SpectrumData/
                      SweepArrays/BroadbandData                       (src/types.jl:16-19)
        │  (wrapped/extracted only at QPSTools layer, by convention — no type contract)
        ▼
ANALYSIS LAYER  —  OpticalSpectroscopy.jl
  AbstractSpectroscopyData (root interface)                          (src/types.jl:42)
  ├── Spectrum            x,y,metadata::Dict{Symbol,Any}            (src/types.jl:1158)  ← NEW (PR #39)
  ├── KineticTrace        time,signal,wavelength,metadata           (src/types.jl:151)
  ├── TASpectrum          wavenumber,signal,time_delay,metadata     (src/types.jl:271)
  ├── GatedSpectrum       wavelength,signal,t_range,metadata        (src/types.jl:1054)
  ├── TimeResolvedMatrix  time,wavelength,data,metadata (2D)        (src/types.jl:900)
  └── PLMap               intensity,spectra,x,y,pixel,metadata      (src/plmap.jl:24)
                          (2D; metadata::Dict{String,Any} — OUTLIER)
  SweepData  (X,Y,DC) — exported, NOT <: AbstractSpectroscopyData   (src/types.jl:240)
        │  (subtyped + constructed cross-repo)
        ▼
LAB LAYER  —  QPSTools.jl
  AnnotatedSpectrum (abstract) <: AbstractSpectroscopyData          (src/types.jl:66)
  └── CavitySpectrum = JASCOSpectrum + sample + path                (src/cavity.jl:30)
  StreakPL = StreakImage + sample + path  (NOT in hierarchy)        (src/streak.jl:22)
  PumpProbeData (LVM intermediate, not in hierarchy)                (src/types.jl:37)
        │  (subtyped cross-repo)
        ▼
GUI LAYER  —  QPSLab/server
  LabSpectrum <: AnnotatedSpectrum = JASCOSpectrum + sample + path  (src/spectrum_project.jl:13)
  Workspace structs (plain, not in hierarchy):
    SpectrumSubDataset / SpectrumProject                           (src/spectrum_project.jl:39, :49)
    TASubDataset / TAProject                                       (src/ta_project.jl:6, :15)
    Correction / Dataset / Session                                (src/sessions.jl:14, :21, :34)

CAVITY (mid-merge into OpticalSpectroscopy — see merge spec/plan)
  CavitySpectroscopy.jl:
    CavitySpectrum (plain, x,y,metadata::Dict{String,Any})         (src/types.jl:31)   ← DROPPED by merge
    CavityFitResult                                                (src/types.jl:114)  ← MOVES to OpticalSpec
    DispersionFitResult                                            (src/types.jl:274)  ← MOVES to OpticalSpec
```

### Fit-result and helper types (OpticalSpectroscopy, NOT `<: AbstractSpectroscopyData`)

`TAPeak` (`types.jl:346`), `TASpectrumFit` (`types.jl:385`, defines `xdata`/`wavenumber` but is not a subtype), `ExpDecayFit` (`types.jl:473`), `MultiexpDecayFit` (`types.jl:523`), `StretchedDecayFit` (`types.jl:593`), `LifetimeSpectrumResult` (`types.jl:647`), `PeakFitResult` (`types.jl:671`), `MultiPeakFitResult` (`types.jl:732`, defines `xdata`/`ydata` but is not a subtype), `GlobalFitResult` (`types.jl:1248`), `FitMapResult` (`types.jl:840`), `PeakInfo` (`peakdetection.jl:25`), `ChirpCalibration` (`chirp.jl:26`, 7 fields, no convenience constructor), `DecompositionResult` (`decomposition.jl:18`), `CosmicRayResult`/`CosmicRayMapResult`/`CosmicRayMatrixResult` (`cosmic_rays.jl:31,47,579`).

### The interface (defined `src/types.jl:44–122`)

10 methods, **2 mandatory** (`xdata`, `ydata` — default *errors*), **8 defaulted** (`zdata→nothing`, `xlabel→"X"`, `ylabel→"Y"`, `zlabel→"Signal"`, `is_matrix→false`, `source_file→""`, `npoints→length(xdata(d))`, `title→source_file(d)`). The generic consumer layer (`_check_1d` guard at `types.jl:125`; `find_peaks`/`band_area`/`calc_fwhm`/`estimate_snr`/`fit_peaks`/arithmetic; the Makie `convert_arguments` hook at `ext/OpticalSpectroscopyMakieExt.jl:8`) reads only **`xdata`, `ydata`, `is_matrix`** generically — the contract is effectively 3 methods wide. The other 7 are consumed only by `show` and external layers.

---

## COUPLING / BLAST RADIUS

Full ranking and per-type analysis: see the COUPLING / BLAST-RADIUS MAP analysis (summarized here). Touch order matters because Julia enforces none of these contracts at compile time — most breakage is **silent runtime** or `FieldError` at save time.

### Tier 1 — CATASTROPHIC: `AbstractSpectroscopyData` (`src/types.jl:42`)
Root interface; ~12 generic dispatches + 6 OpticalSpectroscopy subtypes + the cross-repo subtype chain + the Makie hook. **Adding** new generic methods or new *optional* (defaulted) interface methods = SAFE. **Renaming/re-signing** existing interface methods = DANGEROUS: silent runtime breakage across three repos, because the defaults absorb missing overrides (no compile error). Inserting a supertype above it = SAFE.

### Tier 2 — CRITICAL cross-repo: `JASCOSpectrum` (`JASCOFiles.jl/src/types.jl:41`)
Structurally required by QPSTools `AnnotatedSpectrum`/`CavitySpectrum` AND QPSLab `LabSpectrum`; field-pierced (`.data.x/.y/.date/.spectrometer/.xunits/.yunits/.datatype`) at 20+ sites with **no type contract**. The 9-arg positional constructor `JASCOSpectrum(title,date,spectrometer,datatype,xunits,yunits,x,y,metadata)` is load-bearing for HDF5 read (`QPSLab/server/src/handlers/spectrum_hdf5.jl:199`). Renaming/reordering any field = silent cross-repo break.

### Tier 3 — CRITICAL cross-repo: `AnnotatedSpectrum` (`QPSTools.jl/src/types.jl:66`)
Abstract base; ~25 QPSTools dispatches + QPSLab `LabSpectrum` subtype + eLabFTW glue. The implied **3-field positional constructor `T(jasco, sample, path)`** is required by QPSLab `_reconstruct_spectrum` (`server/src/sessions.jl:434`) and by every type-preserving op. Not enforced — a subtype with a different constructor shape errors at runtime.

### Tier 4–8 — HIGH / MEDIUM-HIGH
`TimeResolvedMatrix` (`types.jl:900`; indexing factory produces `KineticTrace`/`TASpectrum`; constructed in 4 repos; metadata read with mixed Symbol+String keys in QPSLab), `KineticTrace` (`types.jl:151`; decay-fit family + `predict` + indexing target; in QPSLab `TASubDataset.data` union), **`Spectrum`** (`types.jl:1158`; ~20 Spectrum-in/Spectrum-out transforms but **zero external consumers yet** — the safest high-coupling type to refactor *now*), `PLMap` (`plmap.jl:24`; the sole String-keyed metadata outlier), `TASpectrum` (`types.jl:271`; field named `.wavenumber` can hold nm values — latent semantic bug).

### Lowest blast radius (touch freely)
`SweepData` (zero dispatch), `PumpProbeData` (QPSTools-internal), CavitySpectroscopy `CavitySpectrum` (being dropped), the QPSScanFormat NamedTuple aliases. **Adding new generic methods on `AbstractSpectroscopyData` is always safe.**

### Cross-repo edges (the dangerous boundaries)

| # | Edge | Site | Mechanism | Risk |
|---|---|---|---|---|
| E1 | `LabSpectrum <: AnnotatedSpectrum <: AbstractSpectroscopyData` | `QPSLab/.../spectrum_project.jl:13` | type hierarchy (2 repo hops) | renaming either abstract → QPSLab fails to load |
| E2 | All 3 lab spectrum types require `.data::JASCOSpectrum` | field decls `QPSTools/src/cavity.jl:31`, `QPSLab/.../spectrum_project.jl:14` (convention documented at `QPSTools/src/types.jl:56`) | field-layout convention, **no type contract** | JASCOSpectrum field rename → silent break, 20+ sites, 2 repos |
| E3 | QPSLab `_reconstruct_spectrum` needs `T(jasco,sample,path)` | `QPSLab/.../sessions.jl:434` | positional constructor convention | new subtype w/ different ctor → runtime error |
| E4 | `StreakPL.data → StreakImage → TimeResolvedMatrix` | `QPSTools/src/streak.jl:22,121` | field pierce `.time/.wavelength/.counts` | StreakImage field rename → QPSTools break |
| E5 | QPSTools/QPSLab construct OS analysis types positionally/kw | `QPSTools/src/io.jl:84,376,520`, `plmap.jl:189`; QPSLab handlers | constructors | OS ctor signature change → downstream break |
| E6 | QPSTools `_with_analysis_types` consumes `Loaded*` | `QPSTools/src/scan_loading.jl:34,44,55` | NamedTuple field extract | Loaded* field rename → QPSTools break |
| E7 | QPSTools bridges `format_results(::Cavity/Dispersion)` | `QPSTools/src/cavity.jl:113,115` | generic forwarding | **deleted by in-flight merge** |
| E8 | QPSTools re-exports CavitySpectroscopy physics names | `QPSTools/src/QPSTools.jl:54-67` | re-export | merge moves source; re-exports removed |
| E9 | Two distinct `CavitySpectrum` types in QPSTools scope | `QPSTools/src/cavity.jl:30` vs CavitySpectroscopy's | name collision, scope-managed | merge friction; documented |
| E10 | QPSLab reads `TimeResolvedMatrix.metadata` Symbol-keyed | `QPSLab/.../handlers/export.jl:346` | key-type inconsistency | latent — Symbol access on String dict silently returns default |

---

## INTERFACE + METADATA CONTRACT

### Accessor interface
10 methods (above). **Load-bearing generically:** `xdata`, `ydata`, `is_matrix` only. Conformance is **silent and non-enforced** — Julia has no interface check; a subtype that omits `xlabel` displays `"X"` forever; one that omits `xdata` errors only on first call.

**Conformance gaps to fix:**
- **`xdata` is not semantically uniform.** Docstring promises primary axis / signal-or-secondary, but `TimeResolvedMatrix` has `xdata`=wavelength, `ydata`=time (signal is `zdata`); `PLMap` has `xdata`/`ydata`=spatial axes and `zdata`=integrated scalar map (NOT the 3D cube). `_check_1d` exists precisely as a blunt guard against this.
- **`source_file` uses three Symbol-key conventions across the four 1D types, plus a String outlier — and one of them is an outright bug.** `KineticTrace` (`types.jl:174`) and `Spectrum` (`types.jl:1188`) check `:filename` then `:source`; `TASpectrum` (`types.jl:294`) checks `:filename` **only**; `GatedSpectrum` (`types.jl:1076`) checks `:source` **only** (as does the 2D `TimeResolvedMatrix`, `types.jl:929`); `PLMap` uses the String key `"source_file"` (`plmap.jl:48`, default `"unknown"`). The bug: matrix-extracted TASpectra carry only `:extracted_from`/`:requested_time` (set at `types.jl:1020-1021`) — neither `:filename` nor `:source` — so `source_file(::TASpectrum)` **always returns `""`** for them, while the sibling `KineticTrace` extraction path is fine. Arch 2 touches all of these; consolidate them onto one `source_file` key convention (and demote `PLMap`'s String outlier as part of the metadata unification).
- **`TASpectrumFit`/`MultiPeakFitResult` define `xdata`/`ydata` but are not subtypes** — accessor overlap by coincidence, no `_check_1d`/Makie benefit.
- **`AnnotatedSpectrum` supplies no interface methods at the abstract level** — every concrete subtype re-implements `xdata`/`ydata`/`xlabel`/`ylabel`/`source_file`. `LabSpectrum` omits `source_file`/`npoints`/`title` entirely (server reads `sub.data.path` and computes `npoints` inline). `CavitySpectrum` implements `source_file` but not `title` (its rich `_cavity_title` at `cavity.jl:141` is wired only into plotting).

### The Symbol/String metadata split
Conceptual rule (mostly followed): **analysis/instrument metadata = `Dict{Symbol,Any}`** (set by our code); **provenance/sample metadata = `Dict{String,Any}`** (crosses HDF5/JSON/file boundaries where keys are natively strings). Defensible because Symbol keys round-trip poorly through HDF5 attributes and JSON.

**Three concrete violations:**
1. **`PLMap.metadata::Dict{String,Any}`** (`plmap.jl:30`) — the *only* String-keyed metadata among OpticalSpectroscopy's own analysis types. `source_file(::PLMap)` reads String key `"source_file"`, default `"unknown"` (vs `""` everywhere).
2. **`StreakPL`→`TimeResolvedMatrix` nests a `Dict{String,Any}` under a Symbol key** (`QPSTools/src/streak.jl:119`: `:sample => copy(s.sample)`) — defeats uniform traversal.
3. **QPSLab server (String-only) reads `TimeResolvedMatrix.metadata` with a Symbol key** (`get(data.metadata, :signal_label, "ΔA")`, `handlers/export.jl:346`) — works only because the upstream dict happens to be Symbol-keyed; a String-keyed loader path would silently fall to default.

The `Spectrum` constructor already symbolizes String keys on the way in (`types.jl:1170`) — the one existing one-way bridge and a precedent for unification.

### Labels / units / technique — four incompatible encodings, no single source of truth
- **A — metadata keys:** `KineticTrace.xlabel` from `:time_unit`; `Spectrum.xlabel` from `:xlabel` or heuristic. Units are free-form strings, no type, no validation.
- **B — type-level hardcoding:** `TASpectrum.xlabel` hardcoded `"Wavenumber (cm⁻¹)"` (`types.jl:292`) even though the field can hold nm; QPSTools `CavitySpectrum` hardcodes labels ignoring `JASCOSpectrum.yunits`.
- **C — data-range heuristic:** `_detect_spectral_unit` (`types.jl:960`) returns `"cm⁻¹"` if `1200 < min` and `max < 5000` else `"nm"` — a guess, fragile at NIR overlap/calibrated-pixel maps.
- **D — JASCO datatype predicates (technique):** `isftir`/`israman`/`isuvvis` (`JASCOFiles/src/utils.jl:6,13,24`) string-match `JASCOSpectrum.datatype`. **Confirmed cross-package string drift:** `_jasco_technique_tag` keys on `"UV/VISIBLE SPECTRUM"` (`QPSTools/src/types.jl:148`) while `isuvvis` checks `"UV/VIS SPECTRUM"` (`JASCOFiles/src/utils.jl:25`) — two canonical strings for the same technique, neither validated. QPSLab papers over this with an `occursin`-based fallback in `_spectrum_kind` (`QPSLab/.../handlers/spectrum.jl:24-29`) — a third site that hard-codes both spellings, confirming the drift is real and load-bearing.

**Technique is structurally homeless in the analysis layer** — no field, no predicate; everything technique-aware is bolted on at QPSTools/QPSLab via JASCO-specific predicates. A non-JASCO source (streak PL, CSV import) has no path to declare its technique. **Recommended encoding for unification:** a typed triple — `technique::Symbol` (`:ftir`/`:raman`/`:uvvis`/`:cavity`/`:pl`/`:unknown`), `xunit::Symbol`, `yunit::Symbol` — promoted to interface accessors, with display labels *derived* from them; JASCO maps into this once at the wrapping layer; the range heuristic demotes to the `:unknown` fallback.

---

## PERSISTENCE / BACK-COMPAT CONSTRAINTS

There are **six** serialization surfaces. Existing `.h5`/`.json` files MUST keep loading. Full detail in the Persistence & Back-Compat Map analysis.

| Surface | Owner | Format | Persists |
|---|---|---|---|
| A | `QPSLab/server/src/handlers/spectrum_hdf5.jl` | HDF5 | `SpectrumProject` → `LabSpectrum`(→`JASCOSpectrum`) + corrections + fits |
| B | `QPSLab/server/src/handlers/ta_hdf5.jl` | HDF5 | `TAProject` → `KineticTrace`/`TASpectrum` + corrections + fits |
| C | `QPSLab/server/src/handlers/ta_matrix_hdf5.jl`, `streak_pl_hdf5.jl` | HDF5 | `TimeResolvedMatrix` + `ChirpCalibration` + corrections + fits |
| D | `QPSLab/server/src/handlers/hdf5io.jl` | HDF5 | `PLMap` + corrections + `FitMapResult`-derived dict |
| E | `QPSScanFormat.jl/src/{write,read,helpers,schema}.jl` | HDF5 | instrument scans (duck-typed, NOT analysis types) |
| F | `OpticalSpectroscopy.jl/src/chirp.jl:503` | JSON | `ChirpCalibration` standalone (7-key, only fully round-tripping serializer) |

### Constructors that MUST be preserved (each load-bearing for a reader)
- `JASCOSpectrum(title,date,spectrometer,datatype,xunits,yunits,x,y,metadata)` 9-pos — `spectrum_hdf5.jl:199`
- `LabSpectrum(jasco, sample, path)` 3-pos — `spectrum_hdf5.jl:213`, `sessions.jl:434`
- `KineticTrace(time, signal; wavelength=…)` — `ta_hdf5.jl:140`
- `TASpectrum(wavenumber, signal; time_delay=…)` — `ta_hdf5.jl:145`
- `TimeResolvedMatrix(time, wavelength, data, meta)` 4-pos AND `(…; metadata=…)` kw — `ta_matrix_hdf5.jl:96`, `streak_pl_hdf5.jl:159`
- `PLMap(intensity, spectra, x, y, pixel, metadata)` 6-pos — `hdf5io.jl:166`
- `ExpDecayFit`/`MultiexpDecayFit`/`StretchedDecayFit` 8-pos exact-order, `LifetimeSpectrumResult` 6-pos — `streak_pl_hdf5.jl:205,216,227,244`

### Field names accessed RAW by writers (renaming = `FieldError` on save)
`KineticTrace.time/.signal/.wavelength`; `TASpectrum.wavenumber/.signal/.time_delay`; `TimeResolvedMatrix.time/.wavelength/.data/.metadata`; `PLMap.intensity/.x/.y/.pixel/.spectra/.metadata`; `LabSpectrum.data/.sample/.path`; decay-fit fields by name; and via duck-typing `.time/.signal/.wavenumber/.wavelength/.data/.X/.Y/.DC` on QPSScanFormat inputs (`QPSScanFormat/src/write.jl:42,46,84,122-124`).

### Metadata key-type per type (reader reconstructs the exact key type)
`TimeResolvedMatrix.metadata → Dict{Symbol,Any}` (`ta_matrix_hdf5.jl:89`); `PLMap.metadata → Dict{String,Any}` (`hdf5io.jl:158`); `LabSpectrum.sample`/`.data.metadata → Dict{String,Any}` (`spectrum_hdf5.jl:200,203`). **Unifying `PLMap.metadata` to Symbol keys requires a read-time re-symbolization shim for old files.**

### On-disk discriminator strings (cannot rename without a shim)
`type` attr: `spectrum_project`/`raman_project`(alias)/`ta_project`/`ta_matrix`/`streak_pl` (`hdf5io.jl:428-435`); `data_type`: `KineticTrace`/`TASpectrum`; `fit_type`/`kind` tags; `scan_type` values; `format="QPSDrive/1.0"`.

### Known traps / pre-existing bugs (the re-architecture is the moment to decide)
- **`ChirpCalibration` HDF5 read is BROKEN.** `ta_matrix_hdf5.jl:110` calls `ChirpCalibration(...)` with **5 positional args**, but the struct (`chirp.jl:26`) has **7 fields and no 5-arg constructor** → `MethodError` reading any TA-matrix `.h5` with a `chirp_cal` group. No test covers it. Fix by passing 7 args (`r_squared=NaN`, `metadata=Dict{Symbol,Any}()`) OR adding a 5-arg constructor.
- **Fits round-trip lossy as `Dict{String,Any}`** (surfaces A, B) — `serialize_fit_result` dispatches on `MultiPeakFitResult` but gets a `Dict` post-reopen (untested failure path). Any change making residuals/grid mandatory needs a shim injecting empties (the streak path already does this).
- **Must-keep shims:** dataset aliases `shift→x`/`intensity→y`/`raman_path→spectrum_path` for old `raman_project` files; field-default backfills (`jasco_datatype→"RAMAN SPECTRUM"`, `spectrum_kind→"raman"`, `sigma→0.0`); `fit_all` 2D→3D center-array upgrade (`hdf5io.jl:209-222`); the QPSScanFormat `format` major/minor version gate (the only formal versioning anywhere — QPSLab files use a bare `version=1` attr written by the shared top-level save `handle_save_hdf5` at `hdf5io.jl:335` for all four QPSLab surfaces A–D, and never read back as a gate on load).

---

## TARGET DESIGNS

Three candidate architectures. Full trade-off analysis in the Target Architectures cross-cutting doc; condensed here.

### Architecture 1 — "Annotated relocated" (moderate)
Relocate/de-duplicate without touching the JASCO binding. **Delete** QPSLab `LabSpectrum`; introduce a single technique-flexible `AnnotatedSpectrum` concrete type in QPSTools (option 1b: split into `FileSpectrum <: AnnotatedSpectrum` for generic JASCO-backed spectra + a thin `CavitySpectrum` adding `xreversed`/cavity convenience). Provenance still wraps a `JASCOSpectrum`. Derived spectra become plain `Spectrum` (kills the fake-JASCO fabrication). Leave the Symbol/String split documented; fix only `PLMap.metadata → Dict{Symbol,Any}`.
- **Fixes:** duplicate-type, fake-JASCO bug, the LabSpectrum interface gaps, partial PLMap.
- **Leaves:** the JASCO hard-binding (root cause of most smells), GatedSpectrum/Spectrum overlap, fit-result `xdata` types.

### Architecture 2 — "Spectrum-core provenance" (largest) — RECOMMENDED TARGET
`AnnotatedSpectrum` becomes **format-agnostic and MOVES to OpticalSpectroscopy** (abstract only; no file-format deps). Its contract: every subtype has `core::Spectrum` + `provenance::Provenance`, where `Provenance` = `{source_format::Symbol, source_file::String, sample::Dict{String,Any}, raw::Any}`. Interface methods delegate to `s.core`. QPSTools `FileSpectrum = Spectrum + Provenance` is the concrete type; `CavitySpectrum` dissolves into `FileSpectrum` + `:cavity_length` metadata (exactly what the merge spec reserves). `StreakPL` folds onto the shared `Provenance` (kills its duplicated eLabFTW dispatch). All ops operate on `s.core` and rebuild `FileSpectrum(new_core, prov)` — type-preserving; collapses the NamedTuple-return smells. **Metadata:** unify analysis metadata on `Dict{Symbol,Any}`; keep exactly one String-keyed zone — `Provenance.sample` — justified by its eLabFTW/HDF5 boundary.
- **Fixes:** everything in Arch 1 plus the JASCO binding, type-preservation, technique homelessness (pairs with the typed `(technique, xunit, yunit)` triple).
- **Cost:** largest; entangles with the cavity-merge QPSTools PR and requires rewriting QPSLab `spectrum_hdf5.jl` to persist a `Spectrum` core + `Provenance`.

### Architecture 3 — "Derived-only, defer the rest" (smallest)
Surgical: route only QPSLab computed/derived spectra (Tauc, KK, Kubelka-Munk, spectral math) to `Spectrum` instead of a fabricated `JASCOSpectrum`-backed `LabSpectrum`. Nothing deleted/renamed; fully independent of the merge.
- **Fixes:** the fake-JASCO bug (produces *wrong output today* — a Tauc plot labeled "Transmittance") + the NamedTuple plumbing.
- **Leaves:** duplicate-type, layering, JASCO-binding.

### Recommendation
**Adopt Architecture 2 as the target, reached in two shipped increments: Architecture 3 now, then Architecture 2 after the cavity merge.** Do not stop at Architecture 1 — it relocates duplication but preserves the JASCO binding, so you'd re-open the same QPSLab-HDF5 files to finish later, paying the cost twice. The merge spec already commits the project to Arch 2's end-state (its lines 64–70 hand the type workstream the `fit_cavity_spectrum(::<NewType>)` dispatch and "the likely dissolution of QPSTools' `CavitySpectrum`"). Arch 3 first is near-free insurance: the fabrication fix is one handler, but routing a derived `Spectrum` through QPSLab obliges matching `::Spectrum` methods on the serializer, figure-render, and correction-replay paths (see Phase 1) — a contained, additive change, and a strict subset of Arch 2.

---

## RECOMMENDED PLAN OF ATTACK

Phased, ordered. Re-confirm with the user in brainstorming before locking — especially the OPEN QUESTIONS.

**Phase 0 — Brainstorm + decide (no code).** Resolve OPEN QUESTIONS. Lock the target architecture. Decide the ChirpCalibration-bug fix shape (it forces a decision either way). *Risk: none.*

**Phase 1 — Architecture 3 increment (ship now, merge-independent).** QPSLab: in `handle_spectrum_save_transform` (`server/src/handlers/spectrum.jl:248`), route results to `Spectrum` instead of the JASCO-fabrication block at 285–300; make workspace `.data` field `Union{AnnotatedSpectrum, Spectrum}`. Then teach every path that today assumes the wrapped-JASCO shape to also accept a bare `Spectrum`:
- **Figure render:** `_render_figure(::AnnotatedSpectrum)` (`handlers/export.jl:306`) — add a `::Spectrum` method.
- **JSON serializer:** `_serialize_spectrum_sub_dataset` (`handlers/spectrum.jl:326`) calls `_spectrum_kind(sub.data)`, which is defined only on `AnnotatedSpectrum` (`handlers/spectrum.jl:20`) and pierces `data.data.datatype` — add a `_spectrum_kind(::Spectrum)` method (read `metadata[:technique]`/label, fall back `"unknown"`).
- **Correction replay (the under-stated blast radius):** `_apply_correction(::AnnotatedSpectrum)` (`server/src/sessions.jl:446`) and `_reconstruct_spectrum(::T<:AnnotatedSpectrum)` (`sessions.jl:434`) rebuild via `spec.data`/`spec.sample`/`spec.path` (a fabricated `JASCOSpectrum` + `T(new_jasco, sample, path)`). A derived `Spectrum` has none of those fields, so applying any correction to a derived spectrum needs a **parallel `_apply_correction(::Spectrum)` + `Spectrum`-rebuild path** — not just a serialization branch.

*Risk: medium — not "low–medium". Beyond the one handler + serializer branch, the entire correction-replay pipeline assumes the wrapped-JASCO shape; a derived `Spectrum` that is then corrected must round-trip through a new reconstruction path. Existing transform tests pin the fabrication behavior but do not exercise corrections-on-derived-spectra; add that coverage.*

**Phase 2 — Cavity merge lands (its own spec/plan).** Pure code move of `CavityFitResult`/`DispersionFitResult` + vector-in `fit_cavity_spectrum`/`fit_dispersion` into OpticalSpectroscopy; drop CavitySpectroscopy `CavitySpectrum`; delete QPSTools `format_results` bridges (E7) and physics re-exports (E8). **This must precede Arch 2's cavity-fit dispatch** so it isn't done twice. *Risk: low (ported tests). Verify E7/E8/E9 before and after.*

**Phase 3 — Architecture 2, analysis layer (additive).** OpticalSpectroscopy: add `Provenance` struct + format-agnostic abstract `AnnotatedSpectrum` (delegating to `core::Spectrum`) + interface methods + `fit_cavity_spectrum(::Spectrum)` reading `metadata[:cavity_length]` + the typed `(technique, xunit, yunit)` accessors with derived labels. `PLMap.metadata → Dict{Symbol,Any}`. Consolidate `source_file` onto one key convention across the four 1D types + `TimeResolvedMatrix` + `PLMap` (fixes the `TASpectrum` always-`""` bug along the way). *Risk: medium — new public surface, new tests, but purely additive except the PLMap sweep (wide: grep `"pixel_range"`/`"source_file"`/`"background_positions"` and `.metadata[` on PLMap across all three repos).*

**Phase 4 — Architecture 2, lab layer (do in the SAME PR as the Phase-2 QPSTools rewire).** QPSTools: replace `CavitySpectrum`/`AnnotatedSpectrum` with `FileSpectrum` over a `Spectrum` core; rewrite `subtract_spectrum`/`correct_baseline`/`find_peaks` to go through `s.core` (today they pierce `spec.data.x/.y`); rewire `load_cavity`, eLabFTW glue (moves `spec.data.spectrometer`/`.date` reads to `provenance.raw`), plot titles; fold `StreakPL` onto shared `Provenance`. Re-pin every `CavitySpectrum` test + the `transmittance_to_absorbance` round-trip. *Risk: HIGH — bulk of the work; coordinate with merge PR to avoid churning `cavity.jl`/test files twice.*

**Phase 5 — Architecture 2, GUI layer.** QPSLab: delete `LabSpectrum`; workspace `.data` becomes `Union{FileSpectrum, Spectrum}`; rewrite `spectrum_hdf5.jl` to persist a `Spectrum` core + `Provenance` (currently JASCO-field-oriented: `jasco_title`/`jasco_xunits`/…). **Fix the ChirpCalibration 5-vs-7-arg bug and the lossy-fit round-trip (smell S6) while here.** Keep all back-compat shims (Section above). *Risk: HIGH — HDF5 schema rewrite; good test coverage (`test_spectrum_project.jl`) makes regressions catchable.*

**Cross-phase rule:** each phase gates on `Pkg.test()` green in every touched repo (`superpowers:verification-before-completion`). Use worktrees; `[sources]` path-switch for live sibling edits, revert before commit.

---

## OPEN QUESTIONS FOR THE USER

1. **Target architecture:** confirm Arch 2 (via 3→merge→2), or prefer the cheaper Arch 1 / Arch 3-only endpoint?
2. **`GatedSpectrum` fate:** it is structurally `Spectrum` + `t_range`. Collapse into `Spectrum` (with `t_range` as reserved metadata) or keep as a distinct type for time-resolved-extraction semantics?
3. **`Provenance.raw` retention:** keep the original `JASCOSpectrum`/`StreakImage` parked for byte-faithful eLabFTW export, or drop it and reconstruct provenance headers from typed fields only?
4. **Technique encoding:** typed field(s) on the struct, a reserved metadata key with a canonical enum, or a Holy trait? And what is the canonical enum + the one canonical technique string (resolving the `"UV/VIS"` vs `"UV/VISIBLE"` drift)?
5. **Metadata unification depth:** full Symbol-internal convergence (migrate `PLMap` + write a re-symbolization shim for old files), or the lighter "typed-accessor facade over existing dicts" that stops the silent-miss bugs without migrating storage?
6. **ChirpCalibration bug fix shape:** add a 5-arg convenience constructor (accidentally backward-compatible) or fix the reader to pass 7 args? And should the HDF5 embed persist `r_squared`/`metadata` to match the JSON serializer?
7. **`SweepData` placement:** it is exported from `types.jl` but not in the hierarchy and has zero dispatch — leave, move to a raw-acquisition namespace, or drop the export?
8. **Fit-result types that define `xdata`/`ydata` (`TASpectrumFit`, `MultiPeakFitResult`):** make them real `AbstractSpectroscopyData` subtypes (gain `_check_1d`/Makie) or leave the accessor overlap coincidental?
9. **Versioning lever:** introduce a real read-checked `version`/`schema` attr on the QPSLab HDF5 surfaces before changing the schema, so future migrations have a branch point? Today `handle_save_hdf5` writes a bare `version=1` (`QPSLab/server/src/handlers/hdf5io.jl:335`) that no load path reads — the bump-and-gate is a one-line write plus a read-time check.

---

## KICKOFF PROMPT

```
Re-architect the spectroscopy type system across the QPS ecosystem.

Read the hand-off doc FIRST and treat it as ground truth (it has verified file:line
citations from a prior analysis session you have no memory of):
/Users/garrek/Developer/OpticalSpectroscopy.jl/docs/superpowers/2026-06-15-type-system-rearchitecture-handoff.md

Use the ultracode flow: superpowers:brainstorming -> superpowers:writing-plans ->
superpowers:executing-plans. Start in brainstorming and walk me through the OPEN
QUESTIONS section before locking any design. The doc recommends Architecture 2
(Spectrum-core provenance) reached as 3 -> cavity-merge -> 2, but that is a
recommendation, not a decision — surface the trade-offs and let me choose.

Repos in play (siblings under ~/Developer/, all garrekstemo/):
  OpticalSpectroscopy.jl  — analysis layer (AbstractSpectroscopyData, Spectrum)
  QPSTools.jl             — lab layer (AnnotatedSpectrum/CavitySpectrum/StreakPL)
  QPSLab/server           — GUI server (LabSpectrum, all HDF5 persistence)
  CavitySpectroscopy.jl   — mid-merge into OpticalSpectroscopy
  JASCOFiles/HamamatsuStreakFiles/QPSScanFormat — leaf readers (contract-only)

Hard constraints from the doc:
  - Existing HDF5/JSON files MUST keep loading (six serialization surfaces; preserve
    the listed constructors, raw field names, discriminator strings, and shims).
  - The cavity merge (PR-track, see its spec/plan) lands BEFORE Architecture 2's
    cavity-fit dispatch.
  - There is a pre-existing ChirpCalibration HDF5-read bug (5 args vs 7-field struct)
    to decide on, not ignore.
  - Julia enforces none of these type contracts — most breakage is silent runtime or
    FieldError-on-save. Use TDD + verification-before-completion; gate each phase on
    Pkg.test() green in every touched repo. Work in git worktrees.

PR #39 (the Spectrum type) is OPEN on feat/spectrum-type and is the foundation —
confirm its merge status before building on it.
```

---

## REFERENCES

- **This hand-off:** `/Users/garrek/Developer/OpticalSpectroscopy.jl/docs/superpowers/2026-06-15-type-system-rearchitecture-handoff.md`
- **Ecosystem analysis (sibling):** `/Users/garrek/Developer/QPSLab/docs/dev/spectrum-types-ecosystem-analysis.md`
- **Cavity merge — design spec:** `/Users/garrek/Developer/OpticalSpectroscopy.jl/docs/superpowers/specs/2026-06-12-cavityspectroscopy-merge-design.md`
- **Cavity merge — plan:** `/Users/garrek/Developer/OpticalSpectroscopy.jl/docs/superpowers/plans/2026-06-12-cavityspectroscopy-merge.md`
- **Spectrum type — design spec:** `/Users/garrek/Developer/OpticalSpectroscopy.jl/docs/superpowers/specs/2026-06-12-spectrum-type-design.md`
- **Spectrum type — plan:** `/Users/garrek/Developer/OpticalSpectroscopy.jl/docs/superpowers/plans/2026-06-12-spectrum-type.md`
- **PR #39 — "Add Spectrum: generic 1D steady-state spectrum type":** OPEN on branch `feat/spectrum-type` — https://github.com/garrekstemo/OpticalSpectroscopy.jl/pull/39 . This is the foundation type for the whole re-architecture; confirm its merge status before Phase 1.
