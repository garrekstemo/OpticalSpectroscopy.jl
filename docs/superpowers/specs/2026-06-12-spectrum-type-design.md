# Design: `Spectrum` — generic 1D steady-state spectrum type

**Date:** 2026-06-12
**Status:** Approved (brainstorm with Garrek, 2026-06-12)

## Motivation

OpticalSpectroscopy.jl has no generic 1D steady-state spectrum type. The
`AbstractSpectroscopyData` family covers time-resolved and spatial data
(`KineticTrace`, `TASpectrum`, `GatedSpectrum`, `TimeResolvedMatrix`, `PLMap`),
so all steady-state 1D work (FTIR, Raman, UV-Vis, PL emission, cavity
transmission) goes through plain x/y vectors. A typed spectrum enables
spectrum-in → spectrum-out workflows, metadata that travels with the data, and
typed dispatches like the upcoming `fit_cavity_spectrum(::Spectrum)`.

The CavitySpectroscopy.jl merge (separate, future effort) drops that package's
generic `CavitySpectrum(x, y, metadata)` wrapper; `Spectrum` recovers the same
capability generally. **This work lands first**; the cavity merge then adds
`fit_cavity_spectrum(::Spectrum)` reading `:cavity_length` from metadata.

## Decisions (from brainstorm)

| Question | Decision |
|----------|----------|
| Name | `Spectrum` (collisions limited to NeXLSpectrum.jl / archived JuliaAstro Spectra.jl — acceptable) |
| Fields | `x`, `y`, `metadata` only; labels live in metadata |
| Metadata keys | `Dict{Symbol,Any}`, consistent with the rest of the package |
| Dispatch strategy | Hybrid: read-only analysis generic on `AbstractSpectroscopyData`; spectrum-producing transformations concrete on `::Spectrum` |
| `correct_baseline(::Spectrum)` | Returns a new `Spectrum` only (baseline vector available via the vector API) |
| Sequencing | `Spectrum` lands before the CavitySpectroscopy merge |
| Plotting | Family-wide `convert_arguments` — **already exists** in the Makie extension; needs only a test |

## The type

In `src/types.jl`, alongside the family:

```julia
struct Spectrum <: AbstractSpectroscopyData
    x::Vector{Float64}
    y::Vector{Float64}
    metadata::Dict{Symbol,Any}
end
```

- Inner constructor validates `length(x) == length(y)` (ArgumentError, message
  style matching `GatedSpectrum`) and converts inputs via `Float64.()`.
- Outer constructors:
  - `Spectrum(x, y)` — empty metadata.
  - `Spectrum(x, y, metadata::AbstractDict)` — canonical.
  - `Spectrum(x, y; kwargs...)` — kwargs splat into the metadata dict
    (`Spectrum(nu, T; cavity_length=12e-4, mirror="Au")`), borrowed from
    CavitySpectroscopy's `CavitySpectrum`.

### Interface implementation

| Function | Implementation |
|----------|----------------|
| `xdata(s)` | `s.x` |
| `ydata(s)` | `s.y` |
| `xlabel(s)` | `get(s.metadata, :xlabel, ...)`, falling back to `_detect_spectral_unit(s.x)` → `"Wavenumber (cm⁻¹)"` / `"Wavelength (nm)"` |
| `ylabel(s)` | `get(s.metadata, :ylabel, "Signal")` (String-converted) |
| `source_file(s)` | `:filename` then `:source` metadata keys, like `KineticTrace` |
| `is_matrix`, `zdata`, `npoints`, `title` | abstract-type defaults |
| `signal(s)` | `s.y` — family-consistent semantic accessor |

Deliberate divergence: metadata keys are `:xlabel`/`:ylabel` (self-describing
for a generic type) rather than the time-resolved family's `:signal_label`.

Two `show` methods (one-line and `text/plain`) mirroring `GatedSpectrum`:
point count, x-range with detected unit, plus metadata-driven extras
(source, and any of a short list of common keys if present).

The docstring documents the `:cavity_length` convention so the cavity merge
can rely on it, and `:xlabel`/`:ylabel` for display.

## Generic analysis dispatches (read-only)

Already exist (no change to behavior, but **add `is_matrix` guards** — today
`fit_peaks(::TimeResolvedMatrix)` would silently fit the time axis):

- `fit_peaks(spec::AbstractSpectroscopyData, region)` / `fit_peaks(spec)` —
  `src/peakfitting.jl:294,312`
- `subtract_spectrum`, `add_spectra`, `divide_spectra`, `multiply_spectrum`
  on `AbstractSpectroscopyData` (return NamedTuples) — `src/spectroscopy.jl`

New generic methods, each forwarding `xdata`/`ydata` to the vector API with an
`is_matrix` guard (shared internal helper `_check_1d` raising a clear
ArgumentError):

- `find_peaks(spec::AbstractSpectroscopyData; kwargs...)` → `Vector{PeakInfo}`
- `band_area(spec::AbstractSpectroscopyData, lo, hi)` → scalar
- `calc_fwhm(spec::AbstractSpectroscopyData; kwargs...)` → scalar
- `estimate_snr(spec::AbstractSpectroscopyData)` → scalar

QPSTools' more-specific `AnnotatedSpectrum` methods keep winning dispatch where
they exist; its other subtypes gain these for free.

## Concrete `::Spectrum` transformations (constructing)

Spectrum-in → Spectrum-out. Result metadata = `copy` of the input's (first
argument's, for two-spectrum ops). Transforms that change the y-quantity set
`metadata[:ylabel]`; everything else propagates metadata untouched.

| Method | Notes |
|--------|-------|
| `smooth_data(s; window)` | |
| `savitzky_golay_smooth(s; window, order)` | |
| `derivative(s; order, ...)` | uses the `(x, y)` vector form |
| `normalize_area(s)` | |
| `normalize_to_peak(s, position; tolerance)` | |
| `correct_baseline(s; method, kwargs...)` | returns corrected `Spectrum`; baseline vector via vector API |
| `subtract_spectrum(s::Spectrum, ref::Spectrum; scale, interpolate)` | overrides generic NamedTuple method |
| `add_spectra(a::Spectrum, b::Spectrum; interpolate)` | ditto |
| `divide_spectra(a::Spectrum, b::Spectrum; interpolate)` | ditto |
| `multiply_spectrum(s::Spectrum, factor)` | ditto |
| `average_spectra(specs::Spectrum...; interpolate)` | first spectrum's metadata |
| `interpolate_spectrum(s::Spectrum, new_x)` | result x = `new_x` |
| `transmittance_to_absorbance(s; percent)` | sets `:ylabel => "Absorbance"` |
| `absorbance_to_transmittance(s; percent)` | sets `:ylabel` to `"Transmittance"`, or `"Transmittance (%)"` when `percent=true` |
| `snv(s)` | metadata propagated unchanged (same quantity, standardized) |
| `kubelka_munk(s)` | y must be reflectance; sets `:ylabel => "F(R)"` |

Mixed-type arithmetic (`subtract_spectrum(::Spectrum, ::GatedSpectrum)`) falls
through to the existing generic NamedTuple methods — acceptable and documented.

Functions intentionally **not** given `Spectrum` methods now: `kramers_kronig`
(multi-quantity signature), `tauc_plot` / `urbach_tail` (return fit results,
candidates for later generic analysis methods), the five raw `*_baseline`
functions (low-level; `correct_baseline` is the typed entry point),
`fit_ta_spectrum` (TA-specific).

## Plotting

No new code: the extension's existing
`Makie.convert_arguments(::PointBased, ::AbstractSpectroscopyData)` makes
`lines(spec)` / `scatter(spec)` work the moment `Spectrum` subtypes
`AbstractSpectroscopyData`. Add a test.

## Exports

`Spectrum` added to the types export line in `src/OpticalSpectroscopy.jl`.
No new function names — every dispatch extends an existing export.

## Testing (all synthetic, in `test/runtests.jl`)

1. Construction: all three constructor forms; length-mismatch ArgumentError;
   integer/range inputs convert to `Vector{Float64}`.
2. Interface: `xdata`/`ydata`/`zdata`/`is_matrix`/`npoints`/`title`;
   `xlabel`/`ylabel` with and without metadata keys (detection fallback);
   `source_file` from `:filename`/`:source`; `signal`.
3. `show`: both MIME forms, smoke-level.
4. Generic analysis: each new method on a `Spectrum` and on one other family
   type matches the vector-API result; `is_matrix` guard errors on
   `TimeResolvedMatrix` (including for the pre-existing generic `fit_peaks`
   and arithmetic methods).
5. Transformations: each returns `Spectrum`, y matches vector API, x correct,
   metadata propagated (and not aliased — mutating result metadata must not
   affect the input), `:ylabel` updates where specified.
6. Makie: `lines(Spectrum(...))` works in the existing extension testset.

## Docs and housekeeping

- Docstrings for the type, constructors, and every new method (repo enforces
  `missing_docs`).
- Add `Spectrum` to the docs API reference page and to the type-hierarchy
  diagrams in CLAUDE.md and `src/types.jl`'s `AbstractSpectroscopyData`
  docstring if it lists members.
- Version stays **0.1.0** (initial registration not yet triggered).
- Feature branch + PR to `main`.

## Out of scope / follow-ups

- CavitySpectroscopy merge and `fit_cavity_spectrum(::Spectrum)`.
- QPSTools deleting its forwarding dispatches (it gains the generic analysis
  methods automatically once it updates).
- A `rebuild(spec, x, y)` interface for fully generic transformations —
  revisit only if real usage demands it.
