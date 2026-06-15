# Merge CavitySpectroscopy.jl into OpticalSpectroscopy.jl

**Date:** 2026-06-12
**Status:** Approved design, pending implementation plan
**Repos touched:** OpticalSpectroscopy.jl (receives code), QPSTools.jl (rewired), QPSLab (dep removal), CavitySpectroscopy.jl (archived). This document is the coordinating spec; it lives in OpticalSpectroscopy because that is where the code lands.

## Context

CavitySpectroscopy.jl (~1,000 LOC: Fabry-Pérot physics, polariton models, spectrum/dispersion fitting) was split out on the theory that cavity polariton analysis would grow into a standalone public package. In practice the boundary bought no focus and created friction:

- Its dependencies (CurveFit, CurveFitModels, LinearAlgebra, Statistics) are a strict subset of OpticalSpectroscopy's — the split spares no consumer anything.
- Its audience (polariton researchers) needs OpticalSpectroscopy's baselines, peak fitting, and transforms anyway — nobody wants one without the other.
- Because the two packages share no common base, they cannot share generic functions. This produced two unrelated `wavenumber` generics, two `CavitySpectrum` types (a plain x/y/metadata one in CavitySpectroscopy, a JASCO-backed one in QPSTools), and a hand-written bridge in QPSTools routing `OpticalSpectroscopy.format_results` to `CavitySpectroscopy.format_results`.
- It costs a standalone CI, docs site, registration slot, and `[sources]` entries in QPSTools and QPSLab (the latter never even uses it in source).

Both packages are unregistered at 0.1.0, so merging now costs nothing in deprecation cycles.

**Boundary principle going forward:** extract a domain package only when it accumulates heavy dependencies or a genuinely separate audience — not by topic alone. The roadmap's "domain packages at maturity" vision (TwoDIR, RamanTools, …) stays alive for cases that meet that bar.

## Decision

Merge CavitySpectroscopy's **physics and fitting** into OpticalSpectroscopy as a by-topic source file. **Drop the generic `CavitySpectrum` container** (and its `transmittance` accessor) — it does not survive the merge. The cavity API in OpticalSpectroscopy is vector-in, result-out, consistent with the package's existing convention for 1D steady-state data.

QPSTools' JASCO-backed `CavitySpectrum <: AnnotatedSpectrum` is unchanged by this merge, but is understood to be **transitional**: a separate workstream is designing a generic 1D steady-state spectrum type for OpticalSpectroscopy, and QPSTools' `CavitySpectrum` is expected to dissolve into it later. This merge therefore adds nothing new on top of `CavitySpectrum`, and keeps the cavity fitting container-agnostic so that workstream can add a dispatch rather than refactor.

## Changes by repo

### OpticalSpectroscopy.jl

- **New `src/cavity.jl`** (~650 LOC), following the existing flat by-topic file convention. Contents, moved verbatim where possible:
  - From `physics.jl`: `cavity_transmittance`, `compute_cavity_transmittance`, `refractive_index`, `extinction_coeff`, `cavity_mode_energy`, `polariton_branches`, `polariton_eigenvalues`, `hopfield_coefficients`.
  - From `types.jl`: `CavityFitResult`, `DispersionFitResult`, their `show` methods, and their `predict`/`residuals`/`format_results` methods. The `format_results` methods attach to OpticalSpectroscopy's **existing** generic. NOT moved: the generic `CavitySpectrum` struct, its constructors, `wavenumber`/`transmittance` accessors, and `show` methods.
  - From `fitting.jl`: `fit_cavity_spectrum(nu, T; ...)` and all `fit_dispersion` methods (including `fit_dispersion(::Vector{CavityFitResult})`). NOT moved: the `fit_cavity_spectrum(::CavitySpectrum)` dispatch.
- **No new dependencies.** Required imports (`Symmetric`/`eigvals` from LinearAlgebra, `mean` from Statistics, `dielectric_real`/`dielectric_imag` from CurveFitModels, CurveFit's problem/solve API) are already available in the module.
- **New exports:** the eight physics names above, `CavityFitResult`, `DispersionFitResult`, `fit_cavity_spectrum`, `fit_dispersion`. No collisions: `predict`/`residuals`/`format_results`/`wavenumber` are already exported generics gaining methods, and `transmittance` is not carried over.
- **Module docstring** gains a cavity/polariton bullet.
- **Tests:** port `test_physics.jl` and `test_fitting.jl` into the existing test suite layout. The one container-based test case (`fit_cavity_spectrum(spec::CavitySpectrum)` with percent transmittance and metadata cavity length) is rewritten against plain vectors with explicit `L`. `test_types.jl` (which tests the dropped container) is not ported. Aqua runs over the merged package.
- **Docs:** fold CavitySpectroscopy's Documenter content (index + lib reference) into the existing docs tree — cavity entries in `api.md` plus a reference page for the physics chain. Numerics documentation moves as-is; no behavior changes to document.
- **Version stays 0.1.0** (unregistered; pre-registration breaking changes are free).

### QPSTools.jl

- `Project.toml`: remove CavitySpectroscopy from `[deps]` and `[sources]` (no `[compat]` entry exists, per the no-compat-for-sources rule).
- `src/QPSTools.jl`: delete the `using CavitySpectroscopy: ...` block; change `import CavitySpectroscopy: fit_cavity_spectrum, fit_dispersion` to `import OpticalSpectroscopy: fit_cavity_spectrum, fit_dispersion`. The blanket `using OpticalSpectroscopy` already brings the cavity vocabulary into scope.
- **Delete the cavity re-export exception**: drop `export CavityFitResult, DispersionFitResult, fit_cavity_spectrum, fit_dispersion, compute_cavity_transmittance, cavity_transmittance, cavity_mode_energy, polariton_branches, polariton_eigenvalues, hopfield_coefficients`. Students get these from `using OpticalSpectroscopy`, which the documented loading pattern already includes. QPSTools keeps exporting only what it owns: `CavitySpectrum` (its JASCO-backed type), `load_cavity`, and the plotting functions.
- `src/cavity.jl`: delete the `format_results` bridge methods (one generic now). The `fit_cavity_spectrum(::CavitySpectrum)` dispatch stays, now extending OpticalSpectroscopy's function. Update the file header docstring.
- Tests: drop `import CavitySpectroscopy` from `testsetup.jl`; in `runtests.jl`, the stale-binding checks and qualified references (`CavitySpectroscopy.CavityFitResult` etc.) point at `OpticalSpectroscopy` instead.
- `CLAUDE.md`: update the scope section (cavity layer now backed by OpticalSpectroscopy; re-export exception removed) and fix the stale `cavity_transmittance` mention in the structure listing. Update any docs-page references to the CavitySpectroscopy package.

### QPSLab

- `server/Project.toml`: remove the CavitySpectroscopy `[deps]` and `[sources]` entries (declared but unused in source).

### CavitySpectroscopy.jl repo

- Final commit replaces README content with a pointer: merged into OpticalSpectroscopy.jl (2026-06), history preserved here. Then archive the GitHub repo (`gh repo archive`) — matching the QPSView/LVM precedent. Archival is gated on the OpticalSpectroscopy and QPSTools changes landing.

### Ecosystem documentation

- `~/Developer/QPSLab/docs/dev/ecosystem-roadmap.md`: remove CavitySpectroscopy from the dependency diagram and package table; annotate the "domain packages at maturity" section with the extraction bar (heavy deps or separate audience).
- Showcase site repo (`garrekstemo/qps-ecosystem`): remove the CavitySpectroscopy entry / fold its description into OpticalSpectroscopy's.
- Global `~/.claude/CLAUDE.md`: update the ecosystem diagram (cavity line moves under OpticalSpectroscopy's description), the registration-status list, and the "Where code goes" table row for cavity/polariton physics.

## Relation to the 1D steady-state spectrum type workstream

A separate session (spawned task) is brainstorming a generic 1D steady-state spectrum type for OpticalSpectroscopy. Division of labor:

- **This merge** delivers the vector-based cavity API and deletes the generic `CavitySpectrum`. It deliberately avoids entrenching QPSTools' `CavitySpectrum` (no new methods on it).
- **That workstream** owns: the new type itself; a `fit_cavity_spectrum(::<NewType>)` dispatch reading `cavity_length` from spectrum metadata (recovering, generally, the convenience the dropped container provided); and the likely dissolution of QPSTools' `CavitySpectrum` into the new type (with `load_cavity`, eLabFTW glue, and plot-title machinery rewired).
- **Sequencing:** this merge lands first; the type workstream builds on the merged cavity code.

## Execution order

1. **OpticalSpectroscopy PR** — add `src/cavity.jl`, exports, tests, docs. Pure code move; ported tests must pass with unchanged numerics.
2. **QPSTools PR** — rewire imports/exports, drop the dep, update tests and CLAUDE.md. Depends on (1) being merged to main, since the `[sources]` URL tracks the default branch.
3. **QPSLab dep removal** — independent; any time.
4. **Archive + ecosystem docs** — after (1) and (2) land.

## Out of scope

- The 1D spectrum type (separate workstream, above).
- Any change to physics or fitting numerics — this is a code move, verified by the ported test suite.
- Registration of OpticalSpectroscopy (still pending Garrek's go, unchanged by this merge).
