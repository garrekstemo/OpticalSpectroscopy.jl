# OpticalSpectroscopy.jl

Analysis layer of the lab spectroscopy stack (see global CLAUDE.md ecosystem map). Pure analysis — instrument I/O, sample registries, eLabFTW glue, and Makie themes (`qps_theme`, `publication_theme`) stay in QPSTools.jl.

## Scope

In scope: peak fitting/detection, baseline correction, exponential decay (single/multi/global + IRF convolution), unit conversions, normalization/smoothing/spectral math, chirp correction and background subtraction for broadband TA, the typed `AbstractSpectroscopyData` hierarchy, transforms (Kramers-Kronig, Kubelka-Munk, Tauc, SNV), PL/Raman map decomposition (PCA/NMF), cosmic-ray removal, **cavity/polariton physics + fitting** (Hopfield, Rabi, dispersion — `src/cavity.jl`), and the **token-driven TA/gated semantics** (`src/tokens.jl`, `src/timeresolved.jl`).

Dependency budget: target **< 15 direct deps**.

File-reader packages (JASCOFiles, HamamatsuStreakFiles) own no transforms or axis labels — they emit raw data plus the instrument's native unit strings. All unit conversions (including transmittance↔absorbance) and axis labels live here, in the analysis layer.

## Types

`AbstractSpectroscopyData` root, with concrete `Spectrum` (generic 1D; also carries TA slices and gated spectra via metadata tokens), `KineticTrace`, `TimeResolvedMatrix`, `PLMap`. `SweepData` is a separate (non-`AbstractSpectroscopyData`) struct in `types.jl`.

Interface contract — every `AbstractSpectroscopyData` type implements: `xdata`, `ydata`, `zdata`, `xlabel`, `ylabel`, `is_matrix`, `source_file`, `npoints`, `title`.

`AnnotatedSpectrum` (with FTIR/Raman subtypes) is defined in **QPSTools.jl, NOT here** — do not add it to this package.

Metadata tokens: every axis is a `(quantity, unit)` pair of `Symbol` tokens; display labels are *derived* from tokens, never stored as prose or guessed from data magnitudes. See `docs/superpowers/2026-06-15-metadata-token-contract.md`.

## Cavity / polaritons (`src/cavity.jl`)

Live API is vector-based: `polariton_branches(E_cav, E_vib, Omega)`, `hopfield_coefficients(E_cav, E_vib, Omega)`, `fit_cavity_spectrum(nu, T; ...)` with a `fit_cavity_spectrum(s::Spectrum; ...)` method dispatching on the generic `Spectrum`. This layer was merged in from the now-archived CavitySpectroscopy.jl (June 2026); the old `CavitySpectrum` type and `load_cavity` loader were NOT carried over. (Unrelated: the Variable-Rabi-Splitting-VSC-Project repo has its own local module also named `CavitySpectroscopy` — never conflate them.)

## API conventions

- **Dual interface**: functions accept typed spectroscopy data (preferred) or raw vectors.
- **Model functions come from CurveFitModels.jl** — never define fitting functions inline.
- **Fit results are structs** with `predict` / `residuals` / `report` accessors.
- **Plotting (Makie weakdep ext)**: aesthetics-free, no themes here (themes are in QPSTools). Inline styling only for semantic distinction (e.g. fit vs data color).

## CurveFit.jl integration

Extend `CurveFit.residuals`, `CurveFit.predict`, `CurveFit.fitted` for this package's fit-result types via `import CurveFit: residuals, predict, fitted` (not `using` — needed for method extension). CurveFit provides no R² — compute `1 - rss(sol) / ss_tot` yourself.

## Source layout

Entry point `src/OpticalSpectroscopy.jl`; browse `src/`. Baseline algorithms (`src/baseline.jl`): arPLS, SNIP, rubberband, iModPoly, rolling ball.

## Development

- Version 0.1.0, not yet registered (first registers as 0.1).
- Tests use synthetic data only — no local file dependencies.
