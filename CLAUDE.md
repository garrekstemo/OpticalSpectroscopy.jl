# OpticalSpectroscopy.jl

General-purpose spectroscopy analysis tools for Julia. Public, registerable package — lab-specific features (instrument I/O, sample registries, eLabFTW) stay in QPSTools.jl.

## Package Scope

### In Scope

- Peak fitting (Gaussian, Lorentzian, Pseudo-Voigt) via CurveFit.jl + CurveFitModels.jl
- Peak detection via Peaks.jl
- Baseline correction (arPLS, SNIP, rubberband, iModPoly, rolling ball)
- Exponential decay fitting (single/multi-exponential) with optional IRF convolution
- Global fitting with shared parameters across traces
- Unit conversions (wavenumber/wavelength/energy, linewidth/decay-time)
- Normalization, smoothing, spectral math
- Chirp correction for broadband TA (detection, correction, serialization)
- Background subtraction for TA matrices
- Typed spectroscopy data (`AbstractSpectroscopyData` hierarchy)
- Plotting via Makie extension (no themes — users set their own)

### Out of Scope (stays in QPSTools.jl)

- Instrument-specific I/O (LabVIEW .lvm, JASCO CSV)
- Sample registries and metadata lookup
- eLabFTW integration
- Lab themes (`qps_theme`, `publication_theme`)

## Dependencies

**Target: < 15 direct dependencies.** Currently 11 (8 non-stdlib).

**Compat entries required**: All non-stdlib dependencies MUST have `[compat]` entries in Project.toml (required for General registry registration).

### Extensions (Weak Dependencies)

| Extension | Trigger | Purpose |
|-----------|---------|---------|
| `OpticalSpectroscopyMakieExt` | CairoMakie or GLMakie | Plotting functions |

## Type Hierarchy

```
AbstractSpectroscopyData (root interface)
├── Spectrum            (generic 1D steady-state spectrum: signal vs spectral axis)
├── KineticTrace        (kinetics: signal vs time at fixed wavelength)
├── TASpectrum          (spectrum: signal vs wavenumber at fixed time)
├── GatedSpectrum       (spectrum extracted from a time window)
├── TimeResolvedMatrix  (2D: time × wavelength heatmap)
└── PLMap               (2D: spatial PL/Raman map)
```

`AnnotatedSpectrum` (with metadata, FTIR/Raman subtypes) is defined in QPSTools.jl, not here.

All types implement: `xdata`, `ydata`, `zdata`, `xlabel`, `ylabel`, `is_matrix`, `source_file`, `npoints`, `title`.

## API Design Principles

- **Dual interface**: Functions accept typed spectroscopy data (preferred) or raw vectors.
- **Model functions from CurveFitModels.jl**: Never define fitting functions inline.
- **Fit results are structs**: Every fitting function returns a proper result type with accessors (`predict`, `residuals`, `report`).
- **No themes in plotting**: Extension provides functions, not aesthetics. Only inline styling for semantic distinction (e.g., `color=:red` for fit vs data).

## CurveFit.jl Integration

OpticalSpectroscopy extends `CurveFit.residuals`, `CurveFit.predict`, and `CurveFit.fitted` with methods for its own fit result types. Use `import CurveFit: residuals, predict, fitted` (not just `using`) to enable method extension. CurveFit does NOT provide R² — compute as `1 - rss(sol) / ss_tot`.

## Three-Package Boundary

| Package | Owns |
|---------|------|
| **CurveFitModels.jl** | Model functions (`gaussian`, `lorentzian`, etc.). Zero dependencies. |
| **OpticalSpectroscopy.jl** | Types, fitting, baseline, peak detection, chirp, units. Uses CurveFitModels internally. |
| **QPSTools.jl** | Instrument I/O, sample registry, eLabFTW, Makie themes. Re-exports OpticalSpectroscopy. |

- OpticalSpectroscopy does NOT re-export model constructors that are only used internally.
- QPSTools `import`s OpticalSpectroscopy functions to extend with `AnnotatedSpectrum` dispatches.
- Lab members use `using QPSTools` — OpticalSpectroscopy is invisible to them.

## Package Structure

```
src/
  OpticalSpectroscopy.jl  # Main module
  types.jl                # AbstractSpectroscopyData + fit result types
  fitting.jl              # Exponential decay (single/multi/global + IRF)
  peakfitting.jl          # Multi-peak fitting + TA spectrum fitting
  peakdetection.jl        # Peak finding
  baseline.jl             # arPLS, SNIP, rubberband, iModPoly, rolling ball
  spectroscopy.jl         # Normalize, conversions, smoothing
  transforms.jl           # Kramers-Kronig, Kubelka-Munk, Tauc, SNV
  units.jl                # Unitful conversions
  chirp.jl                # Chirp detection, correction, serialization
  plmap.jl                # PLMap type, fit_map, intensity masks
  decomposition.jl        # PCA / NMF map decomposition
  cosmic_rays.jl          # Cosmic ray detection and removal
```

## Development

- **Not yet registered.**
- All tests use synthetic data — no local file dependencies.
- CI runs on Ubuntu only (Julia 1.10 and 1.12).
