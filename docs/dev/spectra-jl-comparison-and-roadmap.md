# OpticalSpectroscopy.jl vs Spectra.jl — Comparison & Improvement Roadmap

*Internal strategy doc. Not for the published docs site.*
*Snapshot date: 2026-05-13. Spectra.jl reference version: v2.1.1 (April 2026).*

> **Naming note:** This package was previously `SpectroscopyTools.jl`. Renamed to `OpticalSpectroscopy.jl` on 2026-05-13 after the scope analysis below. The in-repo rename (Project.toml, src/, ext/, exports, docstrings, examples, tests, docs, CLAUDE.md, AGENTS.md) is complete as of 2026-05-25 on branch `claude/dazzling-heisenberg-d84dd0`. Still outstanding: GitHub repo rename, local directory rename (`~/Developer/SpectroscopyTools.jl` → `~/Developer/OpticalSpectroscopy.jl`), and downstream sweep across `QPSTools.jl`, `QPSDrive.jl`, and `QPSLab/server`. See [Suggested order of execution](#suggested-order-of-execution).

## Why this exists

A side-by-side audit of [OpticalSpectroscopy.jl](https://github.com/garrekstemo/OpticalSpectroscopy.jl) (formerly `SpectroscopyTools.jl`) and [charlesll/Spectra.jl](https://github.com/charlesll/Spectra.jl) — the only other Julia package in the optical-spectroscopy preprocessing/fitting niche.
Purpose: prove the package is worth pushing on, identify *closable* feature gaps, and avoid the structural traps the comparison surfaced.

## TL;DR

OpticalSpectroscopy.jl is **structurally better** than Spectra.jl on type system, dispatch, ecosystem layering, dependency hygiene, plotting via extension, and CI.
Spectra.jl is **feature-richer in the overlap** (more baselines, bootstrap, posterior covariance, auto-λ) and **actually used** (registered, ~40 stars, downloads).
They have **almost-disjoint domains within optical spectroscopy**: TA/ultrafast/PL/chirp here; Raman preprocessing + Raman-specific corrections there.
Closing the overlap-feature gaps + completing the rename + registering is the move. The architecture does not need to change.

## Scope

OpticalSpectroscopy.jl targets photon-based spectroscopy of condensed matter and molecular systems:

- **Steady-state**: UV-Vis, FTIR, Raman, photoluminescence
- **Time-resolved**: femtosecond transient absorption, broadband pump-probe
- **Spatial**: PL/Raman mapping, hyperspectral imaging primitives

Explicitly out of scope: NMR, EPR, X-ray (XRD/XPS/XAS), mass spec, atomic/plasma, astronomical, attosecond. These have different data conventions and analysis traditions that don't fit the optical workflow or the `AbstractSpectroscopyData` type hierarchy.

The shared algorithmic primitives — peak fitting, baseline correction, smoothing, unit conversions — operate on any 1D signal data and are usable across domains as needed. Naming doesn't gatekeep imports.

## What Spectra.jl does that we don't (the closable list)

Ordered by priority.

### High priority

1. **Bootstrap uncertainty estimation on fit results.** Spectra.jl's `bootstrap.jl` resamples residuals and refits, returning parameter distributions. Add as `fit_peaks(...; bootstrap=1000)` returning a populated `bootstrap` field on `MultiPeakFitResult` (and analogously on `ExpDecayFit`, `MultiexpDecayFit`, `GlobalFitResult`).
2. **More baseline algorithms.** Currently 5 (ARPLS, SNIP, rubberband, iModPoly, rolling ball). Add:
   - **ALS** (Eilers & Boelens, the original asymmetric least squares — predecessor to ARPLS, still widely used)
   - **drPLS** (doubly reweighted PLS — newer than arPLS, sometimes better on slopes)
   - **Whittaker smoother + baseline** with auto-λ via L-curve (Spectra.jl added this in v2.1.0 — auto-λ is the differentiator, users hate guessing)
   - **Polynomial baseline** (plain, not iModPoly — sometimes you want simple)
   These all share infrastructure with what's already in [`baseline.jl`](../../src/baseline.jl). Same `correct_baseline(y; method=:als, ...)` dispatch surface.
3. **Auto-tuning for ARPLS λ.** Same L-curve approach. Currently users have to pick `λ=1e5` blind.
4. **Posterior covariance / proper parameter uncertainty.** `PeakFitResult` already has `ci` and `errors` from CurveFit's `stderror`/`confint`. Spectra.jl computes a full Hessian-based posterior covariance via IPNewton + ForwardDiff with explicit priors. Worth evaluating whether CurveFit can give us comparable quality, or whether to add a Bayesian backend option.

### Medium priority

5. **Complete the rename sweep, then register.** README is done. Outstanding: `Project.toml`, `src/SpectroscopyTools.jl` → `src/OpticalSpectroscopy.jl`, module declaration, exports, docstrings, examples, tests, `CLAUDE.md`, docs (`make.jl` + pages), GitHub repo rename, downstream `QPSTools.jl` (`[deps]`, `[sources]`, every `using SpectroscopyTools`). Then register as `OpticalSpectroscopy` — AutoMerge clears cleanly (verified no name collision in General).
6. **Split the long modules.** Internal smell. Both ~800 lines:
   - [`fitting.jl`](../../src/fitting.jl) → `exp_decay.jl`, `global_fit.jl`, `ta_spectrum.jl`
   - [`peakfitting.jl`](../../src/peakfitting.jl) — split into single-peak / multi-peak / model-glue if the seams are clean
7. **Consolidate duplicated helpers.** `_polyeval`, `_polyfit`, `_lls_transform`-style helpers appear across [`baseline.jl`](../../src/baseline.jl) and [`chirp.jl`](../../src/chirp.jl). Move to `utils.jl` (or `_internal.jl`).
8. **Surface the ultrafast subdomain — partially resolved by the rename.** The scope-honest name already telegraphs that ultrafast is a first-class feature, not a hidden tax buried in a generic "tools" package. The README's Scope section makes the bounds explicit. Submodule organization (`OpticalSpectroscopy.TA` for TA-specific types and chirp) remains an option but is now lower priority.

### Low priority / evaluate later

9. **Smoothing API parity.** Spectra.jl has GCV-spline smoothing, Whittaker smoother, plus all the standard windows (hanning/hamming/bartlett/blackman) via DSP. We have Savitzky-Golay only. Probably fine — users with smoothing needs reach for DSP.jl directly. Worth adding Whittaker smoother since the baseline version pulls in the same machinery.
10. **`peak_table` enhancements.** Spectra.jl returns peak metadata in a uniform way; ours does too. Not a gap, just verify the columns are comparable for users coming from there.

## Things to deliberately *not* do

These are scope traps:

- **Don't add NMR or XAS support.** Spectra.jl claims them and has nothing. Different domain (FID, lineshape conventions, chemical shift referencing for NMR; absorption-edge fitting and EXAFS for XAS). Not worth touching — and now actively against the optical scope the name commits to.
- **Don't add Raman TL correction (Long / Galeener / Hehlen).** It exists in Spectra.jl, works, and is correct. Reimplementing it splinters effort. If users need it: `using Spectra; tlcorrection(...)` is a one-liner — they can use both packages.
- **Don't add a Python ML wrapper.** Spectra.jl's `ml.jl` calls `rampy` via PythonCall. We don't want PythonCall as a dep, and machine-learning-on-spectra is a different package's problem.
- **Don't add diamond anvil cell utilities.** Spectra.jl has them. Geochemistry-specific.
- **Don't add XRD-specific analysis (Bragg's law, Rietveld, indexing).** XRD users can use the shared primitives (peak fitting, baseline correction) from this package, but XRD-specific workflows belong in a hypothetical future `XRD.jl`, not here. Same reasoning for any non-optical sub-field that asks: primitives yes, integrated workflow no.
- **Don't expand beyond the optical scope.** The name commits the package to optical spectroscopy. Stretching it to cover NMR/XRD/MS/EELS would be the inverse of Spectra.jl's promises trap — overdelivering past the scope the name claims, which carries its own kind of confusion.

## Architectural strengths to protect

The audit confirmed these are the load-bearing assets. They are the reason the package is worth pushing.

- **`AbstractSpectroscopyData` hierarchy** with uniform `xdata`/`ydata`/`is_matrix`/`title`/etc. interface. This is what lets QPSTools.jl extend with `AnnotatedSpectrum` and what would let downstream packages add their own spectrum types. Spectra.jl has nothing like it.
- **Typed result structs** (`MultiPeakFitResult`, `ExpDecayFit`, `GlobalFitResult`, `TASpectrumFit`, `FitMapResult`) with proper accessors. Not tuples, not Dicts.
- **Symbol/type dispatch over string-keyed methods.** Spectra.jl uses `method="arPLS"` strings throughout. Stay with `method=:arpls`.
- **Makie as a weak/extension dependency.** Spectra.jl has Plots as a hard dep — multi-minute precompile tax for every user. Don't regress.
- **Three-layer separation:** CurveFitModels.jl (pure math) → OpticalSpectroscopy.jl (algorithms + types) → QPSTools.jl (lab I/O + themes + eLabFTW). Spectra.jl is one flat namespace mixing model functions, Raman physics corrections, DAC utilities, and Python ML wrappers.
- **`f(p, x)` model signatures** (ForwardDiff-friendly, CurveFit convention). Spectra.jl uses `gaussian(x, amplitude, center, hwhm)` — lmfit-style, harder to compose.

## The naming decision

The package was renamed from `SpectroscopyTools.jl` to `OpticalSpectroscopy.jl` after evaluating alternatives in a structured four-perspective debate (bold-brand, pragmatic-conservative, scope-honest-realist, convention-purist). Three observations drove the outcome:

1. **The Python spectroscopy ecosystem** (30+ years mature) has no canonical `spectroscopy` package. Successful packages chose scope-honest names: `specutils` for Astropy spectroscopy, `hyperspy` for multi-dimensional microscopy signals (closest architectural analog to ours), `pybaselines` for baselines, `lmfit` for fitting, `rampy` for Raman. The unified-domain naming pattern Julia favors in `DataFrames`/`Plots`/`Makie` hasn't worked in spectroscopy *anywhere* because the sub-fields are too disjoint.

2. **The actual scope is optical.** ~60-65% of the source is ultrafast/PL/spatial specialized; the rest is optical-spectroscopy primitives that happen to be domain-general. The package does not cover magnetic resonance, X-ray, mass spec, plasma, or astronomical — which the bare noun "spectroscopy" implies. `OpticalSpectroscopy.jl` accurately describes the slice.

3. **The scope-realist critique of `*Base` naming.** `SpectroscopyBase.jl` would claim foundation status before any ecosystem exists to be the foundation of. `*Base` packages in Julia are extracted from working multi-package ecosystems (StatsBase, SciMLBase, DiffEqBase, MakieCore); declaring a base layer without one is premature. If a second cross-domain consumer ever materializes, *that* is the moment to extract a shared primitives layer.

Alternatives considered and rejected:

| Name | Why rejected |
|---|---|
| `SpectroscopyTools.jl` (kept) | "Tools" undersells the typed architecture; generic-utility connotation |
| `Spectroscopy.jl` | Claims too much scope; AutoMerge friction with `Spectra.jl` (prefix containment); promises-trap risk |
| `SpectroscopyBase.jl` | Foundation claim premature — no ecosystem to be the base of yet |
| `UltrafastSpectroscopy.jl` | Too narrow — alienates the Raman/FTIR/PL/UV-Vis users the package actually serves |
| `SpectraBase.jl` | Parasitic on Spectra.jl's namespace; AutoMerge gray zone |

## The "promises trap" — partially resolved by the rename

Spectra.jl's README claims support for "Raman, Infrared, Nuclear Magnetic Resonance, XAS..." In practice ~all dedicated features are Raman. NMR has no FID handling, no chemical shift referencing, no peak picking model for NMR. XAS has no edge fitting, no EXAFS extraction. The 532 nm laser default in `invcm_to_nm` and the Long/Galeener/Hehlen correction give away the actual domain.

The trap: a user who arrives wanting NMR support, finds nothing, and bounces. Worse, they tell others "Spectra.jl doesn't really do NMR." The package's perceived value drops below its real value. This is also a maintenance trap — once you've claimed a domain, every issue filed against it costs you to triage even if the answer is "out of scope."

**The rename to `OpticalSpectroscopy.jl` resolves this structurally.** The name no longer claims domain coverage it doesn't have. The README's Scope section makes the bounds explicit. Discipline still applies — don't add NMR/XAS features to chase users — but the trap is no longer load-bearing.

What remains the same: be explicit about features in the README, list techniques specifically (UV-Vis, FTIR, Raman, PL, fs-TA), don't gesture at domains you don't actually serve.

## Splinter-risk vs Spectra.jl

This is a real concern. Mitigations:

1. **The packages are actually complementary.** TA/PL/chirp doesn't exist in Spectra.jl. Raman TL correction and DAC utilities don't exist here. State this clearly in the README. Cross-link. The disjoint-scope framing is now reinforced by the `OpticalSpectroscopy` name making the boundaries less ambiguous.
2. **Don't duplicate Spectra.jl's distinctive features.** Specifically: no Raman TL correction, no DAC utilities, no Python ML wrapper. Let `using Spectra` cover those for users who need them.
3. **Coordinate, don't compete on shared turf.** Opening an issue on Spectra.jl when adding/improving baselines ("we're adding drPLS — happy to cross-link or share test cases") is collegial and signals intent. The rename gives a natural opening to introduce yourself ("I'm registering OpticalSpectroscopy.jl with a deliberately disjoint scope...").
4. **Resist the urge to absorb Spectra.jl's scope.** Their package will keep existing and being maintained. That's good.
5. **Adoption is the real anti-splinter strategy.** Two packages with disjoint scope and clear documentation isn't splintering — it's the ecosystem working. Splintering is two packages doing the same thing with the same scope competing for the same users.

## Open questions

- Does CurveFit.jl already produce a posterior covariance comparable to Spectra.jl's Hessian-based approach? If yes, item 4 in the high-priority list is mostly a *documentation* gap — show users how to get it, not a feature gap.
- Should `AbstractSpectroscopyData` be extracted into its own micro-package (`SpectroscopyInterfaces.jl`) so QPSTools.jl and any future ecosystem package can depend on the interface without pulling all of OpticalSpectroscopy.jl? Probably not yet — wait for a second downstream consumer. **If a future cross-domain Julia spectroscopy package (XRD, NMR, etc.) wants to share primitives, *that* is the extraction moment, not before.**
- Submodule vs flat namespace for TA-specific features. Defer until registration is done; don't break API for organizational reasons pre-1.0. The rename reduces but doesn't eliminate the question.

## Suggested order of execution

1. ~~**README revision: explicit feature list, cross-link to Spectra.jl, scope discipline.**~~ ✓ Done 2026-05-13.
2. **Complete the rename sweep** — `Project.toml`, `src/`, module name, exports, docstrings, examples, tests, `CLAUDE.md`, docs (`make.jl` + page contents), this doc's internal references where stale, GitHub repo rename, downstream `QPSTools.jl`. One coordinated commit/PR.
3. **Register the package** as `OpticalSpectroscopy`. AutoMerge should clear cleanly.
4. **Introduce the package to Spectra.jl's maintainer.** Open a friendly issue on `charlesll/Spectra.jl` noting the scope distinction and offering to cross-link. Ten-minute investment, real anti-splinter signal.
5. **Add ALS + drPLS + Whittaker + polynomial baselines + auto-λ.** (One PR; baseline.jl is well-organized.)
6. **Add bootstrap to fit results.** (One PR; touches all fit result types.)
7. **Split `fitting.jl` and `peakfitting.jl`; consolidate internal helpers into `utils.jl`.** (Pure refactor; before 1.0.)
8. **Evaluate posterior covariance story; document or add as appropriate.**

## References

- OpticalSpectroscopy.jl audit results: in conversation transcript, 2026-05-13. Audit was originally performed on `SpectroscopyTools.jl` pre-rename; findings still apply.
- Spectra.jl audit: same date. Sources: https://github.com/charlesll/Spectra.jl, https://charlesll.github.io/Spectra.jl/stable/.
- Python spectroscopy ecosystem survey (conversation, 2026-05-13): HyperSpy (multi-dimensional microscopy signals with typed `BaseSignal` → `Signal1D`/`Signal2D` hierarchy — closest architectural analog), specutils (Astropy spectroscopy), pybaselines (50+ baseline algorithms), lmfit (canonical Python fitting), rampy (Le Losq's Python Raman package — the Python analog of Spectra.jl).
- Naming debate (4-agent two-round structured debate, 2026-05-13): bold-brand / pragmatic-conservative / scope-honest-realist / convention-purist perspectives. Round 1 produced 4 distinct top picks; Round 2 critique pass plus the Python ecosystem evidence converged on `OpticalSpectroscopy.jl`.
- Three-package ecosystem context: [`CLAUDE.md`](../../CLAUDE.md) in this repo and `/Users/garrek/Developer/QPSLab/docs/dev/ecosystem-roadmap.md`.
