# CavitySpectroscopy → OpticalSpectroscopy Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CavitySpectroscopy.jl's physics and fitting into OpticalSpectroscopy.jl as `src/cavity.jl`, rewire QPSTools/QPSLab off the retired package, and archive its repo.

**Architecture:** Pure code move — no numerics change. The generic `CavitySpectrum` container (and its accessors) is dropped; the cavity API in OpticalSpectroscopy is vector-in/result-out. `format_results`, `predict`, `residuals`, `wavenumber` methods attach to OpticalSpectroscopy's existing generics, which deletes QPSTools' bridge code. Spec: `docs/superpowers/specs/2026-06-12-cavityspectroscopy-merge-design.md`.

**Tech Stack:** Julia 1.10+, CurveFit.jl, CurveFitModels.jl, Documenter.jl, Aqua.jl.

**Repos and branches:**
- `~/Developer/OpticalSpectroscopy.jl` — branch `merge-cavityspectroscopy` (exists; holds the spec commit). Phase 1.
- `~/Developer/QPSTools.jl` — branch `streamline-api-surface` (existing unmerged branch; this work is API-surface cleanup, so it stacks there). Phase 2.
- `~/Developer/QPSLab` — Phase 3.
- `~/Developer/CavitySpectroscopy.jl` + ecosystem docs — Phase 4.

**Conventions for every Julia command:** run from the repo root with `PATH=~/.juliaup/bin:$PATH` prepended (bare `julia` is not on PATH in tool shells).

**What does NOT move (drop list — verify none of these appear in the new code):**
- `struct CavitySpectrum` (generic container), its constructors, `Base.show` methods
- `wavenumber(s::CavitySpectrum)`, `transmittance(s::CavitySpectrum)`, `transmittance(r::CavityFitResult)`
- `function format_results end` + its generic docstring (OpticalSpectroscopy already declares this generic at `src/types.jl:1223`)
- `fit_cavity_spectrum(spec::CavitySpectrum; ...)` dispatch
- The `transmittance` export entirely

`wavenumber(r::CavityFitResult)` DOES move — it matches the existing `wavenumber(r::TASpectrumFit)` fit-result-accessor convention in OpticalSpectroscopy.

---

## Phase 1 — OpticalSpectroscopy.jl

### Task 1: Port the test suite (tests first — they must fail before the source exists)

**Files:**
- Create: `~/Developer/OpticalSpectroscopy.jl/test/test_cavity.jl`
- Modify: `~/Developer/OpticalSpectroscopy.jl/test/runtests.jl` (final lines)

- [ ] **Step 1: Confirm you are on the `merge-cavityspectroscopy` branch**

```bash
cd ~/Developer/OpticalSpectroscopy.jl && git status --short --branch
```
Expected: `## merge-cavityspectroscopy`, clean tree.

- [ ] **Step 2: Assemble `test/test_cavity.jl` from the CavitySpectroscopy test files**

Source line ranges (verified against the current files): `test_physics.jl` line 1 is `using LinearAlgebra: eigen`, the `@testset "Physics"` starts at line 3. `test_fitting.jl` lines 99–123 are the `"fit_cavity_spectrum on CavitySpectrum"` testset (container-based — dropped; its two behaviors, percent-normalization and metadata-L, live in QPSTools' JASCO dispatch and are tested in QPSTools' own test_cavity.jl).

```bash
cd ~/Developer/OpticalSpectroscopy.jl
CSTEST=~/Developer/CavitySpectroscopy.jl/test
{
  cat <<'EOF'
# Cavity & polariton physics and fitting (merged from CavitySpectroscopy.jl).
# Ported with the original suite's seed so the noisy-fit tolerances are
# exercised on the same draws.
Random.seed!(20260611)

using LinearAlgebra: eigen

EOF
  sed -n '3,$p' "$CSTEST/test_physics.jl"
  echo
  sed -n '1,98p' "$CSTEST/test_fitting.jl"
  sed -n '124,$p' "$CSTEST/test_fitting.jl"
} > test/test_cavity.jl
```

- [ ] **Step 3: Adapt qualified names and testset titles**

The "Cavity transmittance reflectance fit" testset qualifies CurveFit re-exports as `CavitySpectroscopy.NonlinearCurveFitProblem` / `.solve` / `.coef`; OpticalSpectroscopy re-exports all three unqualified. The generic testset names "Physics"/"Fitting" are too vague inside the 3,000-line combined suite.

```bash
cd ~/Developer/OpticalSpectroscopy.jl
sed -i '' \
  -e 's/CavitySpectroscopy\.NonlinearCurveFitProblem/NonlinearCurveFitProblem/' \
  -e 's/CavitySpectroscopy\.solve/solve/' \
  -e 's/CavitySpectroscopy\.coef/coef/' \
  test/test_cavity.jl
sed -i '' 's/^@testset "Physics" begin/@testset "Cavity \& polariton physics" begin/' test/test_cavity.jl
sed -i '' 's/^@testset "Fitting" begin/@testset "Cavity \& dispersion fitting" begin/' test/test_cavity.jl
grep -n "CavitySpectroscopy\.\|CavitySpectrum" test/test_cavity.jl
```
Expected: the final grep prints **nothing** — no qualified `CavitySpectroscopy.` calls and no references to the dropped `CavitySpectrum` type survive. (The unqualified package name in the header comment is fine and not matched.)

- [ ] **Step 4: Include the new file in `runtests.jl`**

The file ends with the last testset's `end`, a blank line, and the final `end` closing `@testset "OpticalSpectroscopy.jl"`. Insert the include before that final `end` (use Edit with the closing lines as anchor; `import Random` at the top of runtests.jl already makes `Random.seed!` available):

```julia
    include("test_cavity.jl")

end
```

- [ ] **Step 5: Run the cavity tests — verify they FAIL with undefined names**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
PATH=~/.juliaup/bin:$PATH julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -20
```
Expected: FAILURE — `UndefVarError: cavity_transmittance not defined` (or similar) inside "Cavity & polariton physics". Pre-existing testsets still pass.

- [ ] **Step 6: Commit the failing tests**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
git add test/test_cavity.jl test/runtests.jl
git commit -m "test: port cavity physics + fitting tests from CavitySpectroscopy

Container-based testset dropped (covered at the QPSTools layer).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 2: Move the source into `src/cavity.jl` and wire the module

**Files:**
- Create: `~/Developer/OpticalSpectroscopy.jl/src/cavity.jl`
- Modify: `~/Developer/OpticalSpectroscopy.jl/src/OpticalSpectroscopy.jl:5,35,80`

- [ ] **Step 1: Assemble `src/cavity.jl` from verified line ranges**

Verified ranges in `~/Developer/CavitySpectroscopy.jl/src/`:
- `physics.jl` — entire file (232 lines, no container references)
- `types.jl` 91–134 — `CavityFitResult` section header, docstring, struct, `wavenumber(::CavityFitResult)` accessor (skips 1–89: container; skips 136–141: `transmittance` accessors)
- `types.jl` 143–204 — `predict`/`residuals` methods + `Base.show` (skips 206–213: duplicate `function format_results end` generic + docstring)
- `types.jl` 214–end — `format_results(::CavityFitResult)`, `DispersionFitResult` section, `format_results(::DispersionFitResult)`
- `fitting.jl` 1–181 — `_find_local_maxima` + vector `fit_cavity_spectrum` (skips 183–209: container dispatch)
- `fitting.jl` 211–end — all three `fit_dispersion` methods

```bash
cd ~/Developer/OpticalSpectroscopy.jl
CSSRC=~/Developer/CavitySpectroscopy.jl/src
{
  cat <<'EOF'
# Cavity & polariton spectroscopy: Fabry-Pérot physics, coupled-oscillator
# polariton models, and spectrum/dispersion fitting.
# Merged from CavitySpectroscopy.jl (repo archived); the generic
# CavitySpectrum container was deliberately not carried over — this API is
# vector-in, result-out, like the rest of the 1D steady-state layer.

EOF
  cat "$CSSRC/physics.jl"
  echo
  sed -n '91,134p'  "$CSSRC/types.jl"
  echo
  sed -n '143,204p' "$CSSRC/types.jl"
  echo
  sed -n '214,$p'   "$CSSRC/types.jl"
  echo
  sed -n '1,181p'   "$CSSRC/fitting.jl"
  echo
  sed -n '211,$p'   "$CSSRC/fitting.jl"
} > src/cavity.jl
```

- [ ] **Step 2: Verify the assembly is complete and clean**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
grep -n "CavitySpectrum\|transmittance(s::\|transmittance(r::\|function format_results end" src/cavity.jl
grep -c "^function\|^struct" src/cavity.jl
```
Expected: first grep prints **only** docstring/comment lines if any (no `struct CavitySpectrum`, no accessor definitions, no generic redeclaration — if a docstring cross-reference like `[\`CavitySpectrum\`](@ref)` survives in the `fit_dispersion(::Vector{CavityFitResult})` docstring or elsewhere, fix those references to plain text now). Second: ~15 definitions.

Then scan the two docstrings that referenced the dropped container: in `fit_cavity_spectrum`'s vector-method docstring and the module-level mentions, no `CavitySpectrum` @ref may remain:

```bash
grep -n "@ref" src/cavity.jl | grep -i cavityspectrum
```
Expected: nothing (edit any hit to remove the dangling reference).

- [ ] **Step 3: Wire the module file** (`src/OpticalSpectroscopy.jl`)

Three edits:

(a) Module docstring — extend the capability sentence (line 4–5):

```julia
Provides data types, fitting routines, baseline correction, peak detection,
unit conversions, and utility functions for spectroscopic data analysis,
plus cavity & polariton analysis (Fabry-Pérot transmittance, coupled-oscillator
dispersion fitting, Hopfield coefficients).
```

(b) Include — after `include("fitting.jl")`:

```julia
include("fitting.jl")
include("cavity.jl")
```

(c) Exports — insert a new section after the `export fit_exp_decay, ...` / `export mean_lifetime` fitting block:

```julia
# ==========================================================================
# Exports — Cavity & polariton spectroscopy
# ==========================================================================
export cavity_transmittance, compute_cavity_transmittance
export refractive_index, extinction_coeff
export cavity_mode_energy, polariton_branches, polariton_eigenvalues
export hopfield_coefficients
export CavityFitResult, DispersionFitResult
export fit_cavity_spectrum, fit_dispersion
```

- [ ] **Step 4: Run the cavity tests — verify they PASS**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
PATH=~/.juliaup/bin:$PATH julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -15
```
Expected: PASS, including the Aqua testset (no new deps, no piracy — all moved methods extend functions OpticalSpectroscopy imports or owns, with types it now owns). If a noisy-fit tolerance fails from RNG interaction, the re-seed at the top of test_cavity.jl is missing — check Step 2 of Task 1.

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
git add src/cavity.jl src/OpticalSpectroscopy.jl
git commit -m "feat: merge cavity & polariton physics and fitting from CavitySpectroscopy

Pure code move, vector-based API. The generic CavitySpectrum container is
dropped; format_results/predict/residuals/wavenumber methods attach to the
existing generics.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 3: Documentation

**Files:**
- Create: `~/Developer/OpticalSpectroscopy.jl/docs/src/reference/cavity.md`
- Modify: `~/Developer/OpticalSpectroscopy.jl/docs/make.jl` (Reference pages list)
- Modify: `~/Developer/OpticalSpectroscopy.jl/docs/src/index.md` (tagline paragraph)

- [ ] **Step 1: Create the reference page**

Content adapted from CavitySpectroscopy's `index.md` + `lib/public.md`, minus the dropped container, installation section, and module @docs (covered by api.md). The `@docs` entries for `predict(::CavityFitResult)`, `residuals(::CavityFitResult)`, and `wavenumber(::CavityFitResult)` are REQUIRED — `checkdocs=:exports` errors on uncollected docstrings of exported names. Write exactly:

````markdown
# Cavity & Polariton Spectroscopy

Fabry-Pérot cavity and polariton analysis for light–matter strong coupling
experiments (vibrational strong coupling, exciton polaritons), covering the
chain from raw transmittance data to Rabi splitting and Hopfield coefficients:

1. Multi-oscillator Lorentz dielectric function (via CurveFitModels.jl)
2. Complex refractive index ``n``, ``k`` from the dielectric function
3. Fabry-Pérot Airy transmittance with an absorbing intracavity medium
4. Coupled oscillator polariton model: branches, eigenvalues, mixing fractions
5. Nonlinear least-squares fitting (via CurveFit.jl) of spectra and dispersion

## Quick start

### Polariton physics

```julia
using OpticalSpectroscopy

# Cavity photon energy vs incidence angle (radians)
E_cav = cavity_mode_energy([2040.0, 1.5], deg2rad.(0:5:30))

# Upper and lower polariton branches (2-level coupled oscillator model)
LP, UP = polariton_branches(E_cav, 2055.0, 25.0)

# Light-matter mixing fractions
h = hopfield_coefficients(E_cav, 2055.0, 25.0)
h.photon_LP   # photon fraction of the lower polariton at each angle
```

### Fitting a cavity transmission spectrum

```julia
result = fit_cavity_spectrum(nu, T;
    oscillators = [(nu0 = 2055.0, Gamma = 23.0)],
    L = 12.0e-4,      # cavity length (cm)
    n_bg = 1.4)       # background refractive index

result.R                  # fitted mirror reflectance
result.polariton_peaks    # auto-extracted peak positions
predict(result)           # fitted curve on the data grid
```

### Fitting polariton dispersion

```julia
result = fit_dispersion(angles, lp_positions, up_positions;
    molecular_modes = 2055.0)

result.rabi_splitting     # Rabi splitting (with result.rabi_err)
result.hopfield_zero      # mixing fractions at zero detuning
```

## Conventions

- Wavenumber units (cm⁻¹) throughout; angles in radians.
- Model functions follow the `fn(p, x)` signature and are
  ForwardDiff-compatible, so they can be used directly with CurveFit.jl.
- Hopfield convention: with detuning ``\\delta = E_{cav} - E_{vib}`` and
  ``\\theta = \\tfrac{1}{2}\\,\\mathrm{atan2}(\\Omega, \\delta)``, the photon
  fraction of the lower polariton is
  ``\\sin^2\\theta = \\tfrac{1}{2}(1 - \\delta/\\sqrt{\\delta^2 + \\Omega^2})``
  — at far positive detuning the LP converges to the bare vibration
  (matter-like). Verified against direct Hamiltonian diagonalization in
  the test suite.

## Types

```@docs
CavityFitResult
DispersionFitResult
```

## Physics

```@docs
cavity_transmittance
compute_cavity_transmittance
refractive_index
extinction_coeff
cavity_mode_energy
polariton_branches
polariton_eigenvalues
hopfield_coefficients
```

## Fitting

```@docs
fit_cavity_spectrum
fit_dispersion
predict(::CavityFitResult)
residuals(::CavityFitResult)
wavenumber(::CavityFitResult)
```
````

- [ ] **Step 2: Register the page in `docs/make.jl`**

In the `"Reference" => [...]` list, before `"Additional API" => "api.md",` add:

```julia
            "Cavity & Polaritons" => "reference/cavity.md",
```

- [ ] **Step 3: Extend the `docs/src/index.md` tagline paragraph**

Append to the paragraph ending `...chirp correction for broadband pump-probe experiments.`:

```markdown
For light–matter strong coupling work, it provides cavity and polariton analysis (Fabry-Pérot transmittance modeling, dispersion fitting, Hopfield coefficients).
```

- [ ] **Step 4: Build the docs — verify no errors**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
PATH=~/.juliaup/bin:$PATH julia --project=docs -e 'using Pkg; Pkg.instantiate()' && \
PATH=~/.juliaup/bin:$PATH julia --project=docs docs/make.jl 2>&1 | tail -15
```
Expected: build completes; only cross-reference warnings allowed (`warnonly=[:cross_references]`). Any `checkdocs` error means a moved docstring isn't collected — add its entry to the cavity page.

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
git add docs/src/reference/cavity.md docs/make.jl docs/src/index.md
git commit -m "docs: cavity & polariton reference page

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 4: PR and merge (gates Phase 2)

- [ ] **Step 1: Push and open the PR**

```bash
cd ~/Developer/OpticalSpectroscopy.jl
git push -u origin merge-cavityspectroscopy
gh pr create --title "Merge CavitySpectroscopy into OpticalSpectroscopy" --body "Moves cavity & polariton physics and fitting into src/cavity.jl (pure code move, vector-based API, zero new deps). Drops the generic CavitySpectrum container per the design spec (docs/superpowers/specs/2026-06-12-cavityspectroscopy-merge-design.md). CavitySpectroscopy.jl will be archived once downstream rewiring lands.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 2: Wait for CI green, then CHECKPOINT — confirm with Garrek before merging**

```bash
gh pr checks --watch
```

- [ ] **Step 3: Merge (after approval)**

```bash
cd ~/Developer/OpticalSpectroscopy.jl && gh pr merge --squash --delete-branch
```

---

## Phase 2 — QPSTools.jl (requires Phase 1 merged to main)

All on branch `streamline-api-surface`.

### Task 5: Drop the dependency and rewire the module

**Files:**
- Modify: `~/Developer/QPSTools.jl/Project.toml:7,25`
- Modify: `~/Developer/QPSTools.jl/src/QPSTools.jl:54-67,133-139`
- Modify: `~/Developer/QPSTools.jl/src/cavity.jl:1-9,81-96,111-116`

- [ ] **Step 1: Remove the dep from `Project.toml`**

Delete the `CavitySpectroscopy = "23dbb5c6-..."` line from `[deps]` and the `CavitySpectroscopy = {url = ...}` line from `[sources]`. (No `[compat]` entry exists — sources deps carry none.)

- [ ] **Step 2: Replace the import block in `src/QPSTools.jl`**

Replace lines 54–67 (the `# Cavity physics + fitting live in the public CavitySpectroscopy.jl package.` comment block, the `using CavitySpectroscopy: ...` statement, and the `import CavitySpectroscopy: fit_cavity_spectrum, fit_dispersion` line) with:

```julia
# Cavity physics + fitting live in OpticalSpectroscopy (its src/cavity.jl);
# the blanket `using OpticalSpectroscopy` above brings the polariton
# vocabulary into scope. QPSTools owns only the JASCO-backed CavitySpectrum
# and the JASCO-aware dispatches in src/cavity.jl.
import OpticalSpectroscopy: fit_cavity_spectrum, fit_dispersion
```

- [ ] **Step 3: Trim the export block**

Replace lines 133–139 (`# Cavity types and analysis` through `export refractive_index, extinction_coeff`) with:

```julia
# Cavity (QPSTools owns the JASCO-backed type; physics + fitting names
# come from OpticalSpectroscopy, which students load alongside)
export CavitySpectrum
```

- [ ] **Step 4: Update `src/cavity.jl`**

(a) Header docstring (lines 1–9): replace the sentence `The physics chain (dielectric function → refractive index → Fabry-Perot transmittance), polariton branches/Hopfield coefficients, and the fitting numerics live in the public CavitySpectroscopy.jl package.` with `The physics chain, polariton models, and fitting numerics live in OpticalSpectroscopy (its src/cavity.jl).`

(b) Section comment (lines 81–88): replace the block with:

```julia
# =============================================================================
# Physics + fitting: OpticalSpectroscopy
# =============================================================================
# The cavity physics (cavity_transmittance, polariton branches/eigenvalues,
# Hopfield coefficients, dispersion model) and the fitting layer
# (fit_cavity_spectrum, fit_dispersion, CavityFitResult, DispersionFitResult)
# live in OpticalSpectroscopy. QPSTools adds the JASCO-aware dispatch below.
```

(c) In the `fit_cavity_spectrum(spec::CavitySpectrum)` docstring, change `Numerics from \`CavitySpectroscopy\`.` to `Numerics from \`OpticalSpectroscopy\`.`

(d) Delete the bridge (lines 111–116):

```julia
# Markdown reporting: route OpticalSpectroscopy's format_results generic
# (the one QPSTools users have loaded) to CavitySpectroscopy's methods.
OpticalSpectroscopy.format_results(r::CavityFitResult) =
    CavitySpectroscopy.format_results(r)
OpticalSpectroscopy.format_results(r::DispersionFitResult) =
    CavitySpectroscopy.format_results(r)
```

(delete entirely — one generic now owns these methods).

- [ ] **Step 5: Resolve the environment against the new OpticalSpectroscopy main**

```bash
cd ~/Developer/QPSTools.jl
PATH=~/.juliaup/bin:$PATH julia --project=. -e 'using Pkg; Pkg.update("OpticalSpectroscopy"); Pkg.precompile()' 2>&1 | tail -5
```
Expected: resolves cleanly; CavitySpectroscopy disappears from the Manifest; QPSTools precompiles.

### Task 6: Update QPSTools tests and run the suite

**Files:**
- Modify: `~/Developer/QPSTools.jl/test/testsetup.jl:7`
- Modify: `~/Developer/QPSTools.jl/test/runtests.jl:8-21`
- Modify: `~/Developer/QPSTools.jl/test/test_cavity.jl` (comments + one testset name)

- [ ] **Step 1: `testsetup.jl`** — delete the line `import CavitySpectroscopy`.

- [ ] **Step 2: `runtests.jl` Aqua block** — remove `:CavitySpectroscopy,` from the `deps_compat` ignore list, and delete the whole `piracies=(treat_as_own=[...],)` keyword with its preceding `# Deliberate glue:` comment block (the bridge it excused is gone).

- [ ] **Step 3: `test_cavity.jl` comment/name updates** (behavior-neutral):
  - Header comment (lines 4–9): replace with `# The cavity physics and fitting numerics are tested in OpticalSpectroscopy's suite. This file covers only the QPSTools layer: the JASCO-backed CavitySpectrum, load_cavity, the JASCO-aware fit_cavity_spectrum dispatch, format_results on cavity result types, and plotting.`
  - Accessor comment (lines 22–24): replace `(not CavitySpectroscopy's same-named export, which stays qualified)` with `(the single wavenumber generic, shared with the cavity fit results)`.
  - Dispatch comment (lines 28–30): `the numerics run in CavitySpectroscopy.` → `the numerics run in OpticalSpectroscopy.`
  - Rename `@testset "format_results bridge"` → `@testset "format_results on cavity results"` and replace its comment (`Bare format_results here is OpticalSpectroscopy's generic... routes it to CavitySpectroscopy's methods...`) with `# format_results is one generic in OpticalSpectroscopy; the cavity result methods moved there with the merge.`

- [ ] **Step 4: Run the full QPSTools suite**

```bash
cd ~/Developer/QPSTools.jl
PATH=~/.juliaup/bin:$PATH julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -15
```
Expected: PASS, including Aqua (stale-binding config now references only live deps) and the cavity testsets.

- [ ] **Step 5: Commit**

```bash
cd ~/Developer/QPSTools.jl
git add Project.toml src/QPSTools.jl src/cavity.jl test/testsetup.jl test/runtests.jl test/test_cavity.jl
git commit -m "refactor: source cavity physics from OpticalSpectroscopy, drop CavitySpectroscopy dep

Deletes the format_results bridge and the cavity re-export exception; the
vocabulary now arrives via 'using OpticalSpectroscopy'.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

### Task 7: QPSTools docs + CLAUDE.md

**Files:**
- Modify: `~/Developer/QPSTools.jl/CLAUDE.md` (Scope bullet + Package Structure lines)
- Modify: `~/Developer/QPSTools.jl/docs/src/reference/cavity.md:5`
- Modify: `~/Developer/QPSTools.jl/docs/src/index.md:40`

- [ ] **Step 1: CLAUDE.md Scope bullet** — replace the cavity bullet with:

```markdown
- Lab-side cavity polariton layer: JASCO-backed `CavitySpectrum`, `load_cavity`, JASCO-aware `fit_cavity_spectrum` dispatch. The physics + fitting numerics live in [OpticalSpectroscopy.jl](https://github.com/garrekstemo/OpticalSpectroscopy.jl)'s cavity layer (no re-export — students load OpticalSpectroscopy alongside)
```

- [ ] **Step 2: CLAUDE.md Package Structure** — update two lines:
  - `spectroscopy.jl   # JASCOSpectrum/AnnotatedSpectrum dispatches, cavity_transmittance` → drop the stale `, cavity_transmittance` (the dispatch was removed in commit 62c1b25).
  - `cavity.jl         # JASCO-backed CavitySpectrum + dispatches into CavitySpectroscopy.jl` → `cavity.jl          # JASCO-backed CavitySpectrum + dispatches into OpticalSpectroscopy's cavity layer`

- [ ] **Step 3: docs/src/reference/cavity.md line 5** — replace the sentence pointing at CavitySpectroscopy.jl with:

```markdown
The physics and fitting (`fit_dispersion`, `cavity_mode_energy`, `polariton_branches`, `polariton_eigenvalues`, `hopfield_coefficients`, `compute_cavity_transmittance`, `cavity_transmittance`, `refractive_index`, `extinction_coeff`, and the `CavityFitResult` / `DispersionFitResult` types) live in [OpticalSpectroscopy.jl](https://garrekstemo.github.io/OpticalSpectroscopy.jl/) — see its Cavity & Polaritons reference page.
```

- [ ] **Step 4: docs/src/index.md line 40** — delete the diagram line `├── CavitySpectroscopy.jl ── polariton analysis (independent, public)` (and its connector line if orphaned; cavity capability is implied by the OpticalSpectroscopy node).

- [ ] **Step 5: If `docs/Manifest.toml` is git-tracked, regenerate it; otherwise skip**

```bash
cd ~/Developer/QPSTools.jl
git ls-files --error-unmatch docs/Manifest.toml 2>/dev/null && \
  PATH=~/.juliaup/bin:$PATH julia --project=docs -e 'using Pkg; Pkg.update()' || echo "untracked — skip"
```

- [ ] **Step 6: Commit, push, CHECKPOINT — ask Garrek whether to open the `streamline-api-surface` PR now or keep stacking**

```bash
cd ~/Developer/QPSTools.jl
git add CLAUDE.md docs/
git commit -m "docs: cavity layer now backed by OpticalSpectroscopy

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin streamline-api-surface
```

---

## Phase 3 — QPSLab

### Task 8: Remove the unused dependency

**Files:**
- Modify: `~/Developer/QPSLab/server/Project.toml:8,25`

- [ ] **Step 1: Check repo state, then branch**

```bash
cd ~/Developer/QPSLab && git status --short --branch
git checkout -b remove-cavityspectroscopy-dep
```
(If the tree is dirty, stop and ask Garrek first.)

- [ ] **Step 2: Edit** — delete the `CavitySpectroscopy = "23dbb5c6-..."` line from `[deps]` and the `CavitySpectroscopy = {url = ...}` line from `[sources]` in `server/Project.toml`. (Verified: no `[compat]` entry, no source usage.)

- [ ] **Step 3: Resolve and sanity-load**

```bash
cd ~/Developer/QPSLab/server
PATH=~/.juliaup/bin:$PATH julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.precompile()' 2>&1 | tail -5
```
Expected: resolves and precompiles cleanly.

- [ ] **Step 4: Commit; integration (merge/PR) per Garrek's preference at the checkpoint**

```bash
cd ~/Developer/QPSLab
git add server/Project.toml
git commit -m "chore: drop unused CavitySpectroscopy dep (merged into OpticalSpectroscopy)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Phase 4 — Archive + ecosystem documentation (after Phases 1–2 land)

### Task 9: Retire the CavitySpectroscopy repo

- [ ] **Step 1: Replace `README.md`** in `~/Developer/CavitySpectroscopy.jl` with exactly:

```markdown
# CavitySpectroscopy.jl

**Merged into [OpticalSpectroscopy.jl](https://github.com/garrekstemo/OpticalSpectroscopy.jl)** (June 2026).

The Fabry-Pérot physics, coupled-oscillator polariton models, and
spectrum/dispersion fitting now live in OpticalSpectroscopy's cavity layer —
see its [Cavity & Polaritons documentation](https://garrekstemo.github.io/OpticalSpectroscopy.jl/).
The generic `CavitySpectrum` container was not carried over; the merged API is
vector-based (`fit_cavity_spectrum(nu, T; ...)`).

This repository is archived; full development history is preserved here.
```

- [ ] **Step 2: Commit to main and push**

```bash
cd ~/Developer/CavitySpectroscopy.jl
git checkout main && git add README.md
git commit -m "docs: point README at OpticalSpectroscopy (package merged, repo archived)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

- [ ] **Step 3: CHECKPOINT — confirm with Garrek, then archive**

```bash
gh repo archive garrekstemo/CavitySpectroscopy.jl --yes
```

### Task 10: Ecosystem docs sweep

**Files:**
- Modify: `~/Developer/QPSLab/docs/dev/ecosystem-roadmap.md` (~lines 24–27, 94, 284)
- Modify: `~/Developer/QPSLab/docs/dev/feature-audit.md` (grep for the mention)
- Modify: `~/.claude/CLAUDE.md` (ecosystem diagram, registration list, code-placement table)

- [ ] **Step 1: Roadmap diagram (~lines 24–27)** — delete the `CavitySpectroscopy.jl ──── polariton analysis (independent, public)` node and its connector lines.

- [ ] **Step 2: Roadmap package table (~line 94)** — delete the row `| CavitySpectroscopy.jl | Polariton analysis | Independent | Julia (public) |`.

- [ ] **Step 3: Roadmap "at maturity" tree (~line 284)** — replace the line `├── CavitySpectroscopy.jl      your lab — polariton analysis` with `│   (polariton analysis lives in OpticalSpectroscopy core — extract a domain package only on heavy deps or a separate audience)`.

- [ ] **Step 4: feature-audit.md** — `grep -n CavitySpectroscopy ~/Developer/QPSLab/docs/dev/feature-audit.md`, then rewrite each hit to attribute the capability to OpticalSpectroscopy's cavity layer (prose edit, keep the surrounding claim intact). Commit both files in QPSLab.

- [ ] **Step 5: Global `~/.claude/CLAUDE.md`** — four edits:
  - Diagram: delete the `CavitySpectroscopy.jl` branch lines; append `cavity polaritons (Hopfield, Rabi, dispersion)` to OpticalSpectroscopy's capability list in the same diagram node.
  - Registration paragraph: `Registration pending Garrek's go: OpticalSpectroscopy (**first registers as 0.1**), CavitySpectroscopy.` → `Registration pending Garrek's go: OpticalSpectroscopy (**first registers as 0.1**; includes the cavity layer from the retired CavitySpectroscopy.jl).`
  - Archived list: append `CavitySpectroscopy.jl (merged into OpticalSpectroscopy)`.
  - "Where code goes" table: `| Cavity/polariton physics and fitting | CavitySpectroscopy |` → `| Cavity/polariton physics and fitting | OpticalSpectroscopy |`.

- [ ] **Step 6: Showcase site** — clone `garrekstemo/qps-ecosystem` to `~/Developer/qps-ecosystem` (not currently checked out), grep for `CavitySpectroscopy`, fold its package card/mentions into OpticalSpectroscopy's entry, commit. CHECKPOINT — show Garrek the diff before pushing (plain git deploy publishes immediately).

```bash
git clone https://github.com/garrekstemo/qps-ecosystem ~/Developer/qps-ecosystem
grep -rn "CavitySpectroscopy" ~/Developer/qps-ecosystem --include="*.html" --include="*.md" --include="*.js" | grep -v ".git/"
```

---

## Final verification (after all phases)

- [ ] OpticalSpectroscopy main: `Pkg.test()` green, docs build green.
- [ ] QPSTools `streamline-api-surface`: `Pkg.test()` green; `grep -rn CavitySpectroscopy ~/Developer/QPSTools.jl/src ~/Developer/QPSTools.jl/test ~/Developer/QPSTools.jl/Project.toml ~/Developer/QPSTools.jl/CLAUDE.md ~/Developer/QPSTools.jl/docs/src` → no hits.
- [ ] QPSLab: `grep -n CavitySpectroscopy ~/Developer/QPSLab/server/Project.toml` → no hits.
- [ ] CavitySpectroscopy repo archived on GitHub.
- [ ] Spawned-task coordination: the "Add 1D steady-state spectrum type" session builds on merged main — no action needed here beyond landing Phase 1 first.
