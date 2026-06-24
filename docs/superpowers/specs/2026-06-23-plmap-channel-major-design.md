# PLMap channel-major spectral cube — design

**Issue:** #47 — `perf(PLMap): store spectra channel-major for contiguous, alloc-free per-pixel access`
**Date:** 2026-06-23
**Scope:** lockstep clean break across three repos — OpticalSpectroscopy.jl, QPSTools.jl, QPSLab. No backwards-compatibility shims (all repos unreleased).

## Problem

`PLMap.spectra::Array{Float64,3}` is stored `(nx, ny, n_pixel)` — spatial axes first,
spectral axis last. Julia arrays are column-major, so the spectral axis is the
slowest-varying dimension. Consequences on the million-element cubes this type targets
(e.g. 100×100 grid × ~1340 channels ≈ 107 MB):

- One pixel's spectrum `spectra[i, j, :]` is **strided** — consecutive channels are
  `nx*ny` elements (~80 KB) apart, so reading a spectrum touches ~1340 cache lines.
- `spectra[i, j, :]` (colon `getindex`) **allocates a fresh `Vector`** per access —
  ~10⁴ heap allocations + GC pressure per full-map pass.
- Paid on every per-pixel pass: baseline / smoothing / SG / normalize, cosmic-ray
  detect & remove, `extract_spectrum`, neighbor lookups, the per-pixel `fit_map` loop.

## Core decision

Flip the cube to **channel-major `(n_pixel, nx, ny)`**. One pixel's spectrum becomes
`spectra[:, ix, iy]` — the first (fastest) axis, hence a **contiguous, allocation-free**
column (~10.7 KB, fits in L2). `sum(...; dims=1)` for integrated intensity also reduces
*along* the contiguous axis.

**Idiomatic principle:** match memory layout to the dominant access pattern, and hide
the layout behind an accessor/iterator interface so no caller indexes the raw field.
This is the durable deliverable — the layout becomes an implementation detail.

### What flips vs. what does not

- **Flips (spectral cube):** `PLMap.spectra`, `CosmicRayMapResult.mask`.
- **Unchanged (spatial maps — not spectra):** `PLMap.intensity` `(nx, ny)`; `fit_map`
  result arrays `centers`/`fwhms`/`amplitudes` `(nx, ny, n_peaks)` and `r_squareds`
  `(nx, ny)`; decomposition `loadings` `(nx, ny, n_components)`. These are indexed and
  plotted as spatial heatmaps; only the cube moves.

## OS accessor interface (OpticalSpectroscopy.jl)

New exported accessors on `PLMap`; every consumer routes through them.

- `pixel_view(m, ix, iy) -> @view m.spectra[:, ix, iy]` — contiguous, zero-alloc. The
  hot-path primitive.
- `eachpixel(m)` — iterator of pixel views, built on `eachslice(m.spectra; dims=(2,3))`,
  for sequential whole-map passes.
- `spectra_matrix(m) -> reshape(m.spectra, n_pixel, nx*ny)` — contiguous
  `(n_channel, n_spatial)` view for decomposition / vectorized work.
- `extract_spectrum(m, ix, iy)` stays the ergonomic public API (returns a NamedTuple
  with a **copied** `signal`), now implemented as `collect(pixel_view(m, ix, iy))`.

**Inner-constructor guard:** add an inner constructor asserting
`size(spectra) == (length(pixel), length(x), length(y))`. Catches any missed migration
site at construction time — a tripwire for the layout flip. The convenience
`PLMap(intensity, spectra, x, y, pixel)` (empty-metadata) forwarder stays.

## OS internal migration

- `src/plmap.jl`
  - `subtract_background`: background broadcast `reshape(bg, :, 1, 1)`; intensity sum
    `dims=1`; auto-background neighbor reads via `pixel_view`.
  - `integrated_intensity`, `peak_centers`: slice `m.spectra[p1:p2, :, :]`, reduce
    `dims=1`; per-pixel reads via `pixel_view`.
  - `fit_map`: already goes through `extract_spectrum` → free; verify no direct field
    indexing remains.
  - All docstrings/examples updated to `(n_pixel, nx, ny)` and `size(spectra, 1)`.
- `src/decomposition.jl`
  - `_prepare_map_matrix`: return the `(n_channel, n_spatial)` matrix via
    `spectra_matrix` (+ spectral-range restriction on `dims=1`). PCA/NMF operate on the
    transpose (natural column-major SVD). Public `DecompositionResult` shapes unchanged.
- `src/cosmic_rays.jl`
  - `detect_cosmic_rays`, `remove_cosmic_rays`: `nx, ny` from `size(m.intensity)` /
    `(size(m.spectra,2), size(m.spectra,3))`; per-pixel reads/writes via `pixel_view`
    and contiguous column writes.
  - `_neighbor_spectra`: slice `spectra[p1:p2, ix±1, iy]` / `[p1:p2, ix, iy±1]`.
  - `CosmicRayMapResult.mask` flips to `(n_pixel, nx, ny)`; `channel_counts` and
    `affected` loops follow; `_unflag_wide_runs!` indexes `mask[k, ix, iy]`.

## QPSTools.jl migration

- `src/plmap.jl` `load_pl_map` (the **producer**): build channel-major directly from the
  `(nx*ny, n_pixel)` `readdlm` matrix —
  `spectra = reshape(permutedims(data), n_pixel, nx, ny)` (equivalently
  `permutedims(reshape(data, nx, ny, n_pixel), (3, 1, 2))`); intensity sum `dims=1`.
- `src/plotting/plot_plmap.jl`: spectra reads → `pixel_view` / `extract_spectrum`.

## QPSLab migration (server)

- `src/sessions.jl`
  - `_map_spectra`: `new_spectra = similar(m.spectra)`;
    `new_spectra[:, i, j] = fn(pixel_view(m, i, j))`; `new_intensity = dropdims(sum(new_spectra; dims=1); dims=1)`.
  - `_apply_correction`: `crop_spectrum` → `m.spectra[mask, :, :]`; `crop_map` →
    `m.spectra[:, rows, cols]`; `normalize` → `n_channels = size(m.spectra, 1)`.
- `src/handlers/view.jl` (~8 sites): every `m.spectra[i, j, :]` / `m.spectra[:, :, p1:p2]`
  / `m.spectra[r1:r2, c1:c2, :]` flips axis order; `_event_channel_counts(cr.mask, …)`
  updated for the channel-major mask.
- `src/handlers/codegen.jl`: `size(m.spectra, 1)` / `size(m.spectra, 2)` (nx/ny) become
  `size(m.spectra, 2)` / `size(m.spectra, 3)`; emitted user-facing code uses the
  channel-major convention and accessors.
- `src/handlers/export.jl`, `src/handlers/load.jl`: reconstruct/pass-through `PLMap`
  — verify they pass `m.spectra` straight through (no axis assumptions).
- `src/handlers/hdf5io.jl`
  - `_write_plmap_hdf5`: `n_pixel, nx, ny = size(m.spectra)`; dataset `(n_pixel, nx, ny)`;
    `chunk = (n_pixel, min(nx, 16), min(ny, 16))`, `deflate=3`.
  - `_read_plmap_hdf5`: straight `read(g["spectra"])` — already channel-major. **No**
    translate-on-read, **no** layout marker attribute (no back-compat).

## Testing

- **OpticalSpectroscopy** (`test/runtests.jl`): flip all `PLMap(...)` fixtures to
  channel-major `(np, nx, ny)`; assert `size(m.spectra) == (np, nx, ny)`; assert
  `pixel_view` is contiguous & alloc-free (`strides(pv)[1] == 1`,
  `@allocated pixel_view(m, 1, 1) == 0`); add a constructor-guard test (mismatched dims
  throws); existing `extract_spectrum`/`fit_map`/cosmic-ray/decomposition assertions
  updated for the new shapes.
- **QPSTools**: `load_pl_map` round-trip asserts channel-major cube; plotting smoke test.
- **QPSLab** (`server/test`): `test_corrections.jl`, `test_normalize_sites.jl` fixtures
  flip; add HDF5 round-trip test (write → read → identical channel-major cube).
- All three: `Pkg.test()` green. **Run tests from the main loop, not subagents** (silent
  precompile trips the 180 s workflow watchdog).

## Rollout

Three coordinated branches, merged **OS → QPSTools → QPSLab** (downstream depends on the
OS accessors). During local dev, temporarily point QPSTools/QPSLab `[sources]` at
`{path = "../OpticalSpectroscopy.jl"}`, then revert to the committed GitHub-URL
`[sources]` before committing (per Pkg conventions). QPSLab work bases on `main`, not the
in-flight `perf/autosave-copy-forward` branch; coordinate the eventual merge with it.

## Decisions made without asking (flag to reverse)

- Accessor names: `pixel_view` / `eachpixel` / `spectra_matrix`.
- `CosmicRayMapResult.mask` flips to channel-major (consistency + contiguous mask
  access; public-field shape change, acceptable as unreleased).
- 3D `(n_pixel, nx, ny)` storage (not 2D `(n_pixel, nx*ny)`); the 2D matrix view is
  available for free via `spectra_matrix`.
