# PLMap channel-major spectral cube — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement
> this plan task-by-task. (Subagents are disallowed for this work per user preference — run
> inline from the main loop; Julia tests must NOT run inside subagents/workflows, the silent
> precompile trips the 180 s watchdog.) Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store `PLMap.spectra` channel-major `(n_pixel, nx, ny)` so each pixel's spectrum is a
contiguous, allocation-free column, behind a `pixel_view`/`eachpixel`/`spectra_matrix` accessor
interface — migrated in lockstep across OpticalSpectroscopy.jl, QPSTools.jl, and QPSLab.

**Architecture:** Two-step OS change — first introduce the accessor seam (no behavior change),
then flip the storage layout + bulk-op axes behind it. Then migrate the QPSTools producer and the
QPSLab consumers (corrections, view, codegen, export, HDF5). No back-compat shims (all unreleased).

**Tech Stack:** Julia 1.10 LTS; column-major arrays; `eachslice`/`reshape`/`@view`; HDF5.jl;
CurveFit; multi-threaded loops (`Threads.@threads`).

## Global Constraints

- Julia 1.10 LTS minimum. No `##` comments in Julia. Prefer `eachindex`, `something`.
- Spectral cube is **channel-major `(n_pixel, nx, ny)`**. Spatial maps stay `(nx, ny, …)`.
- No back-compat: no HDF5 layout-marker attribute, no translate-on-read.
- Accessor names (public OS API): `pixel_view`, `eachpixel`, `spectra_matrix`.
- `CosmicRayMapResult.mask` flips to channel-major `(n_pixel, nx, ny)`.
- Run `Pkg.test()` from the **main loop**, never a subagent/workflow.
- Local dev wiring: temporarily set downstream `[sources]` to `{path=...}`, revert before commit.
  Never edit `Manifest.toml`.
- Merge order: OS → QPSTools → QPSLab. QPSLab work bases on `main`.

---

## Task 1: OS — introduce the accessor seam (no behavior change)

Add the three accessors **over the current `(nx, ny, n_pixel)` layout** and route every
single-pixel-read site through them. Package stays green; behavior identical.

**Files:**
- Modify: `src/plmap.jl` (add accessors; reimplement `extract_spectrum`)
- Modify: `src/OpticalSpectroscopy.jl:59` (exports)
- Modify: `src/cosmic_rays.jl` (single-pixel reads → `pixel_view`)
- Test: `test/runtests.jl` (new accessor testset)

**Interfaces:**
- Produces: `pixel_view(m::PLMap, ix::Integer, iy::Integer) -> AbstractVector` (a spectrum view);
  `eachpixel(m::PLMap)` (iterator of spectrum views); `spectra_matrix(m::PLMap) -> AbstractMatrix`
  shaped `(n_pixel, nx*ny)`, column = pixel spectrum.

- [ ] **Step 1: Add the three accessors to `src/plmap.jl`** (right after the `intensity(m)` accessor, ~line 70). In THIS task they wrap the *current* layout:

```julia
"""
    pixel_view(m::PLMap, ix::Integer, iy::Integer) -> AbstractVector

View (no copy) of the CCD spectrum at grid index `(ix, iy)`. After the channel-major
migration this is a unit-stride, allocation-free column.
"""
@inline pixel_view(m::PLMap, ix::Integer, iy::Integer) = @view m.spectra[ix, iy, :]

"""
    eachpixel(m::PLMap)

Iterator over per-pixel spectrum views, in column-major grid order (ix fastest).
"""
eachpixel(m::PLMap) = (pixel_view(m, ix, iy) for iy in 1:length(m.y), ix in 1:length(m.x))

"""
    spectra_matrix(m::PLMap) -> AbstractMatrix

Reshape view of the cube as `(n_pixel, nx*ny)` — each column is one pixel's spectrum.
"""
spectra_matrix(m::PLMap) = permutedims(reshape(m.spectra, length(m.x) * length(m.y), length(m.pixel)))
```

(Note: in this task `spectra_matrix`/`eachpixel` bodies are layout-specific and will be
rewritten in Task 2; that is expected. They exist now only so the API and tests are in place.)

- [ ] **Step 2: Reimplement `extract_spectrum` via `pixel_view`** in `src/plmap.jl` (~line 100). Replace the body's `signal=vec(m.spectra[ix, iy, :])` with `signal=collect(pixel_view(m, ix, iy))`:

```julia
function extract_spectrum(m::PLMap, ix::Int, iy::Int)
    1 <= ix <= length(m.x) || error("ix=$ix out of range 1:$(length(m.x))")
    1 <= iy <= length(m.y) || error("iy=$iy out of range 1:$(length(m.y))")
    return (pixel=m.pixel, signal=collect(pixel_view(m, ix, iy)),
            x=m.x[ix], y=m.y[iy])
end
```

- [ ] **Step 3: Route cosmic-ray single-pixel reads through `pixel_view`.** In `src/cosmic_rays.jl`, in `detect_cosmic_rays(m::PLMap)` and `remove_cosmic_rays(m::PLMap, …)`, replace each `@view m.spectra[ix, iy, p1:p2]` with `@view pixel_view(m, ix, iy)[p1:p2]` and `@view m.spectra[ix, iy, :]` with `pixel_view(m, ix, iy)`. Leave `_neighbor_spectra` (it slices a raw array argument, not a `PLMap`) and the mask untouched in this task.

- [ ] **Step 4: Add exports** to `src/OpticalSpectroscopy.jl:59`. Change:

```julia
export PLMap, extract_spectrum, peak_centers, intensity
```
to:
```julia
export PLMap, extract_spectrum, peak_centers, intensity
export pixel_view, eachpixel, spectra_matrix
```

- [ ] **Step 5: Add an accessor testset** to `test/runtests.jl` (near the existing PLMap tests). With the current layout `spectra=rand(np, ...)`? No — current layout is `(nx,ny,np)`, so build accordingly:

```julia
@testset "PLMap accessors (seam)" begin
    nx, ny, np = 3, 4, 7
    spectra = rand(nx, ny, np)
    intensity = dropdims(sum(spectra; dims=3); dims=3)
    m = PLMap(intensity, spectra, collect(1.0:nx), collect(1.0:ny), collect(1.0:np))
    @test collect(pixel_view(m, 2, 3)) == spectra[2, 3, :]
    @test size(spectra_matrix(m)) == (np, nx * ny)
    @test spectra_matrix(m)[:, 2] == spectra[2, 1, :]   # column-major: pixel (2,1)
    @test length(collect(eachpixel(m))) == nx * ny
    @test extract_spectrum(m, 2, 3).signal == spectra[2, 3, :]
end
```

- [ ] **Step 6: Run OS tests, expect green.**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS (all existing tests + new accessor testset).

- [ ] **Step 7: Commit.**

```bash
git add src/plmap.jl src/cosmic_rays.jl src/OpticalSpectroscopy.jl test/runtests.jl
git commit -m "refactor(PLMap): add pixel_view/eachpixel/spectra_matrix accessor seam (#47)"
```

---

## Task 2: OS — flip storage to channel-major

Flip the cube to `(n_pixel, nx, ny)`. Rewrite the accessor bodies, the constructor (add a
dimension guard), the bulk-op sites, the cosmic-ray mask, and all OS test fixtures.

**Files:**
- Modify: `src/plmap.jl` (struct docstring, inner-constructor guard, accessor bodies, bulk ops)
- Modify: `src/decomposition.jl:52` (`_prepare_map_matrix`)
- Modify: `src/cosmic_rays.jl` (axis order, mask shape, `_neighbor_spectra`)
- Test: `test/runtests.jl` (flip every `PLMap(...)` fixture + `.spectra[...]`; add guard + contiguity tests)

**Interfaces:**
- Produces: `PLMap.spectra` is now `(n_pixel, nx, ny)`; `pixel_view(m,ix,iy)` is a unit-stride
  view (`strides == (1,)`); `CosmicRayMapResult.mask` is `(n_pixel, nx, ny)`.

- [ ] **Step 1: Add the inner-constructor dimension guard** in `src/plmap.jl`. Replace the plain struct (lines 27–34) with:

```julia
struct PLMap <: AbstractSpectroscopyData
    intensity::Matrix{Float64}
    spectra::Array{Float64,3}   # channel-major: (n_pixel, nx, ny)
    x::Vector{Float64}
    y::Vector{Float64}
    pixel::Vector{Float64}
    metadata::Dict{Symbol,Any}

    function PLMap(intensity, spectra, x, y, pixel, metadata)
        size(spectra) == (length(pixel), length(x), length(y)) || throw(DimensionMismatch(
            "PLMap.spectra must be channel-major (n_pixel, nx, ny) = " *
            "($(length(pixel)), $(length(x)), $(length(y))), got $(size(spectra))"))
        size(intensity) == (length(x), length(y)) || throw(DimensionMismatch(
            "PLMap.intensity must be (nx, ny) = ($(length(x)), $(length(y))), got $(size(intensity))"))
        return new(intensity, spectra, x, y, pixel, metadata)
    end
end
```

Update the struct's docstring `# Fields` block: `spectra::Array{Float64,3}` — Raw CCD counts
**`(n_pixel, nx, ny)`** (channel-major: one pixel's spectrum is the contiguous first axis). The
5-arg convenience constructor (lines 36–37) is unchanged (it forwards to the 6-arg).

- [ ] **Step 2: Rewrite the accessor bodies** in `src/plmap.jl` for channel-major:

```julia
@inline pixel_view(m::PLMap, ix::Integer, iy::Integer) = @view m.spectra[:, ix, iy]
eachpixel(m::PLMap) = eachslice(m.spectra; dims=(2, 3))
spectra_matrix(m::PLMap) = reshape(m.spectra, length(m.pixel), length(m.x) * length(m.y))
```

- [ ] **Step 3: Flip the bulk-op sites** in `src/plmap.jl`:
  - `subtract_background`: auto-bg accumulation `bg_spectra .+= vec(m.spectra[ix, iy, :])` → `bg_spectra .+= pixel_view(m, ix, iy)`. Background subtraction `corrected = m.spectra .- reshape(bg_spectra, 1, 1, :)` → `corrected = m.spectra .- reshape(bg_spectra, :, 1, 1)`. Intensity recompute `sum(corrected[:, :, p1:p2]; dims=3)` (dropdims 3) → `dropdims(sum((@view corrected[p1:p2, :, :]); dims=1); dims=1)`.
  - `integrated_intensity`: `dropdims(sum(m.spectra[:, :, p1:p2]; dims=3); dims=3)` → `dropdims(sum((@view m.spectra[p1:p2, :, :]); dims=1); dims=1)`.
  - `peak_centers`: `spectra_slice = @view m.spectra[:, :, p1:p2]` → `@view m.spectra[p1:p2, :, :]`; per-pixel `sig = @view spectra_slice[ix, iy, :]` → `@view spectra_slice[:, ix, iy]`.
  - Docstring examples mentioning `(nx, ny, n_pixel)` / `size(spectra, 3)` → `(n_pixel, nx, ny)` / `size(spectra, 1)`.

- [ ] **Step 4: Update `_prepare_map_matrix`** in `src/decomposition.jl` (lines 52–68) to source the existing `(n_spatial, n_spectral)` contract from channel-major storage — `pca_map`/`nmf_map` bodies stay unchanged:

```julia
function _prepare_map_matrix(m::PLMap; pixel_range=nothing)
    n_pixel, nx, ny = size(m.spectra)
    if !isnothing(pixel_range)
        p1 = max(1, pixel_range[1]); p2 = min(n_pixel, pixel_range[2])
        spec_range = p1:p2
    else
        spec_range = 1:n_pixel
    end
    mat = spectra_matrix(m)                       # (n_pixel, nx*ny), contiguous
    sub = @view mat[spec_range, :]                # (n_spectral, n_spatial)
    data = Float64.(permutedims(sub))             # (n_spatial, n_spectral) — matches old contract
    return data, spec_range, nx, ny
end
```

- [ ] **Step 5: Flip `src/cosmic_rays.jl` axis order, `_neighbor_spectra`, and the mask shape.**
  - Struct doc (line 48): `mask::BitArray{3}` is now `(n_pixel, nx, ny)`.
  - `_neighbor_spectra`: slice the channel axis first —
    ```julia
    function _neighbor_spectra(spectra::AbstractArray{<:Any,3}, ix::Int, iy::Int,
                               nx::Int, ny::Int, p1::Int, p2::Int)
        neighbors = typeof(@view spectra[p1:p2, 1, 1])[]
        ix > 1  && push!(neighbors, @view spectra[p1:p2, ix-1, iy])
        ix < nx && push!(neighbors, @view spectra[p1:p2, ix+1, iy])
        iy > 1  && push!(neighbors, @view spectra[p1:p2, ix, iy-1])
        iy < ny && push!(neighbors, @view spectra[p1:p2, ix, iy+1])
        return neighbors
    end
    ```
  - `_unflag_wide_runs!(mask, ix, iy, p1, p2, max_width)`: index `mask[k, ix, iy]` everywhere (was `mask[ix, iy, k]`).
  - `detect_cosmic_rays(m::PLMap)`: `np, nx, ny = size(m.spectra)`; `mask = falses(np, nx, ny)`; signal `@view m.spectra[p1:p2, ix, iy]`; flag write `mask[k + p1 - 1, ix, iy] = true`; fraction-cap clears `mask[k, ix, iy]` for `k in p1:p2`; `remaining = count(@view mask[p1:p2, ix, iy])`; `affected` via `any(@view mask[:, ix, iy])`; `channel_counts[k] = count(@view mask[k, :, :])`.
  - `remove_cosmic_rays(m::PLMap, result)`: `np, nx, ny = size(m.spectra)`; `cleaned = copy(m.spectra)`; gate `any(@view result.mask[:, ix, iy])`; no-neighbor branch `findall(@view result.mask[:, ix, iy])`, `signal = @view m.spectra[:, ix, iy]`, write `cleaned[ch, ix, iy] = interp[ch]`; main branch signal `@view m.spectra[p1:p2, ix, iy]`, scale mask test `result.mask[k + p1 - 1, ix, iy]`, write `cleaned[k + p1 - 1, ix, iy] = scale * msn[k]`; intensity recompute `dropdims(sum((@view cleaned[rp1:rp2, :, :]); dims=1); dims=1)` (and the no-range branch `sum(cleaned; dims=1)`).

- [ ] **Step 6: Flip every OS test fixture** in `test/runtests.jl`. Apply the rule: any `PLMap(intensity, spectra, x, y, pixel[, meta])` must have `size(spectra) == (length(pixel), length(x), length(y))`, and any `spectra[ix, iy, :]` / `m.spectra[i,j,:]` becomes `spectra[:, ix, iy]` / `pixel_view(m, i, j)`. Concrete edits:
  - Line 331: `PLMap(rand(2, 2), rand(2, 2, 3), [0.0,1.0], [0.0,1.0], [1.0,2.0,3.0])` → spectra `rand(3, 2, 2)`.
  - The accessor testset added in Task 1 Step 5: change `spectra = rand(nx, ny, np)` → `rand(np, nx, ny)`; `intensity = dropdims(sum(spectra; dims=3); dims=3)` → `dropdims(sum(spectra; dims=1); dims=1)`; `pixel_view(m,2,3)` expectation `spectra[:, 2, 3]`; `spectra_matrix(m)[:, 2] == spectra[:, 2, 1]`; `extract_spectrum(m,2,3).signal == spectra[:, 2, 3]`.
  - All PLMap fixtures in the 2100–3100 block (build sites + any `.spectra[` indexing and `size(...,3)`-as-n_pixel). Find them with: `grep -n "PLMap(\|\.spectra\[\|spectra\[" test/runtests.jl`. For each construction, build `spectra` as `(np, nx, ny)` (commonly via `permutedims(old, (3,1,2))` if a `(nx,ny,np)` array already exists, or by constructing channel-major directly). For helper closures that fill `spectra[ix,iy,:] = …`, switch to `spectra[:, ix, iy] = …`.

- [ ] **Step 7: Add guard + contiguity tests** to the accessor testset:

```julia
@test strides(pixel_view(m, 1, 1)) == (1,)            # unit-stride / contiguous
@test (@allocated pixel_view(m, 1, 1)) == 0           # alloc-free view
@test_throws DimensionMismatch PLMap(intensity, rand(np, nx, ny + 1),
                                     collect(1.0:nx), collect(1.0:ny), collect(1.0:np))
```

- [ ] **Step 8: Run OS tests, expect green.**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS. If a fixture was missed, the constructor guard throws `DimensionMismatch` with the offending shape — fix that fixture and re-run.

- [ ] **Step 9: Commit.**

```bash
git add src/ test/runtests.jl
git commit -m "perf(PLMap): store spectra channel-major (n_pixel, nx, ny) (#47)"
```

---

## Task 3: QPSTools — channel-major producer + plotting

**Repo:** `../QPSTools.jl` — branch `perf/plmap-channel-major` off `main`.

**Files:**
- Modify: `src/plmap.jl` (`load_pl_map` producer)
- Modify: `src/plotting/plot_plmap.jl` (spectra reads)
- Modify: `Project.toml` `[sources]` (TEMPORARY local path; revert before commit)
- Test: existing QPSTools tests

**Interfaces:**
- Consumes: OS `PLMap` (channel-major), `pixel_view`, `extract_spectrum` from Task 2.

- [ ] **Step 1: Branch.**
```bash
git -C ../QPSTools.jl checkout main && git -C ../QPSTools.jl checkout -b perf/plmap-channel-major
```

- [ ] **Step 2: Point `[sources]` at the local OS branch** (TEMPORARY). In `../QPSTools.jl/Project.toml`, change the OpticalSpectroscopy `[sources]` line to:
```toml
OpticalSpectroscopy = {path = "../OpticalSpectroscopy.jl"}
```

- [ ] **Step 3: Flip the producer** in `src/plmap.jl` (line 147 area). Replace:
```julia
    # Reshape to (nx, ny, n_pixel)
    spectra = reshape(data, nx, ny, n_pixel)
```
with:
```julia
    # Channel-major (n_pixel, nx, ny): one pixel's spectrum is the contiguous first axis.
    # data is (n_points, n_pixel) = (nx*ny, n_pixel); transpose then reshape.
    spectra = reshape(permutedims(data), n_pixel, nx, ny)
```
And the intensity integrations (lines 154, 156):
```julia
        int_matrix = dropdims(sum((@view spectra[p1:p2, :, :]); dims=1); dims=1)
```
```julia
        int_matrix = dropdims(sum(spectra; dims=1); dims=1)
```

- [ ] **Step 4: Flip plotting reads** in `src/plotting/plot_plmap.jl`. Replace any `m.spectra[ix, iy, :]` / `vec(m.spectra[...])` with `extract_spectrum(m, ix, iy).signal` (or `pixel_view(m, ix, iy)` for read-only plotting). Verify with `grep -n "\.spectra" src/plotting/plot_plmap.jl`.

- [ ] **Step 5: Run QPSTools tests, expect green.**

Run: `julia --project=../QPSTools.jl -e 'using Pkg; Pkg.test()'`
Expected: PASS (resolves OS from the local path).

- [ ] **Step 6: Revert `[sources]`** in `../QPSTools.jl/Project.toml` back to the committed GitHub URL:
```toml
OpticalSpectroscopy = {url = "https://github.com/garrekstemo/OpticalSpectroscopy.jl"}
```

- [ ] **Step 7: Commit** (do NOT commit the path-`[sources]` edit).
```bash
git -C ../QPSTools.jl add src/
git -C ../QPSTools.jl commit -m "perf(load_pl_map): emit channel-major PLMap spectra (#47)"
```

---

## Task 4: QPSLab — migrate consumers + HDF5 layout

**Repo:** `../QPSLab` — branch off `main` (NOT `perf/autosave-copy-forward`).

**Files:**
- Modify: `server/src/sessions.jl` (`_map_spectra`, `_apply_correction`)
- Modify: `server/src/handlers/view.jl` (~8 spectra-index sites; `_event_channel_counts`)
- Modify: `server/src/handlers/codegen.jl` (`size(m.spectra, …)` nx/ny; emitted code)
- Modify: `server/src/handlers/export.jl`, `server/src/handlers/load.jl` (pass-through verify)
- Modify: `server/src/handlers/hdf5io.jl` (write/read layout + chunk)
- Modify: `server/Project.toml` `[sources]` (TEMPORARY local paths; revert before commit)
- Test: `server/test/test_corrections.jl`, `server/test/test_normalize_sites.jl`, HDF5 round-trip

**Interfaces:**
- Consumes: OS channel-major `PLMap`, `pixel_view`, `CosmicRayMapResult.mask` `(n_pixel,nx,ny)`;
  QPSTools channel-major `load_pl_map`.

- [ ] **Step 1: Branch.**
```bash
git -C ../QPSLab checkout main && git -C ../QPSLab checkout -b perf/plmap-channel-major
```

- [ ] **Step 2: Point `[sources]` at local OS + QPSTools branches** (TEMPORARY) in `../QPSLab/server/Project.toml`:
```toml
OpticalSpectroscopy = {path = "../../OpticalSpectroscopy.jl"}
QPSTools = {path = "../../QPSTools.jl"}
```
(Confirm the relative depth: `server/Project.toml` → repo siblings are `../../`.)

- [ ] **Step 3: Migrate `sessions.jl`.**
  - `_map_spectra` (lines 194–207):
    ```julia
    function _map_spectra(fn::Function, m::PLMap)
        nx, ny = size(m.intensity)
        new_spectra = similar(m.spectra)
        Threads.@threads for idx in CartesianIndices((nx, ny))
            i, j = idx[1], idx[2]
            new_spectra[:, i, j] = fn(pixel_view(m, i, j))
        end
        new_intensity = dropdims(sum(new_spectra; dims=1); dims=1)
        return PLMap(new_intensity, new_spectra, m.x, m.y, m.pixel, m.metadata)
    end
    ```
    (Ensure `pixel_view` is in scope — it's exported by OpticalSpectroscopy; add to the `using OpticalSpectroscopy: …` import list in `QPSLabServer.jl:20` if not already imported by name.)
  - `_apply_correction` crop_spectrum (line 215): `new_spectra = m.spectra[:, :, mask]` → `m.spectra[mask, :, :]`; intensity `dropdims(sum((@view new_spectra[...]); dims=1); dims=1)` — but simplest: `new_intensity = dropdims(sum(new_spectra; dims=1); dims=1)`.
  - crop_map (line 226): `new_spectra = m.spectra[rows, cols, :]` → `m.spectra[:, rows, cols]` (intensity unchanged — uses `m.intensity[rows, cols]`).
  - normalize (lines 282, 285): `norm_spectra = zeros(size(m.spectra))` stays; `n_channels = size(m.spectra, 3)` → `size(m.spectra, 1)`.

- [ ] **Step 4: Migrate `handlers/view.jl`.** Flip each site (find via `grep -n "m\.spectra\|cr\.mask\|size(mask)" server/src/handlers/view.jl`):
  - `_event_channel_counts(mask, spectra)` (lines 156–): `nx, ny, np = size(mask)` → `np, nx, ny = size(mask)`; inside, `mask[ix, iy, k]` → `mask[k, ix, iy]`, `spectra[ix, iy, k]` → `spectra[k, ix, iy]`.
  - Line ~80: `dropdims(sum(m.spectra[:, :, p1:p2]; dims=3); dims=3)` → `dropdims(sum((@view m.spectra[p1:p2, :, :]); dims=1); dims=1)`.
  - Lines ~267–268 band_ratio sums: same `[:, :, a:b]; dims=3` → `[a:b, :, :]; dims=1` flip.
  - Line ~288: `spec = m.spectra[ix, iy, :]` → `spec = collect(pixel_view(m, ix, iy))`.
  - Lines ~417/419 and ~461/463 region sums: `region = m.spectra[r1:r2, c1:c2, :]` → `m.spectra[:, r1:r2, c1:c2]`; `region = m.spectra` stays; downstream reductions over the channel axis must use `dims=1` (was `dims=3`) — inspect the few lines after each `region = …` and flip the reduction dim.

- [ ] **Step 5: Migrate `handlers/codegen.jl`.** Lines 394–395 and 563: `nx = get(opts, "nx", size(m.spectra, 1))` → `size(m.spectra, 2)`; `ny = …, size(m.spectra, 2))` → `size(m.spectra, 3)`. Review any emitted user-facing snippet string that indexes `.spectra[i,j,:]` and update it to `pixel_view`/`extract_spectrum` and the channel-major convention (find via `grep -n "spectra" server/src/handlers/codegen.jl`).

- [ ] **Step 6: Verify `export.jl` / `load.jl` pass-through.** `export.jl:269` and `load.jl:305,370` reconstruct `PLMap(... m.spectra ...)` unchanged — these just forward the cube, so they are correct as-is once the cube is channel-major. Confirm no axis assumption nearby (`grep -n "spectra" server/src/handlers/export.jl server/src/handlers/load.jl`). No code change expected; the constructor guard will catch a mistake.

- [ ] **Step 7: Migrate `handlers/hdf5io.jl`.**
  - `_write_plmap_hdf5` (lines 19–22):
    ```julia
    n_pixel, nx, ny = size(m.spectra)
    ds = create_dataset(g, "spectra", Float64, (n_pixel, nx, ny);
        chunk=(n_pixel, min(nx, 16), min(ny, 16)), deflate=3)
    ds[:, :, :] = m.spectra
    ```
  - `_read_plmap_hdf5` (line 160): unchanged — `spectra = read(g["spectra"])` is already channel-major. (No marker attribute, no translate.)

- [ ] **Step 8: Flip QPSLab test fixtures.**
  - `server/test/test_normalize_sites.jl:24`: `spectra` for `PLMap(intensity, spectra, [0.0], [0.0,1.0], [1.0,2.0])` must be `(np=2, nx=1, ny=2)`. Line 36: `fill(2.5, 1, 2, 2)` for `x=[0.0]`(nx=1), `y=[0.0,1.0]`(ny=2), `pixel=[1.0,2.0]`(np=2) → `fill(2.5, 2, 1, 2)`.
  - `server/test/test_corrections.jl:154,183`: rebuild `spectra` channel-major for the stated `x`,`y`,`pixel`. Find every `PLMap(`/`.spectra[` in `server/test/` via `grep -rn "PLMap(\|\.spectra\[" server/test/` and flip per the Task 2 Step 6 rule.

- [ ] **Step 9: Add an HDF5 round-trip test** (in the appropriate `server/test` file, e.g. alongside existing hdf5 tests; if none, add to the corrections/project test that already builds a `PLMap`):

```julia
@testset "PLMap HDF5 channel-major round-trip" begin
    np, nx, ny = 5, 3, 4
    spectra = rand(np, nx, ny)
    intensity = dropdims(sum(spectra; dims=1); dims=1)
    m = PLMap(intensity, spectra, collect(1.0:nx), collect(1.0:ny), collect(1.0:np))
    mktemp() do path, io
        close(io)
        HDF5.h5open(path, "w") do f; _write_plmap_hdf5(f, m); end
        m2 = HDF5.h5open(path, "r") do f; _read_plmap_hdf5(f); end
        @test size(m2.spectra) == (np, nx, ny)
        @test m2.spectra == m.spectra
    end
end
```
(Adjust `HDF5`/helper qualification to match the test file's imports.)

- [ ] **Step 10: Run QPSLab server tests, expect green.**

Run: `julia --project=../QPSLab/server -e 'using Pkg; Pkg.test()'`
Expected: PASS (resolves OS + QPSTools from local paths).

- [ ] **Step 11: Revert `[sources]`** in `../QPSLab/server/Project.toml` to committed GitHub URLs:
```toml
OpticalSpectroscopy = {url = "https://github.com/garrekstemo/OpticalSpectroscopy.jl"}
QPSTools = {url = "https://github.com/garrekstemo/QPSTools.jl"}
```

- [ ] **Step 12: Commit** (NOT the path-`[sources]` edit).
```bash
git -C ../QPSLab add server/src server/test
git -C ../QPSLab commit -m "perf(PLMap): migrate consumers + HDF5 to channel-major cube (#47)"
```

---

## Self-Review

**Spec coverage:** §2 accessor interface → T1/T2 (S1-2). §1 storage flip → T2. §3 OS internals
(plmap/decomposition/cosmic_rays) → T2 (S3-5). §3 mask flip → T2 S5. QPSTools producer/plotting
(§4) → T3. QPSLab sessions/view/codegen/export/load (§5) → T4 S3-6. HDF5 (§6) → T4 S7. Testing
(§7) → T1 S5, T2 S6-8, T3 S5, T4 S8-10. Rollout (§8) → per-task branch/`[sources]`/revert steps.
"Spatial maps unchanged" → enforced by leaving `fit_map` summary arrays and `intensity`
construction untouched; constructor guard asserts `intensity` is `(nx,ny)`.

**Placeholder scan:** No TBD/TODO. Mechanical sweeps (T2 S6, T4 S4/S8) give the exact
transformation rule + `grep` locator + concrete enumerated edits rather than pasting every line —
acceptable because the rule is unambiguous and the constructor guard backstops misses.

**Type consistency:** `pixel_view`/`eachpixel`/`spectra_matrix` signatures identical across T1/T2.
`_prepare_map_matrix` keeps its `(data, spec_range, nx, ny)` return contract so `pca_map`/`nmf_map`
are untouched. `CosmicRayMapResult.mask` shape `(n_pixel,nx,ny)` consistent between OS producer
(T2 S5) and QPSLab consumer (T4 S4 `_event_channel_counts`).
