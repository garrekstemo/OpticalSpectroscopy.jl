# OpticalSpectroscopy.jl — TODO

## 1. Improve chirp detection R² on real data

**Status**: Working but inaccurate on noisy CCD data (R² = 0.492 on PB-film)

**Context**: `detect_chirp` works well on synthetic data (R² > 0.98) but struggles with real broadband TA from a CCD camera (2048 wavelengths, 151 time points, ~467–733 nm). The CCD data is noisy and the chirp onset is subtle relative to later dynamics.

**Current implementation** (`src/chirp.jl`):
- Two methods: `:xcorr` (cross-correlation of absolute onset gradients) and `:threshold` (half-maximum crossing)
- Wavelength binning (`bin_width`) averages adjacent columns before detection
- Savitzky-Golay smoothing before gradient/threshold calculation
- MAD-based outlier rejection after polynomial fitting
- `_auto_chirp_range()` restricts detection to the onset region (10% threshold → peak gradient)

**What to try**:
- Better wavelength binning: adaptive bin widths based on signal strength, or allow user to pass pre-binned data
- Weighted polynomial fitting: weight by signal strength or inverse noise estimate per bin
- Two-stage detection: coarse detection with wide bins first, then refine with narrow bins
- Alternative onset detection: fit a step function or error function to each bin's time trace rather than using raw gradient/threshold
- Pre-detection denoising: apply SVD filter (see task 2) before chirp detection

**Test data**: Real CCD data lives in `QPSTools.jl/data/broadbandTA_experiment/`. The example script is `QPSTools.jl/examples/broadband_ta_example.jl`. To test against real data, you'll need both repos checked out. Synthetic tests in `test/runtests.jl` cover correctness; real-data R² is the quality metric.

**Key gotcha**: The chirp polynomial gives `t_shift(λ)` = time the pump arrives at each wavelength relative to reference. To correct: evaluate `original(t + t_shift)`, NOT `original(t - t_shift)`.

---

## ~~2. Implement SVD filter for TA matrix denoising~~ ✓

Implemented in `src/chirp.jl`. Exports: `svd_filter`, `singular_values`.

---

## ~~3. Multi-exponential global analysis for TAMatrix~~ ✓

Implemented. `fit_global` now supports `n_exp` kwarg for multi-exponential global analysis and `TAMatrix` dispatch with DAS extraction via `das()`. `GlobalFitResult` uses `taus::Vector{Float64}` and `amplitudes::Matrix{Float64}` (n_traces × n_exp).

---

## General notes for the bot

- All tests use synthetic data (no local file dependencies). See existing tests in `test/runtests.jl` for patterns.
- `_sg_filter` is the module-level alias for `savitzky_golay` from SavitzkyGolay.jl — use it directly, no `using` needed.
- `Interpolations` and `JSON` are available at module level.
- `LinearAlgebra` is available at module level (for SVD).
- `Statistics: mean, median, std` are available at module level.
- Run tests with: `cd OpticalSpectroscopy.jl && julia --project=. -e 'using Pkg; Pkg.test()'`
- This is a public package — add `[compat]` entries for any new dependencies.
- Follow existing docstring style (see `detect_chirp`, `correct_chirp` for examples).
