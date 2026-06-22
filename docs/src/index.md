# OpticalSpectroscopy.jl

**Spectroscopy analysis for Julia — steady-state and ultrafast.**

For steady-state work, OpticalSpectroscopy provides a generic `Spectrum` type plus peak fitting (Gaussian, Lorentzian, Voigt, Fano), baseline correction, peak detection, spectral transforms (Kramers-Kronig, Tauc, Kubelka-Munk), and unit conversions for FTIR, Raman, and UV-vis data. For ultrafast work, it provides typed data structures (`KineticTrace`, `TimeResolvedMatrix`), global fitting with decay-associated spectra, IRF-convolved exponential fitting, and chirp correction for broadband pump-probe experiments.

## Installation

```julia
using Pkg
Pkg.add("OpticalSpectroscopy")
```

## Quick Start

```julia
using OpticalSpectroscopy
using CurveFitModels  # for lineshape functions
using Unitful

# ============================================================
# Ultrafast example: synthetic biexponential decay + TA matrix
# ============================================================
time = collect(range(0, 20.0, length=400))
signal = 0.6 .* exp.(-time ./ 0.5) .+ 0.4 .* exp.(-time ./ 5.0) .+ 0.01 .* randn(length(time))

trace = KineticTrace(time, signal; wavelength=800.0)
fit = fit_exp_decay(trace; n_exp=2)
report(fit)

# Synthetic 2D TA matrix: rows = time, cols = wavelength
matrix_time = collect(range(0, 15.0, length=200))
matrix_wl   = collect(range(500.0, 700.0, length=40))
matrix_data = zeros(length(matrix_time), length(matrix_wl))
for (j, λ) in pairs(matrix_wl)
    a_fast = exp(-((λ - 550) / 20)^2)
    a_slow = -0.8 * exp(-((λ - 640) / 30)^2)
    matrix_data[:, j] = a_fast .* exp.(-matrix_time ./ 0.8) .+ a_slow .* exp.(-matrix_time ./ 4.0)
end
matrix_data .+= 0.005 .* randn(size(matrix_data))

matrix = TimeResolvedMatrix(matrix_time, matrix_wl, matrix_data)
gfit = fit_global(matrix; n_exp=2)
spectra = das(gfit)   # n_exp × n_wavelengths decay-associated spectra

# ============================================================
# Steady-state example: noisy Lorentzian peak + baseline
# ============================================================
x = collect(range(1900, 2200, length=600))
y = lorentzian([0.45, 2062.0, 24.0, 0.01], x) .+ 0.002 .* randn(length(x))

result = fit_peaks(x, y, (2000, 2150))
report(result)

bl = correct_baseline(x, y; method=:arpls, λ=1e5)
y_corrected = bl.y

# Unit conversions
wavelength_to_wavenumber(1500u"nm")  # -> wavenumber in cm⁻¹
decay_time_to_linewidth(1.0u"ps")    # -> linewidth in meV (default output unit)
```

## Package Overview

| Module | What it does |
|--------|-------------|
| **Data types** | `Spectrum`, `KineticTrace`, `TimeResolvedMatrix` with semantic axis indexing (`m[λ=800]`, `m[t=1.0]`) |
| **Exponential decay** | Single/multi-exponential with optional IRF convolution |
| **Global fitting** | Shared parameters across traces, decay-associated spectra |
| **TA spectrum fitting** | N-peak model with ESA/GSB/SE labels and sign convention |
| **Chirp correction** | GVD detection (cross-correlation, threshold) and correction for broadband TA |
| **SVD filtering** | Matrix denoising for broadband TA data |
| **Peak fitting** | Gaussian, Lorentzian, Voigt, Pseudo-Voigt, Fano via CurveFit.jl |
| **Cavity / polaritons** | Hopfield coefficients, polariton branches, Rabi splitting, dispersion + cavity-spectrum fitting |
| **Peak detection** | Automatic peak finding with prominence filtering |
| **Baseline correction** | arPLS, SNIP, rubber band, iModPoly, rolling ball |
| **Spectral math** | Smoothing, derivatives, band area, normalization, spectral arithmetic |
| **Transforms** | Kramers-Kronig, Kubelka-Munk, Tauc plot, SNV, Urbach tail |
| **Unit conversions** | Wavenumber, wavelength, energy, linewidth interconversion |
| **PL/Raman mapping** | Spatial maps, peak fitting, cosmic ray detection and removal |

## Documentation Layout

This documentation follows the [Diátaxis](https://diataxis.fr/) framework:

- **Tutorials** — step-by-step walkthroughs for complete workflows (chirp correction for broadband TA, FTIR/Raman peak fitting, PL/Raman map analysis)
- **How-To Guides** — focused recipes for specific tasks (tuning detection sensitivity, fitting overlapping peaks, choosing a peak model, etc.)
- **Reference** — complete API documentation grouped by topic
- **Explanation** — background theory and design rationale for fit statistics and baseline algorithms

## Related Packages

- [CurveFitModels.jl](https://garrekstemo.github.io/CurveFitModels.jl/stable/) — lineshape and temporal model functions used for fitting
- [CurveFit.jl](https://github.com/SciML/CurveFit.jl) — the nonlinear least-squares solver
- [QPSTools.jl](https://github.com/garrekstemo/QPSTools.jl) — lab-specific loaders, plotting themes, and eLabFTW integration built on top of OpticalSpectroscopy
