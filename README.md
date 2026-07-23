# OpticalSpectroscopy.jl

[![CI](https://github.com/garrekstemo/OpticalSpectroscopy.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/garrekstemo/OpticalSpectroscopy.jl/actions/workflows/CI.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://garrekstemo.github.io/OpticalSpectroscopy.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://garrekstemo.github.io/OpticalSpectroscopy.jl/dev/)
[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![codecov](https://codecov.io/gh/garrekstemo/OpticalSpectroscopy.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/garrekstemo/OpticalSpectroscopy.jl)

**Integrated tools for optical spectroscopy in Julia — UV-Vis, FTIR, Raman, photoluminescence, and femtosecond transient absorption.**

OpticalSpectroscopy.jl combines typed data structures (`Spectrum`, `KineticTrace`, `TimeResolvedMatrix`, `PLMap`) with peak fitting (Gaussian, Lorentzian, Voigt, Pseudo-Voigt, Fano), baseline correction (arPLS, SNIP, rubber band, iModPoly, rolling ball), peak detection, spectral transforms (Kramers-Kronig, Tauc, Kubelka-Munk), unit conversions, and ultrafast-specific tools including IRF-convolved exponential fitting, global analysis with decay-associated spectra, and chirp correction for broadband pump-probe data.

The core primitives — peak fitting, baseline correction, smoothing, unit conversions — work on any 1D spectrum-like data and are usable across spectroscopic domains as needed.

## Installation

```julia
using Pkg
Pkg.add("OpticalSpectroscopy")
```

## Quick Start

```julia
using OpticalSpectroscopy

# --- Ultrafast: kinetics with IRF-convolved biexponential ---
trace = KineticTrace(time, signal; wavelength=800.0)
fit = fit_exp_decay(trace; n_exp=2, irf=true)
report(fit)

# --- Ultrafast: global fit across a TA matrix ---
matrix = TimeResolvedMatrix(time, wavelength, data)
gfit = fit_global(matrix; n_exp=2)   # shared τ, decay-associated spectra
spectra = das(gfit)                  # n_exp × n_wavelengths matrix

# --- Steady-state: peak fitting (FTIR, Raman, UV-vis) ---
result = fit_peaks(x, y, (2000, 2100))
report(result)

# --- Steady-state: baseline correction ---
bl = correct_baseline(x, y; method=:arpls)
y_corrected = bl.y

# --- Unit conversions (with Unitful) ---
wavelength_to_wavenumber(1500u"nm")
decay_time_to_linewidth(1.0u"ps")
```

## Features

| Module | Description |
|--------|-------------|
| **Data types** | `Spectrum`, `KineticTrace`, `TimeResolvedMatrix` with semantic axis indexing (`m[λ=800]`, `m[t=1.0]`) |
| **Exponential decay** | Single/multi-exponential with optional IRF convolution |
| **Global fitting** | Shared parameters across traces, decay-associated spectra |
| **TA spectrum fitting** | N-peak model with ESA/GSB/SE labels and sign convention |
| **Chirp correction** | OKE-based calibration, in-data GVD detection (cross-correlation, threshold), and correction for broadband TA |
| **SVD filtering** | Matrix denoising for broadband TA data |
| **Peak fitting** | Gaussian, Lorentzian, Voigt, Pseudo-Voigt, Fano via [CurveFit.jl](https://github.com/SciML/CurveFit.jl) |
| **Peak detection** | Automatic peak finding with prominence filtering |
| **Baseline correction** | arPLS, SNIP, rubber band, iModPoly, rolling ball |
| **Spectral math** | Smoothing, derivatives, band area, normalization, spectral arithmetic |
| **Transforms** | Kramers-Kronig, Kubelka-Munk, Tauc plot, SNV, Urbach tail |
| **Unit conversions** | Wavenumber, wavelength, energy, linewidth interconversion |
| **PL/Raman mapping** | Spatial maps, peak fitting, cosmic ray detection and removal |
| **Time-resolved PL (streak camera)** | Slice extraction, binning, cosmic ray removal, stretched-exponential and lifetime-vs-wavelength fitting |

## Data Types

OpticalSpectroscopy provides typed containers for spectroscopy data:

```julia
# Kinetics at a single wavelength
trace = KineticTrace(time, signal; wavelength=800.0)

# Spectrum at a fixed time delay
spectrum = Spectrum(wavenumber, signal; axis=:wavenumber, time_delay=1.0)

# 2D time x wavelength matrix with semantic indexing
matrix = TimeResolvedMatrix(time, wavelength, data)
matrix[λ=800]   # -> KineticTrace at nearest wavelength
matrix[t=1.0]   # -> Spectrum at nearest time delay
```

## Scope

OpticalSpectroscopy.jl targets photon-based spectroscopy of condensed matter and molecular systems. It does not cover magnetic resonance (NMR, EPR), X-ray (XRD, XPS, XAS), mass spectrometry, atomic/plasma spectroscopy, astronomical spectroscopy, or attosecond spectroscopy — those fields have different data conventions and analysis traditions.

The shared algorithmic primitives — peak fitting, baseline correction, smoothing — operate on any 1D signal data and can be used across domains as needed.

## How is this different from Spectra.jl?

[Spectra.jl](https://github.com/charlesll/Spectra.jl) is an array-based toolkit for steady-state Raman/IR processing: plain `x`/`y` arrays through baseline correction, smoothing, and peak fitting, plus Raman-specific corrections (Long/Galeener/Hehlen temperature-excitation corrections, diamond anvil cell utilities).

OpticalSpectroscopy.jl instead provides **typed data structures** (`Spectrum`, `KineticTrace`, `TimeResolvedMatrix`, `PLMap`) for TA/PL/FTIR data with semantic indexing (`matrix[λ=800]`, `matrix[t=1.0]`), and an **integrated transient-absorption pipeline** — chirp calibration (from an OKE run), detection, and correction for broadband pump-probe data, SVD filtering, and IRF-convolved exponential and global fitting with decay-associated spectra — which no registered Julia package offers. The steady-state primitives are the shared foundation; the ultrafast and mapping workflows are the differentiator. The two packages are complementary and coexist in one session.

## Related packages

- [Spectra.jl](https://github.com/charlesll/Spectra.jl) — Raman preprocessing and fitting with Raman-specific temperature/laser corrections (Long, Galeener, Hehlen) and diamond anvil cell utilities. OpticalSpectroscopy.jl does not implement these; users needing them can use both packages together.
- [Peaks.jl](https://github.com/halleysfifthinc/Peaks.jl) — Lower-level peak detection primitives (used internally).
- [CurveFit.jl](https://github.com/SciML/CurveFit.jl) — Nonlinear fitting backend (used internally).
- [CurveFitModels.jl](https://github.com/garrekstemo/CurveFitModels.jl) — Lineshape model functions (used internally).

## Dependencies

OpticalSpectroscopy uses [CurveFit.jl](https://github.com/SciML/CurveFit.jl) as the nonlinear fitting backend and [CurveFitModels.jl](https://github.com/garrekstemo/CurveFitModels.jl) for model functions. Unit conversions use [Unitful.jl](https://github.com/PainterQubits/Unitful.jl).
