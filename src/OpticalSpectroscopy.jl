"""
OpticalSpectroscopy.jl — General-purpose spectroscopy analysis toolkit.

Provides data types, fitting routines, baseline correction, peak detection,
unit conversions, and utility functions for spectroscopic data analysis.
"""
module OpticalSpectroscopy

# Dependencies
using CurveFit
import CurveFit: residuals, predict, fitted
using CurveFitModels
using Statistics
using LinearAlgebra
using SparseArrays
using Unitful
using Peaks: findmaxima, findminima, peakproms!, peakwidths!
using SavitzkyGolay: savitzky_golay as _sg_filter
using Interpolations
using JSON
import PhysicalConstants.CODATA2022: h, c_0, ħ
using SpecialFunctions: gamma, erfc, erfcx

# Source files (order matters: tokens (pure vocab) before types; types before
# functions that use them)
include("tokens.jl")
include("types.jl")
include("timeresolved.jl")
include("units.jl")
include("spectroscopy.jl")
include("baseline.jl")
include("transforms.jl")
include("peakdetection.jl")
include("peakfitting.jl")
include("fitting.jl")
include("cavity.jl")
include("chirp.jl")
include("plmap.jl")
include("decomposition.jl")
include("cosmic_rays.jl")

# ==========================================================================
# Exports — Types & Interface
# ==========================================================================
export AbstractSpectroscopyData
export xdata, ydata, zdata, xlabel, ylabel, zlabel, is_matrix
export source_file, npoints, title
export xdata_unitful, ydata_unitful, guess_units!
export axis_label, is_canonical, validate_tokens, normalize_unit, normalize_quantity
export Spectrum, KineticTrace, TimeResolvedMatrix, SweepData
export delay, signal, wavenumber, wavelength

# Time-resolved slice extraction
export kinetic_trace, spectral_slice, integrate_time, bin_matrix

# PL/Raman spatial mapping
export PLMap, extract_spectrum, peak_centers, intensity
export pixel_view, eachpixel, spectra_matrix
export integrated_intensity, intensity_mask, fit_map, FitMapResult

# Cosmic ray detection/removal (1D, PLMap, TimeResolvedMatrix)
# Deliberate asymmetry: CosmicRayMatrixResult carries `threshold` for
# cache-invalidation in downstream apps; the 1D/PLMap results do not.
export CosmicRayResult, CosmicRayMapResult, CosmicRayMatrixResult
export detect_cosmic_rays, remove_cosmic_rays

# PL Map decomposition (PCA, NMF)
export DecompositionResult, pca_map, nmf_map

# ==========================================================================
# Exports — Fit Results
# ==========================================================================
export ExpDecayFit, MultiexpDecayFit, GlobalFitResult, StretchedDecayFit
export LifetimeSpectrumResult
export MultiPeakFitResult, PeakFitResult
export TAPeak, TASpectrumFit, anharmonicity, das

# ==========================================================================
# Exports — Fitting
# ==========================================================================
export fit_exp_decay, fit_global, fit_lifetime_spectrum
export fit_peaks, initial_peak_guesses, predict_peak, predict_baseline
export fit_ta_spectrum
export report, format_results, polynomial
export mean_lifetime

# ==========================================================================
# Exports — Cavity & polariton spectroscopy
# ==========================================================================
export cavity_transmittance, compute_cavity_transmittance
export refractive_index, extinction_coeff
export cavity_mode_energy, polariton_branches, polariton_eigenvalues
export hopfield_coefficients
export CavityFitResult, DispersionFitResult
export fit_cavity_spectrum, fit_dispersion

# ==========================================================================
# Exports — Chirp correction
# ==========================================================================
export ChirpCalibration, calibrate_chirp, detect_chirp, correct_chirp, subtract_background
export save_chirp, load_chirp
export svd_filter, singular_values, estimate_n_components

# ==========================================================================
# Exports — Peak detection
# ==========================================================================
export PeakInfo, find_peaks, peak_table, peak_bounds

# ==========================================================================
# Exports — Baseline correction
# ==========================================================================
export arpls_baseline, snip_baseline, rubberband_baseline
export imodpoly_baseline, rolling_ball_baseline
export correct_baseline

# ==========================================================================
# Exports — Spectroscopy utilities
# ==========================================================================
# NOTE: `normalize` (max-abs scaling) is deliberately NOT exported — it would
# collide with LinearAlgebra.normalize in any session loading both packages.
# Use `OpticalSpectroscopy.normalize`, or the explicit variants below.
export normalize_intensity, subtract_spectrum
export smooth_data, calc_fwhm
export transmittance_to_absorbance, absorbance_to_transmittance
export savitzky_golay_smooth, derivative
export band_area, normalize_area, normalize_to_peak
export estimate_snr
# Spectral arithmetic
export add_spectra, divide_spectra, multiply_spectrum, average_spectra
export interpolate_spectrum
# Transforms
export kramers_kronig, kubelka_munk, tauc_plot
export reflectance_to_absorbance, transmittance_to_reflectance, absorbance_to_reflectance
export snv, beer_lambert
export urbach_tail, thickness_from_fringes

# ==========================================================================
# Exports — Unit conversions
# ==========================================================================
export wavenumber_to_wavelength, wavelength_to_wavenumber
export wavenumber_to_energy, wavelength_to_energy, energy_to_wavelength
export linewidth_to_decay_time, decay_time_to_linewidth

# ==========================================================================
# Re-exports from CurveFit.jl
# ==========================================================================
export solve, NonlinearCurveFitProblem
export coef, residuals, predict, fitted, stderror, confint, rss, mse, nobs, isconverged

# ==========================================================================
# Re-exports from CurveFitModels.jl
# ==========================================================================
export gaussian, lorentzian, pseudo_voigt, single_exponential
export fano, voigt, log_normal
export stretched_exponential

# ==========================================================================
# Plotting — methods live in OpticalSpectroscopyMakieExt (loaded with Makie)
# ==========================================================================
"""
    plot_fit(fit::MultiPeakFitResult; residuals=true, components=false,
             baseline=false, xlabel="x", ylabel="Signal",
             figure=(;), axis=(;)) -> Figure

Plot data, fitted curve, and (optionally) residuals, per-peak components,
and the fitted baseline for a multi-peak fit.

Requires a Makie backend: load CairoMakie or GLMakie first to activate the
plotting extension. Returns a `Makie.Figure`.

# Keywords
- `residuals::Bool=true` — Add a residuals panel below the main axis
- `components::Bool=false` — Draw each fitted peak separately
- `baseline::Bool=false` — Draw the fitted polynomial baseline
- `xlabel`, `ylabel` — Axis labels
- `figure`, `axis` — NamedTuples forwarded to `Figure`/`Axis`

# Example
```julia
using GLMakie, OpticalSpectroscopy
result = fit_peaks(x, y, (2000, 2100); n_peaks=2)
plot_fit(result; components=true, baseline=true)
```
"""
function plot_fit end
export plot_fit

end # module OpticalSpectroscopy
