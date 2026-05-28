"""
OpticalSpectroscopy.jl — General-purpose spectroscopy analysis toolkit.

Provides data types, fitting routines, baseline correction, peak detection,
unit conversions, and utility functions for spectroscopic data analysis.

Extracted from QPS.jl to serve as a standalone, reusable foundation.
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

# Source files (order matters: types before functions that use them)
include("types.jl")
include("units.jl")
include("spectroscopy.jl")
include("baseline.jl")
include("transforms.jl")
include("peakdetection.jl")
include("peakfitting.jl")
include("fitting.jl")
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
export TATrace, TASpectrum, TAMatrix, SweepData
export delay, signal, wavenumber, wavelength

# PL/Raman spatial mapping
export PLMap, extract_spectrum, peak_centers, intensity
export integrated_intensity, intensity_mask, fit_map, FitMapResult
export CosmicRayResult, CosmicRayMapResult, detect_cosmic_rays, remove_cosmic_rays

# PL Map decomposition (PCA, NMF)
export DecompositionResult, pca_map, nmf_map

# ==========================================================================
# Exports — Fit Results
# ==========================================================================
export ExpDecayFit, MultiexpDecayFit, GlobalFitResult
export MultiPeakFitResult, PeakFitResult
export TAPeak, TASpectrumFit, anharmonicity, das

# ==========================================================================
# Exports — Fitting
# ==========================================================================
export fit_exp_decay, fit_global
export fit_peaks, predict_peak, predict_baseline
export fit_ta_spectrum
export report, format_results, polynomial

# ==========================================================================
# Exports — Chirp correction
# ==========================================================================
export ChirpCalibration, detect_chirp, correct_chirp, subtract_background
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
export normalize, normalize_intensity, subtract_spectrum
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

# ==========================================================================
# Plotting — methods live in OpticalSpectroscopyMakieExt (loaded with Makie)
# ==========================================================================
function plot_fit end
export plot_fit

end # module OpticalSpectroscopy
