# [Additional API Reference](@id api-reference)

This page collects exported names that aren't covered by the grouped reference pages (peak detection, peak fitting, baseline correction, preprocessing, PL/Raman mapping).

## Module

```@docs
OpticalSpectroscopy.OpticalSpectroscopy
```

## Data Types

```@docs
AbstractSpectroscopyData
Spectrum
KineticTrace
TimeResolvedMatrix
SweepData
TASpectrumFit
TAPeak
fit_ta_spectrum
anharmonicity
```

## Time-Resolved Analysis

Slice extraction, binning, cosmic ray removal, and lifetime fitting for streak-camera and broadband TA data.

```@docs
kinetic_trace
spectral_slice
integrate_time
bin_matrix
fit_lifetime_spectrum
mean_lifetime
StretchedDecayFit
LifetimeSpectrumResult
CosmicRayMatrixResult
```

## Data Interface

Uniform accessors implemented by every `AbstractSpectroscopyData` subtype.

```@docs
xdata
ydata
zdata
xlabel
ylabel
zlabel
is_matrix
```

## Semantic Accessors

Type-specific accessors with domain names.

```@docs
delay
signal
wavenumber
wavelength
```

## Plotting

```@docs
plot_fit
```

## Chirp Correction

```@docs
ChirpCalibration
detect_chirp
correct_chirp
polynomial
save_chirp
load_chirp
```

## SVD Filtering

```@docs
svd_filter
singular_values
estimate_n_components
```

## Exponential Decay Fitting

```@docs
fit_exp_decay
fit_global
ExpDecayFit
MultiexpDecayFit
GlobalFitResult
das
```

## Reporting

`report(result)` prints a formatted summary to the terminal; `format_results(result)` returns the same content as a Markdown string suitable for composing into lab-notebook entries (e.g. the body of an eLabFTW experiment).

```@docs
format_results
```

## Spectroscopy Utilities

```@docs
OpticalSpectroscopy.normalize
calc_fwhm
transmittance_to_absorbance
absorbance_to_transmittance
npoints
source_file
title
```

## Spectral Arithmetic

```@docs
add_spectra
divide_spectra
multiply_spectrum
average_spectra
interpolate_spectrum
```

## Transforms

```@docs
kramers_kronig
kubelka_munk
tauc_plot
snv
beer_lambert
urbach_tail
thickness_from_fringes
reflectance_to_absorbance
absorbance_to_reflectance
transmittance_to_reflectance
```

## Unit Conversions

```@docs
wavenumber_to_wavelength
wavelength_to_wavenumber
wavenumber_to_energy
wavelength_to_energy
energy_to_wavelength
linewidth_to_decay_time
decay_time_to_linewidth
```
