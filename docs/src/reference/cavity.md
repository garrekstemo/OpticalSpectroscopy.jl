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
- Hopfield convention: with detuning ``\delta = E_{cav} - E_{vib}`` and
  ``\theta = \tfrac{1}{2}\,\mathrm{atan2}(\Omega, \delta)``, the photon
  fraction of the lower polariton is
  ``\sin^2\theta = \tfrac{1}{2}(1 - \delta/\sqrt{\delta^2 + \Omega^2})``
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
