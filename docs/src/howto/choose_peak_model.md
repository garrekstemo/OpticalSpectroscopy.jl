# Choose a Peak Model

OpticalSpectroscopy.jl supports three peak models from CurveFitModels.jl. This guide explains when to use each one.

## Available Models

| Model | Parameters | Line shape |
|-------|-----------|------------|
| `lorentzian` | `[A, x₀, Γ, offset]` | Natural line shape (homogeneous broadening) |
| `gaussian` | `[A, x₀, σ, offset]` | Doppler / inhomogeneous broadening |
| `pseudo_voigt` | `[A, x₀, σ, η]` | Weighted mix of Gaussian and Lorentzian |

All models use the signature `fn(p, x)` (parameters first) and are ForwardDiff-compatible.

## When to Use Each

Start with some example data:

```julia
using OpticalSpectroscopy
using CurveFitModels

x = collect(range(2000, 2150, length=500))
y = lorentzian([0.45, 2062.0, 24.0, 0.01], x) .+ 0.003 .* randn(length(x))
```

### Lorentzian (default)

Best for peaks broadened by a single mechanism with a well-defined lifetime:
- Solution-phase FTIR (collisional broadening)
- Isolated Raman peaks in crystals (phonon lifetime)
- Fluorescence emission from single emitters

```julia
result = fit_peaks(x, y, (2000, 2150))  # lorentzian is the default
```

The width parameter `Γ` is the full width at half maximum (FWHM).

### Gaussian

Best for peaks broadened by a distribution of environments:
- Gas-phase spectra (Doppler broadening)
- Amorphous or disordered materials
- XRD peaks at moderate resolution

```julia
result = fit_peaks(x, y, (2000, 2150); model=gaussian)
```

The width parameter `σ` is the standard deviation. FWHM = 2√(2 ln 2) × σ ≈ 2.355 σ.

### Pseudo-Voigt

Best when both homogeneous and inhomogeneous broadening contribute:
- Solution-phase spectra with concentration-dependent broadening
- Solid-state spectra with mixed broadening
- When neither Lorentzian nor Gaussian fits well

```julia
result = fit_peaks(x, y, (2000, 2150); model=pseudo_voigt)
```

The mixing parameter `η` (0 to 1) gives the Lorentzian fraction. `η = 0` is pure Gaussian, `η = 1` is pure Lorentzian.

## Comparing Models

Fit the same data with multiple models and compare:

```julia
r_lor = fit_peaks(x, y, (2000, 2150))
r_gau = fit_peaks(x, y, (2000, 2150); model=gaussian)
r_voi = fit_peaks(x, y, (2000, 2150); model=pseudo_voigt)

println("Lorentzian: R² = ", round(r_lor.r_squared, digits=5))
println("Gaussian:   R² = ", round(r_gau.r_squared, digits=5))
println("Voigt:      R² = ", round(r_voi.r_squared, digits=5))
```

Check residuals visually — the model with the smallest systematic patterns in the residuals is the best choice, not necessarily the one with the highest R².

!!! note
    Pseudo-Voigt has one more parameter than Lorentzian or Gaussian. A modest improvement in R²
    may not be statistically meaningful. See [Fitting Statistics](@ref "Fitting Statistics Reference") for model comparison
    criteria (AIC, BIC, F-test).

## See Also

- [`fit_peaks`](@ref) — full API reference
- [Fitting Statistics](@ref "Fitting Statistics Reference") — model comparison criteria
