# Provide Manual Initial Guesses

When auto-detection fails or the fit converges to a wrong solution, supply your own initial guesses.

## Problem

`fit_peaks` auto-detects peaks and generates initial parameters, but sometimes:
- The auto-detected peaks are wrong (e.g., noise spike selected instead of real peak)
- The fit converges to a local minimum instead of the physical solution
- You know the approximate peak parameters from prior measurements

## Solution

Set up example data to work with:

```julia
using OpticalSpectroscopy
using CurveFitModels

x = collect(range(1950, 2150, length=600))
y = lorentzian([0.3, 2040.0, 15.0, 0.0], x) .+
    lorentzian([0.5, 2060.0, 20.0, 0.0], x) .+
    0.01 .+ 0.004 .* randn(length(x))
```

### Option 1: Provide Detected Peaks

Run `find_peaks` separately, filter or modify the results, and pass them to `fit_peaks`:

```julia
# Detect peaks
peaks = find_peaks(x, y, min_prominence=0.02)

# Keep only the peaks you want
selected = filter(p -> 2000 < p.position < 2100, peaks)

# Pass to fit_peaks
result = fit_peaks(x, y, (1950, 2150); peaks=selected)
```

### Option 2: Provide the Full Parameter Vector

For complete control, pass `p0` — the initial parameter vector. The layout depends on the model:

**For Lorentzian or Gaussian** (3 params per peak + baseline):

```
p0 = [A₁, x₁, Γ₁, ..., Aₙ, xₙ, Γₙ, c₀, c₁]
```

- `Aᵢ` — amplitude of peak i
- `xᵢ` — center position of peak i
- `Γᵢ` — width of peak i (FWHM for Lorentzian, σ for Gaussian)
- `c₀, c₁` — baseline coefficients (constant, linear)

**For Pseudo-Voigt** (4 params per peak + baseline):

```
p0 = [A₁, x₁, σ₁, η₁, ..., c₀, c₁]
```

- `ηᵢ` — mixing parameter (0 = Gaussian, 1 = Lorentzian)

**Example — single Lorentzian with linear baseline:**

```julia
p0 = [
    0.5,      # amplitude
    2060.0,   # center (cm⁻¹)
    20.0,     # FWHM (cm⁻¹)
    0.01,     # baseline constant
    0.0       # baseline slope
]

result = fit_peaks(x, y, (1950, 2150); p0=p0, n_peaks=1)
```

**Example — two Lorentzians with constant baseline:**

```julia
p0 = [
    0.3, 2040.0, 15.0,   # peak 1
    0.5, 2060.0, 20.0,   # peak 2
    0.01                  # baseline constant
]

result = fit_peaks(x, y, (1950, 2150); p0=p0, n_peaks=2, baseline_order=0)
```

## Tips

- The `p0` vector length must equal `n_peaks × params_per_peak + baseline_order + 1`
- If `n_peaks` is not specified with `p0`, it is inferred from the vector length
- Good initial guesses don't need to be exact — within a factor of 2 is usually sufficient
- If the fit still doesn't converge, try narrowing the fitting region

## See Also

- [`fit_peaks`](@ref) — `peaks` and `p0` keyword arguments
- [`find_peaks`](@ref) — generate `PeakInfo` objects to pass as initial guesses
