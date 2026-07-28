# Peak Fitting

Unified multi-peak fitting that works on any spectrum type. Uses [CurveFit.jl](https://docs.sciml.ai/CurveFit/stable/) for the solver and [CurveFitModels.jl](https://github.com/garrekstemo/CurveFitModels.jl) for peak model functions.

## Fitting

```@docs
fit_peaks
initial_peak_guesses
```

## Result Types

```@docs
PeakFitResult
MultiPeakFitResult
```

## Prediction and Decomposition

The `predict` and `residuals` functions are re-exported from CurveFit.jl and extended for `MultiPeakFitResult`:

- `predict(result)` — Composite fit curve (all peaks + baseline) evaluated on the fit region
- `predict(result, x)` — Composite fit curve on a custom x-axis
- `residuals(result)` — Data minus fit (`y_data - y_fit`)

```@docs
predict_peak
predict_baseline
```

## Reporting

```@docs
report
```
