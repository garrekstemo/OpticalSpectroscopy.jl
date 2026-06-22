# Fitting Statistics Reference

A practical guide to statistics used in spectroscopic peak fitting. The examples assume `result` is a `MultiPeakFitResult` obtained from `fit_peaks`:

```julia
result = fit_peaks(x, y, (region_min, region_max))
```

Goodness-of-fit statistics are plain fields on the result (`result.r_squared`, `result.rss`, `result.mse`); per-parameter estimates come from the per-peak `PeakFitResult` reached by indexing — `result[i]` is peak `i`, and `result[i][:center]` returns a `(value, err, ci)` named tuple. The only CurveFit accessor extended for this type is [`residuals`](@ref). (To use the solver-level accessors `coef`/`stderror`/`confint`, fit a `NonlinearCurveFitProblem` directly and call them on the returned `CurveFitSolution`.)

## Parameter Estimates

### Coefficients

The fitted parameter values (amplitude, peak position, FWHM, offset) obtained by minimizing the residual sum of squares. Access them per peak, by name:

```julia
peak = result[1]               # PeakFitResult for the first peak
peak[:center].value            # fitted peak position
peak.params                    # available parameter names, e.g. [:amplitude, :center, :fwhm, :offset]
[peak[:center].value for peak in result]   # centers of every peak
```

## Uncertainty Quantification

### Standard Error

The standard error estimates uncertainty in each fitted parameter. Computed from the diagonal of the variance-covariance matrix:

```math
\text{SE}(\hat{p}_i) = \sqrt{\text{Var}(\hat{p}_i)} = \sqrt{C_{ii}}
```

where ``C`` is the covariance matrix.

```julia
se = result[1][:center].err   # standard error of the first peak's center
```

**When to use:** Report as parameter ± SE for quick uncertainty estimates. Assumes errors are symmetric and normally distributed.

### Confidence Intervals

A range likely to contain the true parameter value. The 95% CI means: if we repeated the experiment many times, 95% of computed intervals would contain the true value.

```math
\text{CI} = \hat{p} \pm t_{\alpha/2, \nu} \cdot \text{SE}(\hat{p})
```

where ``t_{\alpha/2, \nu}`` is the t-distribution critical value and ``\nu`` is degrees of freedom.

```julia
ci = result[1][:center].ci  # 95% CI tuple (low, high) for the first peak's center
```

The intervals are computed at the 95% level. For a different level, fit a `NonlinearCurveFitProblem` directly and call `confint(sol; level=0.99)` on the solver result.

**When to use:** Preferred for publications. More informative than SE alone, especially for small sample sizes where t-distribution differs significantly from normal.

### Variance-Covariance Matrix

The full covariance matrix captures correlations between parameters. Off-diagonal elements indicate how uncertainties in different parameters are related.

```math
C = \sigma^2 (J^T J)^{-1}
```

where ``J`` is the Jacobian matrix and ``\sigma^2`` is estimated from MSE.

Per-parameter standard errors (`result[i][name].err`) and confidence intervals (`result[i][name].ci`) are exposed directly; the full covariance matrix is not currently surfaced as a convenience. For error propagation across derived quantities, use the per-parameter `.err` under the independent-parameter approximation, or access the underlying solver result.

**When to use:** Unusually large `stderror` values flag potential parameter correlations — a good diagnostic for overparameterized models (e.g. two overlapping peaks when one suffices).

## Goodness of Fit

### Residual Sum of Squares (RSS)

The sum of squared differences between observed and fitted values:

```math
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

```julia
ss_res = result.rss
```

**When to use:** Comparing fits of the same model to the same data. Lower is better. Not useful for comparing across different datasets or region sizes.

### Mean Squared Error (MSE)

RSS normalized by degrees of freedom:

```math
\text{MSE} = \frac{\text{RSS}}{n - p}
```

where ``n`` is number of points and ``p`` is number of parameters.

```julia
mse_val = result.mse
```

**When to use:** Comparing fits across different region sizes. Estimates the variance of the residuals. Also used internally to compute parameter uncertainties.

### Coefficient of Determination (R²)

Proportion of variance explained by the model:

```math
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
```

```julia
r_squared = result.r_squared          # computed by the fit
# equivalently, from the residual and total sums of squares:
ss_tot = sum((y .- mean(y)).^2)
r_squared == 1 - result.rss / ss_tot
```

**When to use:** Quick assessment of fit quality. R² > 0.99 is typical for good spectroscopic fits. However, R² can be misleading:
- Always high for peaked data (even bad fits)
- Doesn't detect systematic deviations
- Can't compare models with different numbers of parameters

**Caution:** Always check residuals visually. A high R² does not guarantee a good fit.

### Reduced Chi-Squared (χ²/ν)

If measurement uncertainties (``\sigma_i``) are known:

```math
\chi^2_\nu = \frac{1}{\nu} \sum_{i=1}^{n} \frac{(y_i - \hat{y}_i)^2}{\sigma_i^2}
```

where ``\nu = n - p`` is degrees of freedom.

| Value | Interpretation |
|-------|----------------|
| χ²/ν ≈ 1 | Good fit, errors correctly estimated |
| χ²/ν >> 1 | Poor fit or underestimated errors |
| χ²/ν << 1 | Overfitting or overestimated errors |

**When to use:** When you have reliable error estimates for each data point (e.g., from detector noise specifications or repeated measurements).

## Residual Analysis

### Residuals

The differences between observed and fitted values:

```math
r_i = y_i - \hat{y}_i
```

```julia
res = residuals(result)
```

**Visual inspection is essential.** Look for:
- **Random scatter around zero** — good fit
- **Systematic patterns** — wrong model or missing physics
- **Trends** — baseline problems
- **Outliers** — bad data points or model breakdown

### Autocorrelation in Residuals

Consecutive residuals should be independent. Correlated residuals indicate model inadequacy.

**Durbin-Watson statistic:**

```math
d = \frac{\sum_{i=2}^{n}(r_i - r_{i-1})^2}{\sum_{i=1}^{n} r_i^2}
```

| Value | Interpretation |
|-------|----------------|
| d ≈ 2 | No autocorrelation |
| d < 2 | Positive autocorrelation (residuals follow trends) |
| d > 2 | Negative autocorrelation (residuals alternate) |

## Model Comparison

### Akaike Information Criterion (AIC)

Balances goodness of fit against model complexity:

```math
\text{AIC} = n \ln(\text{RSS}/n) + 2p
```

Lower AIC is better. Penalizes extra parameters.

### Bayesian Information Criterion (BIC)

Similar to AIC but with stronger penalty for parameters:

```math
\text{BIC} = n \ln(\text{RSS}/n) + p \ln(n)
```

**When to use AIC vs BIC:**
- AIC: prediction-focused, may favor more complex models
- BIC: tends toward simpler models, better for identifying "true" model

### F-test for Nested Models

Compare a simpler model (fewer parameters) to a more complex one:

```math
F = \frac{(\text{RSS}_1 - \text{RSS}_2)/(p_2 - p_1)}{\text{RSS}_2/(n - p_2)}
```

**Example:** Testing whether a second Lorentzian peak is statistically justified.

## Practical Guidelines

### Minimum Reporting for Publications

1. **Parameter values with uncertainties**: peak = 2055.3 ± 0.2 cm⁻¹
2. **Confidence intervals** (preferred): 95% CI (2054.9, 2055.7)
3. **Fit quality metric**: R² = 0.9998 or χ²/ν = 1.02
4. **Residual plot** in supplementary material

### Quick Quality Check

```julia
result = fit_peaks(spec, (1900, 2200))

# Good fit indicators:
# - R² > 0.999 for clean spectroscopic data
# - Parameter errors << parameter values
# - Residuals show no systematic pattern
# - MSE comparable to expected noise level
```

### Red Flags

- R² high but residuals show clear patterns
- Parameter uncertainty larger than parameter value
- Confidence interval includes physically impossible values (e.g., negative FWHM)
- Fit result changes significantly with small changes to region bounds

## References

1. **Bevington, P.R. & Robinson, D.K.** (2003). *Data Reduction and Error Analysis for the Physical Sciences*, 3rd ed. McGraw-Hill.
   - Classic reference for error analysis in experimental physics

2. **Press, W.H. et al.** (2007). *Numerical Recipes: The Art of Scientific Computing*, 3rd ed. Cambridge University Press.
   - Chapter 15: Modeling of Data (least squares, confidence limits)

3. **Motulsky, H. & Christopoulos, A.** (2004). *Fitting Models to Biological Data Using Linear and Nonlinear Regression*. Oxford University Press.
   - Practical guide with emphasis on proper statistical interpretation

4. **IUPAC** (1997). *Compendium of Analytical Nomenclature* (Orange Book), 3rd ed.
   - Standard definitions for analytical chemistry

5. **Hug, W. & Fedorovsky, M.** (2018). A comprehensive approach to statistical analysis of vibrational spectra. *J. Raman Spectrosc.* 49, 3-16.
   - Modern treatment specific to vibrational spectroscopy
