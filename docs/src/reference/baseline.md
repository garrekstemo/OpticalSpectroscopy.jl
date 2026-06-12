# Baseline Correction

Baseline correction algorithms with a unified `correct_baseline` dispatcher.

For guidance on choosing between algorithms, see [Baseline Algorithms](@ref).

## Unified API

```@docs
correct_baseline(y::AbstractVector{<:Real}; method::Symbol, kwargs...)
correct_baseline(x::AbstractVector, y::AbstractVector{<:Real}; kwargs...)
correct_baseline(s::Spectrum)
```

## Individual Algorithms

```@docs
arpls_baseline
snip_baseline
rubberband_baseline
imodpoly_baseline
rolling_ball_baseline
```
