# Spectral decomposition for PLMap data (PCA, NMF)

# =============================================================================
# DecompositionResult type
# =============================================================================

"""
    DecompositionResult

Result of PCA or NMF decomposition of a PLMap.

# Fields
- `loadings::Array{Float64,3}` — Spatial maps `(nx, ny, n_components)`
- `components::Matrix{Float64}` — Spectral profiles `(n_components, n_spectral)`
- `explained_variance::Vector{Float64}` — Fraction of variance explained (PCA)
  or reconstruction error per component (NMF)
"""
struct DecompositionResult
    loadings::Array{Float64,3}
    components::Matrix{Float64}
    explained_variance::Vector{Float64}
end

function Base.show(io::IO, r::DecompositionResult)
    nc = size(r.components, 1)
    nx, ny, _ = size(r.loadings)
    print(io, "DecompositionResult($(nx)×$(ny) grid, $(nc) components)")
end

function Base.show(io::IO, ::MIME"text/plain", r::DecompositionResult)
    nc = size(r.components, 1)
    nx, ny, _ = size(r.loadings)
    n_spectral = size(r.components, 2)
    println(io, "DecompositionResult")
    println(io, "  Grid:       $(nx) × $(ny) spatial points")
    println(io, "  Components: $(nc)")
    println(io, "  Spectral:   $(n_spectral) channels")
    print(io, "  Variance:   ", [round(v, digits=4) for v in r.explained_variance])
end

# =============================================================================
# Helpers
# =============================================================================

"""
Reshape PLMap spectra to a 2D matrix `(n_pixels_spatial, n_spectral)` and
optionally restrict to the given spectral range.

Returns `(data_matrix, spectral_indices)` where `spectral_indices` is the
range of indices into the original pixel axis.
"""
function _prepare_map_matrix(m::PLMap; pixel_range=nothing)
    nx, ny, n_pixel = size(m.spectra)

    if !isnothing(pixel_range)
        p1 = max(1, pixel_range[1])
        p2 = min(n_pixel, pixel_range[2])
        spec_range = p1:p2
    else
        spec_range = 1:n_pixel
    end

    # Reshape (nx, ny, n_spectral) -> (nx*ny, n_spectral)
    spectra_slice = @view m.spectra[:, :, spec_range]
    data = reshape(spectra_slice, nx * ny, length(spec_range))

    return Float64.(data), spec_range, nx, ny
end

# =============================================================================
# PCA
# =============================================================================

"""
    pca_map(m::PLMap; n_components::Int=3, pixel_range=nothing) -> DecompositionResult

Principal Component Analysis of PLMap spectra via truncated SVD.

Each spatial pixel has a full spectrum. PCA finds the orthogonal spectral
components that capture the most variance across the map, along with their
spatial loading maps.

# Arguments
- `m::PLMap`: Input PL map with `spectra` field of shape `(nx, ny, n_pixel)`.
- `n_components::Int=3`: Number of principal components to retain.
- `pixel_range`: Optional `(start, stop)` tuple to restrict the spectral
  range before decomposition.

# Returns
A [`DecompositionResult`](@ref) with:
- `loadings`: Spatial score maps `(nx, ny, n_components)` — the weight of
  each component at each spatial position.
- `components`: Spectral profiles `(n_components, n_spectral)` — the principal
  spectral shapes (eigenvectors of the covariance matrix).
- `explained_variance`: Fraction of total variance captured by each component.

# Example
```julia
result = pca_map(m; n_components=5, pixel_range=(900, 1100))
# result.loadings[:, :, 1]  — spatial map of the first principal component
# result.components[1, :]    — spectral shape of the first component
# result.explained_variance  — [0.85, 0.08, 0.03, ...]
```
"""
function pca_map(m::PLMap; n_components::Int=3, pixel_range=nothing)
    data, _, nx, ny = _prepare_map_matrix(m; pixel_range=pixel_range)
    n_spatial, n_spectral = size(data)

    n_components < 1 && throw(ArgumentError("n_components must be >= 1"))
    max_components = min(n_spatial, n_spectral)
    n_components > max_components && throw(ArgumentError(
        "n_components ($n_components) exceeds matrix rank ($max_components)"))

    # Center the data (subtract mean spectrum)
    col_means = vec(mean(data, dims=1))
    centered = data .- col_means'

    # SVD: centered = U * S * Vt
    F = svd(centered)

    # Scores (spatial weights) and loadings (spectral components)
    scores = F.U[:, 1:n_components] .* F.S[1:n_components]'
    components = F.Vt[1:n_components, :]

    # Reshape scores back to spatial grid
    loadings = reshape(scores, nx, ny, n_components)

    # Explained variance ratio
    total_var = sum(F.S .^ 2)
    explained = (F.S[1:n_components] .^ 2) ./ total_var

    return DecompositionResult(loadings, components, explained)
end

# =============================================================================
# NMF (multiplicative update rules)
# =============================================================================

"""
    nmf_map(m::PLMap; n_components::Int=3, pixel_range=nothing,
            max_iter::Int=200, tol::Float64=1e-4) -> DecompositionResult

Non-negative Matrix Factorization of PLMap spectra.

Factorizes the non-negative spectra matrix `V ≈ W * H` where `W` contains
spatial weights and `H` contains non-negative spectral components. Unlike PCA,
NMF enforces non-negativity, producing physically interpretable components that
can represent distinct emitters or spectral species.

Uses Lee & Seung multiplicative update rules with Frobenius norm objective.

# Arguments
- `m::PLMap`: Input PL map with `spectra` field of shape `(nx, ny, n_pixel)`.
- `n_components::Int=3`: Number of NMF components.
- `pixel_range`: Optional `(start, stop)` tuple to restrict the spectral
  range before decomposition.
- `max_iter::Int=200`: Maximum number of multiplicative update iterations.
- `tol::Float64=1e-4`: Convergence tolerance on the relative change in
  reconstruction error between iterations.

# Returns
A [`DecompositionResult`](@ref) with:
- `loadings`: Non-negative spatial weight maps `(nx, ny, n_components)`.
- `components`: Non-negative spectral profiles `(n_components, n_spectral)`.
- `explained_variance`: Per-component fraction of the total Frobenius norm
  (`‖wₖ hₖ‖² / ‖V‖²` for each component `k`, not a cumulative sum).

# Example
```julia
result = nmf_map(m; n_components=3, max_iter=500)
# result.loadings[:, :, 2]  — spatial distribution of the second species
# result.components[2, :]    — emission spectrum of the second species
```
"""
function nmf_map(m::PLMap; n_components::Int=3, pixel_range=nothing,
                 max_iter::Int=200, tol::Float64=1e-4)
    data, _, nx, ny = _prepare_map_matrix(m; pixel_range=pixel_range)
    n_spatial, n_spectral = size(data)

    n_components < 1 && throw(ArgumentError("n_components must be >= 1"))
    max_components = min(n_spatial, n_spectral)
    n_components > max_components && throw(ArgumentError(
        "n_components ($n_components) exceeds matrix dimensions ($max_components)"))

    # Clamp negative values to zero (NMF requires non-negative input)
    V = max.(data, 0.0)

    # Initialize W and H with positive random values
    eps_val = 1e-10
    W = abs.(randn(n_spatial, n_components)) .+ eps_val
    H = abs.(randn(n_components, n_spectral)) .+ eps_val

    # Pre-allocate workspace matrices for in-place operations
    WtV  = similar(H)                                            # (n_components, n_spectral)
    WtW  = zeros(eltype(H), n_components, n_components)          # (n_components, n_components)
    WtWH = similar(H)                                            # (n_components, n_spectral)
    VHt  = similar(W)                                            # (n_spatial, n_components)
    HHt  = zeros(eltype(W), n_components, n_components)          # (n_components, n_components)
    WHHt = similar(W)                                            # (n_spatial, n_components)
    WH   = zeros(eltype(V), n_spatial, n_spectral)               # (n_spatial, n_spectral)

    # Multiplicative update rules (Lee & Seung, Frobenius norm)
    prev_error = Inf
    for iter in 1:max_iter
        # Update H: H <- H .* (W'V) ./ (W'W H)
        mul!(WtV, W', V)
        mul!(WtW, W', W)
        mul!(WtWH, WtW, H)
        H .= H .* WtV ./ (WtWH .+ eps_val)

        # Update W: W <- W .* (V H') ./ (W H H')
        mul!(VHt, V, H')
        mul!(HHt, H, H')
        mul!(WHHt, W, HHt)
        W .= W .* VHt ./ (WHHt .+ eps_val)

        # Check convergence every 10 iterations
        if iter % 10 == 0
            mul!(WH, W, H)
            WH .= V .- WH
            error = sum(abs2, WH)
            rel_change = abs(prev_error - error) / (prev_error + eps_val)
            if rel_change < tol
                break
            end
            prev_error = error
        end
    end

    # Normalize: scale each component so H rows have unit norm
    for k in 1:n_components
        h_norm = norm(view(H, k, :))
        if h_norm > eps_val
            H[k, :] ./= h_norm
            W[:, k] .*= h_norm
        end
    end

    # Reshape W to spatial grid
    loadings = reshape(W, nx, ny, n_components)

    # Compute explained variance as fraction of Frobenius norm
    total_norm_sq = sum(V .^ 2)
    explained = Vector{Float64}(undef, n_components)
    for k in 1:n_components
        # Contribution of component k: ||w_k * h_k||^2 (approximate)
        wk = view(W, :, k)
        hk = view(H, k, :)
        component_norm_sq = sum(wk .^ 2) * sum(hk .^ 2)
        explained[k] = component_norm_sq / (total_norm_sq + eps_val)
    end

    return DecompositionResult(loadings, H, explained)
end
