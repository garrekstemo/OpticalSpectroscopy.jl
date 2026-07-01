# PLMap performance benchmark
# Run: julia --project=. -t 12 benchmark/bench_plmap.jl

using OpticalSpectroscopy
using Statistics
using Random

# ============================================================================
# Synthetic PLMap fixture
# ============================================================================

function make_synthetic_plmap(; nx=51, ny=51, n_pixel=512, seed=42)
    rng = MersenneTwister(seed)

    x = collect(range(-50.0, 50.0, length=nx))
    y = collect(range(-50.0, 50.0, length=ny))
    pixel = collect(range(900.0, 1100.0, length=n_pixel))

    # Channel-major cube, matching the PLMap storage contract
    spectra = Array{Float64,3}(undef, n_pixel, nx, ny)

    # Gaussian peak parameters with smooth spatial variation
    for iy in eachindex(y), ix in eachindex(x)
        # Center drifts spatially (simulates strain/composition gradient)
        center = 1000.0 + 10.0 * sin(2pi * x[ix] / 80.0) + 5.0 * cos(2pi * y[iy] / 60.0)
        sigma = 15.0 + 2.0 * cos(2pi * (x[ix] + y[iy]) / 100.0)
        amplitude = 500.0 + 200.0 * exp(-((x[ix])^2 + (y[iy])^2) / (40.0^2))

        for ip in eachindex(pixel)
            spectra[ip, ix, iy] = amplitude * exp(-0.5 * ((pixel[ip] - center) / sigma)^2)
        end
    end

    # Add Poisson-like noise (sqrt of signal + small read noise)
    for idx in eachindex(spectra)
        noise_level = sqrt(max(spectra[idx], 0.0)) + 2.0
        spectra[idx] += noise_level * randn(rng)
    end

    # Inject cosmic rays (~2% of spectra get a spike)
    n_total_spectra = nx * ny
    n_cr_spectra = round(Int, 0.02 * n_total_spectra)
    cr_positions = randperm(rng, n_total_spectra)[1:n_cr_spectra]

    for flat_idx in cr_positions
        iy = (flat_idx - 1) ÷ nx + 1
        ix = (flat_idx - 1) % nx + 1
        # 1-3 spikes per affected spectrum
        n_spikes = rand(rng, 1:3)
        for _ in 1:n_spikes
            ch = rand(rng, 1:n_pixel)
            spike_height = 2000.0 + 3000.0 * rand(rng)
            spectra[ch, ix, iy] += spike_height
        end
    end

    # Compute integrated intensity (sum over the channel axis)
    intensity = dropdims(sum(spectra; dims=1); dims=1)

    metadata = Dict{Symbol,Any}(:source_file => "synthetic_benchmark")

    return PLMap(intensity, spectra, x, y, pixel, metadata)
end

# ============================================================================
# Benchmarking harness
# ============================================================================

struct BenchResult
    name::String
    median_time_s::Float64
    min_time_s::Float64
    bytes::Int64
    allocs::Int64
end

function format_time(s::Float64)
    if s < 1e-3
        return string(round(s * 1e6, digits=1), " us")
    elseif s < 1.0
        return string(round(s * 1e3, digits=1), " ms")
    else
        return string(round(s, digits=3), " s")
    end
end

function format_bytes(b::Int64)
    if b < 1024
        return string(b, " B")
    elseif b < 1024^2
        return string(round(b / 1024, digits=1), " KiB")
    elseif b < 1024^3
        return string(round(b / 1024^2, digits=1), " MiB")
    else
        return string(round(b / 1024^3, digits=2), " GiB")
    end
end

function run_bench(name::String, f::Function; n_runs=3, warmup=true)
    # Warmup run (compile)
    if warmup
        f()
    end

    times = Float64[]
    bytes_list = Int64[]
    allocs_list = Int64[]

    for _ in 1:n_runs
        stats = @timed f()
        push!(times, stats.time)
        push!(bytes_list, stats.bytes)
        push!(allocs_list, Int64(stats.gcstats.allocd > 0 ? stats.gcstats.allocd : stats.bytes))
    end

    med_idx = sortperm(times)[n_runs ÷ 2 + 1]
    return BenchResult(name, times[med_idx], minimum(times), bytes_list[med_idx], allocs_list[med_idx])
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("=" ^ 78)
    println("PLMap Performance Benchmark")
    println("Julia $(VERSION), Threads: $(Threads.nthreads())")
    println("=" ^ 78)
    println()

    # Create synthetic data
    print("Creating synthetic PLMap (51x51x512)... ")
    plmap_time = @elapsed plmap = make_synthetic_plmap()
    println("done ($(round(plmap_time, digits=2))s)")

    # Extract a single spectrum for 1D benchmarks
    spec = extract_spectrum(plmap, 26, 26)
    single_signal = spec.signal

    println()
    println("Running benchmarks (3 runs each, reporting median)...")
    println("-" ^ 78)

    results = BenchResult[]

    # 1. Cosmic ray detection (full map)
    push!(results, run_bench("detect_cosmic_rays(PLMap, threshold=20)", () -> begin
        detect_cosmic_rays(plmap; threshold=20.0)
    end))

    # 2. Cosmic ray removal (full map)
    cr_result = detect_cosmic_rays(plmap; threshold=20.0)
    println("  (CR detection found $(cr_result.count) cosmic ray voxels in $(cr_result.affected_spectra) spectra)")
    push!(results, run_bench("remove_cosmic_rays(PLMap)", () -> begin
        remove_cosmic_rays(plmap, cr_result)
    end))

    # 3. Peak centers (full map)
    push!(results, run_bench("peak_centers(PLMap)", () -> begin
        peak_centers(plmap)
    end))

    # 4. Baseline correction - arPLS (single spectrum)
    push!(results, run_bench("correct_baseline(spectrum, :arpls)", () -> begin
        correct_baseline(single_signal; method=:arpls)
    end))

    # 5. Savitzky-Golay smoothing (single spectrum)
    push!(results, run_bench("savitzky_golay_smooth(spectrum)", () -> begin
        savitzky_golay_smooth(single_signal; window=11, order=3)
    end))

    # 6. Moving average smoothing (single spectrum)
    push!(results, run_bench("smooth_data(spectrum, window=5)", () -> begin
        smooth_data(single_signal; window=5)
    end))

    # 7. NMF decomposition (3 components, full map)
    push!(results, run_bench("nmf_map(PLMap, 3 components)", () -> begin
        nmf_map(plmap; n_components=3, max_iter=200)
    end))

    # Print results table
    println()
    println("=" ^ 78)
    println("Results")
    println("=" ^ 78)

    max_name_len = maximum(length(r.name) for r in results)
    header = rpad("Operation", max_name_len + 2) * rpad("Median", 14) * rpad("Min", 14) * rpad("Memory", 14) * "Allocs"
    println(header)
    println("-" ^ length(header))

    for r in results
        line = rpad(r.name, max_name_len + 2) *
               rpad(format_time(r.median_time_s), 14) *
               rpad(format_time(r.min_time_s), 14) *
               rpad(format_bytes(r.bytes), 14) *
               string(r.allocs)
        println(line)
    end

    println()
    println("Map size: $(length(plmap.x))x$(length(plmap.y)) spatial, $(length(plmap.pixel)) spectral channels")
    println("Spectrum length: $(length(single_signal)) points")

    return results
end

results = main()
