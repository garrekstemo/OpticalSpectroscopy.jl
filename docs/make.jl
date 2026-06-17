using Documenter
using OpticalSpectroscopy

makedocs(
    sitename = "OpticalSpectroscopy.jl",
    modules = [OpticalSpectroscopy],
    checkdocs = :exports,
    warnonly = [:cross_references],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Chirp Correction (TA)" => "tutorials/chirp_correction.md",
            "Peak Fitting (FTIR)" => "tutorials/ftir_peak_fitting.md",
            "Peak Fitting (Raman)" => "tutorials/raman_peak_fitting.md",
            "PL / Raman Map Analysis" => "tutorials/plmap_analysis.md",
        ],
        "How-To Guides" => [
            "Tune Peak Detection Sensitivity" => "howto/peak_detection_sensitivity.md",
            "Fit Overlapping Peaks" => "howto/overlapping_peaks.md",
            "Choose a Peak Model" => "howto/choose_peak_model.md",
            "Fit with Baseline Correction" => "howto/fit_with_baseline.md",
            "Compare Fits Across Samples" => "howto/compare_fits.md",
            "Provide Manual Initial Guesses" => "howto/manual_initial_guesses.md",
        ],
        "Reference" => [
            "Peak Detection" => "reference/peak_detection.md",
            "Peak Fitting" => "reference/peak_fitting.md",
            "Baseline Correction" => "reference/baseline.md",
            "Preprocessing" => "reference/preprocessing.md",
            "PL / Raman Mapping" => "reference/plmap.md",
            "Cavity & Polaritons" => "reference/cavity.md",
            "Additional API" => "api.md",
        ],
        "Explanation" => [
            "Fitting Statistics" => "explanation/fitting_statistics.md",
            "Baseline Algorithms" => "explanation/baseline_algorithms.md",
        ],
    ],
)

deploydocs(
    repo = "github.com/garrekstemo/OpticalSpectroscopy.jl.git",
    push_preview = true,
)
