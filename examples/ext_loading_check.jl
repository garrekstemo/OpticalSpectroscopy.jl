using OpticalSpectroscopy

ext = Base.get_extension(OpticalSpectroscopy, :OpticalSpectroscopyMakieExt)
@assert ext === nothing "ext should not be loaded yet"
@assert !hasmethod(plot_fit, Tuple{MultiPeakFitResult}) "plot_fit should have no methods yet"
println("✓ Before `using GLMakie`: ext is nothing, plot_fit has no methods")

using GLMakie

ext = Base.get_extension(OpticalSpectroscopy, :OpticalSpectroscopyMakieExt)
@assert ext !== nothing "ext should be loaded now"
@assert hasmethod(plot_fit, Tuple{MultiPeakFitResult}) "plot_fit should have a method"
println("✓ After  `using GLMakie`: ext = $ext, plot_fit(::MultiPeakFitResult) defined")
