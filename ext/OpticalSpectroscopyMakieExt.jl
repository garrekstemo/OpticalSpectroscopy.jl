module OpticalSpectroscopyMakieExt

using OpticalSpectroscopy
using OpticalSpectroscopy: AbstractSpectroscopyData, MultiPeakFitResult,
                         xdata, ydata, predict, predict_peak, predict_baseline
using Makie

function Makie.convert_arguments(P::PointBased, d::AbstractSpectroscopyData)
    return Makie.convert_arguments(P, xdata(d), ydata(d))
end

function OpticalSpectroscopy.plot_fit(
    fit::MultiPeakFitResult;
    residuals::Bool = true,
    components::Bool = false,
    baseline::Bool = false,
    xlabel::AbstractString = "x",
    ylabel::AbstractString = "Signal",
    figure = (;),
    axis = (;),
)
    x = xdata(fit)
    y = ydata(fit)
    yhat = predict(fit)

    fig = Figure(; figure...)
    ax_main = Axis(fig[1, 1]; xlabel = xlabel, ylabel = ylabel, axis...)

    scatter!(ax_main, x, y; label = "data", color = :black, markersize = 6)
    lines!(ax_main, x, yhat; label = "fit", color = :crimson, linewidth = 2)

    if components
        for i in eachindex(fit)
            lines!(ax_main, x, predict_peak(fit, i);
                   color = (:crimson, 0.5), linestyle = :dash,
                   linewidth = 1, label = "peak $i")
        end
    end

    if baseline
        lines!(ax_main, x, predict_baseline(fit);
               color = :gray, linestyle = :dot, label = "baseline")
    end

    axislegend(ax_main; position = :rt)

    if residuals
        ax_res = Axis(fig[2, 1]; xlabel = xlabel, ylabel = "residuals")
        scatter!(ax_res, x, y .- yhat; color = :black, markersize = 5)
        hlines!(ax_res, [0.0]; color = :gray, linestyle = :dash)
        linkxaxes!(ax_main, ax_res)
        hidexdecorations!(ax_main; grid = false)
        rowsize!(fig.layout, 2, Relative(0.25))
    end

    return fig
end

end # module
