module TightBindingApproximationMakieExt

using Printf: @sprintf
using Makie: AbstractAxis, Axis, Colorbar, DataAspect, Figure, Heatmap, Label, Lines, Scatter, axislegend, heatmap!, lines!, scatter!, wong_colors, xlims!, ylims!
using QuantumLattices: Assignment, ReciprocalPath, ReciprocalSpace, str
using TightBindingApproximation: BerryCurvature, EnergyBands, FermiSurface, Fukui, Kubo
import Makie: plot, plot!

# 1. EnergyBands - path/scatter plotting (plot! only, plot uses generic Assignment)
function plot!(ax::AbstractAxis, eb::Assignment{<:EnergyBands}; bands=nothing, weightmultiplier=5.0, weightcolors=nothing, weightlabels=nothing, kwargs...)
    ax.title = get(kwargs, :title, str(eb))
    ax.titlesize = get(kwargs, :titlesize, 10)
    rs = eb.data.reciprocalspace
    if length(eb.action.orbitals) > 0
        weightlabels = isnothing(weightlabels) ? [string("Orbital", length(orb)>1 ? "s " : " ", join(orb, ", ")) for orb in eb.action.orbitals] : weightlabels
        return plot!(ax, rs, eb.data.values, eb.data.weights; weightmultiplier=weightmultiplier, weightcolors=weightcolors, weightlabels=weightlabels, kwargs...)
    else
        return plot!(ax, rs, eb.data.values; kwargs...)
    end
end

# 2. BerryCurvature{Fukui{true}} - custom: multi-panel heatmap with band subtitles
function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{true}}}; kwargs...)
    @assert ndims(bc.data.values) == 3 "BerryCurvature{Fukui{true}} requires 3D data (nx, ny, nband)."
    fig = Figure()
    nband = size(bc.data.values, 3)
    nrow = round(Int, sqrt(nband))
    ncol = ceil(Int, nband / nrow)
    rs = bc.action.reciprocalspace
    clims = extrema(bc.data.values)
    subtitles = [string("band ", bc.action.method.bands[i], isnothing(bc.data.chernnumber) ? "" : @sprintf(" (C = %s)", str(bc.data.chernnumber[i]))) for i in 1:nband]
    for i in 1:nband
        ax = Axis(fig[div(i-1, ncol) + 1, (i-1) % ncol + 1], title=subtitles[i], titlesize=get(kwargs, :titlesize, 8))
        plot!(ax, rs, bc.data.values[:, :, i]; colorrange=clims, kwargs...)
    end
    Colorbar(fig[nrow+1, 1:ncol], limits=clims, vertical=false)
    Label(fig[0, :], get(kwargs, :title, str(bc)), tellwidth=false, fontsize=get(kwargs, :titlesize, 10))
    return fig
end

# 3. BerryCurvature{Fukui{false}} - reuse heatmap plotting
function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{false}}}; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    isheatmap = ndims(bc.data.values) == 2
    plot!(ax, bc.action.reciprocalspace, bc.data.values; title=@sprintf("sum of bands %s %s", bc.action.method.bands, isnothing(bc.data.chernnumber) ? "" : @sprintf(" (C = %s)", str(bc.data.chernnumber))), titlesize=get(kwargs, :titlesize, 8), kwargs...)
    isheatmap && Colorbar(fig[1, 2], ax.scene.plots[end])
    Label(fig[0, :], get(kwargs, :title, str(bc)), tellwidth=false, fontsize=get(kwargs, :titlesize, 10))
    return fig
end

# 4. BerryCurvature{Kubo} - reuse heatmap plotting
function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Kubo}}; kwargs...)
    fig = Figure()
    ax = Axis(fig[1, 1])
    isheatmap = ndims(bc.data.values) == 2 && !(bc.action.reciprocalspace isa ReciprocalPath)
    plot!(ax, bc.action.reciprocalspace, bc.data.values; title=@sprintf("bands below %s %s", bc.action.method.μ, isnothing(bc.data.chernnumber) ? "" : @sprintf(" (C = %s)", str(bc.data.chernnumber))), titlesize=get(kwargs, :titlesize, 8), kwargs...)
    isheatmap && Colorbar(fig[1, 2], ax.scene.plots[end])
    Label(fig[0, :], get(kwargs, :title, str(bc)), tellwidth=false, fontsize=get(kwargs, :titlesize, 10))
    return fig
end

# 5. FermiSurface - scatter with weights (plot! only, plot uses generic Assignment)
function plot!(ax::AbstractAxis, fs::Assignment{<:FermiSurface}; fractional=true, weightmultiplier=1.0, weightcolors=nothing, weightlabels=nothing, kwargs...)
    ax.title = get(kwargs, :title, str(fs))
    ax.titlesize = get(kwargs, :titlesize, 10)
    if length(fs.action.orbitals) > 0
        weightlabels = isnothing(weightlabels) ? [string("Orbital", length(orb)>1 ? "s " : " ", join(orb, ", ")) for orb in fs.action.orbitals] : weightlabels
        return plot!(ax, fs.data.values, fs.data.weights; fractional=fractional, weightmultiplier=weightmultiplier, weightcolors=weightcolors, weightlabels=weightlabels, kwargs...)
    else
        return plot!(ax, fs.data.values; fractional=fractional, kwargs...)
    end
end

end # module
