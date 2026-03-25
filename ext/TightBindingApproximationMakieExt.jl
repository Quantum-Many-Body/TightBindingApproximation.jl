module TightBindingApproximationMakieExt

using Printf: @sprintf
using QuantumLattices: Assignment, ReciprocalSpace, str
using TightBindingApproximation: BerryCurvature, EnergyBands, FermiSurface, Fukui, Kubo
import Makie

# 1. EnergyBands
function Makie.plot!(ax::Makie.AbstractAxis, eb::Assignment{<:EnergyBands}; bands::Union{Bool, Nothing}=nothing, weightmultiplier=5.0, weightcolors=nothing, weightlabels=nothing, kwargs...)
    ax.title = get(kwargs, :title, str(eb))
    ax.titlesize = get(kwargs, :titlesize, 10)
    if length(eb.action.orbitals) > 0
        isa(bands, Bool) && bands && Makie.plot!(ax, eb.data.reciprocalspace, eb.data.values; kwargs...)
        isnothing(weightlabels) && (weightlabels = [string("Orbital", length(orb)>1 ? "s " : " ", join(orb, ", ")) for orb in eb.action.orbitals])
        return Makie.plot!(ax, eb.data.reciprocalspace, eb.data.values, eb.data.weights; weightmultiplier, weightcolors, weightlabels, kwargs...)
    else
        return Makie.plot!(ax, eb.data.reciprocalspace, eb.data.values; kwargs...)
    end
end

# 2. BerryCurvature{Fukui{true}}
function Makie.plot!(fig::Makie.Figure, bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{true}}}; kwargs...)
    return Makie.plot!(
        fig, bc.action.reciprocalspace, bc.data.values;
        title=str(bc),
        subtitles=[ith_subtitle(bc.data.chernnumber, i) for i = 1:size(bc.data.values, 3)],
        kwargs...
    )
end
@inline ith_subtitle(chernnumber, i::Int) = "band $i (C = $(str(chernnumber[i])))"
@inline ith_subtitle(::Nothing, i::Int) = "band $i"

# 3. BerryCurvature{Fukui{false}}
function Makie.plot!(ax::Makie.AbstractAxis, bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{false}}}; kwargs...)
    title = string(str(bc), "\n", @sprintf("sum of bands %s %s", bc.action.method.bands, isnothing(bc.data.chernnumber) ? "" : @sprintf(" (C = %s)", str(bc.data.chernnumber))))
    return Makie.plot!(ax, bc.action.reciprocalspace, bc.data.values; title, kwargs...)
end

# 4. BerryCurvature{Kubo}
function Makie.plot!(ax::Makie.AbstractAxis, bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Kubo}}; kwargs...)
    title = string(str(bc), "\n", @sprintf("bands below %s %s", bc.action.method.μ, isnothing(bc.data.chernnumber) ? "" : @sprintf(" (C = %s)", str(bc.data.chernnumber))))
    return Makie.plot!(ax, bc.action.reciprocalspace, bc.data.values; title, kwargs...)
end

# 5. FermiSurface
function Makie.plot!(ax::Makie.AbstractAxis, fs::Assignment{<:FermiSurface}; kwargs...)
    title = str(fs)
    if length(fs.action.orbitals) > 0
        weightlabels = [string("Orbital", length(orb)>1 ? "s " : " ", join(orb, ", ")) for orb in fs.action.orbitals]
        return Makie.plot!(ax, fs.data.values, fs.data.weights; title, weightlabels, kwargs...)
    else
        return Makie.plot!(ax, fs.data.values; title, markersize=3, kwargs...)
    end
end

end # module
