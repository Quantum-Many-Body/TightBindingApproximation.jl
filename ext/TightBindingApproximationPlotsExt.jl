module TightBindingApproximationPlotsExt

using Printf: @sprintf
using RecipesBase: @recipe, @series
using QuantumLattices: Assignment, ReciprocalSpace, ReciprocalScatter
using TightBindingApproximation: BerryCurvature, BerryCurvatureData, EnergyBands, FermiSurface, FermiSurfaceData, Fukui, Kubo
using QuantumLattices: label, ReciprocalZone, BrillouinZone, str

# Plot energy bands
@recipe function plot(eb::Assignment{<:EnergyBands}; bands=nothing, weightmultiplier=5.0, weightcolors=nothing, weightlabels=nothing)
    title --> str(eb)
    titlefontsize --> 10
    if length(eb.action.orbitals) > 0
        @series begin
            seriestype := :scatter
            weightmultiplier := weightmultiplier
            weightcolors := weightcolors
            weightlabels := isnothing(weightlabels) ? [string("Orbital", length(orbitals)>1 ? "s " : " ", join(orbitals, ", ")) for orbitals in eb.action.orbitals] : weightlabels
            eb.data.reciprocalspace, eb.data.values, eb.data.weights
        end
    end
    isnothing(bands) && (bands = length(eb.action.orbitals)==0)
    if bands
        label --> ""
        seriestype := :path
        eb.data.reciprocalspace, eb.data.values
    end
end

# Plot BerryCurvature (Fukui{true})
@recipe function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{true}}})
    plot_title --> str(bc)
    plot_titlefontsize --> 10
    subtitles --> [@sprintf("band %s %s", band, isnothing(bc.data.chernnumber) ? "" : @sprintf("(C = %s)", str(bc.data.chernnumber[i]))) for (i, band) in enumerate(bc.action.method.bands)]
    subtitlefontsize --> 8
    bc.data.reciprocalspace, bc.data.values
end

# Plot BerryCurvature (Fukui{false})
@recipe function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{false}}})
    plot_title --> str(bc)
    plot_titlefontsize --> 10
    layout := (1, 1)
    subplot := 1
    title --> @sprintf("sum of bands %s %s", bc.action.method.bands, isnothing(bc.data.chernnumber) ? "" : @sprintf("(C = %s)", str(bc.data.chernnumber)))
    plot_title --> str(bc, "\n", info)
    titlefontsize --> 8
    bc.data.reciprocalspace, bc.data.values
end

# Plot BerryCurvature (Kubo)
@recipe function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Kubo}})
    plot_title --> str(bc)
    plot_titlefontsize --> 10
    layout := (1, 1)
    subplot := 1
    title --> @sprintf("bands below %s %s", bc.action.method.μ, isnothing(bc.data.chernnumber) ? "" : @sprintf("(C = %s)", str(bc.data.chernnumber)))
    titlefontsize --> 8
    bc.data.reciprocalspace, bc.data.values
end

# Plot FermiSurface
@recipe function plot(fs::Assignment{<:FermiSurface}; fractional=true, weightmultiplier=1.0, weightcolors=nothing, weightlabels=nothing)
    title --> str(fs)
    titlefontsize --> 10
    seriestype := :scatter
    fractional := fractional
    if length(fs.action.orbitals) > 0
        weightmultiplier := weightmultiplier
        weightcolors := weightcolors
        weightlabels := isnothing(weightlabels) ? [string("Orbital", length(orbitals)>1 ? "s " : " ", join(orbitals, ", ")) for orbitals in fs.action.orbitals] : weightlabels
        fs.data.values, fs.data.weights
    else
        autolims := false
        markersize --> 1
        fs.data.values
    end
end

end # module
