module Fitting

using LinearAlgebra: eigvals
using Optim: LBFGS, Options, optimize
using QuantumLattices: Algorithm, Parameters, matrix, update!
using ..TightBindingApproximation: TBA

export SampleNode, deviation, optimize!

"""
    SampleNode(reciprocals::AbstractVector{<:AbstractVector{<:Number}}, position::Vector{<:Number}, bands::AbstractVector{Int}, values::Vector{<:Number}, ratio::Number)
    SampleNode(reciprocals::AbstractVector{<:AbstractVector{<:Number}}, position::Vector{<:Number}, bands::AbstractVector{Int}, values::Vector{<:Number}, ratios::Vector{<:Number}=ones(length(bands)))

A sample node of a momentum-eigenvalues pair.
"""
struct SampleNode
    k::Vector{Float64}
    bands::Vector{Int}
    values::Vector{Float64}
    ratios::Vector{Float64}
end
@inline function SampleNode(reciprocals::AbstractVector{<:AbstractVector{<:Number}}, position::Vector{<:Number}, bands::AbstractVector{Int}, values::Vector{<:Number}, ratio::Number)
    return SampleNode(reciprocals, position, bands, values, ones(length(bands))*ratio)
end
function SampleNode(reciprocals::AbstractVector{<:AbstractVector{<:Number}}, position::Vector{<:Number}, bands::AbstractVector{Int}, values::Vector{<:Number}, ratios::Vector{<:Number}=ones(length(bands)))
    @assert length(reciprocals)==length(position) "SampleNode error: mismatched reciprocals and position."
    @assert length(bands)==length(values)==length(ratios) "SampleNode error: mismatched bands, values and ratios."
    return SampleNode(mapreduce(*, +, reciprocals, position), collect(bands), values, ratios)
end

"""
    deviation(tba::Union{TBA, Algorithm{<:TBA}}, samplenode::SampleNode) -> Float64
    deviation(tba::Union{TBA, Algorithm{<:TBA}}, samplesets::Vector{SampleNode}) -> Float64

Get the deviation of the eigenvalues between the sample points and model points.
"""
function deviation(tba::Union{TBA, Algorithm{<:TBA}}, samplenode::SampleNode)
    diff = eigvals(tba, samplenode.k)[samplenode.bands] .- samplenode.values
    return real(sum(conj(diff) .* diff .* samplenode.ratios))
end
function deviation(tba::Union{TBA, Algorithm{<:TBA}}, samplesets::Vector{SampleNode})
    result = 0.0
    for samplenode in samplesets
        result += deviation(tba, samplenode)
    end
    return result
end

"""
    optimize!(
        tba::Union{TBA, Algorithm{<:TBA}}, samplesets::Vector{SampleNode}, variables=keys(Parameters(tba));
        verbose=false, method=LBFGS(), options=Options()
    ) -> Tuple{typeof(tba), Optim.MultivariateOptimizationResults}

Optimize the parameters of a tight binding system whose names are specified by `variables` so that the total deviations of the eigenvalues between the model points and sample points minimize.
"""
function optimize!(tba::Union{TBA, Algorithm{<:TBA}}, samplesets::Vector{SampleNode}, variables=keys(Parameters(tba)); verbose=false, method=LBFGS(), options=Options())
    v₀ = collect(real(getfield(Parameters(tba), name)) for name in variables)
    function diff(v::Vector)
        parameters = Parameters{variables}(v...)
        update!(tba; parameters...)
        verbose && println(parameters)
        return deviation(tba, samplesets)
    end
    op = optimize(diff, v₀, method, options)
    parameters = Parameters{variables}(op.minimizer...)
    update!(tba; parameters...)
    return tba, op
end

end
