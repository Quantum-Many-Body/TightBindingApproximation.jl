module TightBindingApproximation

using Printf: @sprintf
using TimerOutputs: @timeit
using LinearAlgebra: inv, dot, Hermitian, Diagonal, eigvals, cholesky, Eigen
using QuantumLattices: getcontent, iidtype, rcoord, plain, creation, annihilation, atol, rtol
using QuantumLattices: AbstractLattice, Bonds, Hilbert, Metric, Operators, OIDToTuple, Table, Term, Boundary
using QuantumLattices: Internal, Fock, Phonon, Hopping, Onsite, Pairing, PhononKinetic, PhononPotential, DMPhonon
using QuantumLattices: Engine, Parameters, AbstractGenerator, CompositeGenerator, Entry, Generator, Formulation, Action, Assignment, Algorithm

import LinearAlgebra: eigen, ishermitian
import QuantumLattices: contentnames, statistics, dimension, kind, matrix, update!, prepare!, run!

export TBAKind, AbstractTBA, TBAMatrix, commutator
export TBA, EnergyBands

"""
    TBAKind{K}

The kind of a free quantum lattice system using the tight-binding approximation.
"""
struct TBAKind{K}
    TBAKind(k::Symbol) = new{k}()
end
@inline TBAKind{K}() where K = TBAKind(K)
@inline Base.promote_rule(::Type{TBAKind{K}}, ::Type{TBAKind{K}}) where K = TBAKind{K}
@inline Base.promote_rule(::Type{TBAKind{:TBA}}, ::Type{TBAKind{:BdG}}) = TBAKind{:BdG}
@inline Base.promote_rule(::Type{TBAKind{:BdG}}, ::Type{TBAKind{:TBA}}) = TBAKind{:BdG}

"""
    TBAKind(T::Type{<:Term})

Depending on the kind of a term type, get the corresponding TBA kind.
"""
@inline TBAKind(::Type{T}) where {T<:Term} = error("TBAKind error: not defined for $(kind(T)).")
@inline TBAKind(::Type{T}) where {T<:Union{Hopping, Onsite}} = TBAKind(:TBA)
@inline TBAKind(::Type{T}) where {T<:Union{Pairing, PhononKinetic, PhononPotential, DMPhonon}} = TBAKind(:BdG)
@inline @generated function TBAKind(::Type{TS}) where {TS<:Tuple{Vararg{Term}}}
    exprs = []
    for i = 1:fieldcount(TS)
        push!(exprs, :(typeof(TBAKind(fieldtype(TS, $i)))))
    end
    return Expr(:call, Expr(:call, :reduce, :promote_type, Expr(:tuple, exprs...)))
end

"""
    Metric(::TBAKind, hilbert::Hilbert{<:Fock} -> OIDToTuple
    Metric(::TBAKind, hilbert::Hilbert{<:Phonon}) -> OIDToTuple

Get the oid-to-tuple metric for a free fermionic/bosonic system or a free phononic system.
"""
@inline @generated Metric(::TBAKind{:TBA}, hilbert::Hilbert{<:Fock}) = OIDToTuple(fieldnames(keytype(hilbert))..., :orbital, :spin)
@inline @generated Metric(::TBAKind{:BdG}, hilbert::Hilbert{<:Fock}) = OIDToTuple(:nambu, fieldnames(keytype(hilbert))..., :orbital, :spin)
@inline @generated Metric(::TBAKind, hilbert::Hilbert{<:Phonon}) = OIDToTuple(:tag, fieldnames(keytype(hilbert))..., :dir)

"""
    commutator(k::TBAKind, hilbert::Hilbert{<:Internal}) -> Union{AbstractMatrix, Nothing}

Get the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.
"""
@inline commutator(::TBAKind{:TBA}, ::Hilbert{<:Internal}) = nothing
@inline commutator(::TBAKind{:BdG}, ::Hilbert{<:Fock{:f}}) = nothing
@inline commutator(k::TBAKind{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = Diagonal(kron([1, -1], ones(Int64, dimension(hilbert, k)÷2)))
@inline commutator(k::TBAKind{:BdG}, hilbert::Hilbert{<:Phonon}) = Hermitian(kron([0 -1im; 1im 0], Diagonal(ones(Int, dimension(hilbert, k)÷2))))
@inline dimension(hilbert::Hilbert, ::TBAKind{:BdG}) = sum(dimension, values(hilbert))

"""
    AbstractTBA{K, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine

Abstract type for free quantum lattice systems using the tight-binding approximation.
"""
abstract type AbstractTBA{K, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine end
@inline contentnames(::Type{<:AbstractTBA}) = (:H, :commutator)
@inline kind(tba::AbstractTBA) = kind(typeof(tba))
@inline kind(::Type{<:AbstractTBA{K}}) where K = K
@inline Base.valtype(::Type{<:AbstractTBA{K, H} where K}) where {H<:AbstractGenerator} = valtype(eltype(H))
@inline Base.valtype(tba::AbstractTBA, ::Nothing) = valtype(tba)
@inline Base.valtype(tba::AbstractTBA, k) = promote_type(valtype(tba), Complex{Int})
@inline statistics(tba::AbstractTBA) = statistics(typeof(tba))
@inline statistics(::Type{<:AbstractTBA{K, H} where K}) where {H<:CompositeGenerator{<:Entry{<:Operators}}} = statistics(eltype(eltype(H)))
@inline dimension(tba::AbstractTBA{K, <:CompositeGenerator}) where K = length(getcontent(getcontent(tba, :H), :table))
@inline update!(tba::AbstractTBA; k=nothing, kwargs...) = ((length(kwargs)>0 && update!(getcontent(tba, :H); kwargs...)); tba)
@inline Parameters(tba::AbstractTBA) = Parameters(getcontent(tba, :H))

"""
    TBAMatrix{T, H<:AbstractMatrix{T}, G<:Union{AbstractMatrix, Nothing}} <: AbstractMatrix{T}

Matrix representation of a free quantum lattice system using the tight-binding approximation.
"""
struct TBAMatrix{T, H<:AbstractMatrix{T}, G<:Union{AbstractMatrix, Nothing}} <: AbstractMatrix{T}
    H::H
    commutator::G
    function TBAMatrix(H::AbstractMatrix, commutator::Union{AbstractMatrix, Nothing})
        new{eltype(H), typeof(H), typeof(commutator)}(H, commutator)
    end
end
@inline Base.size(m::TBAMatrix) = size(m.H)
@inline Base.getindex(m::TBAMatrix, i::Integer, j::Integer) = m.H[i, j]
@inline ishermitian(m::TBAMatrix) = ishermitian(typeof(m))
@inline ishermitian(::Type{<:TBAMatrix}) = true

"""
    matrix(tba::AbstractTBA; k=nothing, kwargs...) -> TBAMatrix

Get the matrix representation of a free quantum lattice system.
"""
function matrix(tba::AbstractTBA{TBAKind(:TBA)}; k=nothing, kwargs...)
    H = getcontent(tba, :H)
    table = getcontent(H, :table)
    result = zeros(valtype(tba, k), dimension(tba), dimension(tba))
    for op in H
        seq₁, seq₂ = table[op[1].index'], table[op[2].index]
        phase = isnothing(k) ? one(valtype(tba, k)) : convert(valtype(tba, k), exp(-1im*dot(k, rcoord(op))))
        result[seq₁, seq₂] += op.value*phase
    end
    return TBAMatrix(Hermitian(result), getcontent(tba, :commutator))
end
function matrix(tba::AbstractTBA{TBAKind(:BdG)}; k=nothing, kwargs...)
    H = getcontent(tba, :H)
    table = getcontent(H, :table)
    result = zeros(valtype(tba, k), dimension(tba), dimension(tba))
    for op in H
        seq₁, seq₂ = table[op[1].index'], table[op[2].index]
        phase = isnothing(k) ? one(valtype(tba, k)) : convert(valtype(tba, k), exp(-1im*dot(k, rcoord(op))))
        result[seq₁, seq₂] += op.value*phase
        if op[1].index.iid.nambu==creation && op[2].index.iid.nambu==annihilation
            seq₁, seq₂ = table[op[1].index], table[op[2].index']
            sign = statistics(tba)==:f ? -1 : +1
            result[seq₁, seq₂] += sign*op.value*phase'
        end
    end
    return TBAMatrix(Hermitian(result), getcontent(tba, :commutator))
end
@inline function matrix(tba::AbstractTBA{TBAKind(:Analytical)}; kwargs...)
    return TBAMatrix(Hermitian(getcontent(tba, :H)(; kwargs...)), getcontent(tba, :commutator))
end

"""
    eigen(m::TBAMatrix) -> Eigen

Solve the eigen problem of a free quantum lattice system.
"""
@inline eigen(m::TBAMatrix{T, H, Nothing}) where {T, H<:AbstractMatrix{T}} = eigen(m.H)
function eigen(m::TBAMatrix{T, H, G}) where {T, H<:AbstractMatrix{T}, G<:AbstractMatrix}
    W = cholesky(m.H)
    K = eigen(Hermitian(W.U*m.commutator*W.L))
    @assert length(K.values)%2==0 "eigen error: wrong dimension of matrix."
    for i = 1:(length(K.values)÷2)
        K.values[i] = -K.values[i]
    end
    V = inv(W.U)*K.vectors*sqrt(Diagonal(K.values))
    return Eigen(K.values, V)
end

"""
    TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}

The usual tight binding approximation for quantum lattice systems.
"""
struct TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}
    lattice::L
    H::H
    commutator::G
    function TBA{K}(lattice::AbstractLattice, H::AbstractGenerator, commutator::Union{AbstractMatrix, Nothing}) where K
        @assert isa(K, TBAKind) "TBA error: wrong kind."
        if !isnothing(commutator)
            values = eigvals(commutator)
            num₁ = count(isapprox(+1, atol=atol, rtol=rtol), values)
            num₂ = count(isapprox(-1, atol=atol, rtol=rtol), values)
            @assert num₁==num₂==length(values)//2 "TBA error: unsupported input commutator."
        end
        new{K, typeof(lattice), typeof(H), typeof(commutator)}(lattice, H, commutator)
    end
end
@inline contentnames(::Type{<:TBA}) = (:lattice, :H, :commutator)

"""
    TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; boundary::Boundary=plain)

Construct a tight-binding quantum lattice system.
"""
@inline function TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; boundary::Boundary=plain)
    tbakind = TBAKind(typeof(terms))
    table = Table(hilbert, Metric(tbakind, hilbert))
    commt = commutator(tbakind, hilbert)
    return TBA{tbakind}(lattice, Generator(terms, Bonds(lattice), hilbert; half=false, table=table, boundary=boundary), commt)
end

"""
    TBA(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing)

Construct a tight-binding quantum lattice system by providing the analytical expressions of the Hamiltonian.
"""
@inline function TBA(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing)
    return TBA{TBAKind(:Analytical)}(lattice, Formulation(hamiltonian, parameters), commt)
end

"""
    EnergyBands{P} <: Action

Energy bands by tight-binding-approximation for quantum lattice systems.
"""
struct EnergyBands{P} <: Action
    path::P
end
@inline prepare!(eb::EnergyBands, tba::AbstractTBA) = (zeros(Float64, length(eb.path)), zeros(Float64, length(eb.path), dimension(tba)))
@inline Base.nameof(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands}) = @sprintf "%s_%s" repr(tba, ∉(keys(eb.action.path))) eb.id
function run!(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands})
    for (i, params) in enumerate(eb.action.path)
        eb.data[1][i] = length(params)==1 && isa(first(params), Number) ? first(params) : i-1
        update!(tba; params...)
        @timeit tba.timer "matrix" (m = matrix(tba.engine; params...))
        @timeit tba.timer "eigen" (eb.data[2][i, :] = eigen(m).values)
    end
end

end # module
