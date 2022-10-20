module TightBindingApproximation

using LinearAlgebra: Diagonal, Eigen, Hermitian, cholesky, dot, inv, norm
using Optim: LBFGS, optimize
using Printf: @sprintf
using QuantumLattices: expand
using QuantumLattices: plain, Boundary, CompositeIndex, Hilbert, Index, Internal, Metric, Table, Term, statistics
using QuantumLattices: Action, Algorithm, AnalyticalExpression, Assignment, CompositeGenerator, Entry, Frontend, OperatorGenerator, Parameters, RepresentationGenerator
using QuantumLattices: periods
using QuantumLattices: ID, MatrixRepresentation, Operator, Operators, OperatorUnitToTuple, iidtype
using QuantumLattices: annihilation, creation, Elastic, FID, Fock, Hooke, Hopping, Kinetic, Onsite, Pairing, Phonon, PID
using QuantumLattices: AbstractLattice, BrillouinZone, Neighbors, ReciprocalPath, ReciprocalZone, bonds, icoordinate, rcoordinate
using QuantumLattices: atol, rtol, decimaltostr, getcontent
using RecipesBase: RecipesBase, @recipe, @series, @layout
using TimerOutputs: @timeit

import LinearAlgebra: eigen, eigvals, eigvecs, ishermitian
import QuantumLattices: add!, dimension, kind, matrix, update!
import QuantumLattices: initialize, run!
import QuantumLattices: contentnames

export Bosonic, Fermionic, Phononic, TBAKind
export AbstractTBA, TBA, TBAMatrix, TBAMatrixRepresentation, commutator
export BerryCurvature, EnergyBands, InelasticNeutronScatteringSpectra
export SampleNode, deviation, optimize!

"""
    TBAKind{K}

The kind of a free quantum lattice system using the tight-binding approximation.
"""
abstract type TBAKind{K} end

"""
    Phononic <: TBAKind{:BdG}

Phononic quantum lattice system.
"""
struct Phononic <: TBAKind{:BdG} end

"""
    Fermionic{K} <: TBAKind{K}

Fermionic quantum lattice system.
"""
struct Fermionic{K} <: TBAKind{K} end
@inline Fermionic(k::Symbol) = Fermionic{k}()
@inline Base.promote_rule(::Type{Fermionic{K}}, ::Type{Fermionic{K}}) where K = Fermionic{K}
@inline Base.promote_rule(::Type{Fermionic{:TBA}}, ::Type{Fermionic{:BdG}}) = Fermionic{:BdG}
@inline Base.promote_rule(::Type{Fermionic{:BdG}}, ::Type{Fermionic{:TBA}}) = Fermionic{:BdG}

"""
    Bosonic{K} <: TBAKind{K}

Bosonic quantum lattice system.
"""
struct Bosonic{K} <: TBAKind{K} end
@inline Bosonic(k::Symbol) = Bosonic{k}()
@inline Base.promote_rule(::Type{Bosonic{K}}, ::Type{Bosonic{K}}) where K = Bosonic{K}
@inline Base.promote_rule(::Type{Bosonic{:TBA}}, ::Type{Bosonic{:BdG}}) = Bosonic{:BdG}
@inline Base.promote_rule(::Type{Bosonic{:BdG}}, ::Type{Bosonic{:TBA}}) = Bosonic{:BdG}

"""
    TBAKind(T::Type{<:Term}, I::Type{<:Internal})

Depending on the kind of a `Term` type and an `Internal` type, get the corresponding TBA kind.
"""
@inline TBAKind(::Type{T}, ::Type{<:Internal}) where {T<:Term} = error("TBAKind error: not defined behavior for the term with termkind=$(kind(T)).")
@inline TBAKind(::Type{T}, ::Type{<:Fock{:f}}) where {T<:Union{Hopping, Onsite}} = Fermionic(:TBA)
@inline TBAKind(::Type{T}, ::Type{<:Fock{:f}}) where {T<:Pairing} = Fermionic(:BdG)
@inline TBAKind(::Type{T}, ::Type{<:Fock{:b}}) where {T<:Union{Hopping, Onsite}} = Bosonic(:TBA)
@inline TBAKind(::Type{T}, ::Type{<:Fock{:b}}) where {T<:Pairing} = Bosonic(:BdG)
@inline TBAKind(::Type{T}, ::Type{<:Phonon}) where {T<:Union{Kinetic, Hooke, Elastic}} = Phononic()
@inline @generated function TBAKind(::Type{TS}, ::Type{I}) where {TS<:Tuple{Vararg{Term}}, I<:Internal}
    exprs = []
    for i = 1:fieldcount(TS)
        push!(exprs, :(typeof(TBAKind(fieldtype(TS, $i), I))))
    end
    return Expr(:call, Expr(:call, :reduce, :promote_type, Expr(:tuple, exprs...)))
end

"""
    Metric(::Fermionic, hilbert::Hilbert{<:Fock{:f}} -> OperatorUnitToTuple
    Metric(::Bosonic, hilbert::Hilbert{<:Fock{:b}} -> OperatorUnitToTuple
    Metric(::Phononic, hilbert::Hilbert{<:Phonon}) -> OperatorUnitToTuple

Get the index-to-tuple metric for a free fermionic/bosonic/phononic system.
"""
@inline Metric(::TBAKind, ::Hilbert) = error("Metric error: not defined behavior.")
@inline @generated Metric(::Fermionic{:TBA}, hilbert::Hilbert{<:Fock{:f}}) = OperatorUnitToTuple(:site, :orbital, :spin)
@inline @generated Metric(::Bosonic{:TBA}, hilbert::Hilbert{<:Fock{:b}}) = OperatorUnitToTuple(:site, :orbital, :spin)
@inline @generated Metric(::Fermionic{:BdG}, hilbert::Hilbert{<:Fock{:f}}) = OperatorUnitToTuple(:nambu, :site, :orbital, :spin)
@inline @generated Metric(::Bosonic{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = OperatorUnitToTuple(:nambu, :site, :orbital, :spin)
@inline @generated Metric(::Phononic, hilbert::Hilbert{<:Phonon}) = OperatorUnitToTuple(:tag, :site, :direction)

"""
    commutator(k::TBAKind, hilbert::Hilbert{<:Internal}) -> Union{AbstractMatrix, Nothing}

Get the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.
"""
@inline commutator(::TBAKind, ::Hilbert{<:Internal}) = error("commutator error: not defined behavior.")
@inline commutator(::Fermionic, ::Hilbert{<:Fock{:f}}) = nothing
@inline commutator(::Bosonic{:TBA}, ::Hilbert{<:Fock{:b}}) = nothing
@inline commutator(::Bosonic{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = Diagonal(kron([1, -1], ones(Int64, sum(dimension, values(hilbert))÷2)))
@inline commutator(::Phononic, hilbert::Hilbert{<:Phonon}) = Hermitian(kron([0 -1im; 1im 0], Diagonal(ones(Int, sum(dimension, values(hilbert))÷2))))

"""
    TBAMatrix{K<:TBAKind, G<:Union{AbstractMatrix, Nothing}, H<:AbstractMatrix} <: AbstractMatrix{Number}

Matrix representation of a free quantum lattice system using the tight-binding approximation.
"""
struct TBAMatrix{K<:TBAKind, G<:Union{AbstractMatrix, Nothing}, H<:AbstractMatrix} <: AbstractMatrix{Number}
    H::H
    commutator::G
    function TBAMatrix{K}(H::AbstractMatrix, commutator::Union{AbstractMatrix, Nothing}) where {K<:TBAKind}
        new{K, typeof(commutator), typeof(H)}(H, commutator)
    end
end
@inline Base.eltype(::Type{<:TBAMatrix{<:TBAKind, <:Union{AbstractMatrix, Nothing}, H}}) where {H<:AbstractMatrix} = eltype(H)
@inline Base.size(m::TBAMatrix) = size(m.H)
@inline Base.getindex(m::TBAMatrix, i::Integer, j::Integer) = m.H[i, j]
@inline ishermitian(m::TBAMatrix) = ishermitian(typeof(m))
@inline ishermitian(::Type{<:TBAMatrix}) = true

"""
    eigen(m::TBAMatrix) -> Eigen

Solve the eigen problem of a free quantum lattice system.
"""
@inline eigen(m::TBAMatrix{<:TBAKind, Nothing}) = eigen(m.H)
function eigen(m::TBAMatrix{<:TBAKind, <:AbstractMatrix})
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
    eigvals(m::TBAMatrix) -> Vector

Get the eigen values of a free quantum lattice system.
"""
@inline eigvals(m::TBAMatrix{<:TBAKind, Nothing}) = eigvals(m.H)
function eigvals(m::TBAMatrix{<:TBAKind, <:AbstractMatrix})
    values = eigen(m.H*m.commutator).values
    @assert length(values)%2==0 "eigvals error: wrong dimension of matrix."
    for i = 1:(length(values)÷2)
        values[i] = -values[i]
    end
    return values
end

"""
    eigvecs(m::TBAMatrix) -> Matrix

Get the eigen vectors of a free quantum lattice system.
"""
@inline eigvecs(m::TBAMatrix) = eigen(m).vectors

"""
    TBAMatrixRepresentation{K<:TBAKind, V, T, D} <: MatrixRepresentation

Matrix representation of the Hamiltonian of a tight-binding system.
"""
struct TBAMatrixRepresentation{K<:TBAKind, V, T, D} <: MatrixRepresentation
    k::V
    table::T
    gauge::Symbol
    function TBAMatrixRepresentation{K, D}(k, table, gauge::Symbol=:icoordinate) where {K<:TBAKind, D}
        @assert gauge∈(:rcoordinate, :icoordinate) "TBAMatrixRepresentation error: gauge must be :rcoordinate or :icoordinate."
        return new{K, typeof(k), typeof(table), D}(k, table, gauge)
    end
end
@inline TBAMatrixRepresentation{K, D}(table, gauge::Symbol=:icoordinate) where {K<:TBAKind, D} = TBAMatrixRepresentation{K, D}(nothing, table, gauge)
@inline Base.valtype(::Type{<:TBAMatrixRepresentation{<:TBAKind, V, T, D} where {V, T}}, ::Type{<:Union{Operator, Operators}}) where D = Matrix{D}
@inline Base.zero(mr::TBAMatrixRepresentation, O::Union{Operator, Operators}) = zeros(eltype(valtype(mr, O)), length(mr.table), length(mr.table))
@inline (mr::TBAMatrixRepresentation)(m::Operator; kwargs...) = add!(zero(mr, m), mr, m; kwargs...)

"""
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:TBA}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID}}, 2}}; kwargs...) -> typeof(dest)
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:f}}}, 2}}; kwargs...) -> typeof(dest)
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:b}}}, 2}}; atol=atol/5, kwargs...) -> typeof(dest)
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:PID}}, 2}}; atol=atol/5, kwargs...) -> typeof(dest)

Get the matrix representation of an operator and add it to destination.
"""
function add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:TBA}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID}}, 2}}; kwargs...)
    seq₁, seq₂ = mr.table[m[1].index'], mr.table[m[2].index]
    coordinate = mr.gauge==:rcoordinate ? rcoordinate(m) : icoordinate(m)
    phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(1im*dot(mr.k, coordinate)))
    dest[seq₁, seq₂] += m.value*phase
    return dest
end
@inline add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:f}}}, 2}}; kwargs...) = _add!(dest, mr, m, -1; kwargs..., atol=0)
@inline add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:b}}}, 2}}; atol=atol/5, kwargs...) = _add!(dest, mr, m, +1; atol=atol, kwargs...)
function _add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m, sign; atol, kwargs...)
    seq₁, seq₂ = mr.table[m[1].index'], mr.table[m[2].index]
    coordinate = mr.gauge==:rcoordinate ? rcoordinate(m) : icoordinate(m)
    phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(1im*dot(mr.k, coordinate)))
    seq₁==seq₂ || (atol = 0)
    dest[seq₁, seq₂] += m.value*phase+atol
    if m[1].index.iid.nambu==creation && m[2].index.iid.nambu==annihilation
        seq₁, seq₂ = mr.table[m[1].index], mr.table[m[2].index']
        dest[seq₁, seq₂] += sign*m.value*phase'+atol
    end
    return dest
end
function add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:PID}}, 2}}; atol=atol/5, kwargs...)
    if m[1] == m[2]
        seq = mr.table[m[1].index]
        dest[seq, seq] += 2*m.value+atol
    else
        seq₁, seq₂ = mr.table[m[1].index], mr.table[m[2].index]
        coordinate = mr.gauge==:rcoordinate ? rcoordinate(m) : icoordinate(m)
        phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(1im*dot(mr.k, coordinate)))
        dest[seq₁, seq₂] += m.value*phase
        dest[seq₂, seq₁] += m.value'*phase'
    end
    return dest
end

"""
    AbstractTBA{K<:TBAKind, H<:RepresentationGenerator, G<:Union{Nothing, AbstractMatrix}} <: Frontend

Abstract type for free quantum lattice systems using the tight-binding approximation.
"""
abstract type AbstractTBA{K<:TBAKind, H<:RepresentationGenerator, G<:Union{Nothing, AbstractMatrix}} <: Frontend end
@inline contentnames(::Type{<:AbstractTBA}) = (:H, :commutator)
@inline kind(tba::AbstractTBA) = kind(typeof(tba))
@inline kind(::Type{<:AbstractTBA{K}}) where K = K()
@inline Base.valtype(::Type{<:AbstractTBA{<:TBAKind, H}}) where {H<:RepresentationGenerator} = valtype(eltype(H))
@inline dimension(tba::AbstractTBA{<:TBAKind, <:CompositeGenerator}) = length(getcontent(getcontent(tba, :H), :table))
@inline update!(tba::AbstractTBA; k=nothing, kwargs...) = ((length(kwargs)>0 && update!(getcontent(tba, :H); kwargs...)); tba)
@inline Parameters(tba::AbstractTBA) = Parameters(getcontent(tba, :H))

"""
    TBAMatrixRepresentation(tba::AbstractTBA, k=nothing; gauge::Symbol=:icoordinate)

Construct the matrix representation transformation of a free quantum lattice system using the tight-binding approximation.
"""
@inline function TBAMatrixRepresentation(tba::AbstractTBA, k=nothing; gauge::Symbol=:icoordinate)
    return TBAMatrixRepresentation{typeof(kind(tba)), datatype(valtype(tba), k)}(k, getcontent(getcontent(tba, :H), :table), gauge)
end
@inline datatype(::Type{D}, ::Nothing) where D = D
@inline datatype(::Type{D}, ::Any) where D = promote_type(D, Complex{Int})

"""
    matrix(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; k=nothing, gauge=:icoordinate, kwargs...) -> TBAMatrix

Get the matrix representation of a free quantum lattice system.
"""
@inline function matrix(tba::AbstractTBA; k=nothing, gauge=:icoordinate, kwargs...)
    return TBAMatrix{typeof(kind(tba))}(Hermitian(TBAMatrixRepresentation(tba, k; gauge=gauge)(expand(getcontent(tba, :H)); kwargs...)), getcontent(tba, :commutator))
end
@inline function matrix(tba::AbstractTBA{<:TBAKind, <:AnalyticalExpression}; kwargs...)
    return TBAMatrix{typeof(kind(tba))}(Hermitian(getcontent(tba, :H)(; kwargs...)), getcontent(tba, :commutator))
end
@inline matrix(tba::Algorithm{<:AbstractTBA}; kwargs...) = matrix(tba.frontend; kwargs...)

"""
    eigen(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Eigen

Get the eigen values and eigen vectors of a free quantum lattice system.
"""
@inline eigen(tba::AbstractTBA; kwargs...) = eigen(matrix(tba; kwargs...))
@inline function eigen(tba::Algorithm{<:AbstractTBA}; kwargs...)
    @timeit tba.timer "eigen" begin
        @timeit tba.timer "matrix" (m = matrix(tba; kwargs...))
        @timeit tba.timer "diagonalization" (eigensystem = eigen(m))
    end
    return eigensystem
end

"""
    eigvals(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Vector

Get the eigen values of a free quantum lattice system.
"""
@inline eigvals(tba::AbstractTBA; kwargs...) = eigvals(matrix(tba; kwargs...))
@inline function eigvals(tba::Algorithm{<:AbstractTBA}; kwargs...)
    @timeit tba.timer "eigvals" begin
        @timeit tba.timer "matrix" (m = matrix(tba; kwargs...))
        @timeit tba.timer "values" (eigenvalues = eigvals(m))
    end
    return eigenvalues
end

"""
    eigvecs(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Matrix

Get the eigen vectors of a free quantum lattice system.
"""
@inline eigvecs(tba::AbstractTBA; kwargs...) = eigvecs(matrix(tba; kwargs...))
@inline function eigvecs(tba::Algorithm{<:AbstractTBA}; kwargs...)
    @timeit tba.timer "eigvecs" begin
        @timeit tba.timer "matrix" (m = matrix(tba; kwargs...))
        @timeit tba.timer "vectors" (eigenvectors = eigvecs(m))
    end
    return eigenvectors
end

"""
    TBA{K, L<:AbstractLattice, H<:RepresentationGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}

The usual tight binding approximation for quantum lattice systems.
"""
struct TBA{K, L<:AbstractLattice, H<:RepresentationGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}
    lattice::L
    H::H
    commutator::G
    function TBA{K}(lattice::AbstractLattice, H::RepresentationGenerator, commutator::Union{AbstractMatrix, Nothing}) where {K<:TBAKind}
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
    TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)

Construct a tight-binding quantum lattice system.
"""
@inline function TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)
    tbakind = TBAKind(typeof(terms), valtype(hilbert))
    table = Table(hilbert, Metric(tbakind, hilbert))
    commt = commutator(tbakind, hilbert)
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, terms))
    return TBA{typeof(tbakind)}(lattice, OperatorGenerator(terms, bonds(lattice, neighbors), hilbert; half=false, table=table, boundary=boundary), commt)
end

"""
    TBA{K}(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}

Construct a tight-binding quantum lattice system by providing the analytical expressions of the Hamiltonian.
"""
@inline function TBA{K}(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    return TBA{K}(lattice, AnalyticalExpression(hamiltonian, parameters), commt)
end

"""
    EnergyBands{P, L<:Union{Colon, Vector{Int}}, O} <: Action

Energy bands by tight-binding-approximation for quantum lattice systems.
"""
struct EnergyBands{P, L<:Union{Colon, Vector{Int}}, O} <: Action
    path::P
    levels::L
    options::O
end
@inline EnergyBands(path, levels::Union{Colon, Vector{Int}}=Colon(); options...) = EnergyBands(path, levels, convert(Dict{Symbol, Any}, options))
@inline initialize(eb::EnergyBands{P, Colon}, tba::AbstractTBA) where P = (collect(Float64, 0:(length(eb.path)-1)), zeros(Float64, length(eb.path), dimension(tba)))
@inline initialize(eb::EnergyBands{P, Vector{Int}}, tba::AbstractTBA) where P = (collect(Float64, 0:(length(eb.path)-1)), zeros(Float64, length(eb.path), length(eb.levels)))
@inline Base.nameof(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands}) = @sprintf "%s_%s" repr(tba, ∉(keys(eb.action.path))) eb.id
function run!(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands})
    atol = get(eb.action.options, :atol, 10^-12)
    for (i, params) in enumerate(pairs(eb.action.path))
        update!(tba; params...)
        length(params)==1 && isa(first(params), Number) && (eb.data[1][i] = first(params))
        eigenvalues = eigvals(tba; params..., eb.action.options...)[eb.action.levels]
        @assert norm(imag(eigenvalues))<atol "run! error: imaginary eigen energies at $(params) with the norm of all imaginary parts being $(norm(imag(eigenvalues)))."
        eb.data[2][i, :] = real(eigenvalues)
    end
end

"""
    BerryCurvature{B<:Union{BrillouinZone, ReciprocalZone}, O} <: Action

Berry curvature of energy bands with the spirit of a momentum space discretization method by [Fukui et al, JPSJ 74, 1674 (2005)](https://journals.jps.jp/doi/10.1143/JPSJ.74.1674).
"""
struct BerryCurvature{B<:Union{BrillouinZone, ReciprocalZone}, O} <: Action
    reciprocalspace::B
    levels::Vector{Int}
    options::O
end
@inline BerryCurvature(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, levels::Vector{Int}; options...) = BerryCurvature(reciprocalspace, levels, convert(Dict{Symbol, Any}, options))

# For the Berry curvature and Chern number on the first Brillouin zone
@inline function initialize(bc::BerryCurvature{<:BrillouinZone}, tba::AbstractTBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "initialize error: Berry curvature should be defined for 2d systems."
    N₁, N₂ = periods(eltype(bc.reciprocalspace))
    x = collect(Float64, 0:(N₁-1))/N₁
    y = collect(Float64, 0:(N₂-1))/N₂
    z = zeros(Float64, length(y), length(x), length(bc.levels))
    n = zeros(Float64, length(bc.levels))
    return (x, y, z, n)
end
function eigvecs(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature{<:BrillouinZone}})
    N₁, N₂ = length(bc.data[1]), length(bc.data[2])
    eigenvectors = zeros(ComplexF64, N₁+1, N₂+1, dimension(tba.frontend), length(bc.action.levels))
    for i = 1:(N₁+1), j=1:(N₂+1)
        momentum = expand(eltype(bc.action.reciprocalspace)(i, j), bc.action.reciprocalspace.reciprocals)
        eigenvectors[i, j, :, :] = eigvecs(tba; k=momentum, bc.action.options...)[:, bc.action.levels]
    end
    return eigenvectors
end

# For the Berry curvature on a specific zone in the reciprocal space
@inline function initialize(bc::BerryCurvature{<:ReciprocalZone}, tba::AbstractTBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "initialize error: Berry curvature should be defined for 2d systems."
    x = collect(bc.reciprocalspace.bounds[1])[1:end-1]
    y = collect(bc.reciprocalspace.bounds[2])[1:end-1]
    z = zeros(Float64, length(y), length(x), length(bc.levels))
    return (x, y, z)
end
function eigvecs(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature{<:ReciprocalZone}})
    N₁, N₂ = length(bc.data[1]), length(bc.data[2])
    indices = CartesianIndices((1:(N₂+1), 1:(N₁+1)))
    eigenvectors = zeros(ComplexF64, N₁+1, N₂+1, dimension(tba.frontend), length(bc.action.levels))
    for (index, momentum) in enumerate(bc.action.reciprocalspace)
        j, i = Tuple(indices[index])
        eigenvectors[i, j, :, :] = eigvecs(tba; k=momentum, bc.action.options...)[:, bc.action.levels]
    end
    return eigenvectors
end

# Compute the Berry curvature and optionally, the Chern number
function run!(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature})
    @timeit tba.timer "eigenvectors" eigenvectors = eigvecs(tba, bc)
    g = isnothing(getcontent(tba.frontend, :commutator)) ? Diagonal(ones(Int, dimension(tba.frontend))) : inv(getcontent(tba.frontend, :commutator))
    @timeit tba.timer "Berry curvature" for i = 1:length(bc.data[1]), j = 1:length(bc.data[2])
        vs₁ = eigenvectors[i, j, :, :]
        vs₂ = eigenvectors[i+1, j, :, :]
        vs₃ = eigenvectors[i+1, j+1, :, :]
        vs₄ = eigenvectors[i, j+1, :, :]
        for k = 1:length(bc.action.levels)
            p₁ = vs₁[:, k]'*g*vs₂[:, k]
            p₂ = vs₂[:, k]'*g*vs₃[:, k]
            p₃ = vs₃[:, k]'*g*vs₄[:, k]
            p₄ = vs₄[:, k]'*g*vs₁[:, k]
            bc.data[3][j, i, k] = angle(p₁*p₂*p₃*p₄)
            length(bc.data)==4 && (bc.data[4][k] += bc.data[3][j, i, k]/2pi)
        end
    end
    length(bc.data)==4 && @info (@sprintf "Chern numbers: %s" join([@sprintf "%s(%s)" cn level for (cn, level) in zip(bc.data[4], bc.action.levels)], ", "))
end

# Plot the Berry curvature and optionally, the Chern number
@recipe function plot(pack::Tuple{Algorithm{<:AbstractTBA}, Assignment{<:BerryCurvature}})
    titles = if length(pack[2].data)==4
        [@sprintf("level %s (C = %s)", level, decimaltostr(chn)) for (level, chn) in zip(pack[2].action.levels, pack[2].data[4])]
    else
        [@sprintf("level %s", level) for level in pack[2].action.levels]
    end
    nr = round(Int, sqrt(length(titles)))
    nc = ceil(Int, length(titles)/nr)
    layout := @layout [(nr, nc); b{0.05h}]
    Δ₁ = pack[2].data[1][2]-pack[2].data[1][1]
    Δ₂ = pack[2].data[2][2]-pack[2].data[2][1]
    xlims = (pack[2].data[1][1]-Δ₁, pack[2].data[1][end]+Δ₁)
    ylims = (pack[2].data[2][1]-Δ₂, pack[2].data[2][end]+Δ₂)
    clims = extrema(pack[2].data[3])
    for i = 1:length(titles)
        @series begin
            seriestype := :heatmap
            title := titles[i]
            titlefontsize := 8
            colorbar := false
            aspect_ratio := :equal
            subplot := i
            xlims := xlims
            ylims := ylims
            clims := clims
            xlabel := "k₁"
            ylabel := "k₂"
            pack[2].data[1], pack[2].data[2], pack[2].data[3][:, :, i]
        end
    end
    plot_title --> nameof(pack[1], pack[2])
    plot_titlefontsize --> 10
    seriestype := :heatmap
    colorbar := false
    subplot := nr*nc+1
    yticks := (0:1, ("", ""))
    LinRange(clims..., 100), [0, 1], [LinRange(clims..., 100)'; LinRange(clims..., 100)']
end

"""
    InelasticNeutronScatteringSpectra{P<:ReciprocalPath, E<:AbstractVector, O} <: Action

Inelastic neutron scattering spectra.
"""
struct InelasticNeutronScatteringSpectra{P<:ReciprocalPath, E<:AbstractVector, O} <: Action
    path::P
    energies::E
    options::O
    function InelasticNeutronScatteringSpectra(path::ReciprocalPath, energies::AbstractVector, options)
        @assert keys(path)==(:k,) "InelasticNeutronScatteringSpectra error: the name of the momenta in the path must be :k."
        new{typeof(path), typeof(energies), typeof(options)}(path, energies, options)
    end
end
@inline InelasticNeutronScatteringSpectra(path::ReciprocalPath, energies::AbstractVector; options...) = InelasticNeutronScatteringSpectra(path, energies, options)
@inline function initialize(inss::InelasticNeutronScatteringSpectra, tba::AbstractTBA)
    x = collect(Float64, 0:(length(inss.path)-1))
    y = collect(Float64, inss.energies)
    z = zeros(Float64, length(y), length(x))
    return (x, y, z)
end

# Inelastic neutron scattering spectra for phonons.
function run!(tba::Algorithm{<:AbstractTBA{Phononic}}, inss::Assignment{<:InelasticNeutronScatteringSpectra})
    dim = dimension(tba.frontend)
    σ = get(inss.action.options, :fwhm, 0.1)/2/√(2*log(2))
    check = get(inss.action.options, :check, true)
    sequences = Dict(site=>[tba.frontend.H.table[Index(site, PID('u', Char(Int('x')+i-1)))] for i=1:phonon.ndirection] for (site, phonon) in pairs(tba.frontend.H.hilbert))
    for (i, momentum) in enumerate(inss.action.path)
        eigenvalues, eigenvectors = eigen(tba; k=momentum, inss.action.options...)
        check && @timeit tba.timer "check" check_polarizations(@views(eigenvectors[(dim÷2+1):dim, 1:(dim÷2)]), @views(eigenvectors[(dim÷2+1):dim, dim:-1:(dim÷2+1)]), momentum./pi)
        @timeit tba.timer "spectra" begin
            for j = 1:dim
                factor = 0
                for (site, sequence) in pairs(sequences)
                    factor += dot(momentum, eigenvectors[sequence, j])*exp(1im*dot(momentum, tba.frontend.lattice[site]))
                end
                factor = abs2(factor)/√(2pi)/σ
                for (nₑ, e) in enumerate(inss.action.energies)
                    # instead of the Lorentz broadening of δ function, the convolution with a FWHM Gaussian is used.
                    inss.data[3][nₑ, i] += factor*exp(-(e-eigenvalues[j])^2/2/σ^2)
                end
            end
        end
    end
    inss.data[3][:, :] = get(inss.action.options, :scale, identity).(inss.data[3].+1)
end
@inline function check_polarizations(qs₁::AbstractMatrix, qs₂::AbstractMatrix, momentum)
    inner = mapreduce((e₁, e₂)->norm(conj(e₁)*e₂), +, qs₁, qs₂)/norm(qs₁)/norm(qs₂)
    isapprox(inner, 1; atol=100*atol, rtol=100*rtol) || begin
        @warn("check_polarizations: small inner product $inner at π*$momentum, indication of degeneracy, otherwise inconsistent polarization vectors.")
    end
end

"""
    SampleNode(reciprocals::AbstractVector{<:AbstractVector}, position::Vector, levels::Vector{Int}, values::Vector, ratio::Number)
    SampleNode(reciprocals::AbstractVector{<:AbstractVector}, position::Vector, levels::Vector{Int}, values::Vector, ratios::Vector=ones(length(levels)))

A sample node of a momentum-eigenvalues pair.
"""
struct SampleNode
    k::Vector{Float64}
    levels::Vector{Int64}
    values::Vector{Float64}
    ratios::Vector{Float64}
end
@inline function SampleNode(reciprocals::AbstractVector{<:AbstractVector}, position::Vector, levels::Vector{Int}, values::Vector, ratio::Number)
    return SampleNode(reciprocals, position, levels, values, ones(length(levels))*ratio)
end
function SampleNode(reciprocals::AbstractVector{<:AbstractVector}, position::Vector, levels::Vector{Int}, values::Vector, ratios::Vector=ones(length(levels)))
    @assert length(reciprocals)==length(position) "SampleNode error: mismatched reciprocals and position."
    @assert length(levels)==length(values)==length(ratios) "SampleNode error: mismatched levels, values and ratios."
    return SampleNode(mapreduce(*, +, reciprocals, position), levels, values, ratios)
end

"""
    deviation(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplenode::SampleNode) -> Float64
    deviation(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplesets::Vector{SampleNode}) -> Float64

Get the deviation of the eigenvalues between the sample points and model points.
"""
function deviation(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplenode::SampleNode)
    diff = eigvals(matrix(tba; k=samplenode.k))[samplenode.levels] .- samplenode.values
    return real(sum(conj(diff) .* diff .* samplenode.ratios))
end
function deviation(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplesets::Vector{SampleNode})
    result = 0.0
    for samplenode in samplesets
        result += deviation(tba, samplenode)
    end
    return result
end

"""
    optimize!(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}},
        samplesets::Vector{SampleNode},
        variables=keys(Parameters(tba));
        verbose=false,
        method=LBFGS()
    ) -> Tuple{typeof(tba), Optim.MultivariateOptimizationResults}

Optimize the parameters of a tight binding system whose names are specified by `variables` so that the total deviations of the eigenvalues between the model points and sample points minimize.
"""
function optimize!(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplesets::Vector{SampleNode}, variables=keys(Parameters(tba)); verbose=false, method=LBFGS())
    v₀ = collect(getfield(Parameters(tba), name) for name in variables)
    function diff(v::Vector)
        parameters = Parameters{variables}(v...)
        update!(tba; parameters...)
        verbose && println(parameters)
        return deviation(tba, samplesets)
    end
    op = optimize(diff, v₀, method)
    parameters = Parameters{variables}(op.minimizer...)
    update!(tba; parameters...)
    return tba, op
end

end # module
