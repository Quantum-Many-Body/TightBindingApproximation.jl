module TightBindingApproximation

using LinearAlgebra: Diagonal, Eigen, Hermitian, cholesky, dot, inv, norm
using Optim: LBFGS, Options, optimize
using Printf: @sprintf
using QuantumLattices: expand
using QuantumLattices: plain, Boundary, CompositeIndex, Hilbert, Index, Internal, Metric, Table, Term, statistics
using QuantumLattices: Action, Algorithm, AnalyticalExpression, Assignment, CompositeGenerator, Entry, Frontend, OperatorGenerator, Parameters, RepresentationGenerator
using QuantumLattices: periods
using QuantumLattices: ID, MatrixRepresentation, Operator, Operators, OperatorUnitToTuple, iidtype
using QuantumLattices: Elastic, FID, Fock, Hooke, Hopping, Kinetic, Onsite, Pairing, Phonon, PID, isannihilation, iscreation
using QuantumLattices: AbstractLattice, BrillouinZone, Neighbors, ReciprocalPath, ReciprocalSpace, ReciprocalZone, bonds, icoordinate, rcoordinate, shrink
using QuantumLattices: atol, rtol, decimaltostr, getcontent, shape
using RecipesBase: RecipesBase, @recipe, @series, @layout
using TimerOutputs: TimerOutput, @timeit_debug

import LinearAlgebra: eigen, eigvals, eigvecs, ishermitian
import QuantumLattices: add!, dimension, kind, matrix, update!
import QuantumLattices: initialize, run!
import QuantumLattices: contentnames

export Bosonic, Fermionic, Phononic, TBAKind
export AbstractTBA, TBA, TBAMatrix, TBAMatrixRepresentation, commutator
export BerryCurvature, DensityOfStates, EnergyBands, FermiSurface, InelasticNeutronScatteringSpectra
export SampleNode, deviation, optimize!

const tbatimer = TimerOutput()

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
@inline commutator(::Bosonic{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = Diagonal(kron([1, -1], ones(Int64, sum(length, values(hilbert))÷2)))
@inline commutator(::Phononic, hilbert::Hilbert{<:Phonon}) = Hermitian(kron([0 -1im; 1im 0], Diagonal(ones(Int, sum(length, values(hilbert))÷2))))

"""
    TBAMatrix{K<:TBAKind, G<:Union{AbstractMatrix, Nothing}, T, H<:AbstractMatrix{T}} <: AbstractMatrix{T}

Matrix representation of a free quantum lattice system using the tight-binding approximation.
"""
struct TBAMatrix{K<:TBAKind, G<:Union{AbstractMatrix, Nothing}, T, H<:AbstractMatrix{T}} <: AbstractMatrix{T}
    H::H
    commutator::G
    function TBAMatrix{K}(H::AbstractMatrix, commutator::Union{AbstractMatrix, Nothing}) where {K<:TBAKind}
        new{K, typeof(commutator), eltype(H), typeof(H)}(H, commutator)
    end
end
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
    if iscreation(m[1]) && isannihilation(m[2])
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
@inline dimension(tba::AbstractTBA{<:TBAKind, <:AnalyticalExpression}) = dimension(getcontent(tba, :H).expression)
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
@inline matrix(tba::Algorithm{<:AbstractTBA}; kwargs...) = matrix(tba.frontend; kwargs...)
@inline function matrix(tba::AbstractTBA; k=nothing, gauge=:icoordinate, kwargs...)
    return TBAMatrix{typeof(kind(tba))}(Hermitian(TBAMatrixRepresentation(tba, k; gauge=gauge)(expand(getcontent(tba, :H)); kwargs...)), getcontent(tba, :commutator))
end
@inline function matrix(tba::AbstractTBA{<:TBAKind, <:AnalyticalExpression}; kwargs...)
    return TBAMatrix{typeof(kind(tba))}(Hermitian(getcontent(tba, :H)(; kwargs...)), getcontent(tba, :commutator))
end

"""
    eigen(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Eigen
    eigen(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, reciprocalspace::ReciprocalSpace; kwargs...) -> Tuple{Vector{<:Vector{<:Number}}, Vector{<:Matrix{<:Number}}}

Get the eigen values and eigen vectors of a free quantum lattice system.
"""
@inline eigen(tba::Algorithm{<:AbstractTBA}; kwargs...) = eigen(tba.frontend; timer=tba.timer, kwargs...)
@inline function eigen(tba::AbstractTBA; timer::TimerOutput=tbatimer, kwargs...)
    @timeit_debug timer "eigen" begin
        @timeit_debug timer "matrix" (m = matrix(tba; kwargs...))
        @timeit_debug timer "diagonalization" (eigensystem = eigen(m))
    end
    return eigensystem
end
@inline eigen(tba::Algorithm{<:AbstractTBA}, reciprocalspace::ReciprocalSpace; kwargs...) = eigen(tba.frontend, reciprocalspace; timer=tba.timer, kwargs...)
function eigen(tba::AbstractTBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, kwargs...)
    datatype = eltype(eltype(reciprocalspace))
    values, vectors = Vector{datatype}[], Matrix{promote_type(datatype, Complex{Int})}[]
    for momentum in reciprocalspace
        eigensystem = eigen(tba; k=momentum, timer=timer, kwargs...)
        push!(values, eigensystem.values)
        push!(vectors, eigensystem.vectors)
    end
    return values, vectors
end

"""
    eigvals(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Vector
    eigvals(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, reciprocalspace::ReciprocalSpace; kwargs...) -> Vector{<:Vector}

Get the eigen values of a free quantum lattice system.
"""
@inline eigvals(tba::Algorithm{<:AbstractTBA}; kwargs...) = eigvals(tba.frontend; timer=tba.timer, kwargs...)
@inline function eigvals(tba::AbstractTBA; timer::TimerOutput=tbatimer, kwargs...)
    @timeit_debug timer "eigvals" begin
        @timeit_debug timer "matrix" (m = matrix(tba; kwargs...))
        @timeit_debug timer "values" (eigenvalues = eigvals(m))
    end
    return eigenvalues
end
@inline eigvals(tba::Algorithm{<:AbstractTBA}, reciprocalspace::ReciprocalSpace; kwargs...) = eigvals(tba.frontend, reciprocalspace; timer=tba.timer, kwargs...)
@inline eigvals(tba::AbstractTBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, kwargs...) = [eigvals(tba; k=momentum, timer=timer, kwargs...) for momentum in reciprocalspace]

"""
    eigvecs(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Matrix
    eigvecs(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, reciprocalspace::ReciprocalSpace; kwargs...) -> Vector{<:Matrix}

Get the eigen vectors of a free quantum lattice system.
"""
@inline eigvecs(tba::Algorithm{<:AbstractTBA}; kwargs...) = eigvecs(tba.frontend; timer=tba.timer, kwargs...)
@inline function eigvecs(tba::AbstractTBA; timer::TimerOutput=tbatimer, kwargs...)
    @timeit_debug timer "eigvecs" begin
        @timeit_debug timer "matrix" (m = matrix(tba; kwargs...))
        @timeit_debug timer "vectors" (eigenvectors = eigvecs(m))
    end
    return eigenvectors
end
@inline eigvecs(tba::Algorithm{<:AbstractTBA}, reciprocalspace::ReciprocalSpace; kwargs...) = eigvecs(tba.frontend, reciprocalspace; timer=tba.timer, kwargs...)
@inline eigvecs(tba::AbstractTBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, kwargs...) = [eigvecs(tba; k=momentum, timer=timer, kwargs...) for momentum in reciprocalspace]

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
    basespace::P
    levels::L
    options::O
end
@inline EnergyBands(basespace, levels::Union{Colon, Vector{Int}}=Colon(); options...) = EnergyBands(basespace, levels, options)
@inline initialize(eb::EnergyBands{P, Colon}, tba::AbstractTBA) where P = (eb.basespace, zeros(Float64, length(eb.basespace), dimension(tba)))
@inline initialize(eb::EnergyBands{P, Vector{Int}}, ::AbstractTBA) where P = (eb.basespace, zeros(Float64, length(eb.basespace), length(eb.levels)))
@inline Base.nameof(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands}) = @sprintf "%s_%s" repr(tba, ∉(names(eb.action.basespace))) eb.id
function run!(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands})
    atol = get(eb.action.options, :atol, 10^-12)
    for (i, params) in enumerate(pairs(eb.action.basespace))
        update!(tba; params...)
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
@inline function initialize(bc::BerryCurvature{<:BrillouinZone}, ::AbstractTBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "initialize error: Berry curvature should be defined for 2d systems."
    ny, nx = map(length, shape(bc.reciprocalspace))
    z = zeros(Float64, ny, nx, length(bc.levels))
    n = zeros(Float64, length(bc.levels))
    return (bc.reciprocalspace, z, n)
end
function eigvecs(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature{<:BrillouinZone}})
    eigenvectors = eigvecs(tba, bc.action.reciprocalspace; bc.action.options...)
    ny, nx = map(length, shape(bc.action.reciprocalspace))
    result = Matrix{eltype(eigenvectors)}(undef, nx+1, ny+1)
    for i=1:nx+1, j=1:ny+1
        result[i, j] = eigenvectors[Int(keytype(bc.action.reciprocalspace)(i, j))][:, bc.action.levels]
    end
    return result
end

# For the Berry curvature on a specific zone in the reciprocal space
@inline function initialize(bc::BerryCurvature{<:ReciprocalZone}, ::AbstractTBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "initialize error: Berry curvature should be defined for 2d systems."
    ny, nx = map(length, shape(bc.reciprocalspace))
    z = zeros(Float64, ny-1, nx-1, length(bc.levels))
    return (shrink(bc.reciprocalspace, 1:nx-1, 1:ny-1), z)
end
function eigvecs(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature{<:ReciprocalZone}})
    eigenvectors = eigvecs(tba, bc.action.reciprocalspace; bc.action.options...)
    ny, nx = map(length, shape(bc.action.reciprocalspace))
    result = Matrix{eltype(eigenvectors)}(undef, nx, ny)
    count = 1
    for i=1:nx, j=1:ny
        result[i, j] = eigenvectors[count][:, bc.action.levels]
        count += 1
    end
    return result
end

# Compute the Berry curvature and optionally, the Chern number
function run!(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature})
    eigenvectors = eigvecs(tba, bc)
    g = isnothing(getcontent(tba.frontend, :commutator)) ? Diagonal(ones(Int, dimension(tba.frontend))) : inv(getcontent(tba.frontend, :commutator))
    @timeit_debug tba.timer "Berry curvature" for i = 1:size(eigenvectors)[1]-1, j = 1:size(eigenvectors)[2]-1
        vs₁ = eigenvectors[i, j]
        vs₂ = eigenvectors[i+1, j]
        vs₃ = eigenvectors[i+1, j+1]
        vs₄ = eigenvectors[i, j+1]
        for k = 1:length(bc.action.levels)
            p₁ = vs₁[:, k]'*g*vs₂[:, k]
            p₂ = vs₂[:, k]'*g*vs₃[:, k]
            p₃ = vs₃[:, k]'*g*vs₄[:, k]
            p₄ = vs₄[:, k]'*g*vs₁[:, k]
            bc.data[2][j, i, k] = angle(p₁*p₂*p₃*p₄)
            length(bc.data)==3 && (bc.data[3][k] += bc.data[2][j, i, k]/2pi)
        end
    end
    length(bc.data)==4 && @info (@sprintf "Chern numbers: %s" join([@sprintf "%s(%s)" cn level for (cn, level) in zip(bc.data[4], bc.action.levels)], ", "))
end

# Plot the Berry curvature and optionally, the Chern number
@recipe function plot(pack::Tuple{Algorithm{<:AbstractTBA}, Assignment{<:BerryCurvature}})
    subtitles --> if length(pack[2].data)==3
        [@sprintf("level %s (C = %s)", level, decimaltostr(chn)) for (level, chn) in zip(pack[2].action.levels, pack[2].data[3])]
    else
        [@sprintf("level %s", level) for level in pack[2].action.levels]
    end
    subtitlefontsize --> 8
    plot_title --> nameof(pack[1], pack[2])
    plot_titlefontsize --> 10
    pack[2].data[1:2]
end

function spectralfunction(tbakind::TBAKind, ω::Real, values::Vector{<:Real}, vectors::Matrix{<:Number}, bands::Union{Colon, Vector{Int}}=:, orbitals::Union{Colon, Vector{Int}}=:; σ::Real)
    result = zero(ω)
    if isa(bands, Colon)
        bands = (isa(tbakind, TBAKind{:TBA}) ? 1 : length(values)÷2):length(values)
    end
    for i in bands
        factor = mapreduce(abs2, +, vectors[orbitals, i])
        result += factor*exp(-(ω-values[i])^2/2/σ^2)
    end
    return result/√(2pi)/σ
end
"""
    FermiSurface{B<:Union{BrillouinZone, ReciprocalZone}, A<:Union{Colon, Vector{Int}}, L<:Tuple{Vararg{Union{Colon, Vector{Int}}}}, O} <: Action

Fermi surface of a free fermionic system.
"""
struct FermiSurface{B<:Union{BrillouinZone, ReciprocalZone}, A<:Union{Colon, Vector{Int}}, L<:Tuple{Vararg{Union{Colon, Vector{Int}}}}, O} <: Action
    reciprocalspace::B
    μ::Float64
    bands::A
    orbitals::L
    options::O
end
@inline FermiSurface(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, μ::Real=0.0, bands::Union{Colon, Vector{Int}}=:, orbitals::Union{Colon, Vector{Int}}...=:; options...) = FermiSurface(reciprocalspace, μ, bands, orbitals, options)
function initialize(fs::FermiSurface, ::AbstractTBA)
    @assert length(fs.reciprocalspace.reciprocals)==2 "initialize error: only two dimensional reciprocal spaces are supported."
    ny, nx = map(length, shape(fs.reciprocalspace))
    z = zeros(Float64, ny, nx, length(fs.orbitals))
    return (fs.reciprocalspace, z)
end
function run!(tba::Algorithm{<:AbstractTBA{<:Fermionic{:TBA}}}, fs::Assignment{<:FermiSurface})
    count = 1
    σ = get(fs.action.options, :fwhm, 0.1)/2/√(2*log(2))
    eigenvalues, eigenvectors = eigen(tba, fs.action.reciprocalspace)
    ny, nx = map(length, shape(fs.action.reciprocalspace))
    for i=1:nx, j=1:ny
        for (k, orbitals) in enumerate(fs.action.orbitals)
            fs.data[2][j, i, k] += spectralfunction(kind(tba.frontend), fs.action.μ, eigenvalues[count], eigenvectors[count], fs.action.bands, orbitals; σ=σ)
        end
        count += 1
    end
end
@recipe function plot(pack::Tuple{Algorithm{<:AbstractTBA}, Assignment{<:FermiSurface}})
    if size(pack[2].data[2])[3]==1
        title --> nameof(pack...)
        titlefontsize --> 10
        pack[2].data[1], pack[2].data[2][:, :, 1]
    else
        subtitles --> [@sprintf("orbitals: %s\n bands: %s", tostr(orbitals), tostr(pack[2].action.bands)) for orbitals in pack[2].action.orbitals]
        subtitlefontsize --> 8
        plot_title --> nameof(pack[1], pack[2])
        plot_titlefontsize --> 10
        pack[2].data
    end
end
@inline tostr(::Colon) = "all"
@inline tostr(contents::Vector{Int}) = join(contents, ", ")

"""
    DensityOfStates{B<:BrillouinZone, A<:Union{Colon, Vector{Int}}, L<:Tuple{Vararg{Union{Colon, Vector{Int}}}}, O} <: Action

Density of states of a tight-binding system.
"""
struct DensityOfStates{B<:BrillouinZone, A<:Union{Colon, Vector{Int}}, L<:Tuple{Vararg{Union{Colon, Vector{Int}}}}, O} <: Action
    brillouinzone::B
    bands::A
    orbitals::L
    options::O
end
@inline DensityOfStates(brillouinzone::BrillouinZone, bands::Union{Colon, Vector{Int}}=:, orbitals::Union{Colon, Vector{Int}}...=:; options...) = DensityOfStates(brillouinzone, bands, orbitals, options)
@inline function initialize(dos::DensityOfStates, ::AbstractTBA)
    ne = get(dos.options, :ne, 100)
    x = zeros(Float64, ne)
    z = zeros(Float64, ne, length(dos.orbitals))
    return (x, z)
end
function run!(tba::Algorithm{<:AbstractTBA{<:Fermionic{:TBA}}}, dos::Assignment{<:DensityOfStates})
    σ = get(dos.action.options, :fwhm, 0.1)/2/√(2*log(2))
    eigenvalues, eigenvectors = eigen(tba, dos.action.brillouinzone)
    emin = get(dos.action.options, :emin, mapreduce(minimum, min, eigenvalues))
    emax = get(dos.action.options, :emax, mapreduce(maximum, max, eigenvalues))
    ne = get(dos.action.options, :ne, 100)
    nk = length(dos.action.brillouinzone)
    dE = (emax-emin)/(ne-1)
    for (i, ω) in enumerate(LinRange(emin, emax, ne))
        dos.data[1][i] = ω
        for (j, orbitals) in enumerate(dos.action.orbitals)
            for (values, vectors) in zip(eigenvalues, eigenvectors)
                dos.data[2][i, j] += spectralfunction(kind(tba.frontend), ω, values, vectors, dos.action.bands, orbitals; σ=σ)/nk*dE
            end
        end
    end
end

"""
    InelasticNeutronScatteringSpectra{P<:ReciprocalSpace, E<:AbstractVector, O} <: Action

Inelastic neutron scattering spectra.
"""
struct InelasticNeutronScatteringSpectra{P<:ReciprocalSpace, E<:AbstractVector, O} <: Action
    reciprocalspace::P
    energies::E
    options::O
    function InelasticNeutronScatteringSpectra(reciprocalspace::ReciprocalSpace, energies::AbstractVector, options)
        @assert names(reciprocalspace)==(:k,) "InelasticNeutronScatteringSpectra error: the name of the momenta in the reciprocalspace must be :k."
        new{typeof(reciprocalspace), typeof(energies), typeof(options)}(reciprocalspace, energies, options)
    end
end
@inline InelasticNeutronScatteringSpectra(reciprocalspace::ReciprocalSpace, energies::AbstractVector; options...) = InelasticNeutronScatteringSpectra(reciprocalspace, energies, options)
@inline initialize(inss::InelasticNeutronScatteringSpectra, ::AbstractTBA) = (inss.reciprocalspace, inss.energies, zeros(Float64, length(inss.energies), length(inss.reciprocalspace)))

# Inelastic neutron scattering spectra for phonons.
function run!(tba::Algorithm{<:AbstractTBA{Phononic}}, inss::Assignment{<:InelasticNeutronScatteringSpectra})
    dim = dimension(tba.frontend)
    σ = get(inss.action.options, :fwhm, 0.1)/2/√(2*log(2))
    check = get(inss.action.options, :check, true)
    sequences = Dict(site=>[tba.frontend.H.table[Index(site, PID('u', Char(Int('x')+i-1)))] for i=1:phonon.ndirection] for (site, phonon) in pairs(tba.frontend.H.hilbert))
    eigenvalues, eigenvectors = eigen(tba, inss.action.reciprocalspace; inss.action.options...)
    for (i, (momentum, values, vectors)) in enumerate(zip(inss.action.reciprocalspace, eigenvalues, eigenvectors))
        check && @timeit_debug tba.timer "check" check_polarizations(@views(vectors[(dim÷2+1):dim, 1:(dim÷2)]), @views(vectors[(dim÷2+1):dim, dim:-1:(dim÷2+1)]), momentum./pi)
        @timeit_debug tba.timer "spectra" begin
            for j = 1:dim
                factor = 0
                for (site, sequence) in pairs(sequences)
                    factor += dot(momentum, vectors[sequence, j])*exp(1im*dot(momentum, tba.frontend.lattice[site]))
                end
                factor = abs2(factor)/√(2pi)/σ
                for (nₑ, e) in enumerate(inss.action.energies)
                    # instead of the Lorentz broadening of δ function, the convolution with a FWHM Gaussian is used.
                    inss.data[3][nₑ, i] += factor*exp(-(e-values[j])^2/2/σ^2)
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
    optimize!(
        tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplesets::Vector{SampleNode}, variables=keys(Parameters(tba));
        verbose=false, method=LBFGS(), x_tol=atol, f_tol=atol
    ) -> Tuple{typeof(tba), Optim.MultivariateOptimizationResults}

Optimize the parameters of a tight binding system whose names are specified by `variables` so that the total deviations of the eigenvalues between the model points and sample points minimize.
"""
function optimize!(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplesets::Vector{SampleNode}, variables=keys(Parameters(tba)); verbose=false, method=LBFGS(), options=Options())
    v₀ = collect(getfield(Parameters(tba), name) for name in variables)
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

end # module
