module TightBindingApproximation

using Printf: @sprintf
using TimerOutputs: @timeit
using Optim: optimize, LBFGS
using RecipesBase: RecipesBase, @recipe, @series
using LinearAlgebra: inv, dot, Hermitian, Diagonal, eigvals, cholesky, Eigen
using QuantumLattices: getcontent, iidtype, rcoord, icoord, expand, statistics, plain, creation, annihilation, atol, rtol, periods, decimaltostr
using QuantumLattices: AbstractLattice, AbstractPID, FID, NID, Index, CompositeOID, ID, Bonds, Hilbert, Metric, Operator, Operators, OIDToTuple, Table, Term, Boundary
using QuantumLattices: Internal, Fock, Phonon, Hopping, Onsite, Pairing, PhononKinetic, PhononPotential, BrillouinZone, ReciprocalZone, ReciprocalPath 
using QuantumLattices: MatrixRepresentation, Engine, Parameters, AbstractGenerator, CompositeGenerator, Entry, Generator, Formulation, Action, Assignment, Algorithm

import LinearAlgebra: eigen, eigvals, ishermitian
import QuantumLattices: add!, contentnames, dimension, kind, matrix, update!, prepare!, run!

export TBAKind, Fermionic, Bosonic, Phononic, AbstractTBA, TBAMatrix, TBAMatrixRepresentation, commutator
export TBA, EnergyBands, BerryCurvature, InelasticNeutronScatteringSpectra
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
@inline TBAKind(::Type{T}, ::Type{<:Internal}) where {T<:Term} = error("TBAKind error: not defined behavior.")
@inline TBAKind(::Type{T}, ::Type{<:Fock{:f}}) where {T<:Union{Hopping, Onsite}} = Fermionic(:TBA)
@inline TBAKind(::Type{T}, ::Type{<:Fock{:f}}) where {T<:Pairing} = Fermionic(:BdG)
@inline TBAKind(::Type{T}, ::Type{<:Fock{:b}}) where {T<:Union{Hopping, Onsite}} = Bosonic(:TBA)
@inline TBAKind(::Type{T}, ::Type{<:Fock{:b}}) where {T<:Pairing} = Bosonic(:BdG)
@inline TBAKind(::Type{T}, ::Type{<:Phonon}) where {T<:Union{PhononKinetic, PhononPotential}} = Phononic()
@inline @generated function TBAKind(::Type{TS}, ::Type{I}) where {TS<:Tuple{Vararg{Term}}, I<:Internal}
    exprs = []
    for i = 1:fieldcount(TS)
        push!(exprs, :(typeof(TBAKind(fieldtype(TS, $i), I))))
    end
    return Expr(:call, Expr(:call, :reduce, :promote_type, Expr(:tuple, exprs...)))
end

"""
    Metric(::Fermionic, hilbert::Hilbert{<:Fock{:f}} -> OIDToTuple
    Metric(::Bosonic, hilbert::Hilbert{<:Fock{:b}} -> OIDToTuple
    Metric(::Phononic, hilbert::Hilbert{<:Phonon}) -> OIDToTuple

Get the oid-to-tuple metric for a free fermionic/bosonic/phononic system.
"""
@inline Metric(::TBAKind, ::Hilbert) = error("Metric error: not defined behavior.")
@inline @generated Metric(::Fermionic{:TBA}, hilbert::Hilbert{<:Fock{:f}}) = OIDToTuple(fieldnames(keytype(hilbert))..., :orbital, :spin)
@inline @generated Metric(::Bosonic{:TBA}, hilbert::Hilbert{<:Fock{:b}}) = OIDToTuple(fieldnames(keytype(hilbert))..., :orbital, :spin)
@inline @generated Metric(::Fermionic{:BdG}, hilbert::Hilbert{<:Fock{:f}}) = OIDToTuple(:nambu, fieldnames(keytype(hilbert))..., :orbital, :spin)
@inline @generated Metric(::Bosonic{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = OIDToTuple(:nambu, fieldnames(keytype(hilbert))..., :orbital, :spin)
@inline @generated Metric(::Phononic, hilbert::Hilbert{<:Phonon}) = OIDToTuple(:tag, fieldnames(keytype(hilbert))..., :dir)

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
    AbstractTBA{K<:TBAKind, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine

Abstract type for free quantum lattice systems using the tight-binding approximation.
"""
abstract type AbstractTBA{K<:TBAKind, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine end
@inline contentnames(::Type{<:AbstractTBA}) = (:H, :commutator)
@inline kind(tba::AbstractTBA) = kind(typeof(tba))
@inline kind(::Type{<:AbstractTBA{K}}) where K = K()
@inline Base.valtype(::Type{<:AbstractTBA{<:TBAKind, H}}) where {H<:AbstractGenerator} = valtype(eltype(H))
@inline dimension(tba::AbstractTBA{<:TBAKind, <:CompositeGenerator}) = length(getcontent(getcontent(tba, :H), :table))
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
    TBAMatrixRepresentation{K<:AbstractTBA, V, T} <: MatrixRepresentation

Matrix representation of the Hamiltonian of a tight-binding system.
"""
struct TBAMatrixRepresentation{K<:AbstractTBA, V, T} <: MatrixRepresentation
    k::V
    table::T
    gauge::Symbol
    function TBAMatrixRepresentation{K}(k, table, gauge::Symbol=:icoord) where {K<:AbstractTBA}
        @assert gauge∈(:rcoord, :icoord) "TBAMatrixRepresentation error: gauge must be :rcoord or :icoord."
        return new{K, typeof(k), typeof(table)}(k, table, gauge)
    end
end
@inline TBAMatrixRepresentation{K}(table, gauge::Symbol=:icoord) where {K<:AbstractTBA} = TBAMatrixRepresentation{K}(nothing, table, gauge)
@inline Base.valtype(mr::TBAMatrixRepresentation) = valtype(typeof(mr))
@inline Base.valtype(::Type{<:TBAMatrixRepresentation{K}}) where {K<:AbstractTBA} = Matrix{promote_type(valtype(K), Complex{Int})}
@inline Base.valtype(::Type{<:TBAMatrixRepresentation{K, Nothing}}) where {K<:AbstractTBA} = Matrix{valtype(K)}
@inline Base.valtype(R::Type{<:TBAMatrixRepresentation}, ::Type{<:Union{Operator, Operators}}) = valtype(R)
@inline Base.zero(mr::TBAMatrixRepresentation) = zeros(eltype(valtype(mr)), length(mr.table), length(mr.table))
@inline Base.zero(mr::TBAMatrixRepresentation, ::Union{Operator, Operators}) = zero(mr)
@inline (mr::TBAMatrixRepresentation)(m::Operator; kwargs...) = add!(zero(mr, m), mr, m; kwargs...)

"""
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:TBA}}}, m::Operator; kwargs...) -> typeof(dest)
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:BdG}}}, m::Operator{<:Number, <:ID{CompositeOID{<:Index{<:AbstractPID, <:FID{:f}}}}}; kwargs...) -> typeof(dest)
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:BdG}}}, m::Operator{<:Number, <:ID{CompositeOID{<:Index{<:AbstractPID, <:FID{:b}}}}}; atol=atol/5, kwargs...) -> typeof(dest)
    add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:BdG}}}, m::Operator{<:Number, <:ID{CompositeOID{<:Index{<:AbstractPID, <:NID}}}}; atol=atol/5, kwargs...) -> typeof(dest)

Get the matrix representation of an operator and add it to destination.
"""
function add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:TBA}}}, m::Operator; kwargs...)
    seq₁, seq₂ = mr.table[m[1].index'], mr.table[m[2].index]
    coord = mr.gauge==:rcoord ? rcoord(m) : icoord(m)
    phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(-1im*dot(mr.k, coord)))
    dest[seq₁, seq₂] += m.value*phase
    return dest
end
@inline add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:BdG}}}, m::Operator{<:Number, <:ID{CompositeOID{<:Index{<:AbstractPID, <:FID{:f}}}}}; kwargs...) = _add!(dest, mr, m, -1; kwargs..., atol=0)
@inline add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:BdG}}}, m::Operator{<:Number, <:ID{CompositeOID{<:Index{<:AbstractPID, <:FID{:b}}}}}; atol=atol/5, kwargs...) = _add!(dest, mr, m, +1; atol=atol, kwargs...)
function _add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:BdG}}}, m, sign; atol, kwargs...)
    seq₁, seq₂ = mr.table[m[1].index'], mr.table[m[2].index]
    coord = mr.gauge==:rcoord ? rcoord(m) : icoord(m)
    phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(-1im*dot(mr.k, coord)))
    seq₁==seq₂ || (atol = 0)
    dest[seq₁, seq₂] += m.value*phase+atol
    if m[1].index.iid.nambu==creation && m[2].index.iid.nambu==annihilation
        seq₁, seq₂ = mr.table[m[1].index], mr.table[m[2].index']
        dest[seq₁, seq₂] += sign*m.value*phase'+atol
    end
    return dest
end
function add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:AbstractTBA{<:TBAKind{:BdG}}}, m::Operator{<:Number, <:ID{CompositeOID{<:Index{<:AbstractPID, <:NID}}}}; atol=atol/5, kwargs...)
    if m[1] == m[2]
        seq = mr.table[m[1].index]
        dest[seq, seq] += 2*m.value+atol
    else
        seq₁, seq₂ = mr.table[m[1].index], mr.table[m[2].index]
        coord = mr.gauge==:rcoord ? rcoord(m) : icoord(m)
        phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(-1im*dot(mr.k, coord)))
        dest[seq₁, seq₂] += m.value*phase
        dest[seq₂, seq₁] += m.value'*phase'
    end
    return dest
end

"""
    TBAMatrixRepresentation(tba::AbstractTBA, k=nothing; gauge::Symbol=:icoord)

Construct the matrix representation transformation of a free quantum lattice system using the tight-binding approximation.
"""
@inline function TBAMatrixRepresentation(tba::AbstractTBA, k=nothing; gauge::Symbol=:icoord)
    return TBAMatrixRepresentation{typeof(tba)}(k, getcontent(getcontent(tba, :H), :table), gauge)
end

"""
    matrix(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; k=nothing, gauge=:icoord, kwargs...) -> TBAMatrix

Get the matrix representation of a free quantum lattice system.
"""
@inline function matrix(tba::AbstractTBA; k=nothing, gauge=:icoord, kwargs...)
    H = getcontent(tba, :H)
    commutator = getcontent(tba, :commutator)
    return TBAMatrix(Hermitian(TBAMatrixRepresentation(tba, k; gauge=gauge)(expand(H); kwargs...)), commutator)
end
@inline function matrix(tba::AbstractTBA{<:TBAKind, <:Formulation}; kwargs...)
    return TBAMatrix(Hermitian(getcontent(tba, :H)(; kwargs...)), getcontent(tba, :commutator))
end
@inline matrix(tba::Algorithm{<:AbstractTBA}; kwargs...) = matrix(tba.engine; kwargs...)

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
    eigvals(m::TBAMatrix) -> Vector

Get the eigen values of a free quantum lattice system.
"""
@inline eigvals(m::TBAMatrix{T, H, Nothing}) where {T, H<:AbstractMatrix{T}} = eigvals(m.H)
function eigvals(m::TBAMatrix{T, H, G}) where {T, H<:AbstractMatrix{T}, G<:AbstractMatrix}
    values = eigen(m.H*m.commutator).values
    @assert length(values)%2==0 "eigvals error: wrong dimension of matrix."
    for i = 1:(length(values)÷2)
        values[i] = -values[i]
    end
    return values
end

"""
    TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}

The usual tight binding approximation for quantum lattice systems.
"""
struct TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}
    lattice::L
    H::H
    commutator::G
    function TBA{K}(lattice::AbstractLattice, H::AbstractGenerator, commutator::Union{AbstractMatrix, Nothing}) where {K<:TBAKind}
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
    tbakind = TBAKind(typeof(terms), valtype(hilbert))
    table = Table(hilbert, Metric(tbakind, hilbert))
    commt = commutator(tbakind, hilbert)
    return TBA{typeof(tbakind)}(lattice, Generator(terms, Bonds(lattice), hilbert; half=false, table=table, boundary=boundary), commt)
end

"""
    TBA{K}(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}

Construct a tight-binding quantum lattice system by providing the analytical expressions of the Hamiltonian.
"""
@inline function TBA{K}(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    return TBA{K}(lattice, Formulation(hamiltonian, parameters), commt)
end

"""
    EnergyBands{P, L<:Union{Colon, Vector{Int}}} <: Action

Energy bands by tight-binding-approximation for quantum lattice systems.
"""
struct EnergyBands{P, L<:Union{Colon, Vector{Int}}} <: Action
    path::P
    levels::L
    options::Dict{Symbol, Any}
end
@inline EnergyBands(path, levels::Union{Colon, Vector{Int}}=Colon(); options...) = EnergyBands(path, levels, convert(Dict{Symbol, Any}, options))
@inline prepare!(eb::EnergyBands{P, Colon}, tba::AbstractTBA) where P = (zeros(Float64, length(eb.path)), zeros(Float64, length(eb.path), dimension(tba)))
@inline prepare!(eb::EnergyBands{P, Vector{Int}}, tba::AbstractTBA) where P = (zeros(Float64, length(eb.path)), zeros(Float64, length(eb.path), length(eb.levels)))
@inline Base.nameof(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands}) = @sprintf "%s_%s" repr(tba, ∉(keys(eb.action.path))) eb.id
function run!(tba::Algorithm{<:AbstractTBA}, eb::Assignment{<:EnergyBands})
    for (i, params) in enumerate(pairs(eb.action.path))
        eb.data[1][i] = length(params)==1 && isa(first(params), Number) ? first(params) : i-1
        update!(tba; params...)
        @timeit tba.timer "matrix" (m = matrix(tba.engine; gauge=get(eb.action.options, :gauge, :icoord), atol=get(eb.action.options, :atol, atol/5), params...))
        @timeit tba.timer "eigen" (eb.data[2][i, :] = eigen(m).values[eb.action.levels])
    end
end

"""
    BerryCurvature{B<:Union{BrillouinZone, ReciprocalZone}} <: Action

Berry curvature of energy bands with the spirit of a momentum space discretization method by [Fukui et al, JPSJ 74, 1674 (2005)](https://journals.jps.jp/doi/10.1143/JPSJ.74.1674).
"""
struct BerryCurvature{B<:Union{BrillouinZone, ReciprocalZone}} <: Action
    reciprocalspace::B
    levels::Vector{Int}
    options::Dict{Symbol, Any}
end
@inline BerryCurvature(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, levels::Vector{Int}; options...) = BerryCurvature(reciprocalspace, levels, convert(Dict{Symbol, Any}, options))
@inline function prepare!(bc::BerryCurvature{<:BrillouinZone}, tba::AbstractTBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "prepare! error: Berry curvature should be defined for 2d systems."
    N₁, N₂ = periods(eltype(bc.reciprocalspace))
    x = collect(Float64, 0:(N₁-1))/N₁
    y = collect(Float64, 0:(N₂-1))/N₂
    z = zeros(Float64, length(y), length(x), length(bc.levels))
    n = zeros(Float64, length(bc.levels))
    return (x, y, z, n)
end
function run!(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature{<:BrillouinZone}})
    N₁, N₂ = length(bc.data[1]), length(bc.data[2])
    eigenvectors = zeros(ComplexF64, N₁, N₂, dimension(tba.engine), length(bc.action.levels))
    for momentum in bc.action.reciprocalspace
        coord = expand(momentum, bc.action.reciprocalspace.reciprocals)
        @timeit tba.timer "matrix" (m = matrix(tba.engine; k=coord, gauge=get(bc.action.options, :gauge, :icoord), atol=get(bc.action.options, :atol, atol/5)))
        @timeit tba.timer "eigen" (eigenvectors[Int(momentum[1])+1, Int(momentum[2])+1, :, :] = eigen(m).vectors[:, bc.action.levels])
    end
    g = isnothing(tba.engine.commutator) ? Diagonal(ones(Int, dimension(tba.engine))) : inv(tba.engine.commutator)
    @timeit tba.timer "Berry curvature" for momentum in bc.action.reciprocalspace
        i₁, j₁ = Int(momentum[1]), Int(momentum[2])
        i₂, j₂ = (i₁+1)%N₁, (j₁+1)%N₂
        vs₁ = eigenvectors[i₁+1, j₁+1, :, :]
        vs₂ = eigenvectors[i₂+1, j₁+1, :, :]
        vs₃ = eigenvectors[i₂+1, j₂+1, :, :]
        vs₄ = eigenvectors[i₁+1, j₂+1, :, :]
        for k = 1:length(bc.action.levels)
            p₁ = vs₁[:, k]'*g*vs₂[:, k]
            p₂ = vs₂[:, k]'*g*vs₃[:, k]
            p₃ = vs₃[:, k]'*g*vs₄[:, k]
            p₄ = vs₄[:, k]'*g*vs₁[:, k]
            bc.data[3][j₁+1, i₁+1, k] = angle(p₁*p₂*p₃*p₄)
            bc.data[4][k] += bc.data[3][j₁+1, i₁+1, k]/2pi
        end
    end
    @info (@sprintf "Chern numbers: %s" join([@sprintf "%s(%s)" cn level for (cn, level) in zip(bc.data[4], bc.action.levels)], ", "))
end
@recipe function plot(pack::Tuple{Algorithm{<:AbstractTBA}, Assignment{<:BerryCurvature{<:BrillouinZone}}})
    levels, chernnumbers = pack[2].action.levels, pack[2].data[4]
    layout --> length(levels)
    aspect_ratio --> :equal
    framestyle --> :none
    colorbar --> :bottom
    plot_title --> nameof(pack[1], pack[2])
    plot_titlefontsize --> 10
    title := @sprintf "level %s (C = %s)" first(levels) decimaltostr(first(chernnumbers))
    titlefontsize --> 8
    for (i, (level, chn)) in enumerate(zip(levels, chernnumbers))
        @series begin
            seriestype := :heatmap
            title := @sprintf "level %s (C = %s)" level decimaltostr(chn)
            subplot := i
            pack[2].data[1], pack[2].data[2], pack[2].data[3][:, :, i]
        end
    end
    primary := false
    ()
end

@inline function prepare!(bc::BerryCurvature{<:ReciprocalZone}, tba::AbstractTBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "prepare! error: Berry curvature should be defined for 2d systems."
    x = collect(bc.reciprocalspace.bounds[1])[1:end-1]
    y = collect(bc.reciprocalspace.bounds[2])[1:end-1]
    z = zeros(Float64, length(y), length(x), length(bc.levels))
    return (x, y, z)
end
function run!(tba::Algorithm{<:AbstractTBA}, bc::Assignment{<:BerryCurvature{<:ReciprocalZone}})
    N₁, N₂ = length(bc.data[1]), length(bc.data[2])
    indices = CartesianIndices((1:(N₂+1), 1:(N₁+1)))
    eigenvectors = zeros(ComplexF64, N₁+1, N₂+1, dimension(tba.engine), length(bc.action.levels))
    for (index, momentum) in enumerate(bc.action.reciprocalspace)
        j, i = Tuple(indices[index])
        @timeit tba.timer "matrix" (m = matrix(tba.engine; k=momentum, gauge=get(bc.action.options, :gauge, :icoord), atol=get(bc.action.options, :atol, atol/5)))
        @timeit tba.timer "eigen" (eigenvectors[i, j, :, :] = eigen(m).vectors[:, bc.action.levels])
    end
    g = isnothing(tba.engine.commutator) ? Diagonal(ones(Int, dimension(tba.engine))) : inv(tba.engine.commutator)
    @timeit tba.timer "Berry curvature" for i = 1:N₁, j = 1:N₂
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
        end
    end
end
@recipe function plot(pack::Tuple{Algorithm{<:AbstractTBA}, Assignment{<:BerryCurvature{<:ReciprocalZone}}})
    levels = pack[2].action.levels
    layout --> length(levels)
    aspect_ratio --> :equal
    framestyle --> :none
    colorbar --> :bottom
    plot_title --> nameof(pack[1], pack[2])
    plot_titlefontsize --> 10
    title := @sprintf "level %s" first(levels)
    titlefontsize --> 8
    for (i, level) in enumerate(levels)
        @series begin
            seriestype := :heatmap
            title := @sprintf "level %s" level
            subplot := i
            pack[2].data[1], pack[2].data[2], pack[2].data[3][:, :, i]
        end
    end
    primary := false
    ()
end

"""
    InelasticNeutronScatteringSpectra{P<:ReciprocalPath, E<:AbstractVector} <: Action

Inelastic neutron scattering spectra.
"""
struct InelasticNeutronScatteringSpectra{P<:ReciprocalPath, E<:AbstractVector} <: Action
    path::P
    energies::E
    options::Dict{Symbol, Any}
    function InelasticNeutronScatteringSpectra(path::ReciprocalPath, energies::AbstractVector, options::Dict{Symbol, Any})
        @assert keys(path)==(:k,) "InelasticNeutronScatteringSpectra error: the name of the momenta in the path must be :k."
        new{typeof(path), typeof(energies)}(path, energies, options)
    end
end
@inline function InelasticNeutronScatteringSpectra(path::ReciprocalPath, energies::AbstractVector; options...)
    InelasticNeutronScatteringSpectra(path, energies, convert(Dict{Symbol, Any}, options))
end
@inline function prepare!(ins::InelasticNeutronScatteringSpectra, tba::AbstractTBA)
    x = collect(Float64, 0:(length(ins.path)-1))
    y = collect(Float64, ins.energies)
    z = zeros(Float64, length(y), length(x))
    return (x, y, z)
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
