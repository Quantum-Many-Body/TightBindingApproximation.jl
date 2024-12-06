using LinearAlgebra: Diagonal, Eigen, cholesky, dot, inv, norm, logdet, normalize
using Printf: @printf, @sprintf
using QuantumLattices: atol, lazy, plain, rtol
using QuantumLattices: AbstractLattice, Action, Algorithm, Assignment, BrillouinZone, Boundary, CoordinatedIndex, Elastic, FockIndex, Fock, Formula, Frontend, Generator, Hilbert, Hooke, Hopping, ID, Index, Internal, Kinetic, LinearTransformation, Matrixization, Metric, Neighbors, Onsite, Operator, OperatorPack, OperatorSet, OperatorSum, OperatorUnitToTuple, Pairing, Phonon, PhononIndex, ReciprocalPath, ReciprocalSpace, ReciprocalZone, Term
using QuantumLattices: ⊕, bonds, checkoptions, dtype, expand, icoordinate, idtype, isannihilation, iscreation, optype, parametertype, rank, rcoordinate, shape, shrink, statistics, tostr, volume
using RecipesBase: RecipesBase, @recipe, @series, @layout
using TimerOutputs: TimerOutput, @timeit_debug

import LinearAlgebra: eigen, eigvals, eigvecs, ishermitian, Hermitian
import QuantumLattices: Parameters, Table, add!, dimension, getcontent, initialize, kind, matrix, options, parameternames, run!, update!

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
    infinitesimal(::TBAKind{:TBA}) -> 0
    infinitesimal(::TBAKind{:BdG}) -> atol/5
    infinitesimal(::Fermionic{:BdG}) -> 0

Infinitesimal used in the matrixization of a tight-binding approximation to be able to capture the Goldstone mode.
"""
@inline infinitesimal(::TBAKind{:TBA}) = 0
@inline infinitesimal(::TBAKind{:BdG}) = atol/5
@inline infinitesimal(::Fermionic{:BdG}) = 0

"""
    Metric(::Fermionic, hilbert::Hilbert{<:Fock{:f}}) -> OperatorUnitToTuple
    Metric(::Bosonic, hilbert::Hilbert{<:Fock{:b}}) -> OperatorUnitToTuple
    Metric(::Phononic, hilbert::Hilbert{<:Phonon}) -> OperatorUnitToTuple

Get the index-to-tuple metric for a free fermionic/bosonic/phononic system.
"""
@inline Metric(::TBAKind, ::Hilbert) = error("Metric error: not defined behavior.")
@inline @generated Metric(::Fermionic{:TBA}, hilbert::Hilbert{<:Fock{:f}}) = OperatorUnitToTuple(:site, :orbital, :spin)
@inline @generated Metric(::Bosonic{:TBA}, hilbert::Hilbert{<:Fock{:b}}) = OperatorUnitToTuple(:site, :orbital, :spin)
@inline @generated Metric(::Fermionic{:BdG}, hilbert::Hilbert{<:Fock{:f}}) = OperatorUnitToTuple(:nambu, :site, :orbital, :spin)
@inline @generated Metric(::Bosonic{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = OperatorUnitToTuple(:nambu, :site, :orbital, :spin)
@inline @generated Metric(::Phononic, hilbert::Hilbert{<:Phonon}) = OperatorUnitToTuple(kind, :site, :direction)

"""
    Table(hilbert::Hilbert{Phonon{:}}, by::OperatorUnitToTuple{(kind, :site, :direction)})

Construct a index-sequence table for a phononic system.
"""
@inline function Table(hilbert::Hilbert{Phonon{:}}, by::OperatorUnitToTuple{(kind, :site, :direction)})
    new = Hilbert(site=>filter(PhononIndex{:u}, internal)⊕filter(PhononIndex{:p}, internal) for (site, internal) in hilbert)
    return Table(new, by)
end

"""
    commutator(k::TBAKind, hilbert::Hilbert{<:Internal}) -> Union{AbstractMatrix, Nothing}

Get the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.
"""
@inline commutator(::TBAKind, ::Hilbert{<:Internal}) = error("commutator error: not defined behavior.")
@inline commutator(::Fermionic, ::Hilbert{<:Fock{:f}}) = nothing
@inline commutator(::Bosonic{:TBA}, ::Hilbert{<:Fock{:b}}) = nothing
@inline commutator(::Bosonic{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = Diagonal(kron([1, -1], ones(Int64, sum(length, values(hilbert))÷2)))
@inline commutator(::Phononic, hilbert::Hilbert{<:Phonon}) = Hermitian(kron([0 -1im; 1im 0], Diagonal(ones(Int, sum(length, values(hilbert))))))

"""
    Quadratic{V<:Number, C<:AbstractVector} <: OperatorPack{V, Tuple{Tuple{Int, Int}, C, C}}

The unified quadratic form in tight-binding approximation.
"""
struct Quadratic{V<:Number, C<:AbstractVector} <: OperatorPack{V, Tuple{Tuple{Int, Int}, C, C}}
    value::V
    position::Tuple{Int, Int}
    rcoordinate::C
    icoordinate::C
end
@inline parameternames(::Type{<:Quadratic}) = (:value, :coordinate)
@inline getcontent(m::Quadratic, ::Val{:id}) = (m.position, m.rcoordinate, m.icoordinate)
@inline Quadratic(value::Number, id::Tuple) = Quadratic(value, id...)
@inline Base.show(io::IO, m::Quadratic) = @printf io "%s(%s, %s, %s, %s)" nameof(typeof(m)) tostr(m.value) m.position m.rcoordinate m.icoordinate

"""
    Quadraticization{K<:TBAKind, T<:Table} <: LinearTransformation

The linear transformation that converts a rank-2 operator to its unified quadratic form.
"""
struct Quadraticization{K<:TBAKind, T<:Table} <: LinearTransformation
    table::T
    Quadraticization{K}(table::Table) where {K<:TBAKind} = new{K, typeof(table)}(table)
end
@inline function Base.valtype(::Type{<:Quadraticization{<:TBAKind}}, O::Type{<:Union{Operator, OperatorSet}})
    P = optype(O)
    @assert rank(P)==2 "valtype error: Quadraticization only applies to rank-2 operator."
    M = Quadratic{valtype(P), parametertype(eltype(idtype(P)), :coordination)}
    return OperatorSum{M, idtype(M)}
end
@inline (q::Quadraticization)(m::Operator; kwargs...) = add!(zero(q, m), q, m; kwargs)

"""
    add!(dest::OperatorSum, q::Quadraticization{<:TBAKind{:TBA}}, m::Operator{<:Number, <:ID{CoordinatedIndex{<:Index, 2}}; kwargs...) -> typeof(dest)
    add!(dest::OperatorSum, q::Quadraticization{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CoordinatedIndex{<:Index{<:FockIndex}}, 2}}; kwargs...) -> typeof(dest)
    add!(dest::OperatorSum, q::Quadraticization{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CoordinatedIndex{<:Index{<:PhononIndex}}, 2}}; kwargs...) -> typeof(dest)

Get the unified quadratic form of a rank-2 operator and add it to `dest`.
"""
function add!(dest::OperatorSum, q::Quadraticization{<:TBAKind{:TBA}}, m::Operator{<:Number, <:ID{CoordinatedIndex{<:Index}, 2}}; kwargs...)
    return add!(dest, Quadratic(m.value, (q.table[m[1]'], q.table[m[2]]), rcoordinate(m), icoordinate(m)))
end
function add!(dest::OperatorSum, q::Quadraticization{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CoordinatedIndex{<:Index{<:FockIndex}}, 2}}; kwargs...)
    rcoord, icoord = rcoordinate(m), icoordinate(m)
    add!(dest, Quadratic(m.value, (q.table[m[1]'], q.table[m[2]]), rcoord, icoord))
    if iscreation(m[1]) && isannihilation(m[2])
        sign = statistics(m[1])==statistics(m[2])==:f ? -1 : 1
        add!(dest, Quadratic(m.value*sign, (q.table[m[2]'], q.table[m[1]]), -rcoord, -icoord))
    end
    return dest
end
function add!(dest::OperatorSum, q::Quadraticization{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CoordinatedIndex{<:Index{<:PhononIndex}}, 2}}; kwargs...)
    seq₁, seq₂ = q.table[m[1]], q.table[m[2]]
    rcoord, icoord = rcoordinate(m), icoordinate(m)
    add!(dest, Quadratic(m.value, (seq₁, seq₂), rcoord, icoord))
    add!(dest, Quadratic(m.value', (seq₂, seq₁), -rcoord, -icoord))
    return dest
end

"""
    TBAMatrixization{D<:Number, V<:Union{AbstractVector{<:Number}, Nothing}} <: Matrixization

Matrixization of the Hamiltonian of a tight-binding system.
"""
struct TBAMatrixization{D<:Number, V<:Union{AbstractVector{<:Number}, Nothing}} <: Matrixization
    k::V
    dim::Int
    gauge::Symbol
    function TBAMatrixization{D}(k, dim::Integer, gauge::Symbol=:icoordinate) where {D<:Number}
        @assert gauge∈(:rcoordinate, :icoordinate) "TBAMatrixization error: gauge must be `:rcoordinate` or `:icoordinate`."
        return new{D, typeof(k)}(k, dim, gauge)
    end
end
@inline Base.valtype(::Type{<:TBAMatrixization{D}}, ::Type{<:Union{Quadratic, OperatorSet{<:Quadratic}}}) where {D<:Number} = Matrix{D}
@inline Base.zero(mr::TBAMatrixization, m::Union{Quadratic, OperatorSet{<:Quadratic}}) = zeros(eltype(valtype(mr, m)), mr.dim, mr.dim)
@inline (mr::TBAMatrixization)(m::Quadratic; kwargs...) = add!(zero(mr, m), mr, m; kwargs...)

"""
    add!(dest::AbstractMatrix, mr::TBAMatrixization, m::Quadratic; infinitesimal=0, kwargs...) -> typeof(dest)

Matrixize a quadratic form and add it to `dest`.
"""
function add!(dest::AbstractMatrix, mr::TBAMatrixization, m::Quadratic; infinitesimal=0, kwargs...)
    coordinate = mr.gauge==:rcoordinate ? m.rcoordinate : m.icoordinate
    phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(1im*dot(mr.k, coordinate)))
    dest[m.position...] += m.value*phase
    if m.position[1]==m.position[2]
        dest[m.position...] += infinitesimal
    end
    return dest
end

"""
    TBAMatrix{C<:Union{AbstractMatrix, Nothing}, T, H<:Hermitian{T, <:AbstractMatrix{T}}} <: AbstractMatrix{T}

Matrix representation of a free quantum lattice system using the tight-binding approximation.
"""
struct TBAMatrix{C<:Union{AbstractMatrix, Nothing}, T, H<:Hermitian{T, <:AbstractMatrix{T}}} <: AbstractMatrix{T}
    H::H
    commutator::C
end
@inline Base.size(m::TBAMatrix) = size(m.H)
@inline Base.getindex(m::TBAMatrix, i::Integer, j::Integer) = m.H[i, j]
@inline ishermitian(m::TBAMatrix) = ishermitian(typeof(m))
@inline ishermitian(::Type{<:TBAMatrix}) = true
@inline Hermitian(m::TBAMatrix) = m.H
@inline Base.Matrix(m::TBAMatrix) = m.H.data

"""
    eigen(m::TBAMatrix) -> Eigen

Solve the eigen problem of a free quantum lattice system.
"""
@inline eigen(m::TBAMatrix{Nothing}) = eigen(m.H)
function eigen(m::TBAMatrix{<:AbstractMatrix})
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
@inline eigvals(m::TBAMatrix{Nothing}) = eigvals(m.H)
function eigvals(m::TBAMatrix{<:AbstractMatrix})
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
    TBA{K<:TBAKind, H<:Union{Formula, OperatorSet, Generator, Frontend}, C<:Union{Nothing, AbstractMatrix}} <: Frontend

Abstract type for free quantum lattice systems using the tight-binding approximation.
"""
abstract type TBA{K<:TBAKind, H<:Union{Formula, OperatorSet, Generator, Frontend}, C<:Union{Nothing, AbstractMatrix}} <: Frontend end
@inline kind(tba::TBA) = kind(typeof(tba))
@inline kind(::Type{<:TBA{K}}) where K = K()
@inline Base.valtype(::Type{<:TBA{<:TBAKind, H}}) where {H<:Union{Formula, OperatorSet, Generator, Frontend}} = valtype(H)
@inline getcontent(tba::TBA{<:TBAKind, <:Union{Formula, OperatorSet, Generator, Frontend}, Nothing}, ::Val{:commutator}) = nothing
@inline Parameters(tba::TBA) = Parameters(getcontent(tba, :H))

"""
    dimension(tba::Union{TBA, Algorithm{<:TBA}}) -> Int

Get the dimension of the matrix representation of a free quantum lattice system.
"""
@inline dimension(tba::Algorithm{<:TBA}) = dimension(tba.frontend)
function dimension(tba::TBA{<:TBAKind, <:Formula})
    try
        return dimension(getcontent(tba, :H).expression)
    catch
        m = getcontent(tba, :H)()
        @assert ndims(m)==2 "dimension error: matrix representation is not a matrix."
        m, n = size(m)
        @assert m==n "dimension error: matrix representation is not square."
        return m
    end
end
function dimension(tba::TBA{<:TBAKind, <:Union{OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}})
    result = 0
    for op in getcontent(tba, :H)
        result = max(result, op.position...)
    end
    return result
end
@inline dimension(tba::TBA{<:TBAKind, <:Frontend}) = dimension(getcontent(tba, :H))

"""
    matrix(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba.frontend)), kwargs...) -> TBAMatrix

Get the matrix representation of a free quantum lattice system.
"""
@inline function matrix(tba::Algorithm{<:TBA}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba.frontend)), kwargs...)
    return matrix(tba.frontend, k; gauge=gauge, infinitesimal=infinitesimal, kwargs...)
end
@inline function matrix(tba::TBA{<:TBAKind, <:Formula}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba)), kwargs...)
    m = getcontent(tba, :H)(k; gauge=gauge, infinitesimal=infinitesimal, kwargs...)
    commutator = getcontent(tba, :commutator)
    return TBAMatrix(Hermitian(m), commutator)
end
@inline function matrix(
    tba::TBA{<:TBAKind, <:Union{OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing;
    gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba)), kwargs...
)
    matrixization = TBAMatrixization{datatype(dtype(tba), k)}(k, dimension(tba), gauge)
    m = matrixization(expand(getcontent(tba, :H)); infinitesimal=infinitesimal, kwargs...)
    commutator = getcontent(tba, :commutator)
    return TBAMatrix(Hermitian(m), commutator)
end
@inline function matrix(tba::TBA{<:TBAKind, <:Frontend}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba)), kwargs...)
    return matrix(getcontent(tba, :H); gauge=gauge, infinitesimal=infinitesimal, kwargs...)
end
@inline datatype(::Type{D}, ::Nothing) where D = D
@inline datatype(::Type{D}, ::Any) where D = promote_type(D, Complex{Int})

"""
    eigen(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; kwargs...) -> Eigen
    eigen(tba::Union{TBA, Algorithm{<:TBA}}, reciprocalspace::ReciprocalSpace; kwargs...) -> Tuple{Vector{<:Vector{<:Number}}, Vector{<:Matrix{<:Number}}}

Get the eigen values and eigen vectors of a free quantum lattice system.
"""
@inline eigen(tba::Algorithm{<:TBA}, k::Union{ReciprocalSpace, AbstractVector{<:Number}, Nothing}=nothing; kwargs...) = eigen(tba.frontend, k; timer=tba.timer, kwargs...)
@inline function eigen(tba::TBA, k::Union{AbstractVector{<:Number}, Nothing}=nothing; timer::TimerOutput=tbatimer, kwargs...)
    @timeit_debug timer "eigen" begin
        @timeit_debug timer "matrix" (m = matrix(tba, k; kwargs...))
        @timeit_debug timer "diagonalization" (eigensystem = eigen(m))
    end
    return eigensystem
end
function eigen(tba::TBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, kwargs...)
    datatype = eltype(eltype(reciprocalspace))
    values, vectors = Vector{datatype}[], Matrix{promote_type(datatype, Complex{Int})}[]
    for momentum in reciprocalspace
        eigensystem = eigen(tba, momentum; timer=timer, kwargs...)
        push!(values, eigensystem.values)
        push!(vectors, eigensystem.vectors)
    end
    return values, vectors
end

"""
    eigvals(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; kwargs...) -> Vector{<:Number}
    eigvals(tba::Union{TBA, Algorithm{<:TBA}}, reciprocalspace::ReciprocalSpace; kwargs...) -> Vector{<:Vector{<:Number}}

Get the eigen values of a free quantum lattice system.
"""
@inline eigvals(tba::Algorithm{<:TBA}, k::Union{ReciprocalSpace, AbstractVector{<:Number}, Nothing}=nothing; kwargs...) = eigvals(tba.frontend, k; timer=tba.timer, kwargs...)
@inline function eigvals(tba::TBA, k::Union{AbstractVector{<:Number}, Nothing}=nothing; timer::TimerOutput=tbatimer, kwargs...)
    @timeit_debug timer "eigvals" begin
        @timeit_debug timer "matrix" (m = matrix(tba, k; kwargs...))
        @timeit_debug timer "values" (eigenvalues = eigvals(m))
    end
    return eigenvalues
end
@inline eigvals(tba::TBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, kwargs...) = [eigvals(tba, momentum; timer=timer, kwargs...) for momentum in reciprocalspace]

"""
    eigvecs(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; kwargs...) -> Matrix{<:Number}
    eigvecs(tba::Union{TBA, Algorithm{<:TBA}}, reciprocalspace::ReciprocalSpace; kwargs...) -> Vector{<:Matrix{<:Number}}

Get the eigen vectors of a free quantum lattice system.
"""
@inline eigvecs(tba::Algorithm{<:TBA}, k::Union{ReciprocalSpace, AbstractVector{<:Number}, Nothing}=nothing; kwargs...) = eigvecs(tba.frontend, k; timer=tba.timer, kwargs...)
@inline function eigvecs(tba::TBA, k::Union{AbstractVector{<:Number}, Nothing}=nothing; timer::TimerOutput=tbatimer, kwargs...)
    @timeit_debug timer "eigvecs" begin
        @timeit_debug timer "matrix" (m = matrix(tba, k; kwargs...))
        @timeit_debug timer "vectors" (eigenvectors = eigvecs(m))
    end
    return eigenvectors
end
@inline eigvecs(tba::TBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, kwargs...) = [eigvecs(tba, momentum; timer=timer, kwargs...) for momentum in reciprocalspace]

"""
    SimpleTBA{
        K<:TBAKind,
        L<:Union{AbstractLattice, Nothing},
        H<:Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}},
        C<:Union{AbstractMatrix, Nothing}
    } <:TBA{K, H, C}

Simple tight-binding approximation for quantum lattice systems.
"""
struct SimpleTBA{
        K<:TBAKind,
        L<:Union{AbstractLattice, Nothing},
        H<:Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}},
        C<:Union{AbstractMatrix, Nothing}
    } <:TBA{K, H, C}
    lattice::L
    H::H
    commutator::C
    function SimpleTBA{K}(
        lattice::Union{AbstractLattice, Nothing},
        H::Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}},
        commutator::Union{AbstractMatrix, Nothing}
    ) where {K<:TBAKind}
        checkcommutator(commutator)
        new{K, typeof(lattice), typeof(H), typeof(commutator)}(lattice, H, commutator)
    end
end
@inline function update!(tba::SimpleTBA; parameters...)
    if length(parameters)>0
        update!(tba.H; parameters...)
    end
    return tba
end
@inline checkcommutator(::Nothing) = nothing
function checkcommutator(commutator::AbstractMatrix)
    values = eigvals(commutator)
    num₁ = count(isapprox(+1, atol=atol, rtol=rtol), values)
    num₂ = count(isapprox(-1, atol=atol, rtol=rtol), values)
    @assert num₁==num₂==length(values)//2 "checkcommutator error: unsupported input commutator."
end

"""
    CompositeTBA{
        K<:TBAKind,
        L<:Union{AbstractLattice, Nothing},
        S<:Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}},
        Q<:Quadraticization,
        H<:Union{OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}},
        C<:Union{AbstractMatrix, Nothing}
    } <:TBA{K, H, C}

Composite tight-binding approximation for quantum lattice systems.
"""
struct CompositeTBA{
        K<:TBAKind,
        L<:Union{AbstractLattice, Nothing},
        S<:Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}},
        Q<:Quadraticization,
        H<:Union{OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}},
        C<:Union{AbstractMatrix, Nothing}
    } <:TBA{K, H, C}
    lattice::L
    system::S
    quadraticization::Q
    H::H
    commutator::C
    function CompositeTBA{K}(
        lattice::Union{AbstractLattice, Nothing},
        system::Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}},
        quadraticization::Quadraticization,
        commutator::Union{AbstractMatrix, Nothing}
    ) where {K<:TBAKind}
        checkcommutator(commutator)
        H = quadraticization(system)
        new{K, typeof(lattice), typeof(system), typeof(quadraticization), typeof(H), typeof(commutator)}(lattice, system, quadraticization, H, commutator)
    end
end
@inline dimension(tba::CompositeTBA) = length(tba.quadraticization.table)
@inline function update!(tba::CompositeTBA; parameters...)
    if length(parameters)>0
        update!(tba.system; parameters...)
        update!(tba.H; parameters...)
    end
    return tba
end

"""
    TBA{K}(H::Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    TBA{K}(lattice::Union{AbstractLattice, Nothing}, H::Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    TBA{K}(H::Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}}, q::Quadraticization, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    TBA{K}(lattice::Union{AbstractLattice, Nothing}, H::Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}}, q::Quadraticization, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Union{Term, Tuple{Term, Vararg{Term}}}, boundary::Boundary=plain; neighbors::Union{Nothing, Int, Neighbors}=nothing)

Construct a tight-binding quantum lattice system.
"""
@inline function TBA{K}(H::Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    return TBA{K}(nothing, H, commutator)
end
@inline function TBA{K}(lattice::Union{AbstractLattice, Nothing}, H::Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    return SimpleTBA{K}(lattice, H, commutator)
end
@inline function TBA{K}(H::Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}}, q::Quadraticization, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    return TBA{K}(nothing, H, q, commutator)
end
@inline function TBA{K}(lattice::Union{AbstractLattice, Nothing}, H::Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}}, q::Quadraticization, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    return CompositeTBA{K}(lattice, H, q, commutator)
end
@inline function TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Union{Term, Tuple{Term, Vararg{Term}}}, boundary::Boundary=plain; neighbors::Union{Nothing, Int, Neighbors}=nothing)
    terms = wrapper(terms)
    tbakind = TBAKind(typeof(terms), valtype(hilbert))
    table = Table(hilbert, Metric(tbakind, hilbert))
    commt = commutator(tbakind, hilbert)
    isnothing(neighbors) && (neighbors = maximum(term->term.bondkind, terms))
    H = Generator(terms, bonds(lattice, neighbors), hilbert, boundary, lazy; half=false)
    return TBA{typeof(tbakind)}(lattice, H, Quadraticization{typeof(tbakind)}(table), commt)
end
@inline wrapper(x) = (x,)
@inline wrapper(xs::Tuple) = xs

"""
    const basicoptions = Dict(
        :gauge => "gauge used to perform the Fourier transformation",
        :infinitesimal => "infinitesimal added to the diagonal of the matrix representation of the tight-binding Hamiltonian"
    )

Basic options of tight-binding actions.
"""
const basicoptions = Dict(
    :gauge => "gauge used to perform the Fourier transformation",
    :infinitesimal => "infinitesimal added to the diagonal of the matrix representation of the tight-binding Hamiltonian"
)

"""
    EnergyBands{P<:ReciprocalSpace, L<:Union{Colon, AbstractVector{Int}}, O} <: Action

Energy bands by tight-binding-approximation for quantum lattice systems.
"""
struct EnergyBands{P<:ReciprocalSpace, L<:Union{Colon, AbstractVector{Int}}, O} <: Action
    reciprocalspace::P
    bands::L
    options::O
end
@inline options(::Type{<:EnergyBands}) = merge(basicoptions, Dict(
    :tol => "maximum tolerance of the imaginary part of eigen energies"
))
@inline function EnergyBands(reciprocalspace::ReciprocalSpace, bands::Union{Colon, AbstractVector{Int}}=Colon(); options...)
    checkoptions(EnergyBands; options...)
    return EnergyBands(reciprocalspace, bands, options)
end
@inline initialize(eb::EnergyBands{P, Colon}, tba::TBA) where {P<:ReciprocalSpace} = (eb.reciprocalspace, zeros(Float64, length(eb.reciprocalspace), dimension(tba)))
@inline initialize(eb::EnergyBands{P, <:AbstractVector{Int}}, ::TBA) where {P<:ReciprocalSpace} = (eb.reciprocalspace, zeros(Float64, length(eb.reciprocalspace), length(eb.bands)))
@inline Base.nameof(tba::Algorithm{<:TBA}, eb::Assignment{<:EnergyBands}) = @sprintf "%s_%s" repr(tba; context=:select=>∉(names(eb.action.reciprocalspace))) eb.id
function run!(tba::Algorithm{<:TBA}, eb::Assignment{<:EnergyBands})
    tol = get(eb.action.options, :tol, 10^-12)
    for (i, k) in enumerate(eb.action.reciprocalspace)
        eigenvalues = eigvals(tba, k; eb.action.options...)[eb.action.bands]
        norm(imag(eigenvalues))>tol && @warn("run! warning: imaginary eigen energies at $k with the norm of all imaginary parts being $(norm(imag(eigenvalues))).")
        eb.data[2][i, :] = real(eigenvalues)
    end
end

"""
    abstract type BerryCurvatureMethod end

Abstract type for calculation of Berry curvature.
"""
abstract type BerryCurvatureMethod end

"""
    Fukui <: BerryCurvatureMethod

Fukui method to calculate Berry curvature of energy bands. see [Fukui et al, JPSJ 74, 1674 (2005)](https://journals.jps.jp/doi/10.1143/JPSJ.74.1674). Hall conductivity (single band) is given by 
```math
\\sigma_{xy} = -{e^2\\over h}\\sum_n c_n, \\nonumber \\\\
c_n = {1\\over 2\\pi}\\int_{BZ}{dk_x dk_y Ω_{xy}}, 
\\Omega_{xy}=(\\nabla\\times {\\bm A})_z,
A_{x}=i\\langle u_n|\\partial_x|u_n\\rangle.
```
"""
struct Fukui <: BerryCurvatureMethod
    bands::Vector{Int}
    abelian::Bool
end
@inline Fukui(bands::AbstractVector{Int}; abelian::Bool=true) = Fukui(collect(bands), abelian)

"""
    Kubo{K<:Union{Nothing, Vector{Float64}}} <: BerryCurvatureMethod
    
Kubo method to calculate the total Berry curvature of occupied energy bands. The Kubo formula is given by
```math
\\Omega_{ij}(\\bm k)=\\epsilon_{ijl}\\sum_{n}f(\\epsilon_n(\\bm k))b_n^l(\\bm k)=-2{\\rm Im}\\sum_v\\sum_c{V_{vc,i}(\\bm k)V_{cv,j}(\\bm k)\\over [\\omega_c(\\bm k)-\\omega_v(\\bm k)]^2},
```
where
```math
 V_{cv,j}={\\langle u_{c\\bm k}|{\\partial H\\over \\partial {\\bm k}_j}|u_{v\\bm k}\\rangle}
```
v and c subscripts denote valence (occupied) and conduction (unoccupied) bands, respectively.
Hall conductivity in 2D space is given by
```math
\\sigma_{xy}=-{e^2\\over h}\\int_{BZ}{dk_x dk_y\\over 2\\pi}{\\Omega_{xy}}
```
"""
struct Kubo{K<:Union{Nothing, Vector{Float64}}} <: BerryCurvatureMethod 
    μ::Float64
    d::Float64
    kx::K
    ky::K
end
@inline Kubo(μ::Real; d::Float64=0.1, kx::T=nothing, ky::T=nothing) where {T<:Union{Nothing, Vector{Float64}}} = Kubo(convert(Float64, μ), d, kx, ky)

"""
    BerryCurvature{B<:ReciprocalSpace, M<:BerryCurvatureMethod, O} <: Action

Berry curvature of energy bands.

!!! note
    To obtain a rotation-symmetric Berry curvature, the `:rcoordinate` gauge should be used. Otherwise, artificial slight rotation symmetry breaking will occur.
"""
struct BerryCurvature{B<:ReciprocalSpace, M<:BerryCurvatureMethod, O} <: Action
    reciprocalspace::B
    method::M
    options::O
end
@inline options(::Type{<:BerryCurvature}) = basicoptions
@inline function BerryCurvature(reciprocalspace::ReciprocalSpace, method::BerryCurvatureMethod; gauge=:rcoordinate, options...)
    checkoptions(BerryCurvature; options...)
    return BerryCurvature(reciprocalspace, method, (gauge=gauge, options...))
end
@inline function BerryCurvature(reciprocalspace::ReciprocalSpace, μ::Real, d::Real=0.1, kx::T=nothing, ky::T=nothing; gauge=:rcoordinate, options...) where {T<:Union{Nothing, Vector{Float64}}}
    checkoptions(BerryCurvature; options...)
    return BerryCurvature(reciprocalspace, Kubo(μ, d, kx, ky), (gauge=gauge, options...))
end
@inline function BerryCurvature(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, bands::AbstractVector{Int}, abelian::Bool=true; gauge=:rcoordinate, options...)
    checkoptions(BerryCurvature; options...)
    return BerryCurvature(reciprocalspace, Fukui(bands; abelian=abelian), (gauge=gauge, options...))
end

# For the Berry curvature and Chern number (Berry phase ÷ 2π) on the first Brillouin zone
@inline function initialize(bc::BerryCurvature{<:BrillouinZone, <:Fukui}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "initialize error: Berry curvature should be defined for 2d systems."
    ny, nx = map(length, shape(bc.reciprocalspace))
    z, n = bc.method.abelian ? (zeros(Float64, ny, nx, length(bc.method.bands)), zeros(Float64, length(bc.method.bands))) : (zeros(Float64, ny, nx, 1), zeros(Float64, 1))
    return (bc.reciprocalspace, z, n)
end
function eigvecs(tba::Algorithm{<:TBA}, bc::Assignment{<:BerryCurvature{<:BrillouinZone, <:Fukui}})
    eigenvectors = eigvecs(tba, bc.action.reciprocalspace; bc.action.options...)
    ny, nx = map(length, shape(bc.action.reciprocalspace))
    result = Matrix{eltype(eigenvectors)}(undef, nx+1, ny+1)
    for i=1:nx+1, j=1:ny+1
        result[i, j] = eigenvectors[Int(keytype(bc.action.reciprocalspace)(i, j))][:, bc.action.method.bands]
    end
    return result
end

# For the Berry curvature and Berry phase (÷2π) on a specific zone in the reciprocal space
@inline function initialize(bc::BerryCurvature{<:ReciprocalZone, <:Fukui}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "initialize error: Berry curvature should be defined for 2d systems."
    ny, nx = map(length, shape(bc.reciprocalspace))
    z, n = bc.method.abelian ? (zeros(Float64, ny-1, nx-1, length(bc.method.bands)), zeros(Float64, length(bc.method.bands))) : (zeros(Float64, ny-1, nx-1, 1), zeros(Float64, 1))
    return (shrink(bc.reciprocalspace, 1:nx-1, 1:ny-1), z, n)
end
function eigvecs(tba::Algorithm{<:TBA}, bc::Assignment{<:BerryCurvature{<:ReciprocalZone, <:Fukui}})
    eigenvectors = eigvecs(tba, bc.action.reciprocalspace; bc.action.options...)
    ny, nx = map(length, shape(bc.action.reciprocalspace))
    result = Matrix{eltype(eigenvectors)}(undef, nx, ny)
    count = 1
    for i=1:nx, j=1:ny
        result[i, j] = eigenvectors[count][:, bc.action.method.bands]
        count += 1
    end
    return result
end

# For the Berry curvature and Berry phase (÷2π) on the Brillouin zone or reciprocal zone.
@inline function initialize(bc::BerryCurvature{<:Union{ReciprocalZone, BrillouinZone}, <:Kubo}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "initialize error: Berry curvature should be defined for 2d systems."
    ny, nx = map(length, shape(bc.reciprocalspace))
    z = zeros(Float64, ny, nx, 1)
    n = zeros(1)
    return (bc.reciprocalspace, z, n)
end

# For the Berry curvature on a specific path in the reciprocal space.
@inline function initialize(bc::BerryCurvature{<:ReciprocalPath, <:Kubo}, ::TBA)
    np = length(bc.reciprocalspace)
    z = zeros(Float64, np)
    return (bc.reciprocalspace, z)
end
function _minilength(rs::ReciprocalSpace)
    if typeof(rs) <: ReciprocalPath
        d = minimum([step(rs, i) for i in 1:length(rs)-1])
    else
        ny, nx = map(length, shape(rs))
        d = minimum(norm, [rs.reciprocals[1]/nx, rs.reciprocals[2]/ny])
    end
    return d 
end
function _kubo(tba::Algorithm{<:TBA},  bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Kubo}})
    dim = dimension(bc.action.reciprocalspace)
    @assert dim ∈(2, 3) "_eigendHk error: only two-dimensional and three-dimensional reciprocal spaces are supported."
    d, kx, ky = bc.action.method.d, bc.action.method.kx, bc.action.method.ky
    ml = _minilength(bc.action.reciprocalspace)
    if isa(kx, Nothing)
        dx, dy = dim==2 ? (d*ml*[1.0, 0.0], d*ml*[0.0, 1.]) : (d*ml*[1., 0.0, 0.0], d*ml*[0.0, 1., 0.0])
    else
        dx, dy = ml*d*normalize(kx), ml*d*normalize(ky)
    end
    @assert dot(dx, dy) == 0.0 "_eigendHk error: kx vector and ky vector should be perpendicular to each other in the plane."
    μ = bc.action.method.μ
    Ωxys = Float64[] 
    for momentum in bc.action.reciprocalspace
        eigensystem = eigen(tba, momentum; bc.action.options...)
        mx₁, mx₂ = matrix(tba, momentum+dx; bc.action.options...), matrix(tba, momentum-dx; bc.action.options...)
        my₁, my₂ = matrix(tba, momentum+dy; bc.action.options...), matrix(tba, momentum-dy; bc.action.options...)
        dHx = (mx₂ - mx₁)/norm(2*dx)
        dHy = (my₂ - my₁)/norm(2*dy)
        res = 0.0
        for (i, valv) in enumerate(eigensystem.values)
            valv > μ && continue
            vs₁ = eigensystem.vectors[:, i]
            for (j, valc) in enumerate(eigensystem.values)
                valc < μ && continue
                vs₂ = eigensystem.vectors[:, j]
                velocity_x = vs₁'*dHx*vs₂
                velocity_y = vs₂'*dHy*vs₁
                res += -2*imag(velocity_x*velocity_y/(valc-valv)^2)
            end
        end
        push!(Ωxys, res)
    end
    return Ωxys
end

# Compute the Berry curvature and optionally, the Chern number or Berry phase (÷ 2π)
function run!(tba::Algorithm{<:TBA}, bc::Assignment{<:BerryCurvature})
    alg = bc.action.method
    isa(bc.action.reciprocalspace, BrillouinZone) && (area = volume(bc.action.reciprocalspace.reciprocals)/length(bc.action.reciprocalspace))
    isa(bc.action.reciprocalspace, ReciprocalZone) && (area = bc.action.reciprocalspace.volume/length(bc.action.reciprocalspace))
    if isa(alg, Fukui) 
        eigenvectors = eigvecs(tba, bc)
        g = isnothing(getcontent(tba.frontend, :commutator)) ? Diagonal(ones(Int, dimension(tba))) : inv(getcontent(tba.frontend, :commutator))
        if alg.abelian
            @timeit_debug tba.timer "Berry curvature" for i = 1:size(eigenvectors)[1]-1, j = 1:size(eigenvectors)[2]-1
                vs₁ = eigenvectors[i, j]
                vs₂ = eigenvectors[i+1, j]
                vs₃ = eigenvectors[i+1, j+1]
                vs₄ = eigenvectors[i, j+1]
                for k = 1:length(alg.bands)
                    p₁ = vs₁[:, k]'*g*vs₂[:, k]
                    p₂ = vs₂[:, k]'*g*vs₃[:, k]
                    p₃ = vs₃[:, k]'*g*vs₄[:, k]
                    p₄ = vs₄[:, k]'*g*vs₁[:, k]
                    bc.data[2][j, i, k] = -angle(p₁*p₂*p₃*p₄)/area
                    length(bc.data)==3 && (bc.data[3][k] += bc.data[2][j, i, k]*area/2pi)
                end
            end
        else
            @timeit_debug tba.timer "Berry curvature" for i = 1:size(eigenvectors)[1]-1, j = 1:size(eigenvectors)[2]-1
                vs₁ = eigenvectors[i, j]
                vs₂ = eigenvectors[i+1, j]
                vs₃ = eigenvectors[i+1, j+1]
                vs₄ = eigenvectors[i, j+1]
                p₁ = (vs₁'*g*vs₂)
                p₂ = (vs₂'*g*vs₃)
                p₃ = (vs₃'*g*vs₄)
                p₄ = (vs₄'*g*vs₁)
                bc.data[2][j, i, 1] = -imag(logdet(p₁*p₂*p₃*p₄))/area
                length(bc.data)==3 && (bc.data[3][1] += bc.data[2][j, i, 1]*area/2pi)
            end
            @warn "This method (non-abelian case for `Fukui` method) is not verified in bosonic system."
        end
    else isa(alg, Kubo)
        Ωxys = _kubo(tba, bc)
        if typeof(bc.action.reciprocalspace) <: Union{ReciprocalZone, BrillouinZone}
            ny, nx = map(length, shape(bc.action.reciprocalspace))
            bc.data[2][:, :, 1] = reshape(Ωxys, ny, nx)
            bc.data[3][1] =  sum(Ωxys*area)/2pi
        else 
            bc.data[2][:] = Ωxys[:]
        end
    end
    length(bc.data)==4 && @info (@sprintf "Chern numbers: %s" join([@sprintf "%s(%s)" cn band for (cn, band) in zip(bc.data[4], bc.action.bands)], ", "))
end

# Plot the Berry curvature and optionally, the Chern number or Berry phase (÷2π)
@recipe function plot(pack::Tuple{Algorithm{<:TBA}, Assignment{<:BerryCurvature}})
    subtitles --> if length(pack[2].data)==3
        if isa(pack[2].action.method, Fukui)
            if pack[2].action.method.abelian 
                [@sprintf("band %s (ϕ/2π = %s)", band, tostr(chn)) for (band, chn) in zip(pack[2].action.method.bands, pack[2].data[3])]
            else
                [@sprintf("sum bands %s (ϕ/2π = %s)", pack[2].action.method.bands, tostr(pack[2].data[3][1]))]
            end
        else
            [@sprintf("occupied bands (μ = %s) (ϕ/2π = %s)", pack[2].action.method.μ, tostr(pack[2].data[3][1]))]
        end
    else
        [@sprintf("occupied band (μ = %s)", pack[2].action.method.μ)]
    end
    subtitlefontsize --> 8
    plot_title --> nameof(pack[1], pack[2])
    plot_titlefontsize --> 10
    pack[2].data[1:2]
end

function spectralfunction(tbakind::TBAKind, ω::Real, values::Vector{<:Real}, vectors::Matrix{<:Number}, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::Union{Colon, AbstractVector{Int}}=:; σ::Real)
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
    FermiSurface{B<:Union{BrillouinZone, ReciprocalZone}, A<:Union{Colon, AbstractVector{Int}}, L<:Tuple{Vararg{Union{Colon, AbstractVector{Int}}}}, O} <: Action

Fermi surface of a free fermionic system.
"""
struct FermiSurface{B<:Union{BrillouinZone, ReciprocalZone}, A<:Union{Colon, AbstractVector{Int}}, L<:Tuple{Vararg{Union{Colon, AbstractVector{Int}}}}, O} <: Action
    reciprocalspace::B
    μ::Float64
    bands::A
    orbitals::L
    options::O
end
@inline options(::Type{<:FermiSurface}) = merge(basicoptions, Dict(
    :fwhm => "full width at half maximum for the Gaussian broadening"
))
@inline function FermiSurface(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, μ::Real=0.0, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::Union{Colon, AbstractVector{Int}}...=:; options...)
    checkoptions(FermiSurface; options...)
    return FermiSurface(reciprocalspace, convert(Float64, μ), bands, orbitals, options)
end
function initialize(fs::FermiSurface, ::TBA)
    @assert length(fs.reciprocalspace.reciprocals)==2 "initialize error: only two dimensional reciprocal spaces are supported."
    ny, nx = map(length, shape(fs.reciprocalspace))
    z = zeros(Float64, ny, nx, length(fs.orbitals))
    return (fs.reciprocalspace, z)
end
function run!(tba::Algorithm{<:TBA{<:Fermionic{:TBA}}}, fs::Assignment{<:FermiSurface})
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
@recipe function plot(pack::Tuple{Algorithm{<:TBA}, Assignment{<:FermiSurface}})
    if size(pack[2].data[2])[3]==1
        title --> nameof(pack...)
        titlefontsize --> 10
        pack[2].data[1], pack[2].data[2][:, :, 1]
    else
        subtitles --> [@sprintf("orbitals: %s\n bands: %s", str(orbitals), str(pack[2].action.bands)) for orbitals in pack[2].action.orbitals]
        subtitlefontsize --> 8
        plot_title --> nameof(pack[1], pack[2])
        plot_titlefontsize --> 10
        pack[2].data
    end
end
@inline str(::Colon) = "all"
@inline str(contents::AbstractVector{Int}) = join(contents, ", ")

"""
    DensityOfStates{B<:BrillouinZone, A<:Union{Colon, AbstractVector{Int}}, L<:Tuple{Vararg{Union{Colon, AbstractVector{Int}}}}, O} <: Action

Density of states of a tight-binding system.
"""
struct DensityOfStates{B<:BrillouinZone, A<:Union{Colon, AbstractVector{Int}}, L<:Tuple{Vararg{Union{Colon, AbstractVector{Int}}}}, O} <: Action
    brillouinzone::B
    bands::A
    orbitals::L
    options::O
end
@inline options(::Type{<:DensityOfStates}) = merge(basicoptions, Dict(
    :fwhm => "full width at half maximum for the Gaussian broadening",
    :ne => "number of energy sample points",
    :emin => "minimum value of the energy window",
    :emax => "maximum value of the energy window"
))
@inline function DensityOfStates(brillouinzone::BrillouinZone, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::Union{Colon, AbstractVector{Int}}...=:; options...)
    checkoptions(DensityOfStates; options...)
    return DensityOfStates(brillouinzone, bands, orbitals, options)
end
@inline function initialize(dos::DensityOfStates, ::TBA)
    ne = get(dos.options, :ne, 100)
    x = zeros(Float64, ne)
    z = zeros(Float64, ne, length(dos.orbitals))
    return (x, z)
end
function run!(tba::Algorithm{<:TBA{<:Fermionic{:TBA}}}, dos::Assignment{<:DensityOfStates})
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
@inline options(::Type{<:InelasticNeutronScatteringSpectra}) = merge(basicoptions, Dict(
    :fwhm => "full width at half maximum for the Gaussian broadening",
    :check => "whether the polarization consistency of phonons will be checked",
    :rescale => "function used to rescale the intensity of the spectrum at each energy-momentum point"
))
@inline function InelasticNeutronScatteringSpectra(reciprocalspace::ReciprocalSpace, energies::AbstractVector; options...)
    checkoptions(InelasticNeutronScatteringSpectra; options...)
    return InelasticNeutronScatteringSpectra(reciprocalspace, energies, options)
end
@inline initialize(inss::InelasticNeutronScatteringSpectra, ::TBA) = (inss.reciprocalspace, inss.energies, zeros(Float64, length(inss.energies), length(inss.reciprocalspace)))

# Inelastic neutron scattering spectra for phonons.
function run!(tba::Algorithm{<:CompositeTBA{Phononic, <:AbstractLattice}}, inss::Assignment{<:InelasticNeutronScatteringSpectra})
    dim = dimension(tba)
    σ = get(inss.action.options, :fwhm, 0.1)/2/√(2*log(2))
    check = get(inss.action.options, :check, true)
    sequences = Dict(site=>[tba.frontend.quadraticization.table[Index(site, PhononIndex{:u}(Char(Int('x')+i-1)))] for i=1:phonon.ndirection] for (site, phonon) in pairs(tba.frontend.system.hilbert))
    eigenvalues, eigenvectors = eigen(tba, inss.action.reciprocalspace; inss.action.options...)
    for (i, (momentum, values, vectors)) in enumerate(zip(inss.action.reciprocalspace, eigenvalues, eigenvectors))
        check && @timeit_debug tba.timer "check" checkpolarizations(@views(vectors[(dim÷2+1):dim, 1:(dim÷2)]), @views(vectors[(dim÷2+1):dim, dim:-1:(dim÷2+1)]), momentum./pi)
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
    inss.data[3][:, :] = get(inss.action.options, :rescale, identity).(inss.data[3])
end
@inline function checkpolarizations(qs₁::AbstractMatrix, qs₂::AbstractMatrix, momentum)
    inner = mapreduce((e₁, e₂)->norm(conj(e₁)*e₂), +, qs₁, qs₂)/norm(qs₁)/norm(qs₂)
    isapprox(inner, 1; atol=100*atol, rtol=100*rtol) || begin
        @warn("checkpolarizations: small inner product $inner at π*$momentum, indication of degeneracy, otherwise inconsistent polarization vectors.")
    end
end
