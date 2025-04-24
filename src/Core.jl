using Contour: contour, coordinates, lines
using LinearAlgebra: Diagonal, Eigen, cholesky, dot, inv, norm, logdet, normalize
using Printf: @printf, @sprintf
using QuantumLattices: atol, lazy, plain, rtol
using QuantumLattices: AbstractLattice, Action, Algorithm, Assignment, BrillouinZone, Boundary, CoordinatedIndex, Elastic, FockIndex, Fock, Formula, Frontend, Generator, Hilbert, Hooke, Hopping, ID, Index, Internal, Kinetic, LinearTransformation, Matrixization, Metric, Neighbors, OneOrMore, Onsite, Operator, OperatorIndexToTuple, OperatorPack, OperatorSet, OperatorSum, Pairing, Phonon, PhononIndex, ReciprocalPath, ReciprocalScatter, ReciprocalSpace, ReciprocalZone, Term
using QuantumLattices: ⊕, bonds, expand, icoordinate, idtype, isannihilation, iscreation, label, nneighbor, operatortype, parametertype, rank, rcoordinate, shape, shrink, statistics, tostr, volume
using RecipesBase: RecipesBase, @recipe, @series
using TimerOutputs: TimerOutput, @timeit_debug

import LinearAlgebra: eigen, eigvals, eigvecs, ishermitian, Hermitian
import QuantumLattices: Data, Parameters, Table, add!, dimension, getcontent, kind, matrix, options, parameternames, run!, scalartype, update!

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
@inline Base.promote_rule(::Type{Fermionic{:TBA}}, ::Type{Fermionic{:BdG}}) = Fermionic{:BdG}
@inline Base.promote_rule(::Type{Fermionic{:BdG}}, ::Type{Fermionic{:TBA}}) = Fermionic{:BdG}

"""
    Bosonic{K} <: TBAKind{K}

Bosonic quantum lattice system.
"""
struct Bosonic{K} <: TBAKind{K} end
@inline Bosonic(k::Symbol) = Bosonic{k}()
@inline Base.promote_rule(::Type{Bosonic{:TBA}}, ::Type{Bosonic{:BdG}}) = Bosonic{:BdG}
@inline Base.promote_rule(::Type{Bosonic{:BdG}}, ::Type{Bosonic{:TBA}}) = Bosonic{:BdG}

"""
    TBAKind(T::Type{<:Term}, I::Type{<:Internal})

Depending on the kind of a `Term` type and an `Internal` type, get the corresponding TBA kind.
"""
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
    Metric(::Fermionic, hilbert::Hilbert{<:Fock{:f}}) -> OperatorIndexToTuple
    Metric(::Bosonic, hilbert::Hilbert{<:Fock{:b}}) -> OperatorIndexToTuple
    Metric(::Phononic, hilbert::Hilbert{<:Phonon}) -> OperatorIndexToTuple

Get the index-to-tuple metric for a free fermionic/bosonic/phononic system.
"""
@inline @generated Metric(::Fermionic{:TBA}, hilbert::Hilbert{<:Fock{:f}}) = OperatorIndexToTuple(:site, :orbital, :spin)
@inline @generated Metric(::Bosonic{:TBA}, hilbert::Hilbert{<:Fock{:b}}) = OperatorIndexToTuple(:site, :orbital, :spin)
@inline @generated Metric(::Fermionic{:BdG}, hilbert::Hilbert{<:Fock{:f}}) = OperatorIndexToTuple(:nambu, :site, :orbital, :spin)
@inline @generated Metric(::Bosonic{:BdG}, hilbert::Hilbert{<:Fock{:b}}) = OperatorIndexToTuple(:nambu, :site, :orbital, :spin)
@inline @generated Metric(::Phononic, hilbert::Hilbert{<:Phonon}) = OperatorIndexToTuple(kind, :site, :direction)

"""
    Table(hilbert::Hilbert{Phonon{:}}, by::OperatorIndexToTuple{(kind, :site, :direction)})

Construct a index-sequence table for a phononic system.
"""
@inline function Table(hilbert::Hilbert{Phonon{:}}, by::OperatorIndexToTuple{(kind, :site, :direction)})
    new = Hilbert(site=>filter(PhononIndex{:u}, internal)⊕filter(PhononIndex{:p}, internal) for (site, internal) in hilbert)
    return Table(new, by)
end

"""
    commutator(k::TBAKind, hilbert::Hilbert{<:Internal}) -> Union{AbstractMatrix, Nothing}

Get the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.
"""
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
    P = operatortype(O)
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
    if m.position[1] == m.position[2]
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
    V = inv(W.U) * K.vectors * sqrt(Diagonal(K.values))
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
@inline scalartype(::Type{<:TBA{<:TBAKind, H}}) where {H<:Union{Formula, OperatorSet, Generator, Frontend}} = scalartype(H)
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
    matrix(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba.frontend))) -> TBAMatrix

Get the matrix representation of a free quantum lattice system.
"""
@inline function matrix(tba::Algorithm{<:TBA}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba.frontend)))
    return matrix(tba.frontend, k; gauge=gauge, infinitesimal=infinitesimal)
end
@inline function matrix(tba::TBA{<:TBAKind, <:Formula}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba)))
    m = getcontent(tba, :H)(k; gauge=gauge, infinitesimal=infinitesimal)
    commutator = getcontent(tba, :commutator)
    return TBAMatrix(Hermitian(m), commutator)
end
@inline function matrix(
    tba::TBA{<:TBAKind, <:Union{OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing;
    gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba))
)
    matrixization = TBAMatrixization{datatype(scalartype(tba), k)}(k, dimension(tba), gauge)
    m = matrixization(expand(getcontent(tba, :H)); infinitesimal=infinitesimal)
    commutator = getcontent(tba, :commutator)
    return TBAMatrix(Hermitian(m), commutator)
end
@inline function matrix(tba::TBA{<:TBAKind, <:Frontend}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; gauge=:icoordinate, infinitesimal=infinitesimal(kind(tba)))
    return matrix(getcontent(tba, :H); gauge=gauge, infinitesimal=infinitesimal)
end
@inline datatype(::Type{D}, ::Nothing) where D = D
@inline datatype(::Type{D}, ::Any) where D = promote_type(D, Complex{Int})

"""
    eigen(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; options...) -> Eigen
    eigen(tba::Union{TBA, Algorithm{<:TBA}}, reciprocalspace::ReciprocalSpace; options...) -> Tuple{Vector{<:Vector{<:Number}}, Vector{<:Matrix{<:Number}}}

Get the eigen values and eigen vectors of a free quantum lattice system.
"""
@inline eigen(tba::Algorithm{<:TBA}, k::Union{ReciprocalSpace, AbstractVector{<:Number}, Nothing}=nothing; options...) = eigen(tba.frontend, k; timer=tba.timer, options...)
@inline function eigen(tba::TBA, k::Union{AbstractVector{<:Number}, Nothing}=nothing; timer::TimerOutput=tbatimer, options...)
    @timeit_debug timer "eigen" begin
        @timeit_debug timer "matrix" (m = matrix(tba, k; options...))
        @timeit_debug timer "diagonalization" (eigensystem = eigen(m))
    end
    return eigensystem
end
function eigen(tba::TBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, options...)
    datatype = eltype(eltype(reciprocalspace))
    values, vectors = Vector{datatype}[], Matrix{promote_type(datatype, Complex{Int})}[]
    for momentum in reciprocalspace
        eigensystem = eigen(tba, momentum; timer=timer, options...)
        push!(values, eigensystem.values)
        push!(vectors, eigensystem.vectors)
    end
    return values, vectors
end

"""
    eigvals(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; options...) -> Vector{<:Number}
    eigvals(tba::Union{TBA, Algorithm{<:TBA}}, reciprocalspace::ReciprocalSpace; options...) -> Vector{<:Vector{<:Number}}

Get the eigen values of a free quantum lattice system.
"""
@inline eigvals(tba::Algorithm{<:TBA}, k::Union{ReciprocalSpace, AbstractVector{<:Number}, Nothing}=nothing; options...) = eigvals(tba.frontend, k; timer=tba.timer, options...)
@inline function eigvals(tba::TBA, k::Union{AbstractVector{<:Number}, Nothing}=nothing; timer::TimerOutput=tbatimer, options...)
    @timeit_debug timer "eigvals" begin
        @timeit_debug timer "matrix" (m = matrix(tba, k; options...))
        @timeit_debug timer "values" (eigenvalues = eigvals(m))
    end
    return eigenvalues
end
@inline eigvals(tba::TBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, options...) = [eigvals(tba, momentum; timer=timer, options...) for momentum in reciprocalspace]

"""
    eigvecs(tba::Union{TBA, Algorithm{<:TBA}}, k::Union{AbstractVector{<:Number}, Nothing}=nothing; options...) -> Matrix{<:Number}
    eigvecs(tba::Union{TBA, Algorithm{<:TBA}}, reciprocalspace::ReciprocalSpace; options...) -> Vector{<:Matrix{<:Number}}

Get the eigen vectors of a free quantum lattice system.
"""
@inline eigvecs(tba::Algorithm{<:TBA}, k::Union{ReciprocalSpace, AbstractVector{<:Number}, Nothing}=nothing; options...) = eigvecs(tba.frontend, k; timer=tba.timer, options...)
@inline function eigvecs(tba::TBA, k::Union{AbstractVector{<:Number}, Nothing}=nothing; timer::TimerOutput=tbatimer, options...)
    @timeit_debug timer "eigvecs" begin
        @timeit_debug timer "matrix" (m = matrix(tba, k; options...))
        @timeit_debug timer "vectors" (eigenvectors = eigvecs(m))
    end
    return eigenvectors
end
@inline eigvecs(tba::TBA, reciprocalspace::ReciprocalSpace; timer::TimerOutput=tbatimer, options...) = [eigvecs(tba, momentum; timer=timer, options...) for momentum in reciprocalspace]

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
        update!(tba.H, tba.quadraticization, tba.system; parameters...)
    end
    return tba
end

"""
    TBA{K}(H::Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    TBA{K}(lattice::Union{AbstractLattice, Nothing}, H::Union{Formula, OperatorSet{<:Quadratic}, Generator{<:OperatorSet{<:Quadratic}}}, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}

    TBA{K}(H::Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}}, q::Quadraticization, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}
    TBA{K}(lattice::Union{AbstractLattice, Nothing}, H::Union{OperatorSet{<:Operator}, Generator{<:OperatorSet{<:Operator}}}, q::Quadraticization, commutator::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}

    TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    TBA{K}(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain; neighbors::Union{Int, Neighbors}=nneighbor(terms)) where {K<:TBAKind}

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
@inline function TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    K = typeof(TBAKind(typeof(terms), valtype(hilbert)))
    return TBA{K}(lattice, hilbert, terms, boundary; neighbors=neighbors)
end
@inline function TBA{K}(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, boundary::Boundary=plain; neighbors::Union{Int, Neighbors}=nneighbor(terms)) where {K<:TBAKind}
    H = Generator(bonds(lattice, neighbors), hilbert, terms, boundary, lazy; half=false)
    quadraticization = Quadraticization{K}(Table(hilbert, Metric(K(), hilbert)))
    commt = commutator(K(), hilbert)
    return TBA{K}(lattice, H, quadraticization, commt)
end

"""
    const basicoptions = (
        gauge = "gauge used to perform the Fourier transformation",
        infinitesimal = "infinitesimal added to the diagonal of the matrix representation of the tight-binding Hamiltonian"
    )

Basic options of tight-binding actions.
"""
const basicoptions = (
    gauge = "gauge used to perform the Fourier transformation",
    infinitesimal = "infinitesimal added to the diagonal of the matrix representation of the tight-binding Hamiltonian"
)

"""
    EnergyBands{L<:Tuple{Vararg{AbstractVector{Int}}}, R<:ReciprocalSpace, A<:Union{Colon, AbstractVector{Int}}} <: Action

Energy bands by tight-binding-approximation for quantum lattice systems.
"""
struct EnergyBands{L<:Tuple{Vararg{AbstractVector{Int}}}, R<:ReciprocalSpace, A<:Union{Colon, AbstractVector{Int}}} <: Action
    reciprocalspace::R
    bands::A
    orbitals::L
end
@inline options(::Type{<:Assignment{<:EnergyBands}}) = merge(basicoptions, (
    tol = "maximum tolerance of the imaginary part of eigen energies",
))

"""
    EnergyBands(reciprocalspace::ReciprocalSpace, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::AbstractVector{Int}...)
    EnergyBands(reciprocalspace::ReciprocalSpace, bands::Union{Colon, AbstractVector{Int}}, orbitals::Tuple{Vararg{AbstractVector{Int}}})

Construct an `EnergyBands`.
"""
@inline function EnergyBands(reciprocalspace::ReciprocalSpace, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::AbstractVector{Int}...)
    @assert all(>(0), map(length, orbitals)) "EnergyBands error: empty orbitals."
    return EnergyBands(reciprocalspace, bands, orbitals)
end

"""
    EnergyBandsData{R<:ReciprocalSpace, W<:Union{Vector{Matrix{Float64}}, Nothing}} <: Data

Data of energy bands, including:

1) `reciprocalspace::R`: reciprocal space on which the energy bands are computed.
2) `values::Matrix{Float64}`: eigen energies of bands with each column storing the values on the same reciprocal point.
3) `weights::W`: if not `nothing`, weights of several certain sets of orbitals projected onto the energy bands.
"""
struct EnergyBandsData{R<:ReciprocalSpace, W<:Union{Vector{Matrix{Float64}}, Nothing}} <: Data
    reciprocalspace::R
    values::Matrix{Float64}
    weights::W
end

# Ordinary energy bands
@inline function Data(eb::EnergyBands{Tuple{}}, tba::TBA)
    nb = isa(eb.bands, Colon) ? dimension(tba) : length(eb.bands)
    return EnergyBandsData(eb.reciprocalspace, zeros(Float64, length(eb.reciprocalspace), nb), nothing)
end
function run!(tba::Algorithm{<:TBA}, eb::Assignment{<:EnergyBands{Tuple{}}}; tol::Real=10^-12, options...)
    for (i, k) in enumerate(eb.action.reciprocalspace)
        eigenvalues = eigvals(tba, k; options...)[eb.action.bands]
        norm(imag(eigenvalues))>tol && @warn("run! warning: imaginary eigen energies at $k with the norm of all imaginary parts being $(norm(imag(eigenvalues))).")
        eb.data.values[i, :] = real(eigenvalues)
    end
end

# Fat energy bands
@inline function Data(eb::EnergyBands, tba::TBA)
    nk = length(eb.reciprocalspace)
    nb = isa(eb.bands, Colon) ? dimension(tba) : length(eb.bands)
    no = length(eb.orbitals)
    return EnergyBandsData(eb.reciprocalspace, zeros(Float64, nk, nb), [zeros(Float64, nk, nb) for _ in 1:no])
end
function run!(tba::Algorithm{<:TBA}, eb::Assignment{<:EnergyBands}; options...)
    bands = isa(eb.action.bands, Colon) ? (1:dimension(tba.frontend)) : eb.action.bands
    for (i, k) in enumerate(eb.action.reciprocalspace)
        es, vs = eigen(tba, k; options...)
        for (j, band) in enumerate(bands)
            eb.data.values[i, j] = es[band]
            for (l, orbitals) in enumerate(eb.action.orbitals)
                eb.data.weights[l][i, j] = mapreduce(abs2, +, vs[orbitals, band])
            end
        end
    end
end

# Plot energy bands
@recipe function plot(eb::Assignment{<:EnergyBands}; bands=nothing, weightmultiplier=5.0, weightcolors=nothing, weightlabels=nothing)
    title --> string(eb)
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
struct Fukui{Abelian} <: BerryCurvatureMethod
    bands::Vector{Int}
    function Fukui{Abelian}(bands::AbstractVector{<:Integer}) where Abelian
        @assert isa(Abelian, Bool) "Fukui error: `Abelian` must be `true` or `false`."
        new{Abelian}(collect(Int, bands))
    end
end

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
    BerryCurvature{B<:ReciprocalSpace, M<:BerryCurvatureMethod} <: Action

Berry curvature of energy bands.

!!! note
    To obtain a rotation-symmetric Berry curvature, the `:rcoordinate` gauge should be used. Otherwise, artificial slight rotation symmetry breaking will occur.
"""
struct BerryCurvature{B<:ReciprocalSpace, M<:BerryCurvatureMethod} <: Action
    reciprocalspace::B
    method::M
end
@inline options(::Type{<:Assignment{<:BerryCurvature}}) = basicoptions

"""
    BerryCurvature(reciprocalspace::ReciprocalSpace, method::BerryCurvatureMethod)
    BerryCurvature(reciprocalspace::ReciprocalSpace, μ::Real, d::Real=0.1, kx::T=nothing, ky::T=nothing) where {T<:Union{Nothing, Vector{Float64}}}
    BerryCurvature(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, bands::AbstractVector{Int}, abelian::Bool=true)

Construct a `BerryCurvature`.
"""
@inline function BerryCurvature(reciprocalspace::ReciprocalSpace, μ::Real, d::Real=0.1, kx::T=nothing, ky::T=nothing) where {T<:Union{Nothing, Vector{Float64}}}
    return BerryCurvature(reciprocalspace, Kubo(μ, d, kx, ky))
end
@inline function BerryCurvature(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, bands::AbstractVector{Int}, abelian::Bool=true)
    return BerryCurvature(reciprocalspace, Fukui{abelian}(bands))
end

"""
    BerryCurvatureData{R<:ReciprocalSpace, C<:Array{Float64}, N<:Union{Vector{Float64}, Float64, Nothing}} <: Data

Data of Berry curvature, including:

1) `reciprocalspace::R`: reciprocal space on which the Berry curvature is computed.
2) `values::C`: Berry curvature for a certain set of bands.
3) `chernnumber::N`: if not `nothing`, Chern number of each energy band when it is a vector or total Chern number of all bands when it is a number.
"""
mutable struct BerryCurvatureData{R<:ReciprocalSpace, C<:Array{Float64}, N<:Union{Vector{Float64}, Float64, Nothing}} <: Data
    const reciprocalspace::R
    const values::C
    chernnumber::N
end

# Fukui method
## For the Berry curvature and Chern number on the first Brillouin zone
@inline function Data(bc::BerryCurvature{<:BrillouinZone, <:Fukui{true}}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "Data error: Berry curvature should be defined for 2d systems."
    values = zeros(Float64, map(length, reverse(shape(bc.reciprocalspace)))..., length(bc.method.bands))
    chernnumber = zeros(Float64, length(bc.method.bands))
    return BerryCurvatureData(bc.reciprocalspace, values, chernnumber)
end
@inline function Data(bc::BerryCurvature{<:BrillouinZone, <:Fukui{false}}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "Data error: Berry curvature should be defined for 2d systems."
    values = zeros(Float64, map(length, reverse(shape(bc.reciprocalspace)))...)
    chernnumber = 0.0
    return BerryCurvatureData(bc.reciprocalspace, values, chernnumber)
end
function eigenvectors(tba::TBA, bc::BerryCurvature{<:BrillouinZone, <:Fukui}; options...)
    vectors = eigvecs(tba, bc.reciprocalspace; options...)
    nx, ny = map(length, shape(bc.reciprocalspace))
    result = Matrix{eltype(vectors)}(undef, nx+1, ny+1)
    for i=1:nx+1, j=1:ny+1
        result[i, j] = vectors[Int(keytype(bc.reciprocalspace)(i, j))][:, bc.method.bands]
    end
    return result
end

## For the Berry curvature on a generic reciprocal zone
@inline function Data(bc::BerryCurvature{<:ReciprocalZone, <:Fukui{true}}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "Data error: Berry curvature should be defined for 2d systems."
    nx, ny = map(length, shape(bc.reciprocalspace))
    values = zeros(Float64, ny-1, nx-1, length(bc.method.bands))
    return BerryCurvatureData(shrink(bc.reciprocalspace, 1:nx-1, 1:ny-1), values, nothing)
end
@inline function Data(bc::BerryCurvature{<:ReciprocalZone, <:Fukui{false}}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "Data error: Berry curvature should be defined for 2d systems."
    nx, ny = map(length, shape(bc.reciprocalspace))
    values = zeros(Float64, ny-1, nx-1)
    return BerryCurvatureData(shrink(bc.reciprocalspace, 1:nx-1, 1:ny-1), values, nothing)
end
function eigenvectors(tba::TBA, bc::BerryCurvature{<:ReciprocalZone, <:Fukui}; options...)
    vectors = eigvecs(tba, bc.reciprocalspace; options...)
    nx, ny = map(length, shape(bc.reciprocalspace))
    result = Matrix{eltype(vectors)}(undef, nx, ny)
    count = 1
    for i=1:nx, j=1:ny
        result[i, j] = vectors[count][:, bc.method.bands]
        count += 1
    end
    return result
end

## By use of Fukui method, compute the Berry curvature and optionally the Chern number
function run!(tba::Algorithm{<:TBA}, bc::Assignment{<:BerryCurvature{<:Union{BrillouinZone, ReciprocalZone}, <:Fukui{true}}}; options...)
    area = volume(bc.action.reciprocalspace) / length(bc.action.reciprocalspace)
    vectors = eigenvectors(tba.frontend, bc.action; gauge=default_gauge(bc), options...)
    g = invcommutator(tba.frontend)
    @timeit_debug tba.timer "Berry curvature" for i = 1:size(vectors)[1]-1, j = 1:size(vectors)[2]-1
        v₁, v₂, v₃, v₄ = vectors[i, j], vectors[i+1, j], vectors[i+1, j+1], vectors[i, j+1]
        for k = 1:length(bc.action.method.bands)
            p₁ = v₁[:, k]' * g * v₂[:, k]
            p₂ = v₂[:, k]' * g * v₃[:, k]
            p₃ = v₃[:, k]' * g * v₄[:, k]
            p₄ = v₄[:, k]' * g * v₁[:, k]
            bc.data.values[j, i, k] = -angle(p₁*p₂*p₃*p₄) / area
            isnothing(bc.data.chernnumber) || (bc.data.chernnumber[k] += bc.data.values[j, i, k]*area/2pi)
        end
    end
    isa(bc.action.reciprocalspace, BrillouinZone) && @info string("Chern numbers: ", join((string(cn, "(", band, ")") for (cn, band) in zip(bc.data.chernnumber, bc.action.method.bands)), ", "))
end
function run!(tba::Algorithm{<:TBA}, bc::Assignment{<:BerryCurvature{<:Union{BrillouinZone, ReciprocalZone}, <:Fukui{false}}}; options...)
    area = volume(bc.action.reciprocalspace) / length(bc.action.reciprocalspace)
    vectors = eigenvectors(tba.frontend, bc.action; gauge=default_gauge(bc), options...)
    g = invcommutator(tba.frontend)
    @timeit_debug tba.timer "Berry curvature" for i = 1:size(vectors)[1]-1, j = 1:size(vectors)[2]-1
        v₁, v₂, v₃, v₄ = vectors[i, j], vectors[i+1, j], vectors[i+1, j+1], vectors[i, j+1]
        p₁ = v₁' * g * v₂
        p₂ = v₂' * g * v₃
        p₃ = v₃' * g * v₄
        p₄ = v₄' * g * v₁
        bc.data.values[j, i] = -imag(logdet(p₁*p₂*p₃*p₄)) / area
        isnothing(bc.data.chernnumber) || (bc.data.chernnumber += bc.data.values[j, i]*area/2pi)
    end
    isa(bc.action.reciprocalspace, BrillouinZone) && @info string("Total Chern number: ", bc.data.chernnumber)
end
@inline invcommutator(tba::TBA{<:TBAKind, <:Union{Formula, OperatorSet, Generator, Frontend}, Nothing}) = Diagonal(ones(Int, dimension(tba)))
@inline invcommutator(tba::TBA) = inv(getcontent(tba, :commutator))
@inline default_gauge(::Assignment{<:BerryCurvature{<:BrillouinZone, <:Fukui}}) = :icoordinate
@inline default_gauge(::Assignment{<:BerryCurvature{<:ReciprocalZone, <:Fukui}}) = :rcoordinate
@inline default_gauge(::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Kubo}}) = :rcoordinate

## Plot the Berry curvature and optionally the Chern number obtained by Fukui method
@recipe function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{true}}})
    plot_title --> string(bc)
    plot_titlefontsize --> 10
    subtitles --> [@sprintf("band %s %s", band, isnothing(bc.data.chernnumber) ? "" : @sprintf("(C = %s)", tostr(bc.data.chernnumber[i]))) for (i, band) in enumerate(bc.action.method.bands)]
    subtitlefontsize --> 8
    bc.data.reciprocalspace, bc.data.values
end
@recipe function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Fukui{false}}})
    plot_title --> string(bc)
    plot_titlefontsize --> 10
    layout := (1, 1)
    subplot := 1
    title --> @sprintf("sum of bands %s %s", bc.action.method.bands, isnothing(bc.data.chernnumber) ? "" : @sprintf("(C = %s)", tostr(bc.data.chernnumber)))
    plot_title --> string(bc, "\n", info)
    titlefontsize --> 8
    bc.data.reciprocalspace, bc.data.values
end

# Kubo method
## For the Berry curvature and Chern number on the first Brillouin zone
@inline function Data(bc::BerryCurvature{<:BrillouinZone, <:Kubo}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "Data error: Berry curvature should be defined for 2d systems."
    values = zeros(Float64, map(length, reverse(shape(bc.reciprocalspace)))...)
    return BerryCurvatureData(bc.reciprocalspace, values, 0.0)
end

## For the Berry curvature on a generic reciprocal zone
@inline function Data(bc::BerryCurvature{<:ReciprocalZone, <:Kubo}, ::TBA)
    @assert length(bc.reciprocalspace.reciprocals)==2 "Data error: Berry curvature should be defined for 2d systems."
    values = zeros(Float64, map(length, reverse(shape(bc.reciprocalspace)))...)
    return BerryCurvatureData(bc.reciprocalspace, values, nothing)
end

## For the Berry curvature on a generic path in the reciprocal space
@inline function Data(bc::BerryCurvature{<:ReciprocalPath, <:Kubo}, ::TBA)
    values = zeros(Float64, length(bc.reciprocalspace))
    return BerryCurvatureData(bc.reciprocalspace, values, nothing)
end

## By use of Kubo method, compute the Berry curvature and optionally the Chern number
function run!(tba::Algorithm{<:TBA}, bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Kubo}}; options...)
    dim = dimension(bc.action.reciprocalspace)
    @assert dim ∈(2, 3) "run! error: only two-dimensional and three-dimensional reciprocal spaces are supported."
    d, kx, ky = bc.action.method.d, bc.action.method.kx, bc.action.method.ky
    ml = minilength(bc.action.reciprocalspace)
    dx, dy = if isnothing(kx)
        dim==2 ? (d*ml*[1.0, 0.0], d*ml*[0.0, 1.0]) : (d*ml*[1.0, 0.0, 0.0], d*ml*[0.0, 1.0, 0.0])
    else
        ml*d*normalize(kx), ml*d*normalize(ky)
    end
    @assert isapprox(dot(dx, dy), 0.0; atol=atol, rtol=rtol) "run! error: kx vector and ky vector should be perpendicular to each other in the plane."
    μ = bc.action.method.μ
    kwargs = (gauge=default_gauge(bc), options...)
    for (k, momentum) in enumerate(bc.action.reciprocalspace)
        eigensystem = eigen(tba, momentum; kwargs...)
        mx₁, mx₂ = matrix(tba, momentum+dx; kwargs...), matrix(tba, momentum-dx; kwargs...)
        my₁, my₂ = matrix(tba, momentum+dy; kwargs...), matrix(tba, momentum-dy; kwargs...)
        dHx = (mx₂-mx₁) / norm(2*dx)
        dHy = (my₂-my₁) / norm(2*dy)
        for (i, valv) in enumerate(eigensystem.values)
            valv > μ && continue
            v₁ = eigensystem.vectors[:, i]
            for (j, valc) in enumerate(eigensystem.values)
                valc < μ && continue
                v₂ = eigensystem.vectors[:, j]
                vx = v₁' * dHx * v₂
                vy = v₂' * dHy * v₁
                bc.data.values[k] += -2imag(vx*vy/(valc-valv)^2)
            end
        end
    end
    if isa(bc.action.reciprocalspace, BrillouinZone)
        area = volume(bc.action.reciprocalspace) / length(bc.action.reciprocalspace)
        bc.data.chernnumber = sum(bc.data.values*area) / 2pi
        @info (@sprintf "Total Chern number at %s: %s" μ bc.data.chernnumber)
    end
end
@inline function minilength(rs::Union{ReciprocalZone, BrillouinZone})
    nx, ny = map(length, shape(rs))
    return minimum(norm, [rs.reciprocals[1]/nx, rs.reciprocals[2]/ny])
end
@inline minilength(rs::ReciprocalPath) = minimum(step(rs, i) for i in 1:length(rs)-1)

## Plot the Berry curvature and optionally the Chern number obtained by Kubo method
@recipe function plot(bc::Assignment{<:BerryCurvature{<:ReciprocalSpace, <:Kubo}})
    plot_title --> string(bc)
    plot_titlefontsize --> 10
    layout := (1, 1)
    subplot := 1
    title --> @sprintf("bands below %s %s", bc.action.method.μ, isnothing(bc.data.chernnumber) ? "" : @sprintf("(C = %s)", tostr(bc.data.chernnumber)))
    titlefontsize --> 8
    bc.data.reciprocalspace, bc.data.values
end

"""
    FermiSurface{L<:Tuple{Vararg{AbstractVector{Int}}}, B<:Union{BrillouinZone, ReciprocalZone}, A<:Union{Colon, AbstractVector{Int}}} <: Action

Fermi surface of a free fermionic system.
"""
struct FermiSurface{L<:Tuple{Vararg{AbstractVector{Int}}}, B<:Union{BrillouinZone, ReciprocalZone}, A<:Union{Colon, AbstractVector{Int}}} <: Action
    reciprocalspace::B
    μ::Float64
    bands::A
    orbitals::L
end
@inline options(::Type{<:Assignment{<:FermiSurface}}) = basicoptions

"""
    FermiSurface(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, μ::Real=0.0, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::AbstractVector{Int}...)
    FermiSurface(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, μ::Real, bands::Union{Colon, AbstractVector{Int}}, orbitals::Tuple{Vararg{AbstractVector{Int}}})

Construct a `FermiSurface`.
"""
@inline function FermiSurface(reciprocalspace::Union{BrillouinZone, ReciprocalZone}, μ::Real=0.0, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::AbstractVector{Int}...)
    return FermiSurface(reciprocalspace, convert(Float64, μ), bands, orbitals)
end

"""
    FermiSurfaceData{S<:ReciprocalScatter} <: Data

Data of Fermi surface, including:

1) `values::S`: points of the Fermi surface.
2) `weights::Vector{Vector{Float64}}`: weights of several certain sets of orbitals projected onto the Fermi surface.
"""
struct FermiSurfaceData{S<:ReciprocalScatter} <: Data
    values::S
    weights::Vector{Vector{Float64}}
end
function Data(fs::FermiSurface, ::TBA)
    @assert length(fs.reciprocalspace.reciprocals)==2 "Data error: only reciprocal spaces with two reciprocal vectors are supported."
    values = ReciprocalScatter{label(fs.reciprocalspace)}(fs.reciprocalspace.reciprocals, eltype(fs.reciprocalspace.reciprocals)[])
    weights = Vector{Float64}[Float64[] for _ in 1:length(fs.orbitals)]
    return FermiSurfaceData(values, weights)
end
function run!(tba::Algorithm{<:TBA}, fs::Assignment{<:FermiSurface}; options...)
    bands = isa(fs.action.bands, Colon) ? (1:dimension(tba.frontend)) : fs.action.bands
    es = matrix(eigvals(tba, fs.action.reciprocalspace; options...))[:, bands]
    xs, ys = range(fs.action.reciprocalspace, 1), range(fs.action.reciprocalspace, 2)
    record = Int[]
    empty!(fs.data.values.coordinates)
    for i in axes(es, 2)
        for line in lines(contour(xs, ys, transpose(reshape(es[:, i], length(ys), length(xs))), fs.action.μ))
            for (x, y) in zip(coordinates(line)...)
                push!(record, bands[i])
                push!(fs.data.values.coordinates, (x, y))
            end
        end
    end
    if length(fs.action.orbitals) > 0
        map(empty!, fs.data.weights)
        for (band, k) in zip(record, fs.data.values)
            vs = eigvecs(tba, k)
            for (i, orbitals) in enumerate(fs.action.orbitals)
                push!(fs.data.weights[i], mapreduce(abs2, +, vs[orbitals, band]))
            end
        end
    end
end
@recipe function plot(fs::Assignment{<:FermiSurface}; fractional=true, weightmultiplier=1.0, weightcolors=nothing, weightlabels=nothing)
    title --> string(fs)
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

"""
    DensityOfStates{B<:BrillouinZone, A<:Union{Colon, AbstractVector{Int}}, L<:Tuple{Vararg{Union{Colon, AbstractVector{Int}}}}} <: Action

Density of states of a tight-binding system.
"""
struct DensityOfStates{B<:BrillouinZone, A<:Union{Colon, AbstractVector{Int}}, L<:Tuple{Vararg{Union{Colon, AbstractVector{Int}}}}} <: Action
    brillouinzone::B
    bands::A
    orbitals::L
end
@inline options(::Type{<:Assignment{<:DensityOfStates}}) = merge(basicoptions, (
    fwhm = "full width at half maximum for the Gaussian broadening",
    ne = "number of energy sample points",
    emin = "minimum value of the energy window",
    emax = "maximum value of the energy window"
))

"""
    DensityOfStates(brillouinzone::BrillouinZone, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::Union{Colon, AbstractVector{Int}}...=:)
    DensityOfStates(brillouinzone::BrillouinZone, bands::Union{Colon, AbstractVector{Int}}, orbitals::Tuple{Vararg{Union{Colon, AbstractVector{Int}}}})

Construct a `DensityOfStates`.
"""
@inline function DensityOfStates(brillouinzone::BrillouinZone, bands::Union{Colon, AbstractVector{Int}}=:, orbitals::Union{Colon, AbstractVector{Int}}...=:)
    return DensityOfStates(brillouinzone, bands, orbitals)
end

"""
    DensityOfStatesData <: Data

Data of density of states, including:

1) `energies::Vector{Float64}`: energy sample points.
2) `values::Matrix{Float64}`: density of states projected onto several certain sets of orbitals.
"""
mutable struct DensityOfStatesData <: Data
    const energies::Vector{Float64}
    values::Matrix{Float64}
end
@inline function Data(dos::DensityOfStates, ::TBA)
    energies = zeros(Float64, 0)
    values = zeros(Float64, 0, length(dos.orbitals))
    return DensityOfStatesData(energies, values)
end
function run!(tba::Algorithm{<:TBA{<:Fermionic{:TBA}}}, dos::Assignment{<:DensityOfStates}; fwhm::Real=0.1, ne::Int=100, emin::Real=NaN, emax::Real=NaN, options...)
    fill!(resize!(dos.data.energies, ne), 0.0)
    dos.data.values = zeros(Float64, ne, length(dos.action.orbitals))
    σ = fwhm / 2 /√ (2*log(2))
    eigenvalues, eigenvectors = eigen(tba, dos.action.brillouinzone; options...)
    isnan(emin) && (emin = mapreduce(minimum, min, eigenvalues))
    isnan(emax) && (emax = mapreduce(maximum, max, eigenvalues))
    nk = length(dos.action.brillouinzone)
    dE = (emax-emin) / (ne-1)
    bands = default_bands(kind(tba.frontend), dimension(tba), dos.action.bands)
    for (i, ω) in enumerate(LinRange(emin, emax, ne))
        dos.data.energies[i] = ω
        for (j, orbitals) in enumerate(dos.action.orbitals)
            for (values, vectors) in zip(eigenvalues, eigenvectors)
                dos.data.values[i, j] += spectralfunction(ω, values, vectors, bands, orbitals; σ=σ)/nk*dE
            end
        end
    end
end
function spectralfunction(ω::Real, values::Vector{<:Real}, vectors::Matrix{<:Number}, bands::AbstractVector{Int}, orbitals::Union{Colon, AbstractVector{Int}}; σ::Real)
    result = zero(ω)
    for i in bands
        factor = mapreduce(abs2, +, vectors[orbitals, i])
        result += factor * exp(-(ω-values[i])^2/2/σ^2)
    end
    return result/√(2pi)/σ
end
@inline default_bands(::TBAKind, ::Int, bands::AbstractVector{Int}) = bands
@inline default_bands(::TBAKind{:TBA}, dim::Int, ::Colon) = 1:dim
@inline default_bands(::TBAKind{:BdG}, dim::Int, ::Colon) = dim÷2:dim

"""
    InelasticNeutronScatteringSpectra{R<:ReciprocalSpace} <: Action

Inelastic neutron scattering spectra.
"""
struct InelasticNeutronScatteringSpectra{R<:ReciprocalSpace} <: Action
    reciprocalspace::R
    energies::Vector{Float64}
    function InelasticNeutronScatteringSpectra(reciprocalspace::ReciprocalSpace, energies::AbstractVector{<:Number})
        @assert label(reciprocalspace)==:k "InelasticNeutronScatteringSpectra error: the name of the momenta in the reciprocalspace must be :k."
        new{typeof(reciprocalspace)}(reciprocalspace, collect(Float64, energies))
    end
end
@inline options(::Type{<:Assignment{<:InelasticNeutronScatteringSpectra}}) = merge(basicoptions, (
    fwhm = "full width at half maximum for the Gaussian broadening",
    check = "whether the polarization consistency of phonons will be checked",
    rescale = "function used to rescale the intensity of the spectrum at each energy-momentum point"
))

"""
    InelasticNeutronScatteringSpectraData{R<:ReciprocalSpace} <: Data

Data of inelastic neutron scattering spectra, including:

1) `reciprocalspace::R`: reciprocal space on which the inelastic neutron scattering spectra are computed
1) `energies::Vector{Float64}`: energy sample points.
2) `values::Matrix{Float64}`: rescaled intensity of inelastic neutron scattering spectra.
"""
struct InelasticNeutronScatteringSpectraData{R<:ReciprocalSpace} <: Data
    reciprocalspace::R
    energies::Vector{Float64}
    values::Matrix{Float64}
end
@inline function Data(inss::InelasticNeutronScatteringSpectra, ::TBA)
    return InelasticNeutronScatteringSpectraData(inss.reciprocalspace, inss.energies, zeros(Float64, length(inss.energies), length(inss.reciprocalspace)))
end

# Inelastic neutron scattering spectra for phonons.
function run!(tba::Algorithm{<:CompositeTBA{Phononic, <:AbstractLattice}}, inss::Assignment{<:InelasticNeutronScatteringSpectra}; fwhm::Real=0.1, check::Bool=true, rescale::Function=identity, options...)
    dim = dimension(tba)
    σ = fwhm / 2 / √(2*log(2))
    sequences = Dict(site=>[tba.frontend.quadraticization.table[Index(site, PhononIndex{:u}(Char(Int('x')+i-1)))] for i=1:phonon.ndirection] for (site, phonon) in pairs(tba.frontend.system.hilbert))
    eigenvalues, eigenvectors = eigen(tba, inss.action.reciprocalspace; options...)
    for (i, (momentum, values, vectors)) in enumerate(zip(inss.action.reciprocalspace, eigenvalues, eigenvectors))
        check && @timeit_debug tba.timer "check" checkpolarizations(@views(vectors[(dim÷2+1):dim, 1:(dim÷2)]), @views(vectors[(dim÷2+1):dim, dim:-1:(dim÷2+1)]), momentum./pi)
        @timeit_debug tba.timer "spectra" begin
            for j = 1:dim
                factor = 0
                for (site, sequence) in pairs(sequences)
                    factor += dot(momentum, vectors[sequence, j]) * exp(1im*dot(momentum, tba.frontend.lattice[site]))
                end
                factor = abs2(factor) / √(2pi) / σ
                for (nₑ, e) in enumerate(inss.action.energies)
                    # instead of the Lorentz broadening of δ function, the convolution with a FWHM Gaussian is used.
                    inss.data.values[nₑ, i] += factor * exp(-(e-values[j])^2/2/σ^2)
                end
            end
        end
    end
    inss.data.values[:, :] = rescale.(inss.data.values)
end
@inline function checkpolarizations(qs₁::AbstractMatrix, qs₂::AbstractMatrix, momentum)
    inner = mapreduce((e₁, e₂)->norm(conj(e₁)*e₂), +, qs₁, qs₂) / norm(qs₁) / norm(qs₂)
    isapprox(inner, 1; atol=100*atol, rtol=100*rtol) || begin
        @warn("checkpolarizations: small inner product $inner at π*$momentum, indication of degeneracy, otherwise inconsistent polarization vectors.")
    end
end
