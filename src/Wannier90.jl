module Wannier90

using DelimitedFiles: readdlm
using LinearAlgebra: Hermitian, dot
using QuantumLattices: Hamiltonian, Hilbert, Lattice, Matrixization, OperatorPack, OperatorSet, OperatorSum, OperatorUnitToTuple, SimpleHamiltonian, Table, add!, kind
using StaticArrays: SVector
using ..TightBindingApproximation: AbstractTBA, Fermionic, TBAKind, TBAMatrix

import QuantumLattices: add!, dimension, getcontent, matrix

export W90, W90Hoppings, W90Matrixization, findblock, readbands, readcenters, readhamiltonian, readlattice

"""
    findblock(name::String, content::String) -> Union{Nothing, Vector{SubString{String}}}

Find a named block in the content of Wannier90 configuration files.
"""
function findblock(name::String, content::String)
    r = match(Regex("(?i)begin\\s*"*name*"\\s*\\n([\\s\\S]*)\\s*\\nend\\s*"*name), content)
    isnothing(r) && return
    return strip.(split(r[1], "\n"))
end

"""
    readlattice(path::AbstractString, prefix::AbstractString="wannier90"; name::Symbol=Symbol(prefix), projection::Bool=true) -> Lattice{3, Float64, 3}

Read the lattice from the ".win" configuration file of Wannier90 with a given `name`.

Besides, `projection` specifies whether only the sites with initial projections in the ".win" file are included in the constructed lattice.
"""
function readlattice(path::AbstractString, prefix::AbstractString="wannier90"; name::Symbol=Symbol(prefix), projection::Bool=true)
    content = open(io->read(io, String), path*"\\"*prefix*".win", "r")
    # get translation vectors
    lines = findblock("unit_cell_cart", content)
    isnothing(lines) && error("readlattice error: unable to find unit_cell_cart block in the .win file.")
    @assert length(lines)∈(3, 4) "readlattice error: incorrect unit_cell_cart block."
    pref = coeff(lines)
    lines = map(i->split(lines[i]), start(lines):length(lines))
    vectors = SVector(
        SVector(parse(Float64, lines[1][1])*pref, parse(Float64, lines[1][2])*pref, parse(Float64, lines[1][3])*pref),
        SVector(parse(Float64, lines[2][1])*pref, parse(Float64, lines[2][2])*pref, parse(Float64, lines[2][3])*pref),
        SVector(parse(Float64, lines[3][1])*pref, parse(Float64, lines[3][2])*pref, parse(Float64, lines[3][3])*pref)
    )
    # get atom positions
    lines = findblock("atoms_frac", content)
    if !isnothing(lines)
        lines = map(split, lines)
        coordinates = hcat(vectors...) * [parse(Float64, lines[j][i]) for i=2:4, j=1:length(lines)]
    else
        lines = findblock("atoms_cart", content)
        isnothing(lines) && error("readlattice error: unable to find atoms_frac or atoms_cart block in the .win file.")
        pref = coeff(lines)
        lines = map(i->split(lines[i]), start(lines):length(lines))
        coordinates = [parse(Float64, lines[j][i])*pref for i=2:4, j=1:length(lines)]
    end
    # projection of atoms
    if projection
        atoms = [lowercase(line[1]) for line in lines]
        lines = findblock("projections", content)
        pref = coeff(lines)
        lines = map(line->split(line, ":")[1], lines)
        sites = Int[]
        for i = start(lines):length(lines)
            site = strip(split(lines[i], ":")[1])
            if occursin(r"(?i)[f, c]=(.*)", site)
                coordinate = map(str->parse(Float64, str), split(match(r"(?i)[f, c]=(.*)", site)[1], ","))
                if occursin(r"(?i)f=", site)
                    coordinate = mapreduce(*, +, vectors, coordinate)
                else
                    coordinate = pref*coordinate
                end
                result = findsite(coordinate, coordinates)
                @assert isa(result, Int) "readlattice error: cannot find site($site)."
                push!(sites, result)
            else
                result = findall(==(lowercase(site)), atoms)
                @assert length(result)>0 "readlattice error: cannot find $site atoms."
                append!(sites, result)
            end
        end
        coordinates = coordinates[:, sites]
    end
    return Lattice(name, coordinates::Matrix{Float64}, vectors::SVector{3, SVector{3, Float64}})
end
@inline coeff(lines) = lowercase(lines[1])=="bohr" ? 0.5291772108 : 1.0
@inline start(lines) = lowercase(lines[1])∈("bohr", "ang", "angstrom") ? 2 : 1
function findsite(coordinate, coordinates; atol=10^-6, rtol=10^-6)
    for i = 1:size(coordinates)[2]
        isapprox(coordinate, coordinates[:, i]; atol=atol, rtol=rtol) && return i
    end
end

"""
    readcenters(path::AbstractString, prefix::AbstractString="wannier90") -> Matrix{Float64}

Read the centers of Wannier functions obtained by Wannier90.
"""
function readcenters(path::AbstractString, prefix::AbstractString="wannier90")
    content = open(io->read(io, String), path*"\\"*prefix*"_centres.xyz", "r")
    lines = [content[index] for index in findall(r"X.*", content)]
    length(lines)==0 && error("readcenters error: Wannier centers not found.")
    result = zeros(Float64, 3, length(lines))
    for (i, line) in enumerate(lines)
        parts = split(line)
        result[1, i] = parse(Float64, parts[2])
        result[2, i] = parse(Float64, parts[3])
        result[3, i] = parse(Float64, parts[4])
    end
    return result
end

"""
    W90Hoppings <: OperatorPack{Matrix{Float64}, NTuple{3, Int}}

Hopping amplitudes among the Wannier orbitals in two unitcells with a fixed relative displacement.

Here,
1) the `:value` attribute represents the hopping amplitude matrix of Wannier orbitals,
2) the `:id` attribute represents the relative displacement of the two unitcells in basis of the lattice translation vectors.
"""
struct W90Hoppings <: OperatorPack{Matrix{ComplexF64}, NTuple{3, Int}}
    value::Matrix{ComplexF64}
    id::NTuple{3, Int}
    function W90Hoppings(value::AbstractMatrix{<:Number}, id::NTuple{3, Integer})
        @assert size(value)[1]==size(value)[2] "W90Hoppings error: hopping matrix is not square."
        new(value, id)
    end
end

"""
    readhamiltonian(path::AbstractString, prefix::AbstractString="wannier90") -> OperatorSum{W90Hoppings, NTuple{3, Int}}

Read the hamiltonian from the "_hr.dat" file generated by Wannier90.
"""
function readhamiltonian(path::AbstractString, prefix::AbstractString="wannier90")
    lines = open(readlines, path*"\\"*prefix*"_hr.dat", "r")
    num_wan = parse(Int, lines[2])
    nrpts = parse(Int, lines[3])
    degeneracies = Int[]
    start = 0
    for i in 4:length(lines)
        append!(degeneracies, parse(Int, degeneracy) for degeneracy in split(lines[i]))
        if length(degeneracies) == nrpts
            start = i+1
            break
        end
        length(degeneracies)>nrpts && error("readhamiltonian error: too many degeneracies for the Wigner-Seitz points of the superlattice.")
    end
    index = 1
    ham = OperatorSum{W90Hoppings}()
    deg = Dict{NTuple{3, Int}, Int}()
    for k in start:length(lines)
        info = split(lines[k])
        point = (parse(Int, info[1]), parse(Int, info[2]), parse(Int, info[3]))
        i, j = parse(Int, info[4]), parse(Int, info[5])
        value = parse(Float64, info[6]) + 1im*parse(Float64, info[7])
        if !haskey(ham, point)
            ham.contents[point] = W90Hoppings(zeros(ComplexF64, num_wan, num_wan), point)
            deg[point] = degeneracies[index]
            index += 1
        end
        ham.contents[point].value[i, j] = value/deg[point]
    end
    for point in keys(ham.contents)
        @assert haskey(ham, map(-, point)) "readhamiltonian error: did not find the inverse of point $(point)."
    end
    return ham
end

"""
    W90Matrixization{V<:AbstractVector{<:Real}} <: Matrixization

Matrixization of the Hamiltonian obtained from Wannier90.
"""
struct W90Matrixization{V<:AbstractVector{<:Real}} <: Matrixization
    k::V
    vectors::SVector{3, SVector{3, Float64}}
    centers::Matrix{Float64}
    gauge::Symbol
    function W90Matrixization(k::AbstractVector{<:Real}, vectors::AbstractVector{<:AbstractVector{<:Real}}, centers::AbstractMatrix{<:Real}, gauge::Symbol)
        @assert length(k)==3 "W90Matrixization error: the length of `k` must be 3."
        @assert length(vectors)==3 && all(v->length(v)==3, vectors) "W90Matrixization error: `vectors` must be 3 length-3 vectors."
        @assert size(centers)[1]==3 "W90Matrixization error: the row number of `centers` must be 3."
        @assert gauge∈(:icoordinate, :rcoordinate) "W90Matrixization error: `gauge` must be either `:icoordinate` or `:rcoordinate`."
        new{typeof(k)}(k, vectors, centers, gauge)
    end
end
@inline Base.valtype(::Type{<:W90Matrixization}, ::Type{<:Union{W90Hoppings, OperatorSet{<:W90Hoppings}}}) = Matrix{ComplexF64}
@inline Base.zero(mr::W90Matrixization, m::Union{W90Hoppings, OperatorSet{<:W90Hoppings}}) = zeros(eltype(valtype(mr, m)), size(mr.centers)[2], size(mr.centers)[2])
@inline (mr::W90Matrixization)(m::W90Hoppings; kwargs...) = add!(zero(mr, m), mr, m; kwargs...)

"""
    add!(dest::AbstractMatrix, mr::W90Matrixization, m::W90Hoppings; kwargs...) -> typeof(dest)

Matrixize a group of Wannier hoppings and add it to `dest`.
"""
function add!(dest::AbstractMatrix, mr::W90Matrixization, m::W90Hoppings; kwargs...)
    @assert size(dest)[1]==size(dest)[2]==size(mr.centers)[2]==size(m.value)[1] "add! error: mismatch occurs."
    icoordinate = mapreduce(*, +, mr.vectors, m.id)
    if mr.gauge == :icoordinate
        phase = exp(1im*dot(mr.k, icoordinate))
        for index in eachindex(dest)
            dest[index] += m.value[index]*phase
        end
    else
        for i = 1:size(mr.centers)[2]
            centerᵢ = SVector(mr.centers[1, i], mr.centers[2, i], mr.centers[3, i])
            for j = 1:size(mr.centers)[2]
                centerⱼ = SVector(mr.centers[1, j], mr.centers[2, j], mr.centers[3, j])
                phase = exp(1im*dot(mr.k, icoordinate+centerⱼ-centerᵢ))
                dest[i, j] += m.value[i, j]*phase
            end
        end
    end
    return dest
end

"""
    W90 <: AbstractTBA{Fermionic{:TBA}, SimpleHamiltonian{OperatorSum{W90Hoppings, NTuple{3, Int}}}, Nothing}

A quantum lattice system based on the information obtained from Wannier90.
"""
struct W90 <: AbstractTBA{Fermionic{:TBA}, SimpleHamiltonian{OperatorSum{W90Hoppings, NTuple{3, Int}}}, Nothing}
    lattice::Lattice{3, Float64, 3}
    centers::Matrix{Float64}
    H::OperatorSum{W90Hoppings, NTuple{3, Int}}
    function W90(lattice::Lattice{3, <:Real, 3}, centers::AbstractMatrix{<:Real}, H::OperatorSum{W90Hoppings})
        @assert size(centers)[1]==3 "W90 error: the row number of `centers` must be 3."
        @assert size(centers)[2]==size(first(H).value)[1] "W90 error: mismatched size of Wannier centers and hoppings."
        new(lattice, centers, H)
    end
end
@inline getcontent(wan::W90, ::Val{:H}) = Hamiltonian(wan.H)
@inline getcontent(wan::W90, ::Val{:commutator}) = nothing
@inline dimension(wan::W90) = size(wan.centers)[2]

"""
    matrix(wan::W90; k=SVector(0.0, 0.0, 0.0), gauge=:icoordinate, kwargs...) -> TBAMatrix

Get the matrix representation of a quantum lattice system based on the information obtained from Wannier90.
"""
@inline function matrix(wan::W90; k=SVector(0.0, 0.0, 0.0), gauge=:icoordinate, kwargs...)
    m = W90Matrixization(k, wan.lattice.vectors, wan.centers, gauge)(wan.H; kwargs...)
    return TBAMatrix{typeof(kind(wan))}(Hermitian(m), nothing)
end

"""
    W90(lattice::Lattice, centers::Matrix{<:Real}, H::OperatorSum{W90Hoppings})
    W90(lattice::Lattice, hilbert::Hilbert, H::OperatorSum{W90Hoppings})

Construct a quantum lattice system based on the information obtained from Wannier90.

In general, the Wannier centers could deviate from their corresponding atom positions.
When `centers::Matrix{<:Real}` is used, the the Wannier centers are assigned directly.
When `hilbert::Hilbert` is used, the Wannier centers will be approximated by their corresponding atom positions.
"""
function W90(lattice::Lattice, hilbert::Hilbert, H::OperatorSum{W90Hoppings})
    table = Table(hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
    centers = zeros(Float64, 3, length(table))
    for (key, index) in pairs(table)
        centers[:, index] = lattice[key[1]]
    end
    return W90(lattice, centers, H)
end

"""
    W90(path::AbstractString, prefix::AbstractString="wannier90"; name::Symbol=Symbol(prefix), projection::Bool=true)

Construct a quantum lattice system based on the files obtained from Wannier90.
"""
function W90(path::AbstractString, prefix::AbstractString="wannier90"; name::Symbol=Symbol(prefix), projection::Bool=true)
    lattice = readlattice(path, prefix; name=name, projection=projection)
    centers = readcenters(path, prefix)
    H = readhamiltonian(path, prefix)
    return W90(lattice, centers, H)
end

"""
    readbands(path::AbstractString, prefix::AbstractString="wannier90") -> Tuple{Vector{Float64}, Matrix{Float64}}

Read the band structure from Wannier90 files by setting "bands_plot = .true.".
"""
function readbands(path::AbstractString, prefix::AbstractString="wannier90")
    nk = open(path*"\\"*prefix*"_band.dat", "r") do f
        count = 0
        while !eof(f)
            line = readline(f; keep=true)
            all(isspace, line) && return count
            count += 1
        end
        error("readbands error: mismatched file organization format.")
    end
    content = readdlm(path*"\\"*prefix*"_band.dat")
    return content[1:nk, 1], reshape(content[:, 2], nk, :)
end

end
