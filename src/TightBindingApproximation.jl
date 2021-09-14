module TightBindingApproximation

using QuantumLattices: Lattice, Bonds, Hilbert, Table, Term, Boundary, Generator, plain
using QuantumLattices: App, Engine, Assignment, Algorithm

export TBA

struct TBA{L<:Lattice, G<:Generator} <: Engine
    lattice::L
    H::G
end
@inline function TBA(lattice::Lattice, hilbert::Hilbert, table::Table, terms::Tuple{Vararg{Term}}; boundary::Boundary=plain)
    return TBA(lattice, Generator(terms, Bonds(lattice), hilbert; half=false, table=table, boundary=boundary))
end

end # module
