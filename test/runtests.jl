using Test
using QuantumLattices: Lattice, Point, PID, Hilbert, Fock, Table, Hopping, Onsite, Generator, Bonds, usualfockindextotuple
using TightBindingApproximation

@testset "TightBindingApproximation.jl" begin
    lattice = Lattice("Square", [Point(PID(1), (0.0, 0.0), (0.0, 0.0))],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        neighbors=1
        )
    hilbert = Hilbert(PID(1)=>Fock{:f}(1, 1, 2))
    table = Table(hilbert, usualfockindextotuple)
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 0.5)
    tba = TBA(lattice, hilbert, table, (t, μ))
    @test tba.lattice == lattice
    @test tba.H == Generator((t, μ), Bonds(lattice), hilbert, table=table, half=false)
end
