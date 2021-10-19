using LinearAlgebra: Diagonal, Hermitian, ishermitian
using QuantumLattices: contentnames, kind, statistics, dimension, azimuth, rcoord, update!, matrix!
using QuantumLattices: PID, CPID, Point, Lattice, FID, Fock, NID, Phonon, Index, Hilbert, OIDToTuple
using QuantumLattices: Hopping, Onsite, Pairing, PhononKinetic, PhononPotential, DMPhonon
using QuantumLattices: ReciprocalPath, @rectangle_str
using TightBindingApproximation

@time @testset "prerequisite" begin
    @test promote_type(TBAKind{:TBA}, TBAKind{:TBA}) == TBAKind{:TBA}
    @test promote_type(TBAKind{:BdG}, TBAKind{:BdG}) == TBAKind{:BdG}
    @test promote_type(TBAKind{:TBA}, TBAKind{:BdG}) == promote_type(TBAKind{:BdG}, TBAKind{:TBA}) == TBAKind{:BdG}

    @test TBAKind(Hopping) == TBAKind(Onsite) == TBAKind(:TBA)
    @test TBAKind(Pairing) == TBAKind(PhononKinetic) == TBAKind(PhononPotential) == TBAKind(DMPhonon) == TBAKind(:BdG)
    @test TBAKind(Tuple{Hopping, Onsite}) == TBAKind(:TBA)
    @test TBAKind(Tuple{Hopping, Onsite, Pairing}) == TBAKind(:BdG)

    @test OIDToTuple(TBAKind(:BdG), Index{PID, NID{Char}}) == OIDToTuple(:tag, :site, :dir)
    @test OIDToTuple(TBAKind(:BdG), Index{CPID{Char}, NID{Char}}) == OIDToTuple(:tag, :scope, :site, :dir)
    @test OIDToTuple(TBAKind(:BdG), Index{PID, FID{Int, Int, Int}}) == OIDToTuple(:nambu, :site, :orbital, :spin)
    @test OIDToTuple(TBAKind(:BdG), Index{CPID{Char}, FID{Int, Int, Int}}) == OIDToTuple(:nambu, :scope, :site, :orbital, :spin)
    @test OIDToTuple(TBAKind(:TBA), Index{PID, FID{Int, Int, Int}}) == OIDToTuple(:site, :orbital, :spin)
    @test OIDToTuple(TBAKind(:TBA), Index{CPID{Char}, FID{Int, Int, Int}}) == OIDToTuple(:scope, :site, :orbital, :spin)

    @test commutator(TBAKind(:TBA), Fock{:f}, 2) == nothing
    @test commutator(TBAKind(:TBA), Fock{:b}, 2) == nothing
    @test commutator(TBAKind(:BdG), Fock{:f}, 2) == nothing
    @test commutator(TBAKind(:BdG), Fock{:b}, 4) == Diagonal([1, 1, -1, -1])
    @test commutator(TBAKind(:BdG), Phonon, 4) == Hermitian([0 0 -im 0; 0 0 0 -im; im 0 0 0; 0 im 0 0])

    contentnames(AbstractTBA) == (:H, :commutator)
    contentnames(TBA) == (:lattice, :H, :commutator)
end

@time @testset "Square" begin
    lattice = Lattice("Square", [Point(PID(1), (0.0, 0.0), (0.0, 0.0))],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        neighbors=1
        )
    hilbert = Hilbert(PID(1)=>Fock{:f}(1, 1, 2))
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 0.0, modulate=true)

    tba = TBA(lattice, hilbert, (t, μ))
    @test kind(tba) == kind(typeof(tba)) == TBAKind(:TBA)
    @test eltype(tba) == eltype(typeof(tba)) == eltype(tba.H)
    @test valtype(tba) == valtype(typeof(tba)) == valtype(tba, nothing) == Float64
    @test valtype(tba, [0.0, 0.0]) == Complex{Float64}
    @test statistics(tba) == statistics(typeof(tba)) == :f
    @test dimension(tba) == 1

    m = matrix!(tba)
    @test size(m) == (1, 1)
    @test m[1, 1] == m.H[1, 1]
    @test ishermitian(m) == ishermitian(typeof(m)) == true

    A(t, μ; k) = 2t*cos(k[1])+2t*cos(k[2])+μ
    path = ReciprocalPath(lattice.reciprocals, rectangle"Γ-X-M-Γ", len=4)
    for param in path
        m = matrix!(tba; param...)
        @test m.H ≈ Hermitian(hcat(A(1.0, 0.0; param...)))
    end
    update!(tba, μ=0.5)
    for param in path
        m = matrix!(tba; param...)
        @test m.H ≈ Hermitian(hcat(A(1.0, 0.5; param...)))
    end

    Δ = Pairing(:Δ, Complex(0.1), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))))
    bdg = TBA(lattice, hilbert, (t, μ, Δ))
    @test kind(bdg) == kind(typeof(bdg)) == TBAKind(:BdG)
    @test valtype(bdg) == valtype(typeof(bdg)) == Complex{Float64}

    A(t, μ, Δ; k) = [2t*cos(k[1])+2t*cos(k[2])+μ 2im*Δ*sin(k[1])+2Δ*sin(k[2]); -2im*Δ*sin(k[1])+2Δ*sin(k[2]) -2t*cos(k[1])-2t*cos(k[2])-μ]
    path = ReciprocalPath(lattice.reciprocals, rectangle"Γ-X-M-Γ", len=4)
    for param in path
        m = matrix!(bdg; param...)
        @test m.H ≈ Hermitian(A(1.0, 0.5, 0.1; param...))
    end
end
