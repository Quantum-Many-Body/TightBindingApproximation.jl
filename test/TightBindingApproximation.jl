using Plots: plot, savefig
using LinearAlgebra: Diagonal, Hermitian, ishermitian
using QuantumLattices: contentnames, kind, dimension, azimuth, rcoord, update!, matrix
using QuantumLattices: PID, CPID, Point, Lattice, FID, Fock, NID, Phonon, Index, Hilbert, Metric, OIDToTuple, Parameters
using QuantumLattices: Hopping, Onsite, Pairing, PhononKinetic, PhononPotential
using QuantumLattices: ReciprocalPath, @rectangle_str, BrillouinZone, Algorithm
using TightBindingApproximation

@time @testset "prerequisite" begin
    @test promote_type(TBAKind{:TBA}, TBAKind{:TBA}) == TBAKind{:TBA}
    @test promote_type(TBAKind{:BdG}, TBAKind{:BdG}) == TBAKind{:BdG}
    @test promote_type(TBAKind{:TBA}, TBAKind{:BdG}) == promote_type(TBAKind{:BdG}, TBAKind{:TBA}) == TBAKind{:BdG}

    @test TBAKind(Hopping) == TBAKind(Onsite) == TBAKind(:TBA)
    @test TBAKind(Pairing) == TBAKind(PhononKinetic) == TBAKind(PhononPotential) == TBAKind(:BdG)
    @test TBAKind(Tuple{Hopping, Onsite}) == TBAKind(:TBA)
    @test TBAKind(Tuple{Hopping, Onsite, Pairing}) == TBAKind(:BdG)


    @test Metric(TBAKind(:TBA), Hilbert(PID(1)=>Fock{:b}(1, 1, 1))) == OIDToTuple(:site, :orbital, :spin)
    @test Metric(TBAKind(:TBA), Hilbert(CPID(1, 1)=>Fock{:f}(1, 1, 1))) == OIDToTuple(:scope, :site, :orbital, :spin)
    @test Metric(TBAKind(:BdG), Hilbert(PID(1)=>Fock{:b}(1, 1, 1))) == OIDToTuple(:nambu, :site, :orbital, :spin)
    @test Metric(TBAKind(:BdG), Hilbert(CPID(1, 1)=>Fock{:f}(1, 1, 1))) == OIDToTuple(:nambu, :scope, :site, :orbital, :spin)
    @test Metric(TBAKind(:BdG), Hilbert(PID(1)=>Phonon(2))) == OIDToTuple(:tag, :site, :dir)
    @test Metric(TBAKind(:BdG), Hilbert(CPID(1, 1)=>Phonon(2))) == OIDToTuple(:tag, :scope, :site, :dir)

    @test commutator(TBAKind(:TBA), Hilbert(PID(1)=>Fock{:f}(1, 2, 2))) == nothing
    @test commutator(TBAKind(:TBA), Hilbert(PID(1)=>Fock{:b}(1, 2, 2))) == nothing
    @test commutator(TBAKind(:BdG), Hilbert(PID(1)=>Fock{:f}(1, 2, 2))) == nothing
    @test commutator(TBAKind(:BdG), Hilbert(PID(1)=>Fock{:b}(1, 2, 2))) == Diagonal([1, 1, -1, -1])
    @test commutator(TBAKind(:BdG), Hilbert(PID(1)=>Phonon(2))) == Hermitian([0 0 -im 0; 0 0 0 -im; im 0 0 0; 0 im 0 0])

    contentnames(AbstractTBA) == (:H, :commutator)
    contentnames(TBA) == (:lattice, :H, :commutator)
end

@time @testset "TBA" begin
    lattice = Lattice(:Square,
        [Point(PID(1), [0.0, 0.0])],
        vectors=[[1.0, 0.0], [0.0, 1.0]],
        neighbors=1
        )
    hilbert = Hilbert(PID(1)=>Fock{:f}(1, 1, 2))
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 0.0, modulate=true)

    tba = TBA(lattice, hilbert, (t, μ))
    @test kind(tba) == kind(typeof(tba)) == TBAKind(:TBA)
    @test valtype(tba) == valtype(typeof(tba)) == Float64
    @test dimension(tba) == 1
    @test Parameters(tba) == (t=1.0, μ=0.0)

    m = matrix(tba)
    @test size(m) == (1, 1)
    @test m[1, 1] == m.H[1, 1]
    @test ishermitian(m) == ishermitian(typeof(m)) == true

    A(t, μ; k) = hcat(2t*cos(k[1])+2t*cos(k[2])+μ)
    tbaₐ = TBA(lattice, A, (t=1.0, μ=0.0))
    path = ReciprocalPath(lattice.reciprocals, rectangle"Γ-X-M-Γ", length=8)
    for kv in pairs(path)
        m = matrix(tba; kv...)
        mₐ = matrix(tbaₐ; kv...)
        @test m.H ≈ mₐ.H ≈ Hermitian(A(1.0, 0.0; kv...))
    end
    update!(tba, μ=0.5)
    update!(tbaₐ, μ=0.5)
    for kv in pairs(path)
        m = matrix(tba; kv...)
        mₐ = matrix(tbaₐ; kv...)
        @test m.H ≈ mₐ.H ≈ Hermitian(A(1.0, 0.5; kv...))
    end

    Δ = Pairing(:Δ, Complex(0.1), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))))
    bdg = TBA(lattice, hilbert, (t, μ, Δ))
    @test kind(bdg) == kind(typeof(bdg)) == TBAKind(:BdG)
    @test valtype(bdg) == valtype(typeof(bdg)) == Complex{Float64}
    @test Parameters(bdg) == (t=1.0, μ=0.5, Δ=Complex(0.1))

    A(t, μ, Δ; k) = [2t*cos(k[1])+2t*cos(k[2])+μ 2im*Δ*sin(k[1])+2Δ*sin(k[2]); -2im*Δ*sin(k[1])+2Δ*sin(k[2]) -2t*cos(k[1])-2t*cos(k[2])-μ]
    bdgₐ = TBA(lattice, A, (t=1.0, μ=0.5, Δ=0.1))
    path = ReciprocalPath(lattice.reciprocals, rectangle"Γ-X-M-Γ", length=8)
    for kv in pairs(path)
        m = matrix(bdg; kv...)
        mₐ = matrix(bdgₐ; kv...)
        @test m.H ≈ mₐ.H ≈ Hermitian(A(1.0, 0.5, 0.1; kv...))
    end

    samplesets = [
        SampleNode(lattice.reciprocals, [0.0, 0.0], [1], [-4.0]),
        SampleNode(lattice.reciprocals, [0.5, 0.5], [1], [+4.0])
    ]
    op = optimize!(tba, samplesets)[2]
    @test isapprox(op.minimizer, [-1.0, 0.0], atol=10^-10)
end

@time @testset "plot" begin
    unitcell = Lattice(:Square, [Point(PID(1), [0.0, 0.0])], vectors=[[1.0, 0.0], [0.0, 1.0]], neighbors=1)
    hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)
    t = Hopping(:t, 1.0, 1, modulate=true)
    μ = Onsite(:μ, 3.5, modulate=true)
    Δ = Pairing(:Δ, Complex(0.5), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))), modulate=true)
    sc = Algorithm(Symbol("p+ip"), TBA(unitcell, hilbert, (t, μ, Δ)))
    path = ReciprocalPath(unitcell.reciprocals, rectangle"Γ-X-M-Γ", length=100)
    energybands = sc(:EB, EnergyBands(path))
    plt = plot(energybands)
    display(plt)
    savefig(plt, "eb.png")

    brillouin = BrillouinZone(unitcell.reciprocals, 100)
    berry = sc(:BerryCurvature, BerryCurvature(brillouin, [1, 2]));
    plt = plot(berry)
    display(plt)
    savefig(plt, "bc.png")
end