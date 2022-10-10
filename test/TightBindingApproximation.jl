using LinearAlgebra: Diagonal, Hermitian, ishermitian
using Plots: plot, plot!, savefig
using QuantumLattices: dimension, kind, matrix, update!
using QuantumLattices: Coupling, Hilbert, Metric, OperatorUnitToTuple
using QuantumLattices: Algorithm, Parameters
using QuantumLattices: Elastic, FID, Fock, Hooke, Hopping, Kinetic, Onsite, Pairing, Phonon
using QuantumLattices: BrillouinZone, Lattice, ReciprocalPath, ReciprocalZone, Segment, azimuth, rcoordinate, @rectangle_str
using QuantumLattices: contentnames
using TightBindingApproximation

@time @testset "prerequisite" begin
    @test promote_type(Fermionic{:TBA}, Fermionic{:TBA}) == Fermionic{:TBA}
    @test promote_type(Fermionic{:BdG}, Fermionic{:BdG}) == Fermionic{:BdG}
    @test promote_type(Fermionic{:TBA}, Fermionic{:BdG}) == promote_type(Fermionic{:BdG}, Fermionic{:TBA}) == Fermionic{:BdG}

    @test promote_type(Bosonic{:TBA}, Bosonic{:TBA}) == Bosonic{:TBA}
    @test promote_type(Bosonic{:BdG}, Bosonic{:BdG}) == Bosonic{:BdG}
    @test promote_type(Bosonic{:TBA}, Bosonic{:BdG}) == promote_type(Bosonic{:BdG}, Bosonic{:TBA}) == Bosonic{:BdG}

    @test TBAKind(Hopping, Fock{:f}) == TBAKind(Onsite, Fock{:f}) == Fermionic(:TBA)
    @test TBAKind(Pairing, Fock{:f}) == Fermionic(:BdG)
    @test TBAKind(Hopping, Fock{:b}) == TBAKind(Onsite, Fock{:b}) == Bosonic(:TBA)
    @test TBAKind(Pairing, Fock{:b}) == Bosonic(:BdG)
    @test TBAKind(Kinetic, Phonon) == TBAKind(Hooke, Phonon) == TBAKind(Elastic, Phonon) == Phononic()
    @test TBAKind(Tuple{Hopping, Onsite}, Fock{:f}) == Fermionic(:TBA)
    @test TBAKind(Tuple{Hopping, Onsite, Pairing}, Fock{:f}) == Fermionic(:BdG)
    @test TBAKind(Tuple{Hopping, Onsite}, Fock{:b}) == Bosonic(:TBA)
    @test TBAKind(Tuple{Hopping, Onsite, Pairing}, Fock{:b}) == Bosonic(:BdG)

    @test Metric(Fermionic(:TBA), Hilbert(1=>Fock{:f}(1, 1))) == OperatorUnitToTuple(:site, :orbital, :spin)
    @test Metric(Fermionic(:BdG), Hilbert(1=>Fock{:f}(1, 1))) == OperatorUnitToTuple(:nambu, :site, :orbital, :spin)
    @test Metric(Bosonic(:TBA), Hilbert(1=>Fock{:b}(1, 1))) == OperatorUnitToTuple(:site, :orbital, :spin)
    @test Metric(Bosonic(:BdG), Hilbert(1=>Fock{:b}(1, 1))) == OperatorUnitToTuple(:nambu, :site, :orbital, :spin)
    @test Metric(Phononic(), Hilbert(1=>Phonon(2))) == OperatorUnitToTuple(:tag, :site, :direction)

    @test commutator(Fermionic(:TBA), Hilbert(1=>Fock{:f}(1, 2))) == nothing
    @test commutator(Fermionic(:BdG), Hilbert(1=>Fock{:f}(1, 2))) == nothing
    @test commutator(Bosonic(:TBA), Hilbert(1=>Fock{:b}(1, 2))) == nothing
    @test commutator(Bosonic(:BdG), Hilbert(1=>Fock{:b}(1, 2))) == Diagonal([1, 1, -1, -1])
    @test commutator(Phononic(), Hilbert(1=>Phonon(2))) == Hermitian([0 0 -im 0; 0 0 0 -im; im 0 0 0; 0 im 0 0])

    contentnames(AbstractTBA) == (:H, :commutator)
    contentnames(TBA) == (:lattice, :H, :commutator)
end

@time @testset "TBA" begin
    lattice = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(1=>Fock{:f}(1, 1))
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 0.0)

    tba = TBA(lattice, hilbert, (t, μ))
    @test kind(tba) == kind(typeof(tba)) == Fermionic(:TBA)
    @test valtype(tba) == valtype(typeof(tba)) == Float64
    @test dimension(tba) == 1
    @test Parameters(tba) == (t=1.0, μ=0.0)

    m = matrix(tba)
    @test size(m) == (1, 1)
    @test m[1, 1] == m.H[1, 1]
    @test ishermitian(m) == ishermitian(typeof(m)) == true

    A(t, μ; k) = hcat(2t*cos(k[1])+2t*cos(k[2])+μ)
    tbaₐ = TBA{Fermionic{:TBA}}(lattice, A, (t=1.0, μ=0.0))
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

    Δ = Pairing(:Δ, Complex(0.1), 1, Coupling(:, FID, :, :, (1, 1)); amplitude=bond->exp(im*azimuth(rcoordinate(bond))))
    bdg = TBA(lattice, hilbert, (t, μ, Δ))
    @test kind(bdg) == kind(typeof(bdg)) == Fermionic(:BdG)
    @test valtype(bdg) == valtype(typeof(bdg)) == Complex{Float64}
    @test Parameters(bdg) == (t=1.0, μ=0.5, Δ=Complex(0.1))

    A(t, μ, Δ; k) = [2t*cos(k[1])+2t*cos(k[2])+μ -2im*Δ*sin(k[1])-2Δ*sin(k[2]); 2im*Δ*sin(k[1])-2Δ*sin(k[2]) -2t*cos(k[1])-2t*cos(k[2])-μ]
    bdgₐ = TBA{Fermionic{:BdG}}(lattice, A, (t=1.0, μ=0.5, Δ=0.1))
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
    unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 3.5)
    Δ = Pairing(:Δ, Complex(0.5), 1, Coupling(:, FID, :, :, (1, 1)); amplitude=bond->exp(im*azimuth(rcoordinate(bond))))
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

    reciprocalzone = ReciprocalZone(unitcell.reciprocals, [Segment(-2.0, +2.0, 201, ends=(true, true)), Segment(-2.0, 2.0, 201, ends=(true, true))])
    berry = sc(:BerryCurvatureExtended, BerryCurvature(reciprocalzone, [1, 2]))
    plt = plot(berry)
    display(plt)
    savefig(plt, "bcextended.png")
end

@time @testset "phonon" begin
    unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(site=>Phonon(2) for site=1:length(unitcell))
    T = Kinetic(:T, 0.5)
    V₁ = Hooke(:V₁, 0.5, 1)
    V₂ = Hooke(:V₂, 0.25, 2)
    phonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))
    path = ReciprocalPath(unitcell.reciprocals, rectangle"Γ-X-M-Γ", length=100)

    energybands = phonon(:EB, EnergyBands(path))
    plt = plot(energybands)
    display(plt)
    savefig(plt, "phonon.png")

    inelastic = phonon(:INSS, InelasticNeutronScatteringSpectra(path, range(0.0, 2.5, length=501); fwhm=0.05, scale=log))
    plt = plot()
    plot!(plt, inelastic)
    plot!(plt, energybands, color=:white, linestyle=:dash)
    display(plt)
    savefig("inelastic.png")
end
