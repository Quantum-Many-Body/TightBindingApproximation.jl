using LinearAlgebra: Diagonal, Eigen, Hermitian, eigen, eigvals, eigvecs, ishermitian
using Plots: plot, plot!, savefig
using QuantumLattices: Algorithm, BrillouinZone, Coupling, Elastic, FockIndex, Fock, Hilbert, Hooke, Hopping, Kinetic, Lattice, MatrixCoupling, Metric, Onsite, OperatorUnitToTuple, Pairing, Parameters, Phonon, ReciprocalPath, ReciprocalZone
using QuantumLattices: azimuth, contentnames, dimension, dtype, expand, kind, matrix, reciprocals, rcoordinate, update!, @rectangle_str, @σ_str
using TightBindingApproximation
using TightBindingApproximation.Fitting

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
    @test Metric(Phononic(), Hilbert(1=>Phonon(2))) == OperatorUnitToTuple(kind, :site, :direction)

    @test isnothing(commutator(Fermionic(:TBA), Hilbert(1=>Fock{:f}(1, 2))))
    @test isnothing(commutator(Fermionic(:BdG), Hilbert(1=>Fock{:f}(1, 2))))
    @test isnothing(commutator(Bosonic(:TBA), Hilbert(1=>Fock{:b}(1, 2))))
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
    @test dtype(tba.H) == dtype(typeof(tba.H)) == Float64
    @test dimension(tba) == 1
    @test Parameters(tba) == (t=1.0, μ=0.0)

    tba₁ = TBA{Fermionic{:TBA}}(tba.H.system, tba.H.transformation)
    tba₂ = TBA{Fermionic{:TBA}}(expand(tba.H.system), tba.H.transformation)
    tba₃ = TBA{Fermionic{:TBA}}(expand(tba.H.representation))
    @test dimension(tba₁) == dimension(tba₂) == dimension(tba₃) == 1
    @test Parameters(tba₁) == (t=1.0, μ=0.0)
    @test Parameters(tba₂) == Parameters(tba₃) == NamedTuple()

    m = matrix(tba)
    @test size(m) == (1, 1)
    @test m[1, 1] == m.H[1, 1]
    @test ishermitian(m) == ishermitian(typeof(m)) == true

    A(t, μ; k=[0.0, 0.0], kwargs...) = hcat(2t*cos(k[1])+2t*cos(k[2])+μ)
    tbaₐ = TBA{Fermionic{:TBA}}(A, (t=1.0, μ=0.0))
    @test dimension(tbaₐ) == 1
    @test Parameters(tbaₐ) == (t=1.0, μ=0.0)

    path = ReciprocalPath(reciprocals(lattice), rectangle"Γ-X-M-Γ", length=8)
    for kv in pairs(path)
        m = matrix(tba; kv...)
        m₁ = matrix(tba₁; kv...)
        m₂ = matrix(tba₂; kv...)
        m₃ = matrix(tba₃; kv...)
        mₐ = matrix(tbaₐ; kv...)
        @test m.H ≈ m₁.H ≈ m₂.H ≈ m₃.H ≈ mₐ.H ≈ Hermitian(A(1.0, 0.0; kv...))
    end
    @test eigen(tbaₐ, path) == (eigvals(tbaₐ, path), eigvecs(tbaₐ, path))

    update!(tba, μ=0.5)
    update!(tbaₐ, μ=0.5)
    for kv in pairs(path)
        m = matrix(tba; kv...)
        mₐ = matrix(tbaₐ; kv...)
        @test m.H ≈ mₐ.H ≈ Hermitian(A(1.0, 0.5; kv...))
    end

    Δ = Pairing(:Δ, Complex(0.1), 1, Coupling(:, FockIndex, :, :, (1, 1)); amplitude=bond->exp(im*azimuth(rcoordinate(bond))))
    bdg = TBA(lattice, hilbert, (t, μ, Δ))
    @test kind(bdg) == kind(typeof(bdg)) == Fermionic(:BdG)
    @test dtype(bdg.H) == dtype(typeof(bdg.H)) == Complex{Float64}
    @test Parameters(bdg) == (t=1.0, μ=0.5, Δ=Complex(0.1))

    A(t, μ, Δ; k, kwargs...) = [2t*cos(k[1])+2t*cos(k[2])+μ -2im*Δ*sin(k[1])-2Δ*sin(k[2]); 2im*Δ*sin(k[1])-2Δ*sin(k[2]) -2t*cos(k[1])-2t*cos(k[2])-μ]
    bdgₐ = TBA{Fermionic{:BdG}}(A, (t=1.0, μ=0.5, Δ=0.1))
    path = ReciprocalPath(reciprocals(lattice), rectangle"Γ-X-M-Γ", length=8)
    for kv in pairs(path)
        m = matrix(bdg; kv...)
        mₐ = matrix(bdgₐ; kv...)
        @test m.H ≈ mₐ.H ≈ Hermitian(A(1.0, 0.5, 0.1; kv...))
    end

    samplesets = [
        SampleNode(reciprocals(lattice), [0.0, 0.0], [1], [-4.0]),
        SampleNode(reciprocals(lattice), [0.5, 0.5], [1], [+4.0])
    ]
    op = optimize!(tba, samplesets)[2]
    @test isapprox(op.minimizer, [-1.0, 0.0], atol=10^-10)
end

@time @testset "EnergyBands and BerryCurvature" begin
    unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 3.5)
    Δ = Pairing(:Δ, Complex(0.5), 1, Coupling(:, FockIndex, :, :, (1, 1)); amplitude=bond->exp(im*azimuth(rcoordinate(bond))))
    sc = Algorithm(Symbol("p+ip"), TBA(unitcell, hilbert, (t, μ, Δ)))
    @test eigen(sc) == Eigen(eigvals(sc), eigvecs(sc))

    path = ReciprocalPath(reciprocals(unitcell), rectangle"Γ-X-M-Γ", length=100)
    @test eigen(sc, path) == (eigvals(sc, path), eigvecs(sc, path))
    savefig(plot(sc(:EB, EnergyBands(path))), "eb.png")

    brillouin = BrillouinZone(reciprocals(unitcell), 100)
    savefig(plot(sc(:BerryCurvature, BerryCurvature(brillouin, [1, 2]))), "bc.png")
    savefig(plot(sc(:BerryCurvatureNonabelian, BerryCurvature(brillouin, [1, 2], false))), "bctwobands.png")

    reciprocalzone = ReciprocalZone(reciprocals(unitcell), [-2.0=>2.0, -2.0=>2.0]; length=201, ends=(true, true))
    savefig(plot(sc(:BerryCurvatureExtended, BerryCurvature(reciprocalzone, [1, 2]))), "bcextended.png")
    kubobc = sc(:BerryCurvatureKubo, BerryCurvature(reciprocalzone, Kubo(0; d=0.1, kx=[1., 0], ky=[0, 1.])))
    savefig(plot(kubobc), "bcextended_Kubo.png")

    savefig(plot(sc(:BerryCurvaturePath, BerryCurvature(path, 0.0))), "bcpath.png")
end

@time @testset "FermiSurface and DensityOfStates" begin
    unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(unitcell))
    t = Hopping(:t, 1.0, 1)
    h = Onsite(:h, 0.1, MatrixCoupling(:, FockIndex, :, σ"z", :))
    tba = Algorithm(:tba, TBA(unitcell, hilbert, (t, h)))

    brillouin = BrillouinZone(reciprocals(unitcell), 200)
    savefig(plot(tba(:FermiSurface, FermiSurface(brillouin, 0.0))), "fs-all.png")
    savefig(plot(tba(Symbol("FermiSurface-SpinDependent"), FermiSurface(brillouin, 0.0, :, [1], [2]))), "fs-spin.png")
    savefig(plot(tba(:DensityOfStates, DensityOfStates(brillouin, :, :, [1], [2]; emin=-5.0, emax=5.0, ne=201, fwhm=0.05))), "dos.png")
    @test isapprox(sum(tba.assignments[:DensityOfStates].data[2][:, 1]), 2.0; atol=10^-3)

    reciprocalzone = ReciprocalZone(reciprocals(unitcell), [-2.0=>2.0, -2.0=>2.0]; length=401, ends=(true, true))
    savefig(plot(tba(:FermiSurfaceExtended, FermiSurface(reciprocalzone, 0.0))), "fs-extended-all.png")
    savefig(plot(tba(Symbol("FermiSurfaceExtended-SpinDependent"), FermiSurface(reciprocalzone, 0.0, :, [1], [2]))), "fs-extended-spin.png")
end

@time @testset "InelasticNeutronScatteringSpectra" begin
    unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(site=>Phonon(2) for site=1:length(unitcell))
    T = Kinetic(:T, 0.5)
    V₁ = Hooke(:V₁, 0.5, 1)
    V₂ = Hooke(:V₂, 0.25, 2)
    phonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))
    path = ReciprocalPath(reciprocals(unitcell), rectangle"Γ-X-M-Γ", length=100)

    energybands = phonon(:EB, EnergyBands(path))
    savefig(plot(energybands), "phonon.png")

    inelastic = phonon(:INSS, InelasticNeutronScatteringSpectra(path, range(0.0, 2.5, length=501); fwhm=0.05, scale=log))
    plt = plot()
    plot!(plt, inelastic)
    plot!(plt, energybands, color=:white, linestyle=:dash)
    savefig("inelastic.png")
end
