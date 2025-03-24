using Artifacts
using LinearAlgebra: Diagonal, Eigen, Hermitian, ishermitian
using Pkg
using Plots: plot, plot!, savefig
using QuantumLattices: Algorithm, BrillouinZone, Coupling, Elastic, Fock, Formula, Generator, Hilbert, Hooke, Hopping, Kinetic, Lattice, Metric, Onsite, OperatorSum, OperatorIndexToTuple, Pairing, Parameters, Phonon, ReciprocalPath, ReciprocalZone, Table
using QuantumLattices: atol, 𝕕, 𝕕⁺𝕕, 𝕡, 𝕦, azimuth, bonds, dimension, distance, expand, getcontent, kind, matrix, parameternames, reciprocals, rcoordinate, scalartype, update!, @rectangle_str, @σ_str
using StaticArrays: SVector
using TightBindingApproximation
using TightBindingApproximation.Fitting
using TightBindingApproximation.Wannier90

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

    @test infinitesimal(Fermionic(:TBA)) == infinitesimal(Fermionic(:BdG)) == infinitesimal(Bosonic(:TBA)) == 0
    @test infinitesimal(Bosonic(:BdG)) == infinitesimal(Phononic()) == atol/5

    @test Metric(Fermionic(:TBA), Hilbert(1=>Fock{:f}(1, 1))) == OperatorIndexToTuple(:site, :orbital, :spin)
    @test Metric(Fermionic(:BdG), Hilbert(1=>Fock{:f}(1, 1))) == OperatorIndexToTuple(:nambu, :site, :orbital, :spin)
    @test Metric(Bosonic(:TBA), Hilbert(1=>Fock{:b}(1, 1))) == OperatorIndexToTuple(:site, :orbital, :spin)
    @test Metric(Bosonic(:BdG), Hilbert(1=>Fock{:b}(1, 1))) == OperatorIndexToTuple(:nambu, :site, :orbital, :spin)
    @test Metric(Phononic(), Hilbert(1=>Phonon(2))) == OperatorIndexToTuple(kind, :site, :direction)

    by = OperatorIndexToTuple(kind, :site, :direction)
    @test Table(Hilbert(Phonon(2), 2), by) == Table([𝕦(1, 'x'), 𝕦(2, 'x'), 𝕦(1, 'y'), 𝕦(2, 'y'), 𝕡(1, 'x'), 𝕡(2, 'x'), 𝕡(1, 'y'), 𝕡(2, 'y')], by)

    @test isnothing(commutator(Fermionic(:TBA), Hilbert(1=>Fock{:f}(1, 2))))
    @test isnothing(commutator(Fermionic(:BdG), Hilbert(1=>Fock{:f}(1, 2))))
    @test isnothing(commutator(Bosonic(:TBA), Hilbert(1=>Fock{:b}(1, 2))))
    @test commutator(Bosonic(:BdG), Hilbert(1=>Fock{:b}(1, 2))) == Diagonal([1, 1, -1, -1])
    @test commutator(Phononic(), Hilbert(1=>Phonon(2))) == Hermitian([0 0 -im 0; 0 0 0 -im; im 0 0 0; 0 im 0 0])
end

@time @testset "Quadratic" begin
    m = Quadratic(1.0, (1, 2), [0.0], [0.0])
    @test parameternames(typeof(m)) == (:value, :coordinate)
    @test getcontent(m, :id) == ((1, 2), [0.0], [0.0])
    @test m == Quadratic(1.0, ((1, 2), [0.0], [0.0]))
    @test string(m) == "Quadratic(1.0, (1, 2), [0.0], [0.0])"
end

@time @testset "Quadraticization{<:TBAKind{:TBA}}" begin
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, 2.0)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (t, μ); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test valtype(quadraticization, expand(gen)) == OperatorSum{Quadratic{Float64, SVector{1, Float64}}, Tuple{Tuple{Int, Int}, SVector{1, Float64}, SVector{1, Float64}}}
    @test quadraticization(first(expand(gen))) == OperatorSum(Quadratic(-1.0, (2, 1), [-0.5], [0.0]))
    ops = quadraticization(expand(gen))
    @test ops == OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(-1.0, (2, 1), [0.5], [1.0]),
        Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0])
    )
end

@time @testset "Quadraticization{Fermionic{:BdG}}" begin
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, 2.0)
    Δ = Pairing(:Δ, 0.1, 1, Coupling(𝕕(:, :, :, :), 𝕕(:, :, :, :)); amplitude=bond->rcoordinate(bond)[1]>0 ? 1 : -1)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (t, μ, Δ); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test quadraticization(expand(gen)) == OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(1.0, (3, 4), [0.5], [-0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(1.0, (4, 3), [-0.5], [-0.0]),
        Quadratic(-1.0, (2, 1), [0.5], [1.0]), Quadratic(1.0, (3, 4), [-0.5], [-1.0]), Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(1.0, (4, 3), [0.5], [1.0]),
        Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(-2.0, (3, 3), [-0.0], [-0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0]), Quadratic(-2.0, (4, 4), [-0.0], [-0.0]),
        Quadratic(-0.1, (4, 1), [-0.5], [0.0]), Quadratic(-0.1, (1, 4), [0.5], [0.0]), Quadratic(0.1, (3, 2), [0.5], [0.0]), Quadratic(0.1, (2, 3), [-0.5], [0.0]),
        Quadratic(0.1, (4, 1), [0.5], [1.0]), Quadratic(0.1, (1, 4), [-0.5], [-1.0]), Quadratic(-0.1, (3, 2), [-0.5], [-1.0]), Quadratic(-0.1, (2, 3), [0.5], [1.0])
    )
end

@time @testset "Quadraticization{Bosonic{:BdG}}" begin
    t = Hopping(:t, -1.0, 1)
    μ = Onsite(:μ, 2.0)
    Δ = Pairing(:Δ, 0.1, 1, Coupling(𝕕(:, :, :, :), 𝕕(:, :, :, :)); amplitude=bond->rcoordinate(bond)[1]>0 ? 1 : -1)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Fock{:b}(1, 1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (t, μ, Δ); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test quadraticization(expand(gen)) == OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(-1.0, (3, 4), [0.5], [-0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(-1.0, (4, 3), [-0.5], [-0.0]),
        Quadratic(-1.0, (2, 1), [0.5], [1.0]), Quadratic(-1.0, (3, 4), [-0.5], [-1.0]), Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(-1.0, (4, 3), [0.5], [1.0]),
        Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(2.0, (3, 3), [-0.0], [-0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0]), Quadratic(2.0, (4, 4), [-0.0], [-0.0]),
        Quadratic(-0.1, (4, 1), [-0.5], [0.0]), Quadratic(-0.1, (1, 4), [0.5], [0.0]), Quadratic(0.1, (3, 2), [0.5], [0.0]), Quadratic(0.1, (2, 3), [-0.5], [0.0]),
        Quadratic(0.1, (4, 1), [0.5], [1.0]), Quadratic(0.1, (1, 4), [-0.5], [-1.0]), Quadratic(-0.1, (3, 2), [-0.5], [-1.0]), Quadratic(-0.1, (2, 3), [0.5], [1.0])
    )
end

@time @testset "Quadraticization{Phononic}" begin
    T = Kinetic(:T, 0.5)
    V = Hooke(:V, 2.0, 1)
    lattice = Lattice([0.0], [0.5]; vectors=[[1.0]])
    hilbert = Hilbert(Phonon(1), length(lattice))
    gen = Generator(bonds(lattice, 1), hilbert, (T, V); half=false)
    tbakind = TBAKind(typeof(gen.terms), valtype(gen.hilbert))
    table = Table(gen.hilbert, Metric(tbakind, gen.hilbert))
    quadraticization = Quadraticization{typeof(tbakind)}(table)
    @test quadraticization(expand(gen)) == OperatorSum(
        Quadratic(0.5, (1, 1), [0.0], [0.0]), Quadratic(0.5, (1, 1), [-0.0], [-0.0]), Quadratic(0.5, (2, 2), [0.0], [0.0]), Quadratic(0.5, (2, 2), [-0.0], [-0.0]),
        Quadratic(4.0, (4, 4), [0.0], [0.0]), Quadratic(4.0, (4, 4), [-0.0], [-0.0]), Quadratic(-2.0, (4, 3), [-0.5], [0.0]), Quadratic(-2.0, (3, 4), [0.5], [-0.0]),
        Quadratic(-2.0, (3, 4), [0.5], [0.0]), Quadratic(-2.0, (4, 3), [-0.5], [-0.0]), Quadratic(4.0, (3, 3), [0.0], [0.0]), Quadratic(4.0, (3, 3), [-0.0], [-0.0]),
        Quadratic(-4.0, (4, 3), [0.5], [1.0]), Quadratic(-4.0, (3, 4), [-0.5], [-1.0])
    )
end

@time @testset "TBAMatrixization" begin
    ops = OperatorSum(
        Quadratic(-1.0, (2, 1), [-0.5], [0.0]), Quadratic(-1.0, (1, 2), [0.5], [0.0]), Quadratic(-1.0, (2, 1), [0.5], [1.0]),
        Quadratic(-1.0, (1, 2), [-0.5], [-1.0]), Quadratic(2.0, (1, 1), [0.0], [0.0]), Quadratic(2.0, (2, 2), [0.0], [0.0])
    )
    matrixization = TBAMatrixization{ComplexF64}([0.0], 2, :icoordinate)
    @test valtype(matrixization, ops) == Matrix{ComplexF64}
    @test zero(matrixization, ops) == zeros(ComplexF64, 2, 2)
    @test matrixization(first(ops)) == [0.0+0.0im 0.0+0.0im; -1.0+0.0im 0.0+0.0im]

    @test matrixization(ops) == [2.0+0.0im -2.0+0.0im; -2.0+0.0im 2.0+0.0im]
    @test TBAMatrixization{ComplexF64}([pi/2], 2, :icoordinate)(ops) == [2.0+0.0im -1.0+1.0im; -1.0-1.0im 2.0+0.0im]
    @test TBAMatrixization{ComplexF64}([pi/2], 2, :rcoordinate)(ops) ≈ [2.0+0.0im -√2+0.0im; -√2+0.0im 2.0+0.0im]
    @test TBAMatrixization{Float64}(nothing, 2, :rcoordinate)(ops) == TBAMatrixization{Float64}(nothing, 2, :icoordinate)(ops) == [2.0 -2.0; -2.0 2.0]
end

@time @testset "TBAMatrix" begin
    data = [2.0+0.0im -2.0+0.0im; -2.0+0.0im 2.0+0.0im]
    m = TBAMatrix(Hermitian(data), nothing)
    @test size(m) == (2, 2)
    @test [m[i, j] for i=1:2, j=1:2] == data
    @test ishermitian(m) == ishermitian(typeof(m)) == true
    @test Hermitian(m) == Hermitian(data)
    @test Matrix(m) == data

    es, vs = [0.0, 4.0], [-√2/2 -√2/2; -√2/2 √2/2]
    @test [eigen(m)...] ≈ [es, vs]
    @test eigvals(m) ≈ es
    @test eigvecs(m) ≈ vs

    data = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 8.0 -4.0+4.0im; 0.0 0.0 -4.0-4.0im 8.0]
    commutator = [0 0 -im 0; 0 0 0 -im; im 0 0 0; 0 im 0 0]
    m = TBAMatrix(Hermitian(data), commutator)
    es = [3.695518130045136, 1.5307337294603554, 1.5307337294603642, 3.695518130045147]
    vs = [
        -0.679661508587653-0.6796615085876528im     0.43742624084814996+0.4374262408481499im    -0.43742624084815235-0.43742624084815235im  0.6796615085876522+0.6796615085876521im;
        0.9611865232676121im                        0.6186141223453353im                        -0.6186141223453371im                       -0.9611865232676166im;
        -0.18391507893355885+0.1839150789335589im   0.28576246307864384-0.28576246307864384im   0.28576246307864483-0.28576246307864483im   -0.18391507893355896+0.18391507893355902im
        0.26009519895275734                         0.40412915090295953                         0.40412915090295987                         0.26009519895275796
    ]
    @test [eigen(m)...] ≈ [es, vs]
    @test eigvals(m) ≈ es
    @test eigvecs(m) ≈ vs
end

@time @testset "TBA" begin
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 0.0)
    lattice = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(lattice))
    tba = TBA(lattice, hilbert, (t, μ))
    @test kind(tba) == kind(typeof(tba)) == Fermionic(:TBA)
    @test isnothing(getcontent(tba, :commutator))
    @test scalartype(tba) == scalartype(typeof(tba)) == Float64
    @test Parameters(tba) == (t=1.0, μ=0.0)
    @test dimension(tba) == 1

    A(t, μ, k=[0.0, 0.0]; kwargs...) = [2t*cos(k[1])+2t*cos(k[2])+μ;;]
    another = TBA{Fermionic{:TBA}}(Formula(A, (t=1.0, μ=0.0)))
    @test scalartype(another) == scalartype(typeof(another)) == Float64
    @test Parameters(another) == (t=1.0, μ=0.0)
    @test dimension(another) == 1

    third = TBA{Fermionic{:TBA}}(tba.H)
    @test scalartype(third) == scalartype(typeof(third)) == Float64
    @test Parameters(third) == (t=1.0, μ=0.0)
    @test dimension(third) == 1

    fourth = TBA{Fermionic{:TBA}}(tba.system, tba.quadraticization)
    @test scalartype(fourth) == scalartype(typeof(fourth)) == Float64
    @test Parameters(fourth) == (t=1.0, μ=0.0)
    @test dimension(fourth) == 1

    path = ReciprocalPath(reciprocals(lattice), rectangle"Γ-X-M-Γ", length=8)
    for k in path
        @test matrix(tba, k) ≈ matrix(another, k) ≈ matrix(third, k) ≈ matrix(fourth, k)
        @test [eigen(tba, k)...] ≈ [eigen(another, k)...] ≈ [eigen(third, k)...] ≈ [eigen(fourth, k)...]
        @test eigvals(tba, k) ≈ eigvals(another, k) ≈ eigvals(third, k) ≈ eigvals(fourth, k)
        @test eigvecs(tba, k) ≈ eigvecs(another, k) ≈ eigvecs(third, k) ≈ eigvecs(fourth, k)
    end
    @test [eigen(tba, path)...] ≈ [eigen(another, path)...] ≈ [eigen(third, path)...] ≈ [eigen(fourth, path)...]
    @test eigvals(tba, path) ≈ eigvals(another, path) ≈ eigvals(third, path) ≈ eigvals(fourth, path)
    @test eigvecs(tba, path) ≈ eigvecs(another, path) ≈ eigvecs(third, path) ≈ eigvecs(fourth, path)

    update!(tba; μ=2.2)
    update!(another; μ=2.2)
    update!(third; μ=2.2)
    update!(fourth; μ=2.2)
    @test [eigen(tba, path)...] ≈ [eigen(another, path)...] ≈ [eigen(third, path)...] ≈ [eigen(fourth, path)...]
end

@time @testset "EnergyBands and BerryCurvature" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Fock{:f}(1, 1), length(unitcell))
    t = Hopping(:t, 1.0, 1)
    μ = Onsite(:μ, 3.5)
    Δ = Pairing(:Δ, Complex(0.5), 1, Coupling(𝕕, :, :, :, (1, 1)); amplitude=bond->exp(im*azimuth(rcoordinate(bond))))
    sc = Algorithm(Symbol("p+ip"), TBA(unitcell, hilbert, (t, μ, Δ)))
    @test matrix(sc) == matrix(sc.frontend)
    @test eigen(sc) == Eigen(eigvals(sc), eigvecs(sc))

    path = ReciprocalPath(reciprocals(unitcell), rectangle"Γ-X-M-Γ", length=100)
    @test eigen(sc, path) == (eigvals(sc, path), eigvecs(sc, path))
    savefig(plot(sc(:EB, EnergyBands(path, 1:2))), "EB.png")
    savefig(plot(sc(:EB, EnergyBands(path, :, 1:1, 2:2)); weightlabels=["particle", "hole"], legend=(0.4, 0.1), legendfont=10, legendcolumn=-1), "EBWeighted.png")

    brillouin = BrillouinZone(reciprocals(unitcell), 100)
    savefig(plot(sc(:BerryCurvatureAbelian, BerryCurvature(brillouin, [1, 2]))), "BerryCurvatureAbelian.png")
    savefig(plot(sc(:BerryCurvatureNonabelian, BerryCurvature(brillouin, [1, 2], false))), "BerryCurvatureNonabelian.png")
    savefig(plot(sc(:BerryCurvatureKubo, BerryCurvature(brillouin, Kubo(0; d=0.1, kx=[1.0, 0.0], ky=[0.0, 1.0])))), "BerryCurvatureKubo.png")

    reciprocalzone = ReciprocalZone(reciprocals(unitcell), [-2.0=>2.0, -2.0=>2.0]; length=201, ends=(true, true))
    savefig(plot(sc(:BerryCurvatureExtendedFukui, BerryCurvature(reciprocalzone, [1, 2]))), "BerryCurvatureExtendedFukui.png")
    savefig(plot(sc(:BerryCurvatureExtendedKubo, BerryCurvature(reciprocalzone, Kubo(0; d=0.1, kx=[1., 0], ky=[0, 1.])))), "BerryCurvatureExtendedKubo.png")
    savefig(plot(sc(:BerryCurvaturePath, BerryCurvature(path, 0.0))), "BerryCurvaturePath.png")
end

@time @testset "FermiSurface and DensityOfStates" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Fock{:f}(1, 2), length(unitcell))
    t = Hopping(:t, 1.0, 1)
    h = Onsite(:h, 0.1, 𝕕⁺𝕕(:, :, σ"z", :))
    tba = Algorithm(:tba, TBA(unitcell, hilbert, (t, h)))

    brillouin = BrillouinZone(reciprocals(unitcell), 200)
    savefig(plot(tba(:FermiSurface, FermiSurface(brillouin, 0.0))), "FermiSurface.png")
    savefig(plot(tba(:FermiSurfaceSpinDependent, FermiSurface(brillouin, 0.0, 1:2, [1], [2]))), "FermiSurfaceSpinDependent.png")
    savefig(plot(tba(:DensityOfStates, DensityOfStates(brillouin, :, :, [1], [2]; emin=-5.0, emax=5.0, ne=201, fwhm=0.05))), "DensityOfStates.png")
    @test isapprox(sum(tba.assignments[:DensityOfStates].data[2][:, 1]), 2.0; atol=10^-3)

    reciprocalzone = ReciprocalZone(reciprocals(unitcell), [-2.0=>2.0, -2.0=>2.0]; length=401, ends=(true, true))
    savefig(plot(tba(:FermiSurfaceExtended, FermiSurface(reciprocalzone, 0.0))), "FermiSurfaceExtended.png")
    savefig(plot(tba(:FermiSurfaceExtendedSpinDependent, FermiSurface(reciprocalzone, 0.0, :, [1], [2]))), "FermiSurfaceExtendedSpinDependent.png")
end

@time @testset "InelasticNeutronScatteringSpectra" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    hilbert = Hilbert(Phonon(2), length(unitcell))
    T = Kinetic(:T, 0.5)
    V₁ = Hooke(:V₁, 0.5, 1)
    V₂ = Hooke(:V₂, 0.25, 2)
    phonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))
    path = ReciprocalPath(reciprocals(unitcell), rectangle"Γ-X-M-Γ", length=100)

    energybands = phonon(:EB, EnergyBands(path))
    savefig(plot(energybands), "PhononEB.png")

    inelastic = phonon(:INSS, InelasticNeutronScatteringSpectra(path, range(0.0, 2.5, length=501); fwhm=0.05, rescale=x->log(1+x)))
    plt = plot()
    plot!(plt, inelastic)
    plot!(plt, energybands, color=:white, linestyle=:dash)
    savefig("InelasticNeutronScatteringSpectra.png")
end

@time @testset "Fitting" begin
    lattice = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    A(t, μ, Δ, k=[0.0, 0.0]; kwargs...) = [2t*cos(k[1])+2t*cos(k[2])+μ -2im*Δ*sin(k[1])-2Δ*sin(k[2]); 2im*Δ*sin(k[1])-2Δ*sin(k[2]) -2t*cos(k[1])-2t*cos(k[2])-μ]
    tba = TBA{Fermionic{:BdG}}(Formula(A, (t=2.0, μ=0.3, Δ=1.0)))
    samplesets = [
        SampleNode(reciprocals(lattice), [0.0, 0.0], [1, 2], [-16.0, 16.0], 1.0),
        SampleNode(reciprocals(lattice), [0.25, 0.0], [1, 2], [-10.0, 10.0], 1.0),
        SampleNode(reciprocals(lattice), [0.25, 0.25], [1, 2], [-8.485281374238571, 8.485281374238571], 1.0)
    ]
    op = optimize!(tba, samplesets)[2]
    @test isapprox(op.minimizer, [4.0, 0.0, 3.0], atol=5*10^-10)
end

toml = Artifacts.find_artifacts_toml(@__DIR__)
const dir = Pkg.Artifacts.ensure_artifact_installed("WannierDataSets", toml)
@time @testset "Wannier90" begin
    prefix = "silicon"
    wan = Algorithm(:silicon, W90(dir, prefix))
    @test W90Matrixization([0.0, 0.0, 0.0], wan.frontend.lattice.vectors, wan.frontend.centers, :icoordinate)(wan.frontend.H[1]) ≈ ComplexF64[
        0.016239+4.75e-6im -0.00301675+2.5e-6im -0.0030155-6.0e-6im -0.0030155-3.25e-6im -0.01056425-1.3e-5im 0.00237225+7.5e-7im 0.002374+5.75e-6im 0.0023735+2.5e-6im;
        -0.0030155+3.25e-6im 0.01623875-2.5e-7im -0.003016-1.0e-6im -0.00301625-3.25e-6im 0.0023735+1.0e-6im 0.00477025+2.5e-7im -0.0055565-2.5e-6im -0.00555775+3.25e-6im;
        -0.0030175-6.0e-6im -0.0030155+2.5e-7im 0.01623925+5.0e-7im -0.0030155+5.75e-6im 0.0023755+4.25e-6im -0.0055575-1.25e-6im 0.004769+1.5e-6im -0.0055575-2.75e-6im;
        -0.00301575-1.5e-6im -0.00301625-2.5e-7im -0.00301625+5.75e-6im 0.0162385-1.75e-6im 0.00237375+3.5e-6im -0.00555675+2.75e-6im -0.0055585-4.0e-6im 0.00477075-7.5e-7im;
        0.008246-3.0e-6im 0.0022645+0.0im 0.00226525+4.25e-6im 0.0022645-7.5e-7im 0.016239+4.75e-6im -0.00301475+4.0e-6im -0.00301675-8.25e-6im -0.003016+2.5e-7im;
        0.00226475+7.5e-7im -0.00638575-2.25e-6im -0.00396775+1.5e-6im -0.0039695-1.25e-6im -0.00301675+3.0e-6im 0.016238-5.0e-7im -0.0030145+7.5e-7im -0.0030165-1.25e-6im;
        0.00226725+2.5e-6im -0.00396925+1.75e-6im -0.0063855-3.5e-6im -0.00396775-3.25e-6im -0.0030155-7.5e-6im -0.0030175-7.5e-7im 0.016239+1.0e-6im -0.00301575+5.5e-6im;
        0.0022645+5.0e-7im -0.003968-1.75e-6im -0.0039695-3.25e-6im -0.00638525+4.25e-6im -0.0030165-2.5e-6im -0.0030155-4.25e-6im -0.00301625+7.0e-6im 0.016239-2.0e-6im
    ]
    path = ReciprocalPath(
        reciprocals(wan.frontend.lattice), (0.5, 0.5, 0.5)=>(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)=>(0.5, 0.0, 0.5), (0.5, -0.5, 0.0)=>(0.375, -0.375, 0.0), (0.375, -0.375, 0.0)=>(0.0, 0.0, 0.0);
        labels=("L"=>"Γ", "Γ"=>"X", "X'"=>"K", "K"=>"Γ"), length=100
    )
    plt = plot()
    plot!(plt, readbands(dir, prefix)...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
    plot!(plt, wan(:EB, EnergyBands(path; gauge=:icoordinate)), color=:black)
    savefig("silicon_wannier90_center.png")

    another = Algorithm(:silicon, W90(wan.frontend.lattice, Hilbert(Fock{:f}(4, 1), length(wan.frontend.lattice)), wan.frontend.H))
    update!(another)
    plt =  plot()
    plot!(plt, readbands(dir, prefix)...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
    plot!(plt, another(:EB, EnergyBands(path; gauge=:rcoordinate)), color=:black)
    savefig("silicon_wannier90_atom.png")
end
