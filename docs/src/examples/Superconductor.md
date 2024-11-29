```@meta
CurrentModule = TightBindingApproximation
```

# p+ip Superconductor on Square Lattice

In this section, it is illustrated how the **Berry curvatures** and **Chern numbers** of free energy bands can be computed with the p+ip superconductor on the square lattice as the example.

## Energy bands

The following codes could compute the energy bands of the Bogoliubov quasiparticles of the p+ip topological superconductor on the square lattice.

```@example p+ip
using QuantumLattices
using TightBindingApproximation
using Plots

# define the unitcell of the square lattice
unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])

# define the Hilbert space of the p+ip superconductor (single-orbital spinless complex fermion)
hilbert = Hilbert(Fock{:f}(1, 1), length(unitcell))

# define the terms

## nearest-neighbor hopping
t = Hopping(:t, 1.0, 1)

## onsite energy as the chemical potential
μ = Onsite(:μ, 3.5)

## p+ip pairing term
Δ = Pairing(
    :Δ, Complex(0.5), 1, Coupling(:, 𝕗, :, :, (1, 1));
    amplitude=bond->exp(im*azimuth(rcoordinate(bond)))
)

# define the Bogoliubov-de Gennes formula for the p+ip superconductor
sc = Algorithm(Symbol("p+ip"), TBA(unitcell, hilbert, (t, μ, Δ)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(reciprocals(unitcell), rectangle"Γ-X-M-Γ", length=100)

# compute the energy bands along the above path
energybands = sc(:EB, EnergyBands(path))

# plot the energy bands
plot(energybands)
```

## Berry curvature and Chern number
The Berry curvatures and the Chern numbers of the quasiparticle bands can be calculated in the unitcell of the reciprocal space:
```@example p+ip
# define the Brillouin zone
brillouin = BrillouinZone(reciprocals(unitcell), 100)

# compute the Berry curvatures and Chern numbers of both quasiparticle bands
berry = sc(:BerryCurvature, BerryCurvature(brillouin, [1, 2]));

# plot the Berry curvatures
plot(berry)
```

The Berry curvatures can also be computed on a reciprocal zone beyond the reciprocal unitcell:
```@example p+ip
# define the reciprocal zone
reciprocalzone = ReciprocalZone(
    reciprocals(unitcell), -2.0=>2.0, -2.0=>2.0;
    length=201, ends=(true, true)
)

# compute the Berry curvature
berry = sc(:BerryCurvatureExtended, BerryCurvature(reciprocalzone, [1, 2]))

# plot the Berry curvature
plot(berry)
```

The total Berry curvatures of occupied bands can also be computed on a reciprocal path with Kubo method:
```@example p+ip
# compute the total Berry curvature 
berry = sc(:BerryCurvaturePath, BerryCurvature(path, Kubo(0.0)))

# plot the Berry curvature
plot(berry)
```

## Edge states

With tiny modification on the algorithm, the edge states of the p+ip topological superconductor could also be computed:
```@example p+ip
# define a cylinder geometry
lattice = Lattice(unitcell, (1, 100), ('P', 'O'))

# define the new Hilbert space corresponding to the cylinder
hilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(lattice))

# define the new Bogoliubov-de Gennes formula
sc = Algorithm(Symbol("p+ip"), TBA(lattice, hilbert, (t, μ, Δ)))

# define the new path in the reciprocal space to compute the edge states
path = ReciprocalPath(reciprocals(lattice), line"Γ₁-Γ₂", length=100)

# compute the energy bands along the above path
edgestates = sc(:Edge, EnergyBands(path))

# plot the energy bands
plot(edgestates)
```

Note that when μ>4 or μ<-4, the superconductor is topologically trivial such that there are no gapless edge states on the cylinder geometry:
```@example p+ip
# compute the edge states with a new onsite energy such that the superconductor is trivial
trivial = sc(:Trivial, EnergyBands(path), parameters=(μ=4.5,))

# plot the energy bands
plot(trivial)
```

## Auto-generation of the analytical expression of the Hamiltonian matrix

Combined with [SymPy](https://github.com/JuliaPy/SymPy.jl), it is also possible to get the analytical expression of the free Hamiltonian in the matrix form:
```@example p+ip-analytical
using SymPy: Sym, symbols
using QuantumLattices
using TightBindingApproximation

unitcell = Lattice(
    [zero(Sym), zero(Sym)];
    vectors=[[one(Sym), zero(Sym)], [zero(Sym), one(Sym)]]
)

hilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))

t = Hopping(:t, symbols("t", real=true), 1)
μ = Onsite(:μ, symbols("μ", real=true))
Δ = Pairing(
    :Δ, symbols("Δ", real=true), 1, Coupling(:, 𝕗, :, :, (1, 1));
    amplitude=bond->exp(im*azimuth(rcoordinate(bond)))
)

sc = TBA(unitcell, hilbert, (t, μ, Δ))

k₁ = symbols("k₁", real=true)
k₂ = symbols("k₂", real=true)
m = matrix(sc, [k₁, k₂])
```