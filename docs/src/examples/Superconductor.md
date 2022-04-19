```@meta
CurrentModule = TightBindingApproximation
```

# p+ip Superconductor on Square Lattice

## Energy bands

The following codes could compute the energy bands of the Bogoliubov quasiparticles of the p+ip topological superconductor on the square lattice.

```@example p+ip
using QuantumLattices
using TightBindingApproximation
using Plots; pyplot()

# define the unitcell of the square lattice
unitcell = Lattice(:Square,
    [Point(PID(1), [0.0, 0.0])],
    vectors=[[1.0, 0.0], [0.0, 1.0]],
    neighbors=1
    )

# define the Hilbert space of the p+ip superconductor (single-orbital spinless complex fermion)
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)

# define the terms

## nearest-neighbor hopping
t = Hopping(:t, 1.0, 1, modulate=true)

## onsite energy as the chemical potential
μ = Onsite(:μ, 3.5, modulate=true)

## p+ip pairing term
Δ = Pairing(:Δ, Complex(0.5), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))), modulate=true)

# define the Bogoliubov-de Gennes formula for the p+ip superconductor
sc = Algorithm(Symbol("p+ip"), TBA(unitcell, hilbert, (t, μ, Δ)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(unitcell.reciprocals, rectangle"Γ-X-M-Γ", length=100)

# compute the energy bands along the above path
energybands = sc(:EB, EnergyBands(path))

# plot the energy bands
plot(energybands)
```

## Berry curvature and Chern number
The Berry curvatures and the Chern numbers of the quasiparticle bands can be calculated as follows:
```@example p+ip
# define the Brillouin zone
brillouin = BrillouinZone(unitcell.reciprocals, 100)

# compute the Berry curvatures and Chern numbers of both quasiparticle bands
berry = sc(:BerryCurvature, BerryCurvature(brillouin, [1, 2]));

# plot the Berry curvature
plot(berry)
```

## Edge states

With tiny modification on the algorithm, the edge states of the p+ip topological superconductor could also be computed:
```@example p+ip
# define a cylinder geometry
lattice = Lattice(unitcell, translations"1P-100O")

# define the new Hilbert space corresponding to the cylinder
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)

# define the new Bogoliubov-de Gennes formula
sc = Algorithm(Symbol("p+ip"), TBA(lattice, hilbert, (t, μ, Δ)))

# define new the path in the reciprocal space to compute the edge states
path = ReciprocalPath(lattice.reciprocals, line"Γ₁-Γ₂", length=100)

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

unitcell = Lattice(:Square,
    [Point(PID(1), [zero(Sym), zero(Sym)])],
    vectors=[[one(Sym), zero(Sym)], [zero(Sym), one(Sym)]],
    neighbors=1
    )

hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)

t = Hopping(:t, symbols("t", real=true), 1)
μ = Onsite(:μ, symbols("μ", real=true))
Δ = Pairing(:Δ, symbols("Δ", real=true), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))))

sc = TBA(unitcell, hilbert, (t, μ, Δ))

k₁ = symbols("k₁", real=true)
k₂ = symbols("k₂", real=true)
m = matrix(sc; k=[k₁, k₂])
```