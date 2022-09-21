```@meta
CurrentModule = TightBindingApproximation
```

# Graphene

## Energy bands

The following codes could compute the energy bands of the monolayer graphene.

```@example graphene
using QuantumLattices
using TightBindingApproximation
using Plots; pyplot()

# define the unitcell of the honeycomb lattice
unitcell = Lattice(
    [0.0, 0.0], [0.0, √3/3];
    name=:Honeycomb,
    vectors=[[1.0, 0.0], [0.5, √3/2]]
)

# define the Hilbert space of graphene (single-orbital spin-1/2 complex fermion)
hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(unitcell))

# define the terms, i.e. the nearest-neighbor hopping
t = Hopping(:t, -1.0, 1)

# define the tight-binding-approximation algorithm for graphene
graphene = Algorithm(:Graphene, TBA(unitcell, hilbert, (t,)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(unitcell.reciprocals, hexagon"Γ-K-M-Γ", length=100)

# compute the energy bands along the above path
energybands = graphene(:EB, EnergyBands(path))

# plot the energy bands
plot(energybands)
```

## Edge states

Graphene supports flatband edge states on zigzag boundaries. Only minor modifications are needed to compute them:
```@example graphene
# define a cylinder geometry with zigzag edges
lattice = Lattice(unitcell, translations"1P-100O")

# define the new Hilbert space corresponding to the cylinder
hilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(lattice))

# define the new tight-binding-approximation algorithm
zigzag = Algorithm(:Graphene, TBA(lattice, hilbert, (t,)))

# define new the path in the reciprocal space to compute the edge states
path = ReciprocalPath(lattice.reciprocals, line"Γ₁-Γ₂", length=100)

# compute the energy bands along the above path
edgestates = zigzag(:EB, EnergyBands(path))

# plot the energy bands
plot(edgestates)
```

## Auto-generation of the analytical expression of the Hamiltonian matrix

Combined with [SymPy](https://github.com/JuliaPy/SymPy.jl), it is also possible to get the analytical expression of the free Hamiltonian in the matrix form:
```@example graphene-analytical
using SymPy: Sym, symbols
using QuantumLattices
using TightBindingApproximation

unitcell = Lattice(
    [zero(Sym), zero(Sym)], [zero(Sym), √(one(Sym)*3)/3];
    name=:Honeycomb,
    vectors=[[one(Sym), zero(Sym)], [one(Sym)/2, √(one(Sym)*3)/2]]
)

hilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))

t = Hopping(:t, symbols("t", real=true), 1)

graphene = TBA(unitcell, hilbert, (t,))

k₁ = symbols("k₁", real=true)
k₂ = symbols("k₂", real=true)
m = matrix(graphene; k=[k₁, k₂])
```