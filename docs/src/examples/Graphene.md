```@meta
CurrentModule = TightBindingApproximation
```

# Graphene

## Energy bands

The following codes could compute the energy bands of the monolayer graphene.

```@example graphene
using QuantumLattices
using TightBindingApproximation
using Plots: plot

# define the unitcell of the honeycomb lattice
unitcell = Lattice(:Honeycomb,
    [Point(PID(1), [0.0, 0.0]), Point(PID(2), [0.0, √3/3])],
    vectors=[[1.0, 0.0], [0.5, √3/2]],
    neighbors=1
    )

# define the Hilbert space of graphene (single-orbital spin-1/2 complex fermion)
hilbert = Hilbert(pid=>Fock{:f}(1, 2, 2) for pid in unitcell.pids)

# define the terms, i.e. the nearest-neighbor hopping
t = Hopping(:t, -1.0, 1, modulate=true)

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
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)

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

unitcell = Lattice(:Honeycomb,
    [Point(PID(1), [zero(Sym), zero(Sym)]), Point(PID(2), [zero(Sym), √(one(Sym)*3)/3])],
    vectors=[[one(Sym), zero(Sym)], [one(Sym)/2, √(one(Sym)*3)/2]],
    neighbors=1
    )

hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)

t = Hopping(:t, symbols("t", real=true), 1)

graphene = TBA(unitcell, hilbert, (t,))

k₁ = symbols("k₁", real=true)
k₂ = symbols("k₂", real=true)
m = matrix(graphene; k=[k₁, k₂])
display(m)
```