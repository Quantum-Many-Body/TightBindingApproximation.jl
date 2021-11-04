```@meta
CurrentModule = TightBindingApproximation
```

# Honeycomb lattice

## Energy bands for graphene

The following codes could compute the energy bands of monolayer graphene.

```@example graphene
using QuantumLattices
using TightBindingApproximation
using Plots: plot

# define the unitcell of graphene
unitcell = Lattice("Hexagon", [Point(PID(1), (0.0, 0.0), (0.0, 0.0)), Point(PID(2), (0.0, √3/3), (0.0, 0.0))],
    vectors=[[1.0, 0.0], [0.5, √3/2]],
    neighbors=1
    )

# define the Hilbert space
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)

# define the terms
t = Hopping(:t, -1.0, 1, modulate=true)

# define the tight-binding-approximation algorithm for graphene
tba = Algorithm("Graphene", TBA(unitcell, hilbert, (t,)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(unitcell.reciprocals, hexagon"Γ-K-M-Γ", len=100)

# compute the energy bands along the above path
eb = register!(tba, :EB, TBAEB(path))

# plot the energy bands
plot(eb)
```

To compute the edge states:
```@example graphene
# define a cylinder geometry with zigzag edges
lattice = Lattice(unitcell, translations"1P-100O")

# define the new Hilbert space
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)

# define the new tight-binding-approximation algorithm
tba = Algorithm("Graphene", TBA(lattice, hilbert, (t,)))

# define new the path in the reciprocal space to compute the edge states
path = ReciprocalPath(lattice.reciprocals, line"Γ₁-Γ₂", len=100)

# compute the energy bands along the above path
edgestates = register!(tba, :EB, TBAEB(path))

# plot the energy bands
plot(edgestates)
```