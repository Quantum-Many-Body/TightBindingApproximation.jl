```@meta
CurrentModule = TightBindingApproximation
```

# Square lattice

## Energy bands for p+ip superconductor

```@example p+ip
using QuantumLattices
using TightBindingApproximation
using Plots: plot

# define the unitcell of graphene
unitcell = Lattice("Square", [Point(PID(1), (0.0, 0.0), (0.0, 0.0))],
    vectors=[[1.0, 0.0], [0.0, 1.0]],
    neighbors=1
    )

# define the Hilbert space
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)

# define the terms
t = Hopping(:t, 1.0, 1, modulate=true)
μ = Onsite(:μ, 3.5, modulate=true)
Δ = Pairing(:Δ, Complex(0.5), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))), modulate=true)

# define the tight-binding-approximation algorithm for graphene
tba = Algorithm("p+ip", TBA(unitcell, hilbert, (t, μ, Δ)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(unitcell.reciprocals, rectangle"Γ-X-M-Γ", len=100)

# compute the energy bands along the above path
eb = register!(tba, :EB, TBAEB(path))

# plot the energy bands
plot(eb)
```

To compute the edge states:
```@example p+ip
# define a cylinder geometry with zigzag edges
lattice = Lattice(unitcell, translations"1P-100O")

# define the new Hilbert space
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)

# define the new tight-binding-approximation algorithm
tba = Algorithm("p+ip", TBA(lattice, hilbert, (t, μ, Δ)))

# define new the path in the reciprocal space to compute the edge states
path = ReciprocalPath(lattice.reciprocals, line"Γ₁-Γ₂", len=100)

# compute the energy bands along the above path
edgestates = register!(tba, :EB, TBAEB(path))

# plot the energy bands
plot(edgestates)
```