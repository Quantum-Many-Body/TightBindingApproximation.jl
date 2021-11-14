```@meta
CurrentModule = TightBindingApproximation
```

# Square lattice

## Energy bands for p+ip superconductor

The following codes could compute the energy bands of the Bogoliubov quasiparticles of the p+ip topological superconductor on the square lattice.

```@example p+ip
using QuantumLattices
using TightBindingApproximation
using Plots: plot

# define the unitcell of the square lattice
unitcell = Lattice("Square", [Point(PID(1), (0.0, 0.0), (0.0, 0.0))],
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
sc = Algorithm("p+ip", TBA(unitcell, hilbert, (t, μ, Δ)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(unitcell.reciprocals, rectangle"Γ-X-M-Γ", len=100)

# compute the energy bands along the above path
energybands = register!(sc, :EB, EnergyBands(path))

# plot the energy bands
plot(energybands)
```

With tiny modification on the algorithm, the edge states of the p+ip topological superconductor could also be computed:
```@example p+ip
# define a cylinder geometry
lattice = Lattice(unitcell, translations"1P-100O")

# define the new Hilbert space corresponding to the cylinder
hilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)

# define the new Bogoliubov-de Gennes formula
sc = Algorithm("p+ip", TBA(lattice, hilbert, (t, μ, Δ)))

# define new the path in the reciprocal space to compute the edge states
path = ReciprocalPath(lattice.reciprocals, line"Γ₁-Γ₂", len=100)

# compute the energy bands along the above path
edgestates = register!(sc, :Edge, EnergyBands(path))

# plot the energy bands
plot(edgestates)
```

Note that when μ>4 or μ<-4, the superconductor is topologically trivial such that there are no gapless edge states on the cylinder geometry:
```@example p+ip
# compute the edge states with a new onsite energy such that the superconductor is trivial
trivial = register!(sc, :Trivial, EnergyBands(path), parameters=(μ=4.5,))

# plot the energy bands
plot(trivial)
```