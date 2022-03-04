```@meta
CurrentModule = TightBindingApproximation
```

# Phonons on Square lattice

## Energy bands

The following codes could compute the energy bands of the phonons on the square lattice using the harmonic approximation on the phonon potential.

```@example phonon
using QuantumLattices
using TightBindingApproximation
using Plots: plot

# define the unitcell of the square lattice
unitcell = Lattice(:Square,
    [Point(PID(1), [0.0, 0.0])],
    vectors=[[1.0, 0.0], [0.0, 1.0]],
    neighbors=2
    )

# define the Hilbert space of phonons with 2 vibrant directions
hilbert = Hilbert(pid=>Phonon(2) for pid in unitcell.pids)

# define the terms

## Kinetic energy with the mass M=1
T = PhononKinetic(:T, 0.5)

## Potential energy on the nearest-neighbor bonds with the spring constant k₁=1.0
V₁ = PhononPotential(:V₁, 0.5, 1)

## Potential energy on the next-nearest-neighbor bonds with the spring constant k₂=0.5
V₂ = PhononPotential(:V₂, 0.25, 2)

# define the harmonic approximation of the phonons on square lattice
phonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(unitcell.reciprocals, rectangle"Γ-X-M-Γ", length=100)

# compute the energy bands along the above path
energybands = phonon(:EB, EnergyBands(path))

# plot the energy bands
plot(energybands)
```