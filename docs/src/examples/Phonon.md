```@meta
CurrentModule = TightBindingApproximation
```

# Phonons on Square lattice

## Energy bands

The following codes could compute the energy bands of the phonons on the square lattice using the harmonic approximation on the phonon potential.

```@example phonon
using QuantumLattices
using TightBindingApproximation
using Plots

# define the unitcell of the square lattice
unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])

# define the Hilbert space of phonons with 2 vibrant directions
hilbert = Hilbert(site=>Phonon(2) for site=1:length(unitcell))

# define the terms

## Kinetic energy with the mass M=1
T = Kinetic(:T, 0.5)

## Potential energy on the nearest-neighbor bonds with the spring constant k₁=1.0
V₁ = Hooke(:V₁, 0.5, 1)

## Potential energy on the next-nearest-neighbor bonds with the spring constant k₂=0.5
V₂ = Hooke(:V₂, 0.25, 2)

# define the harmonic approximation of the phonons on square lattice
phonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(unitcell.reciprocals, rectangle"Γ-X-M-Γ", length=100)

# compute the energy bands along the above path
energybands = phonon(:EB, EnergyBands(path))

# plot the energy bands
plot(energybands)
```

## Inelastic neutron scattering spectra

The inelastic neutron scattering spectra of phonons can also be computed:
```@example phonon
# fwhm: the FWHM of the Gaussian to be convoluted
# scale: the scale of the intensity
spectra = phonon(
    :INSS,
    InelasticNeutronScatteringSpectra(path, range(0.0, 2.5, length=501); fwhm=0.05, scale=log)
)
plt = plot()
plot!(plt, spectra)
plot!(plt, energybands, color=:white, linestyle=:dash)
```