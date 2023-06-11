```@meta
CurrentModule = TightBindingApproximation
```

# Free Spinless Fermions on Kagome Lattice

In this section, it is illustrated how the **density of states** (DOS) and **Fermi surfaces** (FS) of tight-binding systems can be computed with the free spinless fermions on Kagome lattice as the example.

## Lattice structure

Let's first define the structure of the Kagome lattice and display it by the [QuantumLattices](https://github.com/Quantum-Many-Body/QuantumLattices.jl) package:

```@example kagome
using QuantumLattices
using Plots

# define the unitcell of the Kagome lattice
unitcell = Lattice(
    [0.0, 0.0], [0.5, 0.0], [0.25, √3/4];
    name=:Kagome,
    vectors=[[1.0, 0.0], [0.5, √3/2]]
)

# define a finite Kagome lattice containing 8×8 unitcells with the open boundary condition
lattice = Lattice(unitcell, (8, 8))

# plot the finite lattice
plot(lattice, 1)
```

## Energy bands

Now let's calculate the energy bands of the Kagome lattice:

```@example kagome
using TightBindingApproximation

# define the Hilbert space (single-orbital spinless complex fermion)
hilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))

# define the terms, i.e., the nearest-neighbor hopping
t = Hopping(:t, -1.0, 1)

# define the tight-binding-approximation algorithm
kagome = Algorithm(:Kagome, TBA(unitcell, hilbert, t))

# define the path in the reciprocal space to compute the energy bands
path = ReciprocalPath(reciprocals(unitcell), hexagon"Γ-K-M-Γ", length=100)

# compute the energy bands along the above path
energybands = kagome(:EB, EnergyBands(path))

# plot the energy bands
plot(energybands)
```

## Density of states

The total DOS can be computed as follows:

```@example kagome
# define the Brillouin zone
brillouinzone = BrillouinZone(reciprocals(unitcell), 120)

# compute the total DOS and plot the results
dos = kagome(
    :TotalDensityOfStates,
    DensityOfStates(brillouinzone; emin=-5.0, emax=5.0, ne=201, fwhm=0.1)
)
plot(dos)
```
There are three peaks in the DOS. The one at 2 reflects the flat upper energy band, and the ones at 0 and -2 implies the Van Hove singularities of the middle and lower energy bands, respectively.

On the other hand, the DOS of a specific generalized orbital on a given sets of bands can also be computed:

```@example kagome
# the total DOS of the lower two bands
dos = kagome(
    :DensityOfStatesWithoutUpperBand,
    DensityOfStates(brillouinzone, 1:2; emin=-5.0, emax=5.0, ne=201, fwhm=0.1)
)
plot(dos)
```

```@example kagome
# the total DOS of the three sublattices, respectively (degenerate)
dos = kagome(
    :SublatticeDensityOfStates,
    DensityOfStates(brillouinzone, :, [1], [2], [3]; emin=-5.0, emax=5.0, ne=201, fwhm=0.1)
)
plot(dos; labels=["Sublattice 1" "Sublattice 2" "Sublattice 3"], legend=true)
```

```@example kagome
# the DOS of the three sublattices on the lower two bands, respectively (degenerate)
dos = kagome(
    :SublatticeDensityOfStatesWithoutUpperBand,
    DensityOfStates(brillouinzone, 1:2, [1], [2], [3]; emin=-5.0, emax=5.0, ne=201, fwhm=0.1)
)
plot(dos; labels=["Sublattice 1" "Sublattice 2" "Sublattice 3"], legend=true)
```

## Fermi surface

When the Fermi level of the system sets at the Van Hove singularity of the lower band, the FS is also in the shape of a Kagome lattice:

```@example kagome
# define a reciprocal zone proper to show the FS
reciprocalzone = ReciprocalZone(
    [[4*pi/√3, 0.0], [0.0, 4*pi/√3]], -1.0=>1.0, -1.0=>1.0;
    length=600, ends=(true, true)
)

# Fermi level at -2, i.e., the lower Van Hove singularity
fs = kagome(:FermiSurface, FermiSurface(reciprocalzone, -2))
plot(fs; size=(450, 400), clims=(0, 10))
```

As is similar to DOS, specific orbital components of a specific set of bands on the Fermi level can also be computed.

When the Fermi level sets at -2, actually, only the lower band contributes to the Fermi surface. Therefore, the contributions of the upper two bands are zero, as will be shown by the following computation:
```@example kagome
# Fermi level at -2, the contributions of the upper two bands to the FS
fs = kagome(:FermiSurfaceWithoutLowerBand, FermiSurface(reciprocalzone, -2, 2:3))
plot(fs; size=(450, 400), clims=(0, 5))
```

The components of the three sublattices at the FS can be computed as follows:
```@example kagome
# Fermi level at -2, the three sublattice components at the FS, respectively
fs = kagome(:SublatticeFermiSurface, FermiSurface(reciprocalzone, -2, :, [1], [2], [3]))
plot(fs; nrow=1, ncol=3, size=(780, 330), clims=(0, 5), xlabelfontsize=8, ylabelfontsize=8)
```
