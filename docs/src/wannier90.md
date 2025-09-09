```@meta
CurrentModule = TightBindingApproximation.Wannier90
```

# Interface With Wannier90

The module `TightBindingApproximation.Wannier90` offers utilities to parse data from [Wannier90](https://wannier.org/) input and output files to ensure the compatibility with both the [QuantumLattices](https://github.com/Quantum-Many-Body/QuantumLattices.jl) and [TightBindingApproximation](https://github.com/Quantum-Many-Body/TightBindingApproximation.jl) frameworks.

## Silicon as a concrete example

Before using the code examples, you need to prepare a set of [Wannier90](https://wannier.org/) input and output files. For convenience, we provide a pre-generated dataset as a [Julia artifact](https://pkgdocs.julialang.org/v1/artifacts/). To use it, include an `Artifacts.toml` file in your working directory with the following content:
```TOML
[WannierDataSets]
git-tree-sha1 = "d69063bceff09edd6f55fcc9f5989c406d37d1b9"

    [[WannierDataSets.download]]
    sha256 = "2205f49c1374b1c834b5f8dd4d3b9c2f97a5f96eb0a4b338bc7da29642787e69"
    url = "https://gist.github.com/waltergu/1821d4285e35de366dbabb5513c23f6f/raw/d69063bceff09edd6f55fcc9f5989c406d37d1b9.tar.gz"
```

Then this artifact can be used by the following codes as if it were a folder:
```@example wannier90
using Artifacts
using Pkg

toml = Artifacts.find_artifacts_toml(@__DIR__)

# Directory to contain Wannier90 output files
dir = Pkg.Artifacts.ensure_artifact_installed("WannierDataSets", toml)
nothing # hide
```

Then a Wannier90 tight-binding system can be constructed by reading the data from Wannier90 input and output files:
```@example wannier90

using Plots
using QuantumLattices
using TightBindingApproximation
using TightBindingApproximation.Wannier90

# seedname of Wannier90 output file
seedname = "silicon"

# Read data from seedname.win, seedname_centres.xyz and seedname_hr.dat,
# and construct the Wannier90 tight-binding system
wan = Algorithm(:silicon, W90(dir, seedname))
nothing # hide
```

Now we plot the energy bands interpolated by Wannier90 versus those computed directly from `wan` for comparison:
```@example wannier90
# Read the reciprocal path from seedname.win
path = readpath(dir, seedname)

# Read the energy bands computed by Wannier90 from seedname_band.dat
bands = readbands(dir, seedname)

# Plot the energy bands computed by Wannier90 versus those by `wan` for comparison
plt = plot()
plot!(plt, bands...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
plot!(plt, wan(:EB, EnergyBands(path)), color=:black)
```

The constructed `wan` can be converted to a normal tight-binding system
```@example wannier90
# Define the hilbert space. Note that `hilbert.nspin` is 1
# This is because in silicon, the spin-orbital coupling is omitted
# Therefore, seedname_hr.dat in fact describes a spinless system
hilbert = Hilbert(Fock{:f}(4, 1), length(wan.frontend.lattice))

# Convert `wan` to a normal tight-binding system
# Note when `complement_spin=false`, the converted system is also spinless
tba = Algorithm(wan, hilbert; complement_spin=false)
plt = plot()
plot!(plt, bands...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
plot!(plt, tba(:EB, EnergyBands(path)), color=:black)
```

```@example wannier90
# When `complement_spin=true`, the converted system is spinful
tba = Algorithm(wan, hilbert; complement_spin=true)
plt = plot()
plot!(plt, bands...; xlim=(0.0, distance(path)), label=false, color=:green, alpha=0.6, lw=2.5)
plot!(plt, tba(:EB, EnergyBands(path)), color=:black)
```

## Manual

New types to implement the underlying mechanism of [QuantumLattices](https://github.com/Quantum-Many-Body/QuantumLattices.jl) and [TightBindingApproximation](https://github.com/Quantum-Many-Body/TightBindingApproximation.jl) specified for Wannier90 data:
```@docs
W90Hoppings
W90Matrixization
W90
```

Read data from Wannier90 input and output files to construct the Wannier90 tight-binding system:
```@docs
readlattice
readcenters
readhamiltonian
```

Read data from Wannier90 input files for more information:
```@docs
readpath
```

Read results from Wannier90 output files:
```@docs
readbands
```

New types and functions to help implement the conversion to a normal tight-binding system:
```@docs
Operatorization
Algorithm
```

Other utilities:
```@docs
findblock
```
