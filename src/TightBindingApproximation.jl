module TightBindingApproximation

# Core of tight binding approximation
include("Core.jl")
export Bosonic, Fermionic, Phononic, TBAKind
export AbstractTBA, Quadratic, Quadraticization, TBA, TBAHamiltonian, TBAMatrix, TBAMatrixization, commutator, infinitesimal
export BerryCurvature, DensityOfStates, EnergyBands, FermiSurface, InelasticNeutronScatteringSpectra, Kubo, Fukui, BerryCurvatureMethod

# Fitting parameters
include("Fitting.jl")

# Integration with Wannier90
include("Wannier90.jl")

end # module
