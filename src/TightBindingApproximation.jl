module TightBindingApproximation

# Core of tight binding approximation
include("Core.jl")
export Bosonic, Fermionic, Phononic, TBAKind
export CompositeTBA, Quadratic, Quadraticization, SimpleTBA, TBA, TBAMatrix, TBAMatrixization, berrycurvature, commutator, eigen, eigvals, eigvecs, infinitesimal
export BerryCurvature, BerryCurvatureData, DensityOfStates, DensityOfStatesData, EnergyBands, EnergyBandsData, FermiSurface, FermiSurfaceData, InelasticNeutronScatteringSpectra, InelasticNeutronScatteringSpectraData
export BerryCurvatureMethod, Fukui, Kubo

# Fitting parameters
include("Fitting.jl")

# Integration with Wannier90
include("Wannier90.jl")

end # module
