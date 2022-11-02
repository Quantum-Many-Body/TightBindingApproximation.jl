var documenterSearchIndex = {"docs":
[{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"Graphene.md\",\n        \"Superconductor.md\",\n        \"Phonon.md\"\n        ]\nDepth = 2","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Superconductor/#pip-Superconductor-on-Square-Lattice","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"","category":"section"},{"location":"examples/Superconductor/#Energy-bands","page":"p+ip Superconductor on Square Lattice","title":"Energy bands","text":"","category":"section"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"The following codes could compute the energy bands of the Bogoliubov quasiparticles of the p+ip topological superconductor on the square lattice.","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots\n\n# define the unitcell of the square lattice\nunitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])\n\n# define the Hilbert space of the p+ip superconductor (single-orbital spinless complex fermion)\nhilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))\n\n# define the terms\n\n## nearest-neighbor hopping\nt = Hopping(:t, 1.0, 1)\n\n## onsite energy as the chemical potential\nμ = Onsite(:μ, 3.5)\n\n## p+ip pairing term\nΔ = Pairing(\n    :Δ, Complex(0.5), 1, Coupling(:, FID, :, :, (1, 1));\n    amplitude=bond->exp(im*azimuth(rcoordinate(bond)))\n)\n\n# define the Bogoliubov-de Gennes formula for the p+ip superconductor\nsc = Algorithm(Symbol(\"p+ip\"), TBA(unitcell, hilbert, (t, μ, Δ)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(reciprocals(unitcell), rectangle\"Γ-X-M-Γ\", length=100)\n\n# compute the energy bands along the above path\nenergybands = sc(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"examples/Superconductor/#Berry-curvature-and-Chern-number","page":"p+ip Superconductor on Square Lattice","title":"Berry curvature and Chern number","text":"","category":"section"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"The Berry curvatures and the Chern numbers of the quasiparticle bands can be calculated in the unitcell of the reciprocal space:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"# define the Brillouin zone\nbrillouin = BrillouinZone(reciprocals(unitcell), 100)\n\n# compute the Berry curvatures and Chern numbers of both quasiparticle bands\nberry = sc(:BerryCurvature, BerryCurvature(brillouin, [1, 2]));\n\n# plot the Berry curvatures\nplot(berry)","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"The Berry curvatures can also be computed on a reciprocal zone beyond the reciprocal unitcell:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"# define the reciprocal zone\nreciprocalzone = ReciprocalZone(\n    reciprocals(unitcell),\n    [Segment(-2.0, +2.0, 201, ends=(true, true)), Segment(-2.0, 2.0, 201, ends=(true, true))]\n)\n\n# compute the Berry curvature\nberry = sc(:BerryCurvatureExtended, BerryCurvature(reciprocalzone, [1, 2]))\n\n# plot the Berry curvature\nplot(berry)","category":"page"},{"location":"examples/Superconductor/#Edge-states","page":"p+ip Superconductor on Square Lattice","title":"Edge states","text":"","category":"section"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"With tiny modification on the algorithm, the edge states of the p+ip topological superconductor could also be computed:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"# define a cylinder geometry\nlattice = Lattice(unitcell, Translations((1, 100), ('P', 'O')))\n\n# define the new Hilbert space corresponding to the cylinder\nhilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(lattice))\n\n# define the new Bogoliubov-de Gennes formula\nsc = Algorithm(Symbol(\"p+ip\"), TBA(lattice, hilbert, (t, μ, Δ)))\n\n# define new the path in the reciprocal space to compute the edge states\npath = ReciprocalPath(reciprocals(lattice), line\"Γ₁-Γ₂\", length=100)\n\n# compute the energy bands along the above path\nedgestates = sc(:Edge, EnergyBands(path))\n\n# plot the energy bands\nplot(edgestates)","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"Note that when μ>4 or μ<-4, the superconductor is topologically trivial such that there are no gapless edge states on the cylinder geometry:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"# compute the edge states with a new onsite energy such that the superconductor is trivial\ntrivial = sc(:Trivial, EnergyBands(path), parameters=(μ=4.5,))\n\n# plot the energy bands\nplot(trivial)","category":"page"},{"location":"examples/Superconductor/#Auto-generation-of-the-analytical-expression-of-the-Hamiltonian-matrix","page":"p+ip Superconductor on Square Lattice","title":"Auto-generation of the analytical expression of the Hamiltonian matrix","text":"","category":"section"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"Combined with SymPy, it is also possible to get the analytical expression of the free Hamiltonian in the matrix form:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"using SymPy: Sym, symbols\nusing QuantumLattices\nusing TightBindingApproximation\n\nunitcell = Lattice(\n    [zero(Sym), zero(Sym)];\n    name=:Square,\n    vectors=[[one(Sym), zero(Sym)], [zero(Sym), one(Sym)]]\n)\n\nhilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))\n\nt = Hopping(:t, symbols(\"t\", real=true), 1)\nμ = Onsite(:μ, symbols(\"μ\", real=true))\nΔ = Pairing(\n    :Δ, symbols(\"Δ\", real=true), 1, Coupling(:, FID, :, :, (1, 1));\n    amplitude=bond->exp(im*azimuth(rcoordinate(bond)))\n)\n\nsc = TBA(unitcell, hilbert, (t, μ, Δ))\n\nk₁ = symbols(\"k₁\", real=true)\nk₂ = symbols(\"k₂\", real=true)\nm = matrix(sc; k=[k₁, k₂])","category":"page"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Phonon/#Phonons-on-Square-lattice","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"","category":"section"},{"location":"examples/Phonon/#Energy-bands","page":"Phonons on Square lattice","title":"Energy bands","text":"","category":"section"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"The following codes could compute the energy bands of the phonons on the square lattice using the harmonic approximation on the phonon potential.","category":"page"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots\n\n# define the unitcell of the square lattice\nunitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])\n\n# define the Hilbert space of phonons with 2 vibrant directions\nhilbert = Hilbert(site=>Phonon(2) for site=1:length(unitcell))\n\n# define the terms\n\n## Kinetic energy with the mass M=1\nT = Kinetic(:T, 0.5)\n\n## Potential energy on the nearest-neighbor bonds with the spring constant k₁=1.0\nV₁ = Hooke(:V₁, 0.5, 1)\n\n## Potential energy on the next-nearest-neighbor bonds with the spring constant k₂=0.5\nV₂ = Hooke(:V₂, 0.25, 2)\n\n# define the harmonic approximation of the phonons on square lattice\nphonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(reciprocals(unitcell), rectangle\"Γ-X-M-Γ\", length=100)\n\n# compute the energy bands along the above path\nenergybands = phonon(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"examples/Phonon/#Inelastic-neutron-scattering-spectra","page":"Phonons on Square lattice","title":"Inelastic neutron scattering spectra","text":"","category":"section"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"The inelastic neutron scattering spectra of phonons can also be computed:","category":"page"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"# fwhm: the FWHM of the Gaussian to be convoluted\n# scale: the scale of the intensity\nspectra = phonon(\n    :INSS,\n    InelasticNeutronScatteringSpectra(path, range(0.0, 2.5, length=501); fwhm=0.05, scale=log)\n)\nplt = plot()\nplot!(plt, spectra)\nplot!(plt, energybands, color=:white, linestyle=:dash)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"#TightBindingApproximation","page":"Home","title":"TightBindingApproximation","text":"","category":"section"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tight binding approximation for free quantum lattice systems based on the QuantumLattices pack.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In Julia v1.8+, please type ] in the REPL to use the package mode, then type this command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add TightBindingApproximation","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples of tight binding approximation for quantum lattice system","category":"page"},{"location":"#Manuals","page":"Home","title":"Manuals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [TightBindingApproximation]","category":"page"},{"location":"#QuantumLattices.DegreesOfFreedom.Metric-Tuple{TBAKind, QuantumLattices.DegreesOfFreedom.Hilbert}","page":"Home","title":"QuantumLattices.DegreesOfFreedom.Metric","text":"Metric(::Fermionic, hilbert::Hilbert{<:Fock{:f}} -> OperatorUnitToTuple\nMetric(::Bosonic, hilbert::Hilbert{<:Fock{:b}} -> OperatorUnitToTuple\nMetric(::Phononic, hilbert::Hilbert{<:Phonon}) -> OperatorUnitToTuple\n\nGet the index-to-tuple metric for a free fermionic/bosonic/phononic system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.AbstractTBA","page":"Home","title":"TightBindingApproximation.AbstractTBA","text":"AbstractTBA{K<:TBAKind, H<:RepresentationGenerator, G<:Union{Nothing, AbstractMatrix}} <: Frontend\n\nAbstract type for free quantum lattice systems using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.BerryCurvature","page":"Home","title":"TightBindingApproximation.BerryCurvature","text":"BerryCurvature{B<:Union{BrillouinZone, ReciprocalZone}, O} <: Action\n\nBerry curvature of energy bands with the spirit of a momentum space discretization method by Fukui et al, JPSJ 74, 1674 (2005).\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.Bosonic","page":"Home","title":"TightBindingApproximation.Bosonic","text":"Bosonic{K} <: TBAKind{K}\n\nBosonic quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.EnergyBands","page":"Home","title":"TightBindingApproximation.EnergyBands","text":"EnergyBands{P, L<:Union{Colon, Vector{Int}}, O} <: Action\n\nEnergy bands by tight-binding-approximation for quantum lattice systems.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.Fermionic","page":"Home","title":"TightBindingApproximation.Fermionic","text":"Fermionic{K} <: TBAKind{K}\n\nFermionic quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.InelasticNeutronScatteringSpectra","page":"Home","title":"TightBindingApproximation.InelasticNeutronScatteringSpectra","text":"InelasticNeutronScatteringSpectra{P<:ReciprocalPath, E<:AbstractVector, O} <: Action\n\nInelastic neutron scattering spectra.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.Phononic","page":"Home","title":"TightBindingApproximation.Phononic","text":"Phononic <: TBAKind{:BdG}\n\nPhononic quantum lattice system.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.SampleNode","page":"Home","title":"TightBindingApproximation.SampleNode","text":"SampleNode(reciprocals::AbstractVector{<:AbstractVector}, position::Vector, levels::Vector{Int}, values::Vector, ratio::Number)\nSampleNode(reciprocals::AbstractVector{<:AbstractVector}, position::Vector, levels::Vector{Int}, values::Vector, ratios::Vector=ones(length(levels)))\n\nA sample node of a momentum-eigenvalues pair.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBA","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA{K, L<:AbstractLattice, H<:RepresentationGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}\n\nThe usual tight binding approximation for quantum lattice systems.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBA-Tuple{QuantumLattices.Spatials.AbstractLattice, QuantumLattices.DegreesOfFreedom.Hilbert, Tuple{Vararg{QuantumLattices.DegreesOfFreedom.Term}}}","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)\n\nConstruct a tight-binding quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.TBA-Union{Tuple{K}, Tuple{QuantumLattices.Spatials.AbstractLattice, Function, NamedTuple{Names, <:Tuple{Vararg{Number}}} where Names}, Tuple{QuantumLattices.Spatials.AbstractLattice, Function, NamedTuple{Names, <:Tuple{Vararg{Number}}} where Names, Union{Nothing, AbstractMatrix}}} where K<:TBAKind","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA{K}(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing) where {K<:TBAKind}\n\nConstruct a tight-binding quantum lattice system by providing the analytical expressions of the Hamiltonian.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.TBAKind","page":"Home","title":"TightBindingApproximation.TBAKind","text":"TBAKind{K}\n\nThe kind of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBAKind-Union{Tuple{T}, Tuple{Type{T}, Type{<:QuantumLattices.DegreesOfFreedom.Internal}}} where T<:QuantumLattices.DegreesOfFreedom.Term","page":"Home","title":"TightBindingApproximation.TBAKind","text":"TBAKind(T::Type{<:Term}, I::Type{<:Internal})\n\nDepending on the kind of a Term type and an Internal type, get the corresponding TBA kind.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.TBAMatrix","page":"Home","title":"TightBindingApproximation.TBAMatrix","text":"TBAMatrix{K<:TBAKind, G<:Union{AbstractMatrix, Nothing}, T, H<:AbstractMatrix{T}} <: AbstractMatrix{T}\n\nMatrix representation of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBAMatrixRepresentation","page":"Home","title":"TightBindingApproximation.TBAMatrixRepresentation","text":"TBAMatrixRepresentation(tba::AbstractTBA, k=nothing; gauge::Symbol=:icoordinate)\n\nConstruct the matrix representation transformation of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBAMatrixRepresentation-2","page":"Home","title":"TightBindingApproximation.TBAMatrixRepresentation","text":"TBAMatrixRepresentation{K<:TBAKind, V, T, D} <: MatrixRepresentation\n\nMatrix representation of the Hamiltonian of a tight-binding system.\n\n\n\n\n\n","category":"type"},{"location":"#LinearAlgebra.eigen-Tuple{AbstractTBA}","page":"Home","title":"LinearAlgebra.eigen","text":"eigen(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Eigen\n\nGet the eigen values and eigen vectors of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.eigen-Tuple{TBAMatrix{var\"#s7\", Nothing, T} where {var\"#s7\"<:TBAKind, T}}","page":"Home","title":"LinearAlgebra.eigen","text":"eigen(m::TBAMatrix) -> Eigen\n\nSolve the eigen problem of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.eigvals-Tuple{AbstractTBA}","page":"Home","title":"LinearAlgebra.eigvals","text":"eigvals(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Vector\n\nGet the eigen values of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.eigvals-Tuple{TBAMatrix{var\"#s7\", Nothing, T} where {var\"#s7\"<:TBAKind, T}}","page":"Home","title":"LinearAlgebra.eigvals","text":"eigvals(m::TBAMatrix) -> Vector\n\nGet the eigen values of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.eigvecs-Tuple{AbstractTBA}","page":"Home","title":"LinearAlgebra.eigvecs","text":"eigvecs(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; kwargs...) -> Matrix\n\nGet the eigen vectors of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.eigvecs-Tuple{TBAMatrix}","page":"Home","title":"LinearAlgebra.eigvecs","text":"eigvecs(m::TBAMatrix) -> Matrix\n\nGet the eigen vectors of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.QuantumOperators.matrix-Tuple{AbstractTBA}","page":"Home","title":"QuantumLattices.QuantumOperators.matrix","text":"matrix(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}; k=nothing, gauge=:icoordinate, kwargs...) -> TBAMatrix\n\nGet the matrix representation of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.add!-Tuple{AbstractMatrix, TBAMatrixRepresentation{<:TBAKind{:TBA}}, QuantumLattices.QuantumOperators.Operator{<:Number, <:Tuple{QuantumLattices.DegreesOfFreedom.CompositeIndex{<:QuantumLattices.DegreesOfFreedom.Index{Int64, <:QuantumLattices.QuantumSystems.FID}}, QuantumLattices.DegreesOfFreedom.CompositeIndex{<:QuantumLattices.DegreesOfFreedom.Index{Int64, <:QuantumLattices.QuantumSystems.FID}}}}}","page":"Home","title":"QuantumLattices.add!","text":"add!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:TBA}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID}}, 2}}; kwargs...) -> typeof(dest)\nadd!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:f}}}, 2}}; kwargs...) -> typeof(dest)\nadd!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:b}}}, 2}}; atol=atol/5, kwargs...) -> typeof(dest)\nadd!(dest::AbstractMatrix, mr::TBAMatrixRepresentation{<:TBAKind{:BdG}}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:PID}}, 2}}; atol=atol/5, kwargs...) -> typeof(dest)\n\nGet the matrix representation of an operator and add it to destination.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.commutator-Tuple{TBAKind, QuantumLattices.DegreesOfFreedom.Hilbert}","page":"Home","title":"TightBindingApproximation.commutator","text":"commutator(k::TBAKind, hilbert::Hilbert{<:Internal}) -> Union{AbstractMatrix, Nothing}\n\nGet the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.deviation-Tuple{Union{QuantumLattices.Frameworks.Algorithm{<:AbstractTBA}, AbstractTBA}, SampleNode}","page":"Home","title":"TightBindingApproximation.deviation","text":"deviation(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplenode::SampleNode) -> Float64\ndeviation(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}}, samplesets::Vector{SampleNode}) -> Float64\n\nGet the deviation of the eigenvalues between the sample points and model points.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.optimize!","page":"Home","title":"TightBindingApproximation.optimize!","text":"optimize!(tba::Union{AbstractTBA, Algorithm{<:AbstractTBA}},\n    samplesets::Vector{SampleNode},\n    variables=keys(Parameters(tba));\n    verbose=false,\n    method=LBFGS()\n) -> Tuple{typeof(tba), Optim.MultivariateOptimizationResults}\n\nOptimize the parameters of a tight binding system whose names are specified by variables so that the total deviations of the eigenvalues between the model points and sample points minimize.\n\n\n\n\n\n","category":"function"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Graphene/#Graphene","page":"Graphene","title":"Graphene","text":"","category":"section"},{"location":"examples/Graphene/#Energy-bands","page":"Graphene","title":"Energy bands","text":"","category":"section"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"The following codes could compute the energy bands of the monolayer graphene.","category":"page"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots\n\n# define the unitcell of the honeycomb lattice\nunitcell = Lattice(\n    [0.0, 0.0], [0.0, √3/3];\n    name=:Honeycomb,\n    vectors=[[1.0, 0.0], [0.5, √3/2]]\n)\n\n# define the Hilbert space of graphene (single-orbital spin-1/2 complex fermion)\nhilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(unitcell))\n\n# define the terms, i.e. the nearest-neighbor hopping\nt = Hopping(:t, -1.0, 1)\n\n# define the tight-binding-approximation algorithm for graphene\ngraphene = Algorithm(:Graphene, TBA(unitcell, hilbert, (t,)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(reciprocals(unitcell), hexagon\"Γ-K-M-Γ\", length=100)\n\n# compute the energy bands along the above path\nenergybands = graphene(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"examples/Graphene/#Edge-states","page":"Graphene","title":"Edge states","text":"","category":"section"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"Graphene supports flatband edge states on zigzag boundaries. Only minor modifications are needed to compute them:","category":"page"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"# define a cylinder geometry with zigzag edges\nlattice = Lattice(unitcell, Translations((1, 100), ('P', 'O')))\n\n# define the new Hilbert space corresponding to the cylinder\nhilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(lattice))\n\n# define the new tight-binding-approximation algorithm\nzigzag = Algorithm(:Graphene, TBA(lattice, hilbert, (t,)))\n\n# define new the path in the reciprocal space to compute the edge states\npath = ReciprocalPath(reciprocals(lattice), line\"Γ₁-Γ₂\", length=100)\n\n# compute the energy bands along the above path\nedgestates = zigzag(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(edgestates)","category":"page"},{"location":"examples/Graphene/#Auto-generation-of-the-analytical-expression-of-the-Hamiltonian-matrix","page":"Graphene","title":"Auto-generation of the analytical expression of the Hamiltonian matrix","text":"","category":"section"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"Combined with SymPy, it is also possible to get the analytical expression of the free Hamiltonian in the matrix form:","category":"page"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"using SymPy: Sym, symbols\nusing QuantumLattices\nusing TightBindingApproximation\n\nunitcell = Lattice(\n    [zero(Sym), zero(Sym)], [zero(Sym), √(one(Sym)*3)/3];\n    name=:Honeycomb,\n    vectors=[[one(Sym), zero(Sym)], [one(Sym)/2, √(one(Sym)*3)/2]]\n)\n\nhilbert = Hilbert(site=>Fock{:f}(1, 1) for site=1:length(unitcell))\n\nt = Hopping(:t, symbols(\"t\", real=true), 1)\n\ngraphene = TBA(unitcell, hilbert, (t,))\n\nk₁ = symbols(\"k₁\", real=true)\nk₂ = symbols(\"k₂\", real=true)\nm = matrix(graphene; k=[k₁, k₂])","category":"page"}]
}
