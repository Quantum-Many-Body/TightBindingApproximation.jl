var documenterSearchIndex = {"docs":
[{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"Graphene.md\",\n        \"Superconductor.md\",\n        \"Phonon.md\"\n        ]\nDepth = 2","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Superconductor/#pip-Superconductor-on-Square-Lattice","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"","category":"section"},{"location":"examples/Superconductor/#Energy-bands","page":"p+ip Superconductor on Square Lattice","title":"Energy bands","text":"","category":"section"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"The following codes could compute the energy bands of the Bogoliubov quasiparticles of the p+ip topological superconductor on the square lattice.","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots: plot\n\n# define the unitcell of the square lattice\nunitcell = Lattice(:Square,\n    [Point(PID(1), [0.0, 0.0])],\n    vectors=[[1.0, 0.0], [0.0, 1.0]],\n    neighbors=1\n    )\n\n# define the Hilbert space of the p+ip superconductor (single-orbital spinless complex fermion)\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)\n\n# define the terms\n\n## nearest-neighbor hopping\nt = Hopping(:t, 1.0, 1, modulate=true)\n\n## onsite energy as the chemical potential\nμ = Onsite(:μ, 3.5, modulate=true)\n\n## p+ip pairing term\nΔ = Pairing(:Δ, Complex(0.5), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))), modulate=true)\n\n# define the Bogoliubov-de Gennes formula for the p+ip superconductor\nsc = Algorithm(Symbol(\"p+ip\"), TBA(unitcell, hilbert, (t, μ, Δ)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(unitcell.reciprocals, rectangle\"Γ-X-M-Γ\", length=100)\n\n# compute the energy bands along the above path\nenergybands = sc(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"examples/Superconductor/#Edge-states","page":"p+ip Superconductor on Square Lattice","title":"Edge states","text":"","category":"section"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"With tiny modification on the algorithm, the edge states of the p+ip topological superconductor could also be computed:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"# define a cylinder geometry\nlattice = Lattice(unitcell, translations\"1P-100O\")\n\n# define the new Hilbert space corresponding to the cylinder\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)\n\n# define the new Bogoliubov-de Gennes formula\nsc = Algorithm(Symbol(\"p+ip\"), TBA(lattice, hilbert, (t, μ, Δ)))\n\n# define new the path in the reciprocal space to compute the edge states\npath = ReciprocalPath(lattice.reciprocals, line\"Γ₁-Γ₂\", length=100)\n\n# compute the energy bands along the above path\nedgestates = sc(:Edge, EnergyBands(path))\n\n# plot the energy bands\nplot(edgestates)","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"Note that when μ>4 or μ<-4, the superconductor is topologically trivial such that there are no gapless edge states on the cylinder geometry:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"# compute the edge states with a new onsite energy such that the superconductor is trivial\ntrivial = sc(:Trivial, EnergyBands(path), parameters=(μ=4.5,))\n\n# plot the energy bands\nplot(trivial)","category":"page"},{"location":"examples/Superconductor/#Auto-generation-of-the-analytical-expression-of-the-Hamiltonian-matrix","page":"p+ip Superconductor on Square Lattice","title":"Auto-generation of the analytical expression of the Hamiltonian matrix","text":"","category":"section"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"Combined with SymPy, it is also possible to get the analytical expression of the free Hamiltonian in the matrix form:","category":"page"},{"location":"examples/Superconductor/","page":"p+ip Superconductor on Square Lattice","title":"p+ip Superconductor on Square Lattice","text":"using SymPy: Sym, symbols\nusing QuantumLattices\nusing TightBindingApproximation\n\nunitcell = Lattice(:Square,\n    [Point(PID(1), [zero(Sym), zero(Sym)])],\n    vectors=[[one(Sym), zero(Sym)], [zero(Sym), one(Sym)]],\n    neighbors=1\n    )\n\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)\n\nt = Hopping(:t, symbols(\"t\", real=true), 1)\nμ = Onsite(:μ, symbols(\"μ\", real=true))\nΔ = Pairing(:Δ, symbols(\"Δ\", real=true), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))))\n\nsc = TBA(unitcell, hilbert, (t, μ, Δ))\n\nk₁ = symbols(\"k₁\", real=true)\nk₂ = symbols(\"k₂\", real=true)\nm = matrix(sc; k=[k₁, k₂])","category":"page"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Phonon/#Phonons-on-Square-lattice","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"","category":"section"},{"location":"examples/Phonon/#Energy-bands","page":"Phonons on Square lattice","title":"Energy bands","text":"","category":"section"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"The following codes could compute the energy bands of the phonons on the square lattice using the harmonic approximation on the phonon potential.","category":"page"},{"location":"examples/Phonon/","page":"Phonons on Square lattice","title":"Phonons on Square lattice","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots: plot\n\n# define the unitcell of the square lattice\nunitcell = Lattice(:Square,\n    [Point(PID(1), [0.0, 0.0])],\n    vectors=[[1.0, 0.0], [0.0, 1.0]],\n    neighbors=2\n    )\n\n# define the Hilbert space of phonons with 2 vibrant directions\nhilbert = Hilbert(pid=>Phonon(2) for pid in unitcell.pids)\n\n# define the terms\n\n## Kinetic energy of the phonons with the mass M=1\nT = PhononKinetic(:T, 0.5)\n\n## Potential energy of the phonons on the nearest-neighbor bonds with the spring constant k₁=1.0\nV₁ = PhononPotential(:V₁, 0.5, 1)\n\n## Potential energy of the phonons on the next-nearest-neighbor bonds with the spring constant k₂=0.5\nV₂ = PhononPotential(:V₂, 0.25, 2)\n\n# define the harmonic approximation of the phonons on square lattice\nphonon = Algorithm(:Phonon, TBA(unitcell, hilbert, (T, V₁, V₂)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(unitcell.reciprocals, rectangle\"Γ-X-M-Γ\", length=100)\n\n# compute the energy bands along the above path\nenergybands = phonon(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"#TightBindingApproximation","page":"Home","title":"TightBindingApproximation","text":"","category":"section"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tight binding approximation for free quantum lattice systems based on the QuantumLattices pack.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"In Julia v1.6+, please type ] in the REPL to use the package mode, then type this command:","category":"page"},{"location":"","page":"Home","title":"Home","text":"pkg> add TightBindingApproximation","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples of tight binding approximation for quantum lattice system","category":"page"},{"location":"#Manuals","page":"Home","title":"Manuals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [TightBindingApproximation]","category":"page"},{"location":"#QuantumLattices.Essentials.DegreesOfFreedom.Metric-Tuple{TBAKind{:TBA}, QuantumLattices.Essentials.DegreesOfFreedom.Hilbert{<:QuantumLattices.Essentials.QuantumSystems.Fock}}","page":"Home","title":"QuantumLattices.Essentials.DegreesOfFreedom.Metric","text":"Metric(::TBAKind, hilbert::Hilbert{<:Fock} -> OIDToTuple\nMetric(::TBAKind, hilbert::Hilbert{<:Phonon}) -> OIDToTuple\n\nGet the oid-to-tuple metric for a free fermionic/bosonic system or a free phononic system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.AbstractTBA","page":"Home","title":"TightBindingApproximation.AbstractTBA","text":"AbstractTBA{K, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine\n\nAbstract type for free quantum lattice systems using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.EnergyBands","page":"Home","title":"TightBindingApproximation.EnergyBands","text":"EnergyBands{P} <: Action\n\nEnergy bands by tight-binding-approximation for quantum lattice systems.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBA","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA(lattice::AbstractLattice, hamiltonian::Function, parameters::Parameters, commt::Union{AbstractMatrix, Nothing}=nothing)\n\nConstruct a tight-binding quantum lattice system by providing the analytical expressions of the Hamiltonian.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBA-2","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}\n\nThe usual tight binding approximation for quantum lattice systems.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBA-Tuple{QuantumLattices.Essentials.Spatials.AbstractLattice, QuantumLattices.Essentials.DegreesOfFreedom.Hilbert, Tuple{Vararg{QuantumLattices.Essentials.DegreesOfFreedom.Term}}}","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; boundary::Boundary=plain)\n\nConstruct a tight-binding quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.TBAKind","page":"Home","title":"TightBindingApproximation.TBAKind","text":"TBAKind{K}\n\nThe kind of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBAKind-Union{Tuple{Type{T}}, Tuple{T}} where T<:QuantumLattices.Essentials.DegreesOfFreedom.Term","page":"Home","title":"TightBindingApproximation.TBAKind","text":"TBAKind(T::Type{<:Term})\n\nDepending on the kind of a term type, get the corresponding TBA kind.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.TBAMatrix","page":"Home","title":"TightBindingApproximation.TBAMatrix","text":"TBAMatrix{T, H<:AbstractMatrix{T}, G<:Union{AbstractMatrix, Nothing}} <: AbstractMatrix{T}\n\nMatrix representation of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBAMatrixRepresentation","page":"Home","title":"TightBindingApproximation.TBAMatrixRepresentation","text":"TBAMatrixRepresentation{K<:AbstractTBA, V, T} <: MatrixRepresentation\n\nMatrix representation of the Hamiltonian of a tight-binding system.\n\n\n\n\n\n","category":"type"},{"location":"#LinearAlgebra.eigen-Union{Tuple{TBAMatrix{T, H, Nothing}}, Tuple{H}, Tuple{T}} where {T, H<:AbstractMatrix{T}}","page":"Home","title":"LinearAlgebra.eigen","text":"eigen(m::TBAMatrix) -> Eigen\n\nSolve the eigen problem of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Essentials.QuantumOperators.matrix-Tuple{AbstractTBA}","page":"Home","title":"QuantumLattices.Essentials.QuantumOperators.matrix","text":"matrix(tba::AbstractTBA; k=nothing, gauge=:rcoord, atol=atol/5, kwargs...) -> TBAMatrix\n\nGet the matrix representation of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.commutator-Tuple{TBAKind{:TBA}, QuantumLattices.Essentials.DegreesOfFreedom.Hilbert}","page":"Home","title":"TightBindingApproximation.commutator","text":"commutator(k::TBAKind, hilbert::Hilbert{<:Internal}) -> Union{AbstractMatrix, Nothing}\n\nGet the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"method"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Graphene/#Graphene","page":"Graphene","title":"Graphene","text":"","category":"section"},{"location":"examples/Graphene/#Energy-bands","page":"Graphene","title":"Energy bands","text":"","category":"section"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"The following codes could compute the energy bands of the monolayer graphene.","category":"page"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots: plot\n\n# define the unitcell of the honeycomb lattice\nunitcell = Lattice(:Honeycomb,\n    [Point(PID(1), [0.0, 0.0]), Point(PID(2), [0.0, √3/3])],\n    vectors=[[1.0, 0.0], [0.5, √3/2]],\n    neighbors=1\n    )\n\n# define the Hilbert space of graphene (single-orbital spin-1/2 complex fermion)\nhilbert = Hilbert(pid=>Fock{:f}(1, 2, 2) for pid in unitcell.pids)\n\n# define the terms, i.e. the nearest-neighbor hopping\nt = Hopping(:t, -1.0, 1, modulate=true)\n\n# define the tight-binding-approximation algorithm for graphene\ngraphene = Algorithm(:Graphene, TBA(unitcell, hilbert, (t,)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(unitcell.reciprocals, hexagon\"Γ-K-M-Γ\", length=100)\n\n# compute the energy bands along the above path\nenergybands = graphene(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"examples/Graphene/#Edge-states","page":"Graphene","title":"Edge states","text":"","category":"section"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"Graphene supports flatband edge states on zigzag boundaries. Only minor modifications are needed to compute them:","category":"page"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"# define a cylinder geometry with zigzag edges\nlattice = Lattice(unitcell, translations\"1P-100O\")\n\n# define the new Hilbert space corresponding to the cylinder\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)\n\n# define the new tight-binding-approximation algorithm\nzigzag = Algorithm(:Graphene, TBA(lattice, hilbert, (t,)))\n\n# define new the path in the reciprocal space to compute the edge states\npath = ReciprocalPath(lattice.reciprocals, line\"Γ₁-Γ₂\", length=100)\n\n# compute the energy bands along the above path\nedgestates = zigzag(:EB, EnergyBands(path))\n\n# plot the energy bands\nplot(edgestates)","category":"page"},{"location":"examples/Graphene/#Auto-generation-of-the-analytical-expression-of-the-Hamiltonian-matrix","page":"Graphene","title":"Auto-generation of the analytical expression of the Hamiltonian matrix","text":"","category":"section"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"Combined with SymPy, it is also possible to get the analytical expression of the free Hamiltonian in the matrix form:","category":"page"},{"location":"examples/Graphene/","page":"Graphene","title":"Graphene","text":"using SymPy: Sym, symbols\nusing QuantumLattices\nusing TightBindingApproximation\n\nunitcell = Lattice(:Honeycomb,\n    [Point(PID(1), [zero(Sym), zero(Sym)]), Point(PID(2), [zero(Sym), √(one(Sym)*3)/3])],\n    vectors=[[one(Sym), zero(Sym)], [one(Sym)/2, √(one(Sym)*3)/2]],\n    neighbors=1\n    )\n\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)\n\nt = Hopping(:t, symbols(\"t\", real=true), 1)\n\ngraphene = TBA(unitcell, hilbert, (t,))\n\nk₁ = symbols(\"k₁\", real=true)\nk₂ = symbols(\"k₂\", real=true)\nm = matrix(graphene; k=[k₁, k₂])","category":"page"}]
}
