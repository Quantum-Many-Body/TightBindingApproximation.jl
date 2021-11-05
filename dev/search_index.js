var documenterSearchIndex = {"docs":
[{"location":"examples/SquareLattice/","page":"Square lattice","title":"Square lattice","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/SquareLattice/#Square-lattice","page":"Square lattice","title":"Square lattice","text":"","category":"section"},{"location":"examples/SquareLattice/#Energy-bands-for-pip-superconductor","page":"Square lattice","title":"Energy bands for p+ip superconductor","text":"","category":"section"},{"location":"examples/SquareLattice/","page":"Square lattice","title":"Square lattice","text":"The following codes could compute the energy bands of the Bogoliubov quasiparticles of the p+ip topological superconductor on the square lattice.","category":"page"},{"location":"examples/SquareLattice/","page":"Square lattice","title":"Square lattice","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots: plot\n\n# define the unitcell of the square lattice\nunitcell = Lattice(\"Square\", [Point(PID(1), (0.0, 0.0), (0.0, 0.0))],\n    vectors=[[1.0, 0.0], [0.0, 1.0]],\n    neighbors=1\n    )\n\n# define the Hilbert space of the p+ip superconductor (single-orbital spinless complex fermion)\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in unitcell.pids)\n\n# define the terms\n\n## nearest-neighbor hopping\nt = Hopping(:t, 1.0, 1, modulate=true)\n\n## onsite energy as the chemical potential\nμ = Onsite(:μ, 3.5, modulate=true)\n\n## p+ip pairing term\nΔ = Pairing(:Δ, Complex(0.5), 1, amplitude=bond->exp(im*azimuth(rcoord(bond))), modulate=true)\n\n# define the Bogoliubov-de Gennes formula for the p+ip superconductor\nsc = Algorithm(\"p+ip\", TBA(unitcell, hilbert, (t, μ, Δ)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(unitcell.reciprocals, rectangle\"Γ-X-M-Γ\", len=100)\n\n# compute the energy bands along the above path\nenergybands = register!(sc, :EB, TBAEB(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"examples/SquareLattice/","page":"Square lattice","title":"Square lattice","text":"With tiny modification on the algorithm, the edge states of the p+ip topological superconductor could also be computed:","category":"page"},{"location":"examples/SquareLattice/","page":"Square lattice","title":"Square lattice","text":"# define a cylinder geometry\nlattice = Lattice(unitcell, translations\"1P-100O\")\n\n# define the new Hilbert space corresponding to the cylinder\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)\n\n# define the new Bogoliubov-de Gennes formula\nsc = Algorithm(\"p+ip\", TBA(lattice, hilbert, (t, μ, Δ)))\n\n# define new the path in the reciprocal space to compute the edge states\npath = ReciprocalPath(lattice.reciprocals, line\"Γ₁-Γ₂\", len=100)\n\n# compute the energy bands along the above path\nedgestates = register!(sc, :Edge, TBAEB(path))\n\n# plot the energy bands\nplot(edgestates)","category":"page"},{"location":"examples/SquareLattice/","page":"Square lattice","title":"Square lattice","text":"Note that when μ>4 or μ<-4, the superconductor is topologically trivial such that there are no gapless edge states on the cylinder geometry:","category":"page"},{"location":"examples/SquareLattice/","page":"Square lattice","title":"Square lattice","text":"# compute the edge states with a new onsite energy such that the superconductor is trivial\ntrivial = register!(sc, :Trivial, TBAEB(path), parameters=(μ=4.5,))\n\n# plot the energy bands\nplot(trivial)","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/Introduction/#examples","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Here are some examples to illustrate how this package could be used.","category":"page"},{"location":"examples/Introduction/","page":"Introduction","title":"Introduction","text":"Pages = [\n        \"HoneycombLattice.md\",\n        \"SquareLattice.md\",\n        ]\nDepth = 2","category":"page"},{"location":"examples/HoneycombLattice/","page":"Honeycomb lattice","title":"Honeycomb lattice","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"examples/HoneycombLattice/#Honeycomb-lattice","page":"Honeycomb lattice","title":"Honeycomb lattice","text":"","category":"section"},{"location":"examples/HoneycombLattice/#Energy-bands-for-graphene","page":"Honeycomb lattice","title":"Energy bands for graphene","text":"","category":"section"},{"location":"examples/HoneycombLattice/","page":"Honeycomb lattice","title":"Honeycomb lattice","text":"The following codes could compute the energy bands of the monolayer graphene.","category":"page"},{"location":"examples/HoneycombLattice/","page":"Honeycomb lattice","title":"Honeycomb lattice","text":"using QuantumLattices\nusing TightBindingApproximation\nusing Plots: plot\n\n# define the unitcell of the honeycomb lattice\nunitcell = Lattice(\"Honeycomb\", [\n        Point(PID(1), (0.0, 0.0), (0.0, 0.0)),\n        Point(PID(2), (0.0, √3/3), (0.0, 0.0))\n        ],\n    vectors=[[1.0, 0.0], [0.5, √3/2]],\n    neighbors=1\n    )\n\n# define the Hilbert space of graphene (single-orbital spin-1/2 complex fermion)\nhilbert = Hilbert(pid=>Fock{:f}(1, 2, 2) for pid in unitcell.pids)\n\n# define the terms, i.e. the nearest-neighbor hopping\nt = Hopping(:t, -1.0, 1, modulate=true)\n\n# define the tight-binding-approximation algorithm for graphene\ngraphene = Algorithm(\"Graphene\", TBA(unitcell, hilbert, (t,)))\n\n# define the path in the reciprocal space to compute the energy bands\npath = ReciprocalPath(unitcell.reciprocals, hexagon\"Γ-K-M-Γ\", len=100)\n\n# compute the energy bands along the above path\nenergybands = register!(graphene, :EB, TBAEB(path))\n\n# plot the energy bands\nplot(energybands)","category":"page"},{"location":"examples/HoneycombLattice/","page":"Honeycomb lattice","title":"Honeycomb lattice","text":"Graphene supports flatband edge states on zigzag boundaries. Only minor modifications are needed to compute them:","category":"page"},{"location":"examples/HoneycombLattice/","page":"Honeycomb lattice","title":"Honeycomb lattice","text":"# define a cylinder geometry with zigzag edges\nlattice = Lattice(unitcell, translations\"1P-100O\")\n\n# define the new Hilbert space corresponding to the cylinder\nhilbert = Hilbert(pid=>Fock{:f}(1, 1, 2) for pid in lattice.pids)\n\n# define the new tight-binding-approximation algorithm\nzigzag = Algorithm(\"Graphene\", TBA(lattice, hilbert, (t,)))\n\n# define new the path in the reciprocal space to compute the edge states\npath = ReciprocalPath(lattice.reciprocals, line\"Γ₁-Γ₂\", len=100)\n\n# compute the energy bands along the above path\nedgestates = register!(zigzag, :EB, TBAEB(path))\n\n# plot the energy bands\nplot(edgestates)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = TightBindingApproximation","category":"page"},{"location":"#TightBindingApproximation","page":"Home","title":"TightBindingApproximation","text":"","category":"section"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Tight binding approximation for free quantum lattice systems based on the QuantumLattices pack.","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Examples","category":"page"},{"location":"#Manuals","page":"Home","title":"Manuals","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [TightBindingApproximation]","category":"page"},{"location":"#QuantumLattices.Essentials.DegreesOfFreedom.OIDToTuple-Union{Tuple{I}, Tuple{TBAKind, Type{I}}} where I<:(QuantumLattices.Essentials.DegreesOfFreedom.Index{var\"#s6\", var\"#s7\"} where {var\"#s6\"<:QuantumLattices.Essentials.Spatials.AbstractPID, var\"#s7\"<:QuantumLattices.Essentials.QuantumSystems.FID})","page":"Home","title":"QuantumLattices.Essentials.DegreesOfFreedom.OIDToTuple","text":"OIDToTuple(::TBAKind, ::Type{I}) where {I<:Index{<:AbstractPID, <:FID}} -> OIDToTuple\nOIDToTuple(::TBAKind, ::Type{I}) where {I<:Index{<:AbstractPID, <:NID}} -> OIDToTuple\n\nGet the oid-to-tuple metric for a free fermionic/bosonic system or a free phononic system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.AbstractTBA","page":"Home","title":"TightBindingApproximation.AbstractTBA","text":"AbstractTBA{K, H<:AbstractGenerator, G<:Union{Nothing, AbstractMatrix}} <: Engine\n\nAbstract type for free quantum lattice systems using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBA","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA{K, L<:AbstractLattice, H<:AbstractGenerator, G<:Union{AbstractMatrix, Nothing}} <: AbstractTBA{K, H, G}\n\nThe usual tight binding approximation for quantum lattice systems.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBA-Tuple{QuantumLattices.Essentials.Spatials.AbstractLattice, QuantumLattices.Essentials.DegreesOfFreedom.Hilbert, Tuple{Vararg{QuantumLattices.Essentials.DegreesOfFreedom.Term, N} where N}}","page":"Home","title":"TightBindingApproximation.TBA","text":"TBA(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}; boundary::Boundary=plain)\n\nConstruct a tight-binding quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.TBAEB","page":"Home","title":"TightBindingApproximation.TBAEB","text":"TBAEB{P} <: Action\n\nEnergy bands by tight-binding-approximation for quantum lattice systems.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBAKind","page":"Home","title":"TightBindingApproximation.TBAKind","text":"TBAKind{K}\n\nThe kind of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#TightBindingApproximation.TBAKind-Union{Tuple{Type{T}}, Tuple{T}} where T<:QuantumLattices.Essentials.DegreesOfFreedom.Term","page":"Home","title":"TightBindingApproximation.TBAKind","text":"TBAKind(T::Type{<:Term})\n\nDepending on the kind of a term type, get the corresponding TBA kind.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.TBAMatrix","page":"Home","title":"TightBindingApproximation.TBAMatrix","text":"TBAMatrix{T, H<:AbstractMatrix{T}, G<:Union{AbstractMatrix, Nothing}} <: AbstractMatrix{T}\n\nMatrix representation of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"type"},{"location":"#LinearAlgebra.eigen-Union{Tuple{TBAMatrix{T, H, Nothing}}, Tuple{H}, Tuple{T}} where {T, H<:AbstractMatrix{T}}","page":"Home","title":"LinearAlgebra.eigen","text":"eigen(m::TBAMatrix) -> Eigen\n\nSolve the eigen problem of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#QuantumLattices.Essentials.QuantumOperators.matrix!-Tuple{AbstractTBA{TBAKind{:TBA}(), H, G} where {H<:QuantumLattices.Essentials.Frameworks.AbstractGenerator, G<:Union{Nothing, AbstractMatrix{T} where T}}}","page":"Home","title":"QuantumLattices.Essentials.QuantumOperators.matrix!","text":"matrix!(tba::AbstractTBA; k=nothing, kwargs...) -> TBAMatrix\n\nGet the matrix representation of a free quantum lattice system.\n\n\n\n\n\n","category":"method"},{"location":"#TightBindingApproximation.commutator-Tuple{TBAKind, Type{var\"#s6\"} where var\"#s6\"<:QuantumLattices.Essentials.DegreesOfFreedom.Internal, Integer}","page":"Home","title":"TightBindingApproximation.commutator","text":"commutator(::TBAKind, ::Type{<:Internal}, n::Integer) -> Union{AbstractMatrix, Nothing}\n\nGet the commutation relation of the single-particle operators of a free quantum lattice system using the tight-binding approximation.\n\n\n\n\n\n","category":"method"}]
}
