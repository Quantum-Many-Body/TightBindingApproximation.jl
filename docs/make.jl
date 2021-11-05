using TightBindingApproximation
using Documenter

DocMeta.setdocmeta!(TightBindingApproximation, :DocTestSetup, :(using TightBindingApproximation); recursive=true)

makedocs(;
    modules=[TightBindingApproximation],
    authors="waltergu <waltergu1989@gmail.com>",
    repo="https://github.com/Quantum-Many-Body/TightBindingApproximation.jl/blob/{commit}{path}#{line}",
    sitename="TightBindingApproximation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/TightBindingApproximation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "examples/Introduction.md",
            "examples/HoneycombLattice.md",
            "examples/SquareLattice.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/TightBindingApproximation.jl",
)
