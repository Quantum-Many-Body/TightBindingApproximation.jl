using TightBindingApproximation
using Documenter

DocMeta.setdocmeta!(TightBindingApproximation, :DocTestSetup, :(using TightBindingApproximation); recursive=true)

makedocs(;
    modules=[TightBindingApproximation],
    authors="waltergu <waltergu1989@gmail.com>",
    sitename="TightBindingApproximation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Quantum-Many-Body.github.io/TightBindingApproximation.jl",
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "examples/Introduction.md",
            "examples/Graphene.md",
            "examples/Superconductor.md",
            "examples/Kagome.md",
            "examples/Phonon.md",
        ],
        "Manual" => "manual.md",
    ],
)

deploydocs(;
    repo="github.com/Quantum-Many-Body/TightBindingApproximation.jl",
)
