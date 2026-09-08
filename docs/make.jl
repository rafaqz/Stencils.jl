using Documenter, Stencils, Statistics

DocMeta.setdocmeta!(Stencils, :DocTestSetup, :(using Stencils, Statistics); recursive=true)

makedocs(
    modules = [Stencils],
    sitename = "Stencils.jl",
    checkdocs = :all,
    format = Documenter.HTML(
        prettyurls=get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Stencils" => "index.md",
    ],
)

deploydocs(
    repo = "github.com/rafaqz/Stencils.jl.git",
)
