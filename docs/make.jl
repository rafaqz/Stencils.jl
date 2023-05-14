using Documenter, Stencils

makedocs(
    modules = [Stencils],
    sitename = "Stencils.jl",
    checkdocs = :all,
    strict = true,
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
