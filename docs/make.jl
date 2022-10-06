using Documenter, Neighborhoods

makedocs(
    modules = [Neighborhoods],
    sitename = "Neighborhoods.jl",
    checkdocs = :all,
    strict = true,
    format = Documenter.HTML(
        prettyurls = CI,
    ),
    pages = [
        "Neighborhoods" => "index.md",
    ],
)

deploydocs(
    repo = "github.com/rafaqz/Neighborhoods.jl.git",
)
