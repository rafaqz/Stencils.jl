using Aqua

Aqua.test_ambiguities([Neighborhoods, Base, Core])
Aqua.test_unbound_args(Neighborhoods)
Aqua.test_undefined_exports(Neighborhoods)
Aqua.test_project_extras(Neighborhoods)
Aqua.test_stale_deps(Neighborhoods)
Aqua.test_deps_compat(Neighborhoods)
Aqua.test_project_toml_formatting(Neighborhoods)

include("neighborhoods.jl")
include("array.jl")
