using Stencils
using Aqua

Aqua.test_ambiguities([Stencils, Base, Core])
Aqua.test_unbound_args(Stencils)
Aqua.test_undefined_exports(Stencils)
Aqua.test_project_extras(Stencils)
Aqua.test_stale_deps(Stencils)
Aqua.test_deps_compat(Stencils)
Aqua.test_project_toml_formatting(Stencils)

include("stencils.jl")
include("array.jl")
