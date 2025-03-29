using Stencils
using Aqua
using Test

@testset "Aqua.jl" begin
    Aqua.test_all(
        Stencils;
        ambiguities=(exclude=[Base.copy!, Base.copyto!],),
        unbound_args=true,
        undefined_exports=true,
        project_extras=true,
        stale_deps=true,
        deps_compat=true,
        piracies=true,
    )
end

include("stencils.jl")
include("array.jl")