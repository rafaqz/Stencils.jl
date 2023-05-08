module Stencils

using Adapt, 
      ConstructionBase,
      KernelAbstractions,
      OffsetArrays,
      Setfield,
      StaticArrays,
      UnicodeGraphics

export Stencil, Window, Kernel, Moore, VonNeumann, Positional, Layered
export StencilArray
export BoundaryCondition, Wrap, Remove
export Padding, Conditional, Halo

export stencil, neighbors, offsets, indices, distances, radius, diameter, kernel, kernelproduct
export broadcast_stencil, broadcast_stencil!

include("stencil.jl")
include("boundary.jl")
include("padding.jl")
include("array.jl")
include("broadcast_stencil.jl")

include("stencils/window.jl")
include("stencils/moore.jl")
include("stencils/vonneumman.jl")
include("stencils/positional.jl")
include("stencils/layered.jl")
include("stencils/kernel.jl")

end # Module Stencils

