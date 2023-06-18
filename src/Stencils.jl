module Stencils

using Adapt, 
      ConstructionBase,
      KernelAbstractions,
      OffsetArrays,
      StaticArrays,
      UnicodeGraphics

import SparseArrays

export Stencil, Window, Kernel, Moore, VonNeumann, Positional, Layered, 
    Circle, Cross, AngledCross, BackSlash, ForwardSlash, Vertical, Horizontal, NamedStencil

export StencilArray, SwitchingStencilArray

export Remove, Use, Wrap

export Conditional, Halo

export stencil, neighbors, offsets, indices, distances, kernelproduct, radius, diameter
export mapstencil, mapstencil!, switch

include("stencil.jl")
include("stencils/window.jl")
include("stencils/moore.jl")
include("stencils/vonneumman.jl")
include("stencils/shapes.jl")
include("stencils/positional.jl")
include("stencils/layered.jl")
include("stencils/kernel.jl")

include("boundary.jl")
include("padding.jl")
include("array.jl")
include("mapstencil.jl")

end # Module Stencils

