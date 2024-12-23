module Stencils

using Adapt, 
      ConstructionBase,
      KernelAbstractions,
      StaticArrays,
      UnicodeGraphics

import SparseArrays

export Stencil, Window, Rectangle, Kernel, Moore, VonNeumann, Positional, Layered,
    Circle, Cross, AngledCross, BackSlash, ForwardSlash, Vertical, Horizontal, Diamond,
    NamedStencil, Annulus, Cardinal, Ordinal

export StencilArray, SwitchingStencilArray

export Remove, Use, Wrap, Reflect

export Conditional, Halo

export stencil, neighbors, offsets, indices, distances, kernelproduct, radius, diameter
export mapstencil, mapstencil!, switch

include("stencil.jl")
include("stencils/window.jl")
include("stencils/moore.jl")
include("stencils/vonneumman.jl")
include("stencils/shapes.jl")
include("stencils/positional.jl")
include("stencils/rectangle.jl")
include("stencils/layered.jl")
include("stencils/kernel.jl")
include("stencils/named.jl")

include("boundary.jl")
include("padding.jl")
include("array.jl")
include("mapstencil.jl")

end # Module Stencils

