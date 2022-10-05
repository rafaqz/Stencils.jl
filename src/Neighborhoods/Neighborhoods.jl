module Neighborhoods

using ConstructionBase, StaticArrays, OffsetArrays, UnicodeGraphics

export Neighborhood, Window, AbstractKernelNeighborhood, Kernel,
       Moore, VonNeumann, AbstractPositionalNeighborhood, Positional, LayeredPositional

export NeighborhoodArray

export BoundaryCondition, Wrap, Remove

export neighbors, neighborhood, kernel, kernelproduct, offsets, positions, radius, distances

export setwindow, updatewindow, unsafe_updatewindow

export pad_axes, inner_axes, pad_array, inner_array, inner_view

export broadcast_neighborhood, broadcast_neighborhood!

include("neighborhood.jl")
include("array.jl")
include("boundary.jl")
include("padding.jl")
include("broadcast_neighborhood.jl")

include("neighborhoods/moore.jl")
include("neighborhoods/positional.jl")
include("neighborhoods/vonneumman.jl")
include("neighborhoods/window.jl")
include("neighborhoods/kernel.jl")

end # Module Neighborhoods

