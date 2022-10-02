module Neighborhoods

using ConstructionBase, StaticArrays, OffsetArrays, UnicodeGraphics

export Neighborhood, Window, AbstractKernelNeighborhood, Kernel,
       Moore, VonNeumann, AbstractPositionalNeighborhood, Positional, LayeredPositional

export NeighborhoodArray

export BoundaryCondition, Wrap, Remove

export neighbors, neighborhood, kernel, kernelproduct, offsets, positions, radius, distances

export setwindow, updatewindow, unsafe_updatewindow

export pad_axes, unpad_axes, pad_array, unpad_array, unpad_view

export broadcast_neighborhood, broadcast_neighborhood!

include("neighborhood.jl")
include("array.jl")
include("boundary.jl")
include("padding.jl")
include("moore.jl")
include("positional.jl")
include("vonneumman.jl")
include("window.jl")
include("kernel.jl")
include("broadcast_neighborhood.jl")

end # Module Neighborhoods

