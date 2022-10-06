module Neighborhoods

using ConstructionBase, StaticArrays, OffsetArrays, UnicodeGraphics

export Neighborhood, Window, Kernel, Moore, VonNeumann, Positional, Layered
export NeighborhoodArray
export BoundaryCondition, Wrap, Remove
export Paddeun, Padded, Unpadded

export neighborhood, kernel, neighbors, offsets, indices, distances, radius, diameter
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

