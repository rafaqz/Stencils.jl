const AxisOffsets = Tuple{Int,Int}

"""
    Rectangle <: Stencil

    Rectangle(offsets::Tuple{Tuple}...)
    Rectangle{O}()

Stencils that can take arbitrary shapes by specifying each coordinate,
as `Tuple{Int,Int}` of the row/column distance (positive and negative)
from the central point.

The stencil radius is calculated from the most distant coordinate,
and the dimensionality `N` of the stencil is taken from the length of
the first coordinate, e.g. `1`, `2` or `3`.

See [`NamedStencil`](@ref) for a similar stencil with named offsets.

## Example

```julia
julia> p = Rectangle((0, -1), (2, 1)) 
Positional{((0, -1), (2, 1), (-1, 1), (0, 1)), 2, 2, 4, Nothing}
   ▄ 
 ▀ ▀ 
   ▀ 
```
"""
struct Rectangle{O,R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::SVector{L,T}
    function Rectangle{O,R,N,L,T}(neighbors::SVector{L,T}) where {O,R,N,L,T} 
        @assert all(map(o -> length(o) == N, O)) "All offsets must be the length `N` of $N, got $O" 
        new{O,R,N,L,T}(neighbors)
    end
end
Rectangle{O,R,N,L}(neighbors::SVector{L,T}) where {O,R,N,L,T} = 
    Rectangle{O,R,N,L,T}(neighbors)
Rectangle{O,R,N,L}() where {O,R,N,L} = 
    Rectangle{O,R,N,L}(SVector(ntuple(_ -> nothing, L)))
function Rectangle{O}(args::SVector...) where O
    N = length(O)
    R = maximum(O) do o
        max(map(abs, o)...)
    end
    L = prod(length ∘ splat(:), O)
    Rectangle{O,R,N,L}(args...)
end
Rectangle(os1::AxisOffsets, offsets::AxisOffsets...) = Rectangle((os1, offsets...))
Rectangle(offsets::CustomOffsets) = Rectangle{offsets}()

offsets(::Type{<:Rectangle{O,R,N,L}}) where {O,R,N,L} =
    SVector{L}(map(Tuple, CartesianIndices(map(splat(:), O))))

function ConstructionBase.constructorof(::Type{Rectangle{O,R,N,L,T}}) where {O,R,N,L,T}
    Rectangle{O,R,N,L}
end

@inline function rebuild(n::Rectangle{O,R,N,L}, neighbors) where {O,R,N,L}
    Rectangle{O,R,N,L}(neighbors)
end
