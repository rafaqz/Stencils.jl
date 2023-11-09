const CustomOffset = Tuple{Int,Vararg{Int}}
const CustomOffsets = Tuple{<:CustomOffset,Vararg{CustomOffset}}

"""
    Positional <: AbstractPositionalStencil

    Positional(offsets::Tuple{Vararg{Int}}...)
    Positional(offsets::Tuple{Tuple{Vararg{Int}}})
    Positional{O}()

Stencils that can take arbitrary shapes by specifying each coordinate,
as `Tuple{Int,Int}` of the row/column distance (positive and negative)
from the central point.

The stencil radius is calculated from the most distant coordinate,
and the dimensionality `N` of the stencil is taken from the length of
the first coordinate, e.g. `1`, `2` or `3`.

See [`NamedStencil`](@ref) for a similar stencil with named offsets.

## Example

```julia
julia> p = Positional((0, -1), (2, 1), (-1, 1), (0, 1)) 
Positional{((0, -1), (2, 1), (-1, 1), (0, 1)), 2, 2, 4, Nothing}
   ▄ 
 ▀ ▀ 
   ▀ 
```
"""
struct Positional{O,R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::SVector{L,T}
    Positional{O,R,N,L,T}(neighbors::SVector{L,T}) where {O,R,N,L,T} = new{O,R,N,L,T}(neighbors)
end
Positional{O,R,N,L}(neighbors::SVector{L,T}) where {O,R,N,L,T} = 
    Positional{O,R,N,L,T}(neighbors)
Positional{O,R,N,L}() where {O,R,N,L} = 
    Positional{O,R,N,L}(SVector(ntuple(_ -> nothing, L)))
function Positional{O}(args::SVector...) where O
    N = length(first(O))
    R = _positional_radii(N, O)
    L = length(O)
    Positional{O,R,N,L}(args...)
end
Positional(os1::CustomOffset, offsets::CustomOffset...) = Positional((os1, offsets...))
Positional(offsets::CustomOffsets) = Positional{offsets}()

function ConstructionBase.constructorof(::Type{Positional{O,R,N,L,T}}) where {O,R,N,L,T}
    Positional{O,R,N,L}
end

offsets(::Type{<:Positional{O}}) where O = SVector(O)

@inline function rebuild(n::Positional{O,R,N,L}, neighbors) where {O,R,N,L}
    Positional{O,R,N,L}(neighbors)
end

# Calculate the maximum absolute value in the offsets to use as the radius
# function _positional_radii(ndims, offsets::Union{AbstractArray,Tuple})
#     ntuple(ndims) do i
#         extrema(o[i] for o in offsets)
#     end
# end
function _positional_radii(ndims, offsets::Union{AbstractArray,Tuple})
    maximum(map(maximum, offsets))
end
