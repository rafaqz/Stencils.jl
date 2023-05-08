"""
    AbstractPositionalStencil <: Stencil

Positional stencils are tuples of coordinates that are specified in relation
to the central point of the current cell. They can be any arbitrary shape or size,
but should be listed in column-major order for performance.
"""
abstract type AbstractPositionalStencil{R,N,L} <: Stencil{R,N,L} end

const CustomOffset = Tuple{Int,Vararg{Int}}
const CustomOffsets = Union{Tuple{<:CustomOffset,Vararg{<:CustomOffset}}}

"""
    Positional <: AbstractPositionalStencil

    Positional(coord::Tuple{Vararg{Int}}...)
    Positional(offsets::Tuple{Tuple{Vararg{Int}}})
    Positional{O}()

Stencils that can take arbitrary shapes by specifying each coordinate,
as `Tuple{Int,Int}` of the row/column distance (positive and negative)
from the central point.

The stencil radius is calculated from the most distant coordinate.
For simplicity the window read from the main grid is a square with sides
`2r + 1` around the central point.

The dimensionality `N` of the stencil is taken from the length of
the first coordinate, e.g. `1`, `2` or `3`.


Example radius `R = 1`:

```
N = 1   N = 2

 ▄▄      ▀▄
          ▀
```

Example radius `R = 2`:

```
N = 1   N = 2

         ▄▄
 ▀ ▀▀   ▀███
           ▀
```

Using the `O` parameter e.g. `Positional{((1, 2), (1, 1))}()` removes any
runtime cost of generating the stencil.
"""
struct Positional{O,R,N,L,T<:Union{Nothing,<:AbstractArray}} <: AbstractPositionalStencil{R,N,L}
    "A tuple of tuples of Int, containing 2-D coordinates relative to the central point"
    _neighbors::T
end
Positional(co::CustomOffset, args::CustomOffset...) = Positional((co, args...))
function Positional(offsets::CustomOffsets, _neighbors::Union{Nothing,AbstractVector}=nothing)
    Positional{offsets}(_neighbors)
end
function Positional{O}(_neighbors::Union{Nothing,AbstractVector}=nothing) where O
    N = length(first(O))
    R = _positional_radii(N, O)
    L = length(O)
    Positional{O,R,N,L}(_neighbors)
end
function Positional{O,R,N,L}(_neighbors::T=nothing) where {O,R,N,L,T<:Union{Nothing,AbstractVector}}
    Positional{O,R,N,L,T}(_neighbors)
end

function ConstructionBase.constructorof(::Type{Positional{O,R,N,L,T}}) where {O,R,N,L,T}
    Positional{O,R,N,L}
end

offsets(::Type{<:Positional{O}}) where O = SVector(O)

@inline function setneighbors(n::Positional{O,R,N,L}, _neighbors::T2) where {O,R,N,L,T2<:StaticVector{L}}
    Positional{O,R,N,L,T2}(_neighbors)
end

# Calculate the maximum absolute value in the offsets to use as the radius
function _positional_radii(ndims, offsets::Union{AbstractArray,Tuple})
    ntuple(ndims) do i
        extrema(o[i] for o in offsets)
    end
end
