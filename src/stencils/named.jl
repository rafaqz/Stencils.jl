const CustomOffset = Tuple{Int,Vararg{Int}}
const CustomOffsets = Tuple{<:CustomOffset,Vararg{CustomOffset}}

"""
    NamedStencil <: AbstractStencil

    NamedStencil(coord::Tuple{Vararg{Int}}...)
    NamedStencil(offsets::Tuple{Tuple{Vararg{Int}}})
    NamedStencil{O}()

A named stencil that can take arbitrary shapes where each offset
position is named. This can make stencil code much easier to
read by removing magic numbers.

The stencil radius is calculated from the most distant coordinate.
For simplicity the window read from the main grid is a square with sides
`2r + 1` around the central point.

The dimensionality `N` of the stencil is taken from the length of
the first coordinate, e.g. `1`, `2` or `3`.
"""
struct NamedStencil{K,O,R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::SVector{L,T}
    function NamedStencil{K,O,R,N,L,T}(neighbors::SVector{L,T}) where {K,O,R,N,L,T}
        length(K) == length(O) == L || throw(ArgumentError("Length of keys must match length of offsets and L parameter"))
        new{K,O,R,N,L,T}(neighbors)
    end
end
NamedStencil{K,O,R,N,L}(neighbors::SVector{L,T}) where {K,O,R,N,L,T} =
    NamedStencil{K,O,R,N,L,T}(neighbors)
NamedStencil{K,O,R,N,L}() where {K,O,R,N,L} =
    NamedStencil{K,O,R,N,L}(SVector(ntuple(_ -> nothing, L)))
function NamedStencil{K,O}(args...) where {K,O}
    N = length(first(O))
    R = _positional_radii(N, O)
    L = length(O)
    NamedStencil{K,O,R,N,L}(args...)
end
NamedStencil(offsets::NamedTuple{K}) where K = NamedStencil{K,values(offsets)}()
NamedStencil(; kw...) = NamedStencil(values(kw))
NamedStencil(ns::NamedStencil) = ns
NamedStencil{K}(offsets) where K = NamedStencil{K,Tuple(offsets)}()
NamedStencil{K}(offsets::AbstractArray) where K = NamedStencil{K,Tuple(offsets)}()

Base.getproperty(a::NamedStencil{K}, x::Symbol) where K = getproperty(NamedTuple{K}(values(a)), x)
Base.getindex(a::NamedStencil, x::Symbol) = getproperty(a, x)
Base.propertynames(a::NamedStencil{K}) where K = K

function ConstructionBase.constructorof(::Type{NamedStencil{K,O,R,N,L}}) where {K,O,R,N,L}
    NamedStencil{K,O,R,N,L}
end

offsets(::Type{<:NamedStencil{<:Any,O}}) where O = SVector(O)

@inline function rebuild(n::NamedStencil{K,O,R,N,L}, neighbors) where {K,O,R,N,L}
    NamedStencil{K,O,R,N,L}(neighbors)
end
