"""
    Layered <: Abstract

    Layered(layers::...)

`Tuple` or `NamedTuple` of neighborhoods that can be used together.

`neighbors` for `Layered` returns a tuple of iterators
for each neighborhood layer.
"""
struct Layered{R,N,L,La,T} <: Neighborhood{R,N,L}
    "A tuple of custom neighborhoods"
    layers::La
end
Layered(layers::...) = Layere(layers)
function Layered(layers::Union{NamedTuple,Tuple}, _neighbors=nothing)
    R = maximum(map(radius, layers))
    N = ndims(first(layers))
    L = map(length, layers)
    Layered{R,N,L}(layers, _neighbors)
end
function Layered{R,N,L}(layers) where {R,N,L}
    Layered{R,N,L,typeof(layers)}(layers)
end

@inline neighbors(hood::Layered) = map(l -> neighbors(l), hood.layers)
@inline offsets(::Type{<:Layered{R,N,L,La}}) where {R,N,L,La} =
    map(p -> offsets(p), tuple_contents(La))
@inline positions(hood::Layered, I::Tuple) = map(l -> positions(l, I...), hood.layers)
@inline function setneighbors(n::Layered{R,N,L}, _neighbors) where {R,N,L}
    Layered{R,N,L}(n.layers)
end

@inline Base.sum(hood::Layered) = map(sum, neighbors(hood))

@inline function unsafe_readneighbors(hood::Layered{R,N}, A::AbstractArray{T,N}, I::NTuple{N,Int}) where {T,R,N}
    map(l -> unsafe_readneighbors(l, A, I), hood)
end
