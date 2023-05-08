"""
    Layered <: Abstract

    Layered(layers::...)

`Tuple` or `NamedTuple` of neighborhoods that can be used together.

`neighbors` for `Layered` returns a tuple of iterators
for each neighborhood layer.
"""
struct Layered{R,N,L,La} <: Neighborhood{R,N,L}
    "A tuple of custom neighborhoods"
    layers::La
end
Layered(layers::Neighborhood...) = Layered(layers)
function Layered(layers::Union{NamedTuple,Tuple}, _neighbors=nothing)
    N = ndims(first(layers))
    R = if length(layers) > 1
        layer_radii = map(l -> _radii(Val{N}(), l), layers)
        # Find the maximum bounds of each dimension
        reduce(layer_radii) do acc, next
            map(acc, next) do a, n
                min(a[1], n[1]), max(a[2], n[2])
            end
        end
    else
        radius(first(layers))
    end
    L = map(length, layers)
    Layered{R,N,L}(layers)
end
Layered{R,N,L}(layers) where {R,N,L} = Layered{R,N,L,typeof(layers)}(layers)
Layered(; layers...) = Layered(values(layers))

layers(hood::Layered) = hood.layers
@inline neighbors(hood::Layered) = map(l -> neighbors(l), hood.layers)
@inline offsets(::Type{<:Layered{R,N,L,La}}) where {R,N,L,La} =
    map(p -> offsets(p), tuple_contents(La))
@inline indices(hood::Layered, I::Tuple) = map(l -> indices(l, I...), hood.layers)
@inline function setneighbors(h::Layered{R,N,L}, layer_neighbors) where {R,N,L}
    Layered{R,N,L}(map(setneighbors, layers(h), layer_neighbors))
end

@inline function unsafe_neighbors(A::AbstractNeighborhoodArray, hood::Layered, I::CartesianIndex)
    map(l -> unsafe_neighbors(A, l, I), layers(hood))
end

Base.map(f, hood::Layered) = map(f, layers(hood))
