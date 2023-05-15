"""
    Layered <: Abstract

    Layered(layers::...)

`Tuple` or `NamedTuple` of stencils that can be used together.

`neighbors` for `Layered` returns a tuple of iterators
for each stencil layer.
"""
struct Layered{R,N,L,T,La}
    "A tuple of custom stencils"
    layers::La
end
Layered{R,N,L,T}(layers) where {R,N,L,T} = Layered{R,N,L,T,typeof(layers)}(layers)
Layered{R,N,L}(layers) where {R,N,L} = Layered{R,N,L,eltype(first(layers))}(layers)

const StencilOrLayered = Union{Stencil,Layered}

Layered(layers::StencilOrLayered...) = Layered(layers)
function Layered(layers::Union{NamedTuple,Tuple})
    N = ndimensions(first(layers))
    R = maximum(map(radius, layers))
    L = map(length, layers)
    Layered{R,N,L}(layers)
end

layers(stencil::Layered) = stencil.layers
@inline offsets(::Type{<:Layered{R,N,L,T,La}}) where {R,N,L,T,La} =
    map(p -> offsets(p), tuple_contents(La))

@inline function setneighbors(h::Layered{R,N,L,T}, layerneighbors) where {R,N,L,T}
    Layered{R,N,L,T}(map(setneighbors, layers(h), layerneighbors))
end

for f in (:neighbors, :offsets, :cartesian_offsets, :distances, :distance_zones)
    @eval $f(layered::Layered) = map($f, layered)
end

radius(layered::Layered{R}) where R = R
ndimensions(layered::Layered{<:Any,N}) where N = N
indices(layered::Layered, I) = map(l -> indices(l, I), layered)

Base.map(f, stencil::Layered) = map(f, layers(stencil))

