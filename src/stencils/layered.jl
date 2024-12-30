"""
    Layered <: Abstract

    Layered(layers::Union{Stencil,Layered}...)
    Layered(; layer_keywords...)
    Layered(layers::Union{Tuple,NamedTuple})

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
Layered(; kw...) = Layered(values(kw))
function Layered(layers::Union{NamedTuple,Tuple})
    N = ndimensions(first(layers))
    R = maximum(map(radius, layers))
    L = map(length, layers)
    Layered{R,N,L}(layers)
end

Base.length(l::Layered) = map(length, layers(l))
Base.getproperty(l::Layered, x::Symbol) = Base.getproperty(layers(l), x::Symbol)
Base.getindex(l::Layered, x::Int) = Base.getindex(layers(l), x)
Base.getindex(l::Layered, x::Symbol) = Base.getindex(layers(l), x)
Base.getindex(l::Layered, x::Tuple) = Base.getindex(layers(l), x)
Base.map(f, stencil::Layered) = map(f, layers(stencil))

layers(stencil::Layered) = Base.getfield(stencil, :layers)
@inline offsets(::Type{<:Layered{R,N,L,T,La}}) where {R,N,L,T,La} =
    map(p -> offsets(p), tuple_contents(La))

@inline function rebuild(h::Layered{R,N,L,T}, layerneighbors, centers) where {R,N,L,T}
    Layered{R,N,L,T}(map(rebuild, layers(h), layerneighbors, centers))
end

for f in (:center, :neighbors, :offsets, :cartesian_offsets, :distances, :distance_zones)
    @eval $f(layered::Layered) = map($f, layered)
end

radius(layered::Layered{R}) where R = R
ndimensions(layered::Layered{<:Any,N}) where N = N
indices(layered::Layered, I::Tuple) = map(l -> indices(l, I), layered)

_zero_values(::Type{T}, l::Layered) where T =
    map(l -> _zero_values(T, l), layers(l))
