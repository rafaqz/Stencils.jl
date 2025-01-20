const CustomOffset = Tuple{Int,Vararg{Int}}
const CustomOffsets = Tuple{<:CustomOffset,Vararg{CustomOffset}}

"""
    NamedStencil <: AbstractStencil

    NamedStencil(; kw...)
    NamedStencil(values::NamedTuple)
    NamedStencil{Keys}(values)

A named stencil that can take arbitrary shapes where each offset
position is named. This can make stencil code easier to read by 
removing magic numbers.

## Example

```jldoctest
julia> using Stencils

julia> ns = NamedStencil(; west=(0, -1), north=(1, 0), south=(-1, 0), east=(0, 1)) 
NamedStencil{(:west, :north, :south, :east), ((0, -1), (1, 0), (-1, 0), (0, 1)), 1, 2, 4, Nothing}
▄▀▄
 ▀ 

julia> A = StencilArray((1:10) * (1:10)', ns);

julia> stencil(A, (5, 5)).east # we can access values by name
30

julia> mapstencil(s -> s.east + s.west, A); # and use them in `mapstencil` functions
```

We can also take some shortcuts, and just name an existing stencil:

```julia
julia> ns = NamedStencil{(:w,:n,:s,:e)}(VonNeumann(1)) 
```

The stencil radius is calculated from the most distant coordinate,
and the dimensionality `N` of the stencil is taken from the length of
the first coordinate, e.g. `1`, `2` or `3`.
"""
struct NamedStencil{K,O,R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::SVector{L,T}
    center::T
    function NamedStencil{K,O,R,N,L,T}(neighbors::SVector{L,T},center::T) where {K,O,R,N,L,T}
        length(K) == length(O) == L || throw(ArgumentError("Length of keys must match length of offsets and L parameter"))
        new{K,O,R,N,L,T}(neighbors, center)
    end
end
NamedStencil{K,O,R,N,L}(neighbors::SVector{L,T},center::T) where {K,O,R,N,L,T} =
    NamedStencil{K,O,R,N,L,T}(neighbors, center)
NamedStencil{K,O,R,N,L}() where {K,O,R,N,L} =
    NamedStencil{K,O,R,N,L}(SVector(ntuple(_ -> nothing, L)), nothing)
function NamedStencil{K,O}() where {K,O}
    N = length(first(O))
    R = _positional_radii(N, O)
    L = length(O)
    NamedStencil{K,O,R,N,L}()
end
NamedStencil(offsets::NamedTuple{K}) where K = NamedStencil{K,values(offsets)}()
NamedStencil(; kw...) = NamedStencil(values(kw))
NamedStencil(ns::NamedStencil) = ns
# Ambiguity
NamedStencil{K}(offsets::Tuple) where K = NamedStencil{K,offsets}()
NamedStencil{K}(offsets::AbstractArray) where K = NamedStencil{K,Tuple(offsets)}()
NamedStencil{K}(offsets::StaticArray) where K = NamedStencil{K,Tuple(offsets)}()
NamedStencil{K}(offsets::Base.Generator) where K = NamedStencil{K,Tuple(offsets)}()
NamedStencil{K}(stencil::Stencil) where K = NamedStencil{K,Tuple(offsets(stencil))}()

Base.getproperty(a::NamedStencil{K}, x::Symbol) where K = getproperty(NamedTuple{K}(values(a)), x)
Base.getindex(a::NamedStencil, x::Symbol) = getproperty(a, x)
Base.propertynames(a::NamedStencil{K}) where K = K
_names(a::Type{NamedStencil{K,O,R,N,L,T}}) where {K,O,R,N,L,T} = K
_names(a::NamedStencil) = _names(typeof(a))

function ConstructionBase.constructorof(::Type{NamedStencil{K,O,R,N,L}}) where {K,O,R,N,L}
    NamedStencil{K,O,R,N,L}
end

offsets(::Type{<:NamedStencil{<:Any,O}}) where O = SVector(O)

@inline function rebuild(::NamedStencil{K,O,R,N,L}, neighbors, center) where {K,O,R,N,L}
    NamedStencil{K,O,R,N,L}(neighbors, center)
end

NamedStencil(s::Cardinal) = NamedStencil{(:E, :S, :N, :W)}(s)
NamedStencil(s::Ordinal) = NamedStencil{(:SE, :NE, :SW, :NW)}(s)

function Base.merge(a::NamedStencil{<:Any,<:Any,<:Any,N}, b::NamedStencil{<:Any,<:Any,<:Any,N}) where N

    names_a = _names(a)
    names_b = _names(b)
    isdisjoint(names_a, names_b) || throw(ArgumentError("Stencils have overlapping names"))

    off_a = offsets(a)
    off_b = offsets(b)
    isdisjoint(off_a, off_b) || throw(ArgumentError("Named stencils have overlapping offsets"))

    na = NamedTuple{names_a}(off_a)
    nb = NamedTuple{names_b}(off_b)
    nt = merge(na, nb)

    NamedStencil(nt)
end
