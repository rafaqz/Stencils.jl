"""
    VonNeumann(1; ndims=2)
    VonNeumann(; radius=1, ndims=2)
    VonNeumann{R,N}()

A 2 dimensionsl Von Neumann stencil is a damond-shaped, omitting the central cell:

Radius `R = 1`:

```
N = 1   N = 2

 ▄ ▄     ▄▀▄
          ▀
```

Radius `R = 2`:

```
N = 1   N = 2

         ▄█▄
▀▀ ▀▀   ▀█▄█▀
          ▀
```

In 1 dimension it is identical to [`Moore`](@ref).

Using `R` and `N` type parameters removes runtime cost of generating the stencil,
compated to passing arguments/keywords.
"""
struct VonNeumann{R,N,L,T<:Union{Nothing,<:AbstractArray}} <: Stencil{R,N,L}
    _neighbors::T
end
VonNeumann(; radius=1, ndims=2) = VonNeumann(radius; ndims)
VonNeumann(radius, _neighbors=nothing; ndims=2) = VonNeumann{radius,ndims}(_neighbors)
VonNeumann{R}(_neighbors=nothing; ndims=2) where R = VonNeumann{R,ndims}(_neighbors)
function VonNeumann{R,N}(_neighbors=nothing) where {R,N}
    L = delannoy(N, R) - 1
    VonNeumann{R,N,L}(_neighbors)
end
VonNeumann{R,N,L}(_neighbors::T=nothing) where {R,N,L,T} = VonNeumann{R,N,L,T}(_neighbors)

@inline setneighbors(n::VonNeumann{R,N,L}, _neighbors::T2) where {R,N,L,T2<:StaticVector{L}} =
    VonNeumann{R,N,L,T2}(_neighbors)

offsets(T::Type{<:VonNeumann}) = SVector(_offsets(T))
@generated function _offsets(::Type{H}) where {H<:VonNeumann{R,N}} where {R,N}
    offsets_expr = Expr(:tuple)
    rngs = ntuple(_ -> -R:R, N)
    for I in CartesianIndices(rngs)
        manhatten_distance = sum(map(abs, Tuple(I)))
        if manhatten_distance in 1:R
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return offsets_expr
end

# Utils

# delannoy 
# Calculate delannoy numbers recursively
# (gives the length of a VonNeumann stencil + center)
function delannoy(a, b)
    (a == 0 || b == 0) && return 1
    return delannoy(a-1, b) + delannoy(a, b-1) + delannoy(a-1, b-1) 
end
