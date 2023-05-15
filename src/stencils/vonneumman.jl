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
struct VonNeumann{R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::StaticVector{L,T}
    VonNeumann{R,N,L,T}(neighbors::StaticVector{L,T}) where {R,N,L,T} = new{R,N,L,T}(neighbors)
end
VonNeumann{R,N,L}(neighbors::StaticVector{L,T}) where {R,N,L,T} = VonNeumann{R,N,L,T}(neighbors)
VonNeumann{R,N,L}() where {R,N,L} = VonNeumann{R,N,L}(SVector(ntuple(_ -> nothing, L))) 
function VonNeumann{R,N}(args::StaticVector...) where {R,N}
    L = delannoy(N, R) - 1
    VonNeumann{R,N,L}(args...)
end
VonNeumann{R}(args::StaticVector...) where R = VonNeumann{R,2}(args...)
VonNeumann(args::StaticVector...; radius=1, ndims=2) = VonNeumann{radius,ndims}(args...)
VonNeumann(radius::Int, ndims::Int=2) = VonNeumann{radius,ndims}()

@inline setneighbors(n::VonNeumann{R,N,L}, neighbors) where {R,N,L} =
    VonNeumann{R,N,L}(neighbors)

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
