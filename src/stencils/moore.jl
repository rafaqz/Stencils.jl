"""
    Moore <: Stencil

    Moore(radius::Int=1; ndims=2)
    Moore(; radius=1, ndims=2)
    Moore{R}(; ndims=2)
    Moore{R,N}()

Moore stencils define the stencil as all cells within a horizontal or
vertical distance of the central cell. The central cell is omitted.

Radius `R = 1`:

```
N = 1   N = 2
 
 ▄ ▄     █▀█
         ▀▀▀
```

Radius `R = 2`:

```
N = 1   N = 2

        █████
▀▀ ▀▀   ██▄██
        ▀▀▀▀▀
```

Using `R` and `N` type parameters removes runtime cost of generating the stencil,
compated to passing arguments/keywords.
"""
struct Moore{R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::StaticVector{L,T}
    Moore{R,N,L,T}(neighbors::StaticVector{L,T}) where {R,N,L,T} = new{R,N,L,T}(neighbors)
end
Moore{R,N,L}(neighbors::StaticVector{L,T}) where {R,N,L,T} = Moore{R,N,L,T}(neighbors)
Moore{R,N,L}() where {R,N,L} = Moore{R,N,L}(SVector(ntuple(_ -> nothing, L)))
function Moore{R,N}(args::StaticVector...) where {R,N}
    L = (2R + 1)^N - 1
    Moore{R,N,L}(args...)
end
Moore{R}(args::StaticVector...) where R = Moore{R,2}(args...)
Moore(args::StaticVector...; radius=1, ndims=2) = Moore{radius,ndims}(args...)
Moore(radius::Int, ndims::Int=2) = Moore{radius,ndims}()

@inline offsets(T::Type{<:Moore}) = SVector(_offsets(T))
@generated function _offsets(::Type{<:Moore{R,N,L}}) where {R,N,L}
    exp = Expr(:tuple)
    # First half
    for I in CartesianIndices(ntuple(_-> -R:R, N))[1:L÷2]
        push!(exp.args, :($(Tuple(I))))
    end
    # Skip the middle position
    # Second half
    for I in CartesianIndices(ntuple(_-> -R:R, N))[L÷2+2:L+1]
        push!(exp.args, :($(Tuple(I))))
    end
    return exp
end
@inline setneighbors(n::Moore{R,N,L}, neighbors) where {R,N,L} = Moore{R,N,L}(neighbors)
