"""
    Moore <: Neighborhood

    Moore(radius::Int=1; ndims=2)
    Moore(; radius=1, ndims=2)
    Moore{R}(; ndims=2)
    Moore{R,N}()

Moore neighborhoods define the neighborhood as all cells within a horizontal or
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

Using `R` and `N` type parameters removes runtime cost of generating the neighborhood,
compated to passing arguments/keywords.
"""
struct Moore{R,N,L,T} <: Neighborhood{R,N,L}
    _neighbors::T
end
Moore(radius::Int=1; ndims=2) = Moore{radius,ndims}()
Moore(args...; radius=1, ndims=2) = Moore{radius,ndims}(args...)
Moore{R}(_neighbors=nothing; ndims=2) where R = Moore{R,ndims,}(_neighbors)
Moore{R,N}(_neighbors=nothing) where {R,N} = Moore{R,N,(2R+1)^N-1}(_neighbors)
Moore{R,N,L}(_neighbors::T=nothing) where {R,N,L,T} = Moore{R,N,L,T}(_neighbors)

@generated function offsets(::Type{<:Moore{R,N,L}}) where {R,N,L}
    exp = Expr(:tuple)
    for I in CartesianIndices(ntuple(_-> -R:R, N))[1:L÷2]
        push!(exp.args, :($(Tuple(I))))
    end
    for I in CartesianIndices(ntuple(_-> -R:R, N))[L÷2+2:L+1]
        push!(exp.args, :($(Tuple(I))))
    end
    return exp
end
@inline setneighbors(n::Moore{R,N,L}, _neighbors::T2) where {R,N,L,T2} = Moore{R,N,L,T2}(_neighbors)
