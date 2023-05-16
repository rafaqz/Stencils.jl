"""
    Window <: Stencil

    Window(; radius=1, ndims=2)
    Window{R}(; ndims=2)
    Window{R,N}()

A neighboorhood of radius R that includes the central cell.

Radius `R = 1`:

```
N = 1   N = 2
        
 ▄▄▄     ███
         ▀▀▀
```

Radius `R = 2`:

```
N = 1   N = 2

        █████
▀▀▀▀▀   █████
        ▀▀▀▀▀
```
"""
struct Window{R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::SVector{L,T}
    Window{R,N,L,T}(neighbors::SVector{L,T}) where {R,N,L,T} = new{R,N,L,T}(neighbors)
end
Window{R,N,L}(neighbors::SVector{L,T}) where {R,N,L,T} = Window{R,N,L,T}(neighbors)
Window{R,N,L}() where {R,N,L} = Window{R,N,L}(SVector(ntuple(_ -> nothing, L)))
function Window{R,N}(args::SVector...) where {R,N} 
    L = (2R + 1)^N
    Window{R,N,L}(args...)
end
Window{R}(args::SVector...) where {R} = Window{R,2}(args...)
Window(args...; radius=1, ndims=2) = Window{radius,ndims}(args...)
Window(radius::Int, ndims::Int=2) = Window{radius,ndims}()
Window(A::AbstractArray) = Window{(size(A, 1) - 1) ÷ 2,ndims(A)}()

# The central cell is included
@generated function offsets(::Type{<:Window{R,N}}) where {R,N}
    D = 2R + 1
    vals = ntuple(i -> (rem(i - 1, D) - R, (i - 1) ÷ D - R), D^N)
    return :(SVector($vals))
end

@inline setneighbors(::Window{R,N,L}, neighbors) where {R,N,L} = Window{R,N,L}(neighbors)
