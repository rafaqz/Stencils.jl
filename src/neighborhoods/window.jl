"""
    Window <: Neighborhood

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
struct Window{R,N,L,T<:Union{Nothing,<:AbstractArray}} <: Neighborhood{R,N,L}
    _neighbors::T
end
Window(; radius=1, ndims=2) = Window{radius,ndims}(args...)
Window(R::Int, args...; ndims=2) = Window{R,ndims}(args...)
Window{R}(_neighbors=nothing; ndims=2) where {R} = Window{R,ndims}(_neighbors)
Window{R,N}(_neighbors=nothing) where {R,N} = Window{R,N,(2R+1)^N}(_neighbors)
Window{R,N,L}(_neighbors::T=nothing) where {R,N,L,T} = Window{R,N,L,T}(_neighbors)
Window(A::AbstractArray) = Window{(size(A, 1) - 1) ÷ 2,ndims(A)}()

# The central cell is included
@inline function offsets(::Type{<:Window{R,N}}) where {R,N}
    D = 2R + 1
    SVector(ntuple(i -> (rem(i - 1, D) - R, (i - 1) ÷ D - R), D^N))
end

@inline setneighbors(::Window{R,N,L}, _neighbors::T2) where {R,N,L,T2} = Window{R,N,L,T2}(_neighbors)
