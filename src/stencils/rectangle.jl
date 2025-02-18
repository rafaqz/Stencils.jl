const AxisOffsets = Tuple{Int,Int}

"""
    Rectangle <: Stencil

    Rectangle(offsets::Tuple{Tuple}...)
    Rectangle{O}()

Rectanglar stencils of arbitrary shapes. These are 
specified with pulles of offsets around the center point, 
one for each dimension.
"""
struct Rectangle{O,R,N,L,T} <: Stencil{R,N,L,T}
    neighbors::SVector{L,T}
    center::T
    function Rectangle{O,R,N,L,T}(neighbors::SVector{L,T}, center::T) where {O,R,N,L,T} 
        new{O,R,N,L,T}(neighbors, center)
    end
end
Rectangle{O,R,N,L}(neighbors::SVector{L,T}, center::T) where {O,R,N,L,T} = 
    Rectangle{O,R,N,L,T}(neighbors, center)
Rectangle{O,R,N,L}() where {O,R,N,L} = 
    Rectangle{O,R,N,L}(SVector(ntuple(_ -> nothing, L)), nothing)
function Rectangle{O}(args::SVector, center) where O
    all(map(o -> length(o) == 2, O)) ||
        throw(ArgumentError("All offset tuples must have length `2`, got $O"))
    N = length(O)
    R = maximum(O) do o
        max(map(abs, o)...)
    end
    L = prod(length ∘ splat(:), O)
    Rectangle{O,R,N,L}(args, center)
end
function Rectangle{O}() where O
    all(map(o -> length(o) == 2, O)) ||
        throw(ArgumentError("All offset tuples must have length `2`, got $O"))
    N = length(O)
    R = maximum(O) do o
        max(map(abs, o)...)
    end
    L = prod(length ∘ splat(:), O)
    Rectangle{O,R,N,L}()
end
Rectangle(os1::AxisOffsets, offsets::AxisOffsets...) = Rectangle((os1, offsets...))
Rectangle(offsets::CustomOffsets) = Rectangle{offsets}()

Base.@assume_effects :foldable offsets(::Type{<:Rectangle{O,R,N,L}}) where {O,R,N,L} =
    SVector{L}(map(Tuple, CartesianIndices(map(splat(:), O))))

function ConstructionBase.constructorof(::Type{Rectangle{O,R,N,L,T}}) where {O,R,N,L,T}
    Rectangle{O,R,N,L}
end

@inline function rebuild(::Rectangle{O,R,N,L}, neighbors, center) where {O,R,N,L}
    Rectangle{O,R,N,L}(neighbors, center)
end
