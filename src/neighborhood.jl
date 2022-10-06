"""
    Neighborhood

Neighborhoods define the pattern of surrounding cells in the "neighborhood"
of the current cell. The `neighbors` function returns the surrounding
cells as an iterable.

The main kinds of neighborhood are demonstrated below:

![Neighborhoods](https://raw.githubusercontent.com/cesaraustralia/DynamicGrids.jl/media/Neighborhoods.png)

Neighborhoods can be used in `NeighborhoodRule` and `SetNeighborhoodRule` -
the same shapes with different purposes. In a `NeighborhoodRule` the neighborhood specifies
which cells around the current cell are returned as an iterable from the `neighbors` function.
These can be counted, summed, compared, or multiplied with a kernel in an
`AbstractKernelNeighborhood`, using [`kernelproduct`](@ref).

In `SetNeighborhoodRule` neighborhoods give the locations of cells around the central cell,
as [`offsets`] and absolute [`positions`](@ref) around the index of each neighbor. These
can then be written to manually.
"""
abstract type Neighborhood{R,N,L} end

ConstructionBase.constructorof(::Type{<:T}) where T <: Neighborhood{R,N,L} where {R,N,L} =
    T.name.wrapper{R,N,L}

"""
    kernelproduct(hood::AbstractKernelNeighborhood)
    kernelproduct(hood::Neighborhood, kernel)

Returns the vector dot product of the neighborhood and the kernel,
although differing from `dot` in that the dot product is not taken for
vector members of the neighborhood - they are treated as scalars.
"""
function kernelproduct end

"""
    radius(neighborhood) -> Int

Return the radius of a neighborhood.
"""
function radius end
radius(hood::Neighborhood{R}) where R = R


"""
    diameter(rule) -> Int

The diameter of a neighborhood is `2r + 1` where `r` is the radius.
"""
diameter(hood::Neighborhood{R}) where R = diameter(R)
diameter(radius::Integer) = 2radius + 1

"""
    neighbors(x::Neighborhood) -> iterable

Returns an indexable iterator for all cells in the neighborhood,
either a `Tuple` of values or a range.

Custom `Neighborhood`s must define this method.
"""
function neighbors end
neighbors(hood::Neighborhood) = hood._neighbors

"""
    offsets(x) -> iterable

Returns an indexable iterable over all cells in the neighborhood,
containing `Tuple`s of the offset from the central cell.

Custom `Neighborhood`s must define this method.
"""
function offsets end
offsets(hood::Neighborhood{<:Any,N,L}) where {N,L} = offsets(typeof(hood))::NTuple{L,NTuple{N,Int}}
    
"""
    indices(x::Union{Neighborhood,NeighborhoodRule}}, I::Tuple) -> iterable

Returns an indexable iterable of `Tuple` indices of each neighbor in the main array.
"""
function indices end
@inline indices(hood::Neighborhood, I::CartesianIndex) = indices(hood, Tuple(I))
@inline indices(hood::Neighborhood, I::Int...) = indices(hood, I)
@generated function indices(hood::Neighborhood, I)
    args = map(O -> :(map(+, $O, I)), offsets(hood)) 
    return Expr(:tuple, args...)
end

struct LazyNeighors{T,N,L,A<:AbstractKernelNeighborhood{T,N,},H<:Neighborhood{<:Any,<:Any,L}}
    A::A
end

"""
    distances(hood::Neighborhood)

Get the center-to-center distance of each neighborhood position from the central cell,
so that horizontally or vertically adjacent cells have a distance of `1.0`, and a
diagonally adjacent cell has a distance of `sqrt(2.0)`.

Values are calculated at compile time, so `distances` can be used with little overhead.
"""
@generated function distances(hood::Neighborhood{R,N,L}) where {R,N,L}
    args = map(offsets(hood)) do O
        sqrt(sum(o -> o^2, O))
    end
    return Expr(:tuple, args...)
end

Base.eltype(hood::Neighborhood) = eltype(neighbors(hood))
Base.length(hood::Neighborhood) = length(typeof(hood))
Base.length(::Type{<:Neighborhood{<:Any,<:Any,L}}) where L = L
Base.ndims(hood::Neighborhood{<:Any,N}) where N = N
# Note: size may not relate to `length` in the same way
# as in an array. A neighborhood does not have to include all cells
# in the area covered by `size` and `axes`.
Base.size(hood::Neighborhood{R,N}) where {R,N} = ntuple(_ -> 2R+1, N)
Base.axes(hood::Neighborhood{R,N}) where {R,N} = ntuple(_ -> SOneTo{2R+1}(), N)
Base.iterate(hood::Neighborhood, args...) = iterate(neighbors(hood), args...)
Base.getindex(hood::Neighborhood, i) = neighbors(hood)[i]
function Base.show(io::IO, mime::MIME"text/plain", hood::Neighborhood{R}) where R
    println(typeof(hood))
    bools = Bool[((i, j) in offsets(hood)) for i in -R:R, j in -R:R]
    print(io, UnicodeGraphics.blockize(bools))
    if !isnothing(neighbors(hood)) 
        println(io)
        show(io, mime, neighbors(hood))
    end
end

# Utils

# Copied from StaticArrays. If they can do it...
Base.@pure function tuple_contents(::Type{X}) where {X<:Tuple}
    return tuple(X.parameters...)
end
tuple_contents(xs::Tuple) = xs

"""
    readneighbors(hood::Neighborhood, A::AbstractArray, I) => SArray

Get a single neighborhood from an array, as a `Tuple`, checking bounds.
"""
readneighbors(hood::Neighborhood, A::AbstractArray, I::Int...) = readneighbors(hood, A, I)
@inline function readneighbors(hood::Neighborhood{R,N}, A::AbstractArray, I) where {R,N}
    for O in ntuple(_ -> (-R, R), N)
        edges = Tuple(I) .+ O
        map(I -> checkbounds(A, I...), edges)
    end
    return unsafe_readneighbors(hood, A, I...)
end

"""
    unsafe_readneighbors(hood::Neighborhood, A::AbstractArray, I) => SArray

Get a single neighborhood from an array, as a `Tuple`, without checking bounds.
"""
@inline unsafe_readneighbors(hood::Neighborhood, A::AbstractArray, I::CartesianIndex) =
    unsafe_readneighbors(hood, A, Tuple(I))
@inline unsafe_readneighbors(hood::Neighborhood, A::AbstractArray, I::Int...) =
    unsafe_readneighbors(hood, A, I)
@inline function unsafe_readneighbors(hood::Neighborhood{R,N}, A::AbstractArray{T,N}, I::NTuple{N,Int}) where {T,R,N}
    map(indices(hood, I)) do P
        neighbor_getindex(A, P...) 
    end
end
function unsafe_readneighbors(
    ::Neighborhood{R,N1}, A::AbstractArray{T,N2}, I::NTuple{N3,Int}
) where {T,R,N1,N2,N3}
    throw(DimensionMismatch("neighborhood has $N1 dimensions while array has $N2 and index has $N3"))
end

"""
    updateneighbors(x, A::AbstractArray, I...) => Neighborhood

Set the neighbors of a neighborhood to values from the array A around index `I`.
Bounds checks will reduce performance, aim to use `unsafe_setneighbors` directly.
"""
@inline function updateneighbors(x, A::AbstractArray, i, I...)
    setneighbors(x, readneighbors(x, A, i, I...))
end

"""
    unsafe_updateneighbors(x, A::AbstractArray, I...) => Neighborhood

Set the neighbors of a neighborhood to values from the array A around index `I`.

No bounds checks occur, ensure that A has padding of at least the neighborhood radius.
"""
@inline function unsafe_updateneighbors(h::Neighborhood, A::AbstractArray, i, I...)
    setneighbors(h, unsafe_readneighbors(h, A, i, I...))
end
