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
offsets(hood::Neighborhood{<:Any,N,L}) where {N,L} = offsets(typeof(hood))::SVector{L,NTuple{N,Int}}
getoffset(hood, i::Int) = offsets(hood)[i]
    
"""
    indices(x::Union{Neighborhood,NeighborhoodRule}}, I::Tuple) -> iterable

Returns an indexable iterable of `Tuple` indices of each neighbor in the main array.
"""
function indices end
@inline indices(hood::Neighborhood, I::CartesianIndex) = indices(hood, Tuple(I))
@inline indices(hood::Neighborhood, I::Int...) = indices(hood, I)
@inline indices(hood::Neighborhood, I) = map(O -> map(+, O, I), offsets(hood)) 
Base.@propagate_inbounds indexat(hood::Neighborhood, center, i) = CartesianIndex(offsets(hood)[i]) + center

"""
    distances(hood::Neighborhood)

Get the center-to-center distance of each neighborhood position from the central cell,
so that horizontally or vertically adjacent cells have a distance of `1.0`, and a
diagonally adjacent cell has a distance of `sqrt(2.0)`.

Values are calculated at compile time, so `distances` can be used with little overhead.
"""
function distances(hood::Neighborhood)
    map(offsets(hood)) do O
        sqrt(sum(o -> o^2, O))
    end
end

"""
    distance_zones(hood::Neighborhood)

List all distance zones as a Tuple
"""
distance_zones(hood::Neighborhood) = Tuple(sort(union(distances)))

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
Base.@propagate_inbounds Base.getindex(hood::Neighborhood, i) = neighbors(hood)[i]
Base.keys(hood::Neighborhood{<:Any,<:Any,L}) where L = StaticArrays.SOneTo(L)
function Base.show(io::IO, mime::MIME"text/plain", hood::Neighborhood{R}) where R
    println(typeof(hood))
    bools = Bool[((i, j) in offsets(hood)) for i in -R:R, j in -R:R]
    print(io, UnicodeGraphics.blockize(bools))
    if !isnothing(neighbors(hood)) 
        println(io)
        if !isnothing(neighbors(hood))
            printstyled(io, "with neighbors:\n", color=:light_black)
            show(io, mime, neighbors(hood))
        end
    end
end

# Utils

# Copied from StaticArrays. If they can do it...
Base.@pure function tuple_contents(::Type{X}) where {X<:Tuple}
    return tuple(X.parameters...)
end
tuple_contents(xs::Tuple) = xs
