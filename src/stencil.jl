"""
    Stencil

Stencils define a pattern of neighboring cells around the current cell. 
The [`neighbors`](@ref) function returns the surrounding cells as an iterable.

Stencils objects are updated to contain the neighbors of a location.
This is so that user functions can be passed a single object from whitch they 
can retreive neighbors, distances to neighbors and other information,
rather than having this in multiple objects.

Stencils are iterators over `neighbors(stencil)`, so additional properties can be ignored.
"""
abstract type Stencil{R,N,L} end

ConstructionBase.constructorof(::Type{<:T}) where T <: Stencil{R,N,L} where {R,N,L} =
    T.name.wrapper{R,N,L}

"""
    radius(stencil) -> Int

Return the radius of a stencil.
"""
function radius end
radius(hood::Stencil{R}) where R = R

"""
    diameter(rule) -> Int

The diameter of a stencil is `2r + 1` where `r` is the radius.
"""
diameter(hood::Stencil{R}) where R = diameter(R)
diameter(radius::Integer) = 2radius + 1

"""
    neighbors(x::Stencil) -> iterable

Returns an indexable iterator for all cells in the stencil,
either a `Tuple` of values or a range.

Custom `Stencil`s must define this method.
"""
function neighbors end
neighbors(hood::Stencil) = hood._neighbors

"""
    setneighbors(x::Stencil, neighbors::StaticArray)

Update the eighbors of a `Stencil`, returning and identical object with new values.
"""
function setneighbors end

"""
    offsets(x) -> iterable

Returns an indexable iterable over all cells in the stencil,
containing `Tuple`s of the offset from the central cell.

Custom `Stencil`s must define this method.
"""
function offsets end
offsets(hood::Stencil) = offsets(typeof(hood))
getoffset(hood, i::Int) = offsets(hood)[i]

@generated cartesian_offsets(hood::Stencil) = map(CartesianIndex, offsets(hood))
    
"""
    indices(x::Stencil, I::Tuple) -> iterable

Returns an indexable iterable of `Tuple` indices of each neighbor in the main array.
"""
function indices end
@inline indices(hood::Stencil, I::CartesianIndex) = indices(hood, Tuple(I))
@inline indices(hood::Stencil, I::Int...) = indices(hood, I)
@inline indices(hood::Stencil, I) = map(O -> map(+, O, I), offsets(hood)) 
Base.@propagate_inbounds indexat(hood::Stencil, center, i) = CartesianIndex(offsets(hood)[i]) + center

"""
    distances(hood::Stencil)

Get the center-to-center distance of each stencil position from the central cell,
so that horizontally or vertically adjacent cells have a distance of `1.0`, and a
diagonally adjacent cell has a distance of `sqrt(2.0)`.

Values are calculated at compile time, so `distances` can be used with little overhead.
"""
@generated function distances(hood::Stencil)
    map(offsets(hood)) do O
        sqrt(sum(o -> o^2, O))
    end
end

"""
    distance_zones(hood::Stencil)

List all distance zones as a Tuple
"""
@generated function distance_zones(hood::Stencil)
    map(o -> sum(map(abs, o)), offsets(hood))
end

"""
    kernelproduct(hood::AbstractKernelStencil)
    kernelproduct(hood::Stencil, kernel)

Returns the vector dot product of the stencil and the kernel,
although differing from `dot` in that the dot product is not taken
iteratively for members of the stencil - they are treated as scalars.
"""
function kernelproduct end


# Base methods
Base.eltype(hood::Stencil) = eltype(neighbors(hood))
Base.length(hood::Stencil) = length(typeof(hood))
Base.length(::Type{<:Stencil{<:Any,<:Any,L}}) where L = L
Base.ndims(hood::Stencil{<:Any,N}) where N = N
# Note: size may not relate to `length` in the same way
# as in an array. A stencil does not have to include all cells
# in the area covered by `size` and `axes`.
Base.size(hood::Stencil{R,N}) where {R,N} = ntuple(_ -> 2R+1, N)
Base.axes(hood::Stencil{R,N}) where {R,N} = ntuple(_ -> SOneTo{2R+1}(), N)
Base.iterate(hood::Stencil, args...) = iterate(neighbors(hood), args...)
Base.@propagate_inbounds Base.getindex(hood::Stencil, i) = neighbors(hood)[i]
Base.keys(hood::Stencil{<:Any,<:Any,L}) where L = StaticArrays.SOneTo(L)

# Show
function Base.show(io::IO, mime::MIME"text/plain", hood::Stencil{R,N}) where {R,N}
    rs = _radii(Val{N}(), R)
    println(typeof(hood))
    bools = _bool_array(hood)
    print(io, UnicodeGraphics.blockize(bools))
    if !isnothing(neighbors(hood)) 
        println(io)
        if !isnothing(neighbors(hood))
            printstyled(io, "with neighbors:\n", color=:light_black)
            show(io, mime, neighbors(hood))
        end
    end
end

# Get a array of Bool for the offsets that ar used by a Stencil
function _bool_array(hood::Stencil{R,1}) where {R}
    rs = _radii(hood)
    Bool[((i,) in offsets(hood)) for i in -rs[1][1]:rs[1][2]]
end
function _bool_array(hood::Stencil{R,2}) where {R}
    rs = _radii(hood)
    Bool[((i, j) in offsets(hood)) for i in -rs[1][1]:rs[1][2], j in -rs[2][1]:rs[2][2]]
end
function _bool_array(hood::Stencil{R,3}) where {R}
    rs = _radii(hood)
    # Just show the center slice
    Bool[((i, j, 0) in offsets(hood)) for i in -rs[1][1]:rs[1][2], j in -rs[2][1]:rs[2][2]]
end

# Utils

# Copied from StaticArrays. If they can do it...
Base.@pure function tuple_contents(::Type{X}) where {X<:Tuple}
    return tuple(X.parameters...)
end
tuple_contents(xs::Tuple) = xs


# radii
# Get the radii of a stencil in N dimensions
# The radius can vary by dimension and side
# NTuple of tuples - end state
_radii(::Val{N}, r::NTuple{N,<:Tuple{<:Integer,<:Integer}}) where N = r
_radii(::Val{0}, r::Tuple{}) = ()
# NTuple of Integers, map so both sides are the same
_radii(::Val{N}, rs::NTuple{N,Integer}) where N = map(r -> (r, r), rs) 
# Integer, make an Ntuple{N,NTuple{2,Integer}}
_radii(::Val{N}, r::Integer) where N = ntuple(_ -> (r, r), N)
_radii(ndims::Val, ::Stencil{R}) where R = _radii(ndims, R)
# Convert array/stencil to `Val{N}` for ndims
_radii(::Stencil{R,N}) where {R,N} = _radii(Val{N}(), R)
_radii(A::AbstractArray{<:Any,N}, r) where N = _radii(Val{N}(), r)
