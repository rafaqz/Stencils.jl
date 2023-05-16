"""
    Stencil <: StaticVector

Stencils define a pattern of neighboring cells around the current cell.
They reduce the dimensions of the neighborhood values into a `StaticVector`
of neighbor values.

Stencils objects are updated to contain the neighbors of a location.
This is so that user functions can be passed a single object from whitch they
can retreive neighbors, distances to neighbors and other information,
rather than having this in multiple objects.

Stencils also provide a range of compile-time utility funcitons like
`distances` and `offsets`.
"""
abstract type Stencil{R,N,L,T} <: StaticVector{L,T} end

ConstructionBase.constructorof(::Type{<:T}) where T <: Stencil{R,N,L} where {R,N,L} =
    T.name.wrapper{R,N,L}

ndimensions(::Stencil{<:Any,N}) where N = N

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
neighbors(hood::Stencil) = hood.neighbors

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

#### Base methods
@inline Base.Tuple(s::Stencil) = Tuple(neighbors(s))
function Base.promote_rule(::Type{<:Stencil{R,N,L,T}}, ::Type{<:Stencil{R,N,L,U}}) where {R,N,L,T,U}
    Stencil{R,N,L,promote_type(T,U)}
end
@inline Base.iterate(hood::Stencil, args...) = iterate(neighbors(hood), args...)
Base.@propagate_inbounds Base.getindex(hood::Stencil{<:Any,<:Any,<:Any,T}, i::Int) where T =
    hood.neighbors[i]::T
Base.parent(hood::Stencil) = neighbors(hood)

# Show
function Base.show(io::IO, mime::MIME"text/plain", hood::Stencil{R,N}) where {R,N}
    rs = _radii(Val{N}(), R)
    println(typeof(hood))
    bools = _bool_array(hood)
    print(io, UnicodeGraphics.blockize(bools))
    if !isnothing(neighbors(hood))
        println(io)
        if !isnothing(first(neighbors(hood)))
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
function _bool_array(hood::Stencil{R,N}) where {R,N} 
    rs = _radii(hood)
    # Just show the center slice
    Bool[((i, j, ntuple(_ -> 0, N-2)...) in offsets(hood)) for i in -rs[1][1]:rs[1][2], j in -rs[2][1]:rs[2][2]]
end



#### Utils

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


macro stencil(name, description)
    docstring = """

        $name <: Stencil

        $name(; radius=1, ndims=2)
        $name(radius, ndims)
        $name{R,N}()
        $name{R,N}()

    $description

    Using `R` and `N` type parameters removes runtime cost of generating the stencil,
    compated to passing arguments/keywords.
    """

    name = esc(name)

    struct_expr = quote
        struct $name{R,N,L,T} <: Stencil{R,N,L,T}
            neighbors::SVector{L,T}
            $name{R,N,L,T}(neighbors::StaticVector{L,T}) where {R,N,L,T} = new{R,N,L,T}(neighbors)
        end
    end
    func_exprs = quote
        $name{R,N,L}(neighbors::StaticVector{L,T}) where {R,N,L,T} = $name{R,N,L,T}(neighbors)
        $name{R,N,L}() where {R,N,L} = $name{R,N,L}(SVector(ntuple(_ -> nothing, L)))
        function $name{R,N}(args::StaticVector...) where {R,N}
            L = length(offsets($name{R,N}))
            $name{R,N,L}(args...)
        end
        $name{R}(args::StaticVector...) where R = $name{R,2}(args...)
        $name(args::StaticVector...; radius=1, ndims=2) = $name{radius,ndims}(args...)
        $name(radius::Int, ndims::Int=2) = $name{radius,ndims}()

        @inline Stencils.setneighbors(n::$name{R,N,L}, neighbors) where {R,N,L} = $name{R,N,L}(neighbors)
    end
    return Expr(:block, :(Base.@doc $docstring $struct_expr), func_exprs)
end