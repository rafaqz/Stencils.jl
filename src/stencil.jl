"""
    Stencil <: StaticVector

Stencils define a pattern of neighboring cells around the current cell.
They reduce the structure and dimensions of the neighborhood into a
`StaticVector` of values.

Stencil objects are updated to contain the neighbors for an array index.

This design is so that user functions can be passed a single object from
which they can retrieve center, neighbors, offsets, distances to neighbors
and other information.

Stencils also provide a range of compile-time utility funcitons like
`distances` and `offsets`.
"""
abstract type Stencil{R,N,L,T} <: StaticVector{L,T} end

ConstructionBase.constructorof(::Type{<:T}) where T <: Stencil{R,N,L} where {R,N,L} =
    T.name.wrapper{R,N,L}

# Usually stencils are isbits so we don't do anything on `adapt`
Adapt.adapt_structure(to, s::Stencil) = s

Base.@assume_effects :foldable StaticArraysCore.Size(::Type{T}) where {T<:Stencil{<:Any,<:Any}} = 
    StaticArrays.Size{(length(offsets(T)),)}()
StaticArraysCore.Size(::Type{T}) where {T<:Stencil{<:Any,<:Any,L}} where L =
    StaticArrays.Size((L,))

ndimensions(::Stencil{<:Any,N}) where N = N

"""
    radius(stencil) -> Int

Return the radius of a stencil.
"""
function radius end
radius(hood::Stencil{R}) where R = R
radius(::Tuple{}) = 0

"""
    diameter(rule) -> Int

The diameter of a stencil is `2r + 1` where `r` is the radius.
"""
diameter(hood::Stencil{R}) where R = diameter(R)
diameter(radius::Integer) = 2radius + 1

"""
    neighbors(x::Stencil) -> iterable

Returns a basic `SVector` of all cells in the stencil.
"""
function neighbors end
neighbors(hood::Stencil) = getfield(hood, :neighbors)

"""
    rebuild(x::Stencil, neighbors::StaticArray)

Rebuild a `Stencil`, returning an stencil of the same
size and shape, with new neighbor values.
"""
function rebuild end

"""
    offsets(x)

Return an `SVector` of `NTuple{N,Int}`, containing all
positions in the stencil as offsets from the central cell.

Custom `Stencil`s must define this method.
"""
function offsets end
offsets(hood::Stencil) = offsets(typeof(hood))
getoffset(hood, i::Int) = offsets(hood)[i]

@generated cartesian_offsets(hood::Stencil) = map(CartesianIndex, offsets(hood))

"""
    indices(x::Stencil, I::Union{Tuple,CartesianIndex})
    indices(x::AbstractStencilArray, I::Union{Tuple,CartesianIndex})

Returns an `SVector` of `CartesianIndices` for each neighbor around `I`.

`indices` for `Stencil` do not know about array boundaries and wil not wrap or reflect.
On `AbstractStencilArray` they will wrap and reflect depending on the boundary condition 
of the array.
"""
function indices end
@inline indices(hood, I::CartesianIndex) = indices(hood, Tuple(I))
@inline indices(hood, I::Int...) = indices(hood, I)
# Allow trailing indices - we can use a Stencil with N smaller than the array N
@inline function indices(hood::Stencil{<:Any,N1}, I::NTuple{N2}) where {N1,N2} 
    map(I1 -> (I1..., I[N1+1:N2]...), indices(hood, I[1:N1])) 
end
@inline indices(hood::Stencil{<:Any,N}, I::NTuple{N}) where N = map(O -> map(+, O, I), offsets(hood))

Base.@propagate_inbounds indexat(hood::Stencil, center, i) = CartesianIndex(offsets(hood)[i]) + center

"""
    center(x::Stencil)

Return the value of the central cell in the stencil.
"""
center(hood::Stencil) = getfield(hood, :center)

"""
    distances(hood::Stencil)

Returns an `SVector` of center-to-center distance of each stencil position from the central
cell, so that horizontally or vertically adjacent cells have a distance of `1.0`, and a
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

Returns an `SVector` of `Int` distance zones for each offset in the Stencil.
"""
@generated function distance_zones(hood::Stencil)
    map(o -> sum(map(abs, o)), offsets(hood))
end

"""
    kernelproduct(hood::AbstractKernelStencil)
    kernelproduct(hood::Stencil, kernel)

Returns the vector dot product of the stencil and the kernel,
although differing from `dot` in that it is not taken
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
    neighbors(hood)[i]::T
Base.parent(hood::Stencil) = neighbors(hood)

# Show
function Base.show(io::IO, mime::MIME"text/plain", hood::Stencil{R,N}) where {R,N}
    rs = _radii(Val{N}(), R)
    show(io, typeof(hood))
    bools = _bool_array(hood)
    println(io)
    UnicodeGraphics.uprint(io, bools, :block)
    if !isnothing(neighbors(hood))
        if !isnothing(first(neighbors(hood)))
            println(io)
            printstyled(io, "with neighbors:\n", color=:light_black)
            show(io, mime, neighbors(hood))
        end
    end
end

function Base.show(io::IO, hood::Stencil{R,N}) where {R,N}
    show(io, typeof(hood))
    if !isnothing(neighbors(hood))
        if !isnothing(first(neighbors(hood)))
            show(io, mime, neighbors(hood))
        end
    end
end

# Get a array of Bool for the offsets that are used by a Stencil
# This is used to build the ascii stencil shape
function _bool_array(hood::Stencil{R,1}) where {R}
    rs = _radii(hood)
    Bool[((i,) in offsets(hood)) for i in -rs[1][1]:rs[1][2]][:, :]
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

# Macro to define stencils
macro stencil(name, description)
    docstring = """

        $name <: Stencil

        $name(; radius=1, ndims=2)
        $name(radius, ndims)
        $name{R,N}()

    $description
    """

    name = esc(name)

    struct_expr = quote
        struct $name{R,N,L,T} <: Stencil{R,N,L,T}
            neighbors::SVector{L,T}
            center::T
            $name{R,N,L,T}(neighbors::StaticVector{L,T}, center::T) where {R,N,L,T} = new{R,N,L,T}(neighbors,center)
        end
    end
    func_exprs = quote
        
        # Filled stencils
        $name{R,N,L}(neighbors::StaticVector{L,T}, center::T) where {R,N,L,T} = $name{R,N,L,T}(neighbors,center)
        $name{R,N}(neighbors::StaticVector{L,T}, center::T) where {R,N,L,T} = $name{R,N,L,T}(neighbors,center)
        $name{R}(args::StaticVector, center) where R = $name{R,2}(args, center)
        $name(args::StaticVector, center; radius=1, ndims=2) = $name{radius,ndims}(args, center)

        # Empty stencils
        $name{R,N,L}() where {R,N,L} = $name{R,N,L}(SVector(ntuple(_ -> nothing, L)), nothing)
        function $name{R,N}() where {R,N}
            L = length(offsets($name{R,N}))
            $name{R,N,L}()
        end
        $name{R}() where R = $name{R,2}()
        $name(; radius=1, ndims=2) = $name{radius,ndims}()
        $name(radius::Int, ndims::Int=2) = $name{radius,ndims}()

        @inline Stencils.rebuild(n::$name{R,N,L}, neighbors, center) where {R,N,L} = $name{R,N,L}(neighbors, center)
    end
    return Expr(:block, :(Base.@doc $docstring $struct_expr), func_exprs)
end
