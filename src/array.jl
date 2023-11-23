
"""
    AbstractStencilArray <: StaticArray

Supertype for arrays with a [`Stencil`](@ref),
a [`BoundaryCondition`](@ref), and [`Padding`](@ref).
"""
abstract type AbstractStencilArray{S,R,T,N,A,H,BC,P} <: AbstractArray{T,N} end

"""
    stencil(A::AbstractStencilArray)
    stencil(A::AbstractStencilArray, I...)

Get a [`Stencil`](@ref) with neighbors updated for indices `I`.

`I` can be a `CartesianIndex`, a `Tuple`, or splatted arguments.

Without passing `I`, the stencil will not be updated, and will 
likely contain `nothing` values - but it may still be useful for its
other methods.
"""
Base.@propagate_inbounds stencil(A::AbstractStencilArray, I::Tuple) =
    stencil(A, CartesianIndex(I))
Base.@propagate_inbounds stencil(A::AbstractStencilArray, I::Union{CartesianIndex,Int}...) =
    stencil(A, CartesianIndex(to_indices(A, I)))
Base.@propagate_inbounds stencil(A::AbstractStencilArray, I::CartesianIndex) =
    stencil(stencil(A), A, I)
Base.@propagate_inbounds stencil(hood::StencilOrLayered, A::AbstractStencilArray, I::CartesianIndex) =
    rebuild(stencil(A), neighbors(A, I))
Base.@propagate_inbounds stencil(A::AbstractStencilArray) = A.stencil

"""
    unsafe_stencil(x, A::AbstractArray, I) => Stencil

Update the neighbors of a stencil to values from the array A around index `I`,
without checking bounds of `I`. Bounds checking of neighbors still occurs, but
with the assumption that `I` is inbounds.
"""
@inline unsafe_stencil(A::AbstractStencilArray, I::CartesianIndex) =
    unsafe_stencil(stencil(A), A, I)
@inline unsafe_stencil(hood::StencilOrLayered, A::AbstractStencilArray, I::CartesianIndex) =
    stencil(hood, unsafe_neighbors(hood, A, I))

"""
    boundary(A::AbstractStencilArray)

Get the [`BoundaryCondition`](@ref) object from an `AbstractStencilArray`.
"""
boundary(A::AbstractStencilArray) = A.boundary

"""
    padding(A::AbstractStencilArray)

Get the [`Padding`](@ref) object from an `AbstractStencilArray`.
"""
padding(A::AbstractStencilArray) = A.padding

"""

    neighbors([hood::Stencil,] A::AbstractStencilArray, I) => SArray

Get stencil neighbors from `A` around center `I` as an `SVector`.
"""
@inline neighbors(A::AbstractStencilArray, I::NTuple{<:Any,Int}) = neighbors(A, I...)
@inline neighbors(A::AbstractStencilArray, I::Int...) = neighbors(A, CartesianIndex(I))
@inline neighbors(A::AbstractStencilArray, I::CartesianIndex) = neighbors(stencil(A), A, I)
@inline function neighbors(stencil::StencilOrLayered, A::AbstractStencilArray{<:Any,R,<:Any,N}, I::CartesianIndex) where {R,N}
    if padding(A) isa Halo  # Conditional Remove has checks internally
        low = CartesianIndex(ntuple(_ -> -R, N))
        high = CartesianIndex(ntuple(_ -> R, N))
        checkbounds(parent(A), I + low)
        checkbounds(parent(A), I + high)
    end
    return unsafe_neighbors(stencil, A, I)
end

"""
    unsafe_neighbors([hood::Stencil,] A::AbstractStencilArray, I::CartesianIndex) => SArray

Get stencil neighbors from `A` around center `I` as an `SVector`, without checking bounds of `I`.
"""
@inline unsafe_neighbors(A::AbstractStencilArray, I::CartesianIndex) =
    unsafe_neighbors(stencil(A), A, I)
@inline function unsafe_neighbors(hood::Stencil, A::AbstractStencilArray, I::CartesianIndex)
    map(indices(hood, I)) do P
        @inbounds getneighbor(A, CartesianIndex(P))
    end
end
@inline function unsafe_neighbors(hood::Layered, A::AbstractStencilArray, I::CartesianIndex)
    map(l -> unsafe_neighbors(l, A, I), hood)
end

"""
    getneighbor(A::AbstractStencilArray, I::CartesianIndex)

Get an array value from a stencil neighborhood.

This method handles boundary conditions.
"""
Base.@propagate_inbounds function getneighbor(A::AbstractStencilArray, I::CartesianIndex)
    getneighbor(A, boundary(A), padding(A), I)
end
# If `Halo` padded we can just use regular `getindex`
# on the parent array, which is an `OffsetArray`
Base.@propagate_inbounds function getneighbor(A::AbstractStencilArray, ::BoundaryCondition, pad::Halo, I::CartesianIndex)
    @boundscheck checkbounds(parent(A), I)
    @inbounds parent(A)[I]
end
# `Conditional` needs handling for specific boundary conditions. 
# For Wrap we swap the side. 
Base.@propagate_inbounds function getneighbor(A::AbstractStencilArray{S}, ::Wrap, pad::Conditional, I::CartesianIndex) where S
    sz = tuple_contents(S)
    wrapped_inds = map(Tuple(I), sz) do i, s
        i < 1 ? i + s : (i > s ? i - s : i)
    end
    @boundscheck checkbounds(parent(A), wrapped_inds...)
    return @inbounds A[wrapped_inds...]
end
# For Remove we use padval if out of bounds
@inline function getneighbor(A::AbstractStencilArray, padding::Remove, pad::Conditional, I::CartesianIndex)
    checkbounds(Bool, A, I) ? (@inbounds A[I]) : padding.padval
end

# update_boundary!
# Reset or wrap boundary where required. This allows us to ignore
# bounds checks on stencils and still use a wraparound grid.
update_boundary!(As::Tuple) = map(update_boundary!, As)
update_boundary!(A::AbstractStencilArray) =
    update_boundary!(A, padding(A), boundary(A))
# Conditional sets boundary conditions on the fly
update_boundary!(A::AbstractStencilArray, ::Conditional, ::BoundaryCondition) = A
update_boundary!(A::AbstractStencilArray, ::Halo{:in}, ::Use) = A
# Halo needs updating
function update_boundary!(A::AbstractStencilArray{S,R}, ::Halo, bc::Remove) where {S<:Tuple{L},R} where {L}
    # Use the inner array so broadcasts over views works on GPU
    # they don't through the `OffsetArray` wrapper
    src = parent(parent(A))
    @inbounds src[vcat(1:R, L+R+1:L+2R)] .= Ref(padval(bc))
    return A
end
function update_boundary!(A::AbstractStencilArray{S,R}, ::Halo, bc::Remove) where {S<:Tuple{Y,X},R} where {Y,X}
    src = parent(parent(A))
    # Sides
    @inbounds src[1:Y+2R, vcat(1:R, X+R+1:X+2R)] .= Ref(padval(bc))
    @inbounds src[vcat(1:R, Y+R+1:Y+2R), 1:X+2R] .= Ref(padval(bc))
    return A
end
function update_boundary!(A::AbstractStencilArray{S,R}, ::Halo, bc::Remove) where {S<:Tuple{Z,Y,X},R} where {Z,Y,X}
    src = parent(parent(A))
    @inbounds src[axes(src, 1), axes(src, 2), vcat(1:R, X+R+1:X+2R)] .= Ref(padval(bc))
    @inbounds src[axes(src, 1), vcat(1:R, Y+R+1:Y+2R), axes(src, 3)] .= Ref(padval(bc))
    @inbounds src[vcat(1:R, Z+R+1:Z+2R), axes(src, 2), axes(src, 3)] .= Ref(padval(bc))
    return A
end
function update_boundary!(A::AbstractStencilArray{S,R}, ::Halo, ::Wrap) where {S<:Tuple{L},R} where {L}
    src = parent(A)
    startpad = 1:R
    endpad = L+R+1:L+2R
    startvals = R+1:2R
    endvals = L+1:L+R
    @assert length(startpad) == length(endvals) == R
    @assert length(endpad) == length(startvals) == R
    @inbounds src[startpad] .= src[endvals]
    @inbounds src[endpad] .= src[startvals]
    return A
end
function update_boundary!(A::AbstractStencilArray{S,R}, ::Halo, ::Wrap) where {S<:Tuple{Y,X},R} where {Y,X}
    src = parent(parent(A))
    n_xs, n_ys = X, Y
    startpad_x = startpad_y = 1:R
    endpad_x = n_xs+R+1:n_xs+2R
    endpad_y = n_ys+R+1:n_ys+2R
    start_x = start_y = R+1:2R
    end_x = n_xs+1:n_xs+R
    end_y = n_ys+1:n_ys+R
    xs = 1:n_xs+2R
    ys = 1:n_ys+2R

    @assert length(startpad_x) == length(start_x) == R
    @assert length(endpad_x) == length(end_x) == R
    @assert length(startpad_y) == length(start_y) == R
    @assert length(endpad_y) == length(end_y) == R
    @assert map(length, (ys, xs)) === size(src)

    CI = CartesianIndices
    # Sides ---
    @inbounds src[CI((ys, startpad_x))] .= src[CI((ys, end_x))]
    @inbounds src[CI((ys, endpad_x))]   .= src[CI((ys, start_x))]
    @inbounds src[CI((startpad_y, xs))] .= src[CI((end_y, xs))]
    @inbounds src[CI((endpad_y, xs))]   .= src[CI((start_y, xs))]

    # Corners ---
    @inbounds src[CI((startpad_y, startpad_x))] .= src[CI((end_y, end_x))]
    @inbounds src[CI((startpad_y, endpad_x))]   .= src[CI((end_y, start_x))]
    @inbounds src[CI((endpad_y, startpad_x))]   .= src[CI((start_y, end_x))]
    @inbounds src[CI((endpad_y, endpad_x))]     .= src[CI((start_y, start_x))]

    return after_update_boundary!(A)
end
function update_boundary!(A::AbstractStencilArray{S,R}, ::Halo, ::Wrap) where {S<:Tuple{Z,Y,X},R} where {Z,Y,X}
    src = parent(parent(A))
    n_xs, n_ys, n_zs = X, Y, Z
    startpad_x = startpad_y = startpad_z = 1:R
    endpad_x = n_xs+R+1:n_xs+2R
    endpad_y = n_ys+R+1:n_ys+2R
    endpad_z = n_ys+R+1:n_zs+2R
    start_x = start_y = start_z = R+1:2R
    end_x = n_xs+1:n_xs+R
    end_y = n_ys+1:n_ys+R
    end_z = n_zs+1:n_zs+R
    xs = 1:n_xs+2R
    ys = 1:n_ys+2R
    zs = 1:n_zs+2R

    @assert length(startpad_x) == length(start_x) == R
    @assert length(endpad_x) == length(end_x) == R
    @assert length(startpad_y) == length(start_y) == R
    @assert length(endpad_y) == length(end_y) == R
    @assert map(length, (zs , ys, xs)) === size(src)

    CI = CartesianIndices
    # Sides ---
    # X
    @inbounds copxto!(src, CI((startpad_y, xs, zs)), src, CI((end_y, xs, zs)))
    @inbounds copxto!(src, CI((endpad_y, xs, zs)), src, CI((start_y, xs, zs)))
    # Y
    @inbounds copxto!(src, CI((ys, startpad_x, zs)), src, CI((ys, end_x, zs)))
    @inbounds copxto!(src, CI((ys, endpad_x, zs)), src, CI((ys, start_x, zs)))
    # Z
    @inbounds copxto!(src, CI((ys, xs, startpad_z)), src, CI((ys, xs, end_z)))
    @inbounds copxto!(src, CI((ys, xs, endpad_z)), src, CI((ys, xs, start_z)))

    # Corners ---
    @inbounds src[CI((startpad_y, startpad_x, startpad_z))] .= src[CI((end_y, end_x, end_z))]
    @inbounds src[CI((startpad_y, startpad_x, endpad_z))] .= src[CI((end_y, end_x, start_z))]
    @inbounds src[CI((startpad_y, endpad_x, startpad_z))] .= src[CI((end_y, start_x, end_z))]
    @inbounds src[CI((startpad_y, endpad_x, endpad_x))] .= src[CI((end_y, start_x, start_z))]
    @inbounds src[CI((endpad_y, endpad_x, endpad_z))] .= src[CI((start_y, start_x, start_z))]
    @inbounds src[CI((endpad_y, startpad_x, endpad_z))] .= src[CI((end_y, start_x, start_z))]
    @inbounds src[CI((endpad_y, endpad_x, startpad_z))] .= src[CI((start_y, start_x, end_z))]
    @inbounds src[CI((endpad_y, startpad_x, startpad_z))] .= src[CI((start_y, end_x, end_z))]
    return after_update_boundary!(A)
end

# Allow additional boundary updating behaviours
after_update_boundary!(A) = A

radii(x::Int, s::NTuple{N}) where N = ntuple(_ -> ntuple(_ -> x, Val{N}()), Val{N}())
radii(x::Tuple, s::Tuple) = x


# Base methods
function Base.copy!(S::AbstractStencilArray{<:Any,R}, A::AbstractArray) where R
    pad_axes = map(ax -> ax .+ R, axes(A))
    copyto!(parent(source(S)), CartesianIndices(pad_axes), A, CartesianIndices(A))
    return 
end
function Base.copy!(A::AbstractArray, S::AbstractStencilArray{<:Any,R}) where R
    pad_axes = map(ax -> ax .+ R, axes(A))
    copyto!(A, CartesianIndices(A), parent(source(S)), CartesianIndices(pad_axes))
    return A
end
function Base.copy!(dst::AbstractStencilArray{<:Any,RD}, src::AbstractStencilArray{<:Any,RS}) where {RD,RS}
    dst_axes = map(s -> RD:s + RD, size(dst))
    src_axes = map(s -> RS:s + RS, size(src))
    copyto!(parent(source(dst)), CartesianIndices(dst_axes),
            parent(source(src)), CartesianIndices(src_axes)
    )
    return dst
end

function Base.show(io::IO, mime::MIME"text/plain", A::AbstractStencilArray)
    show(io, mime, Array(A))
    println()
    println()
    show(io, mime, stencil(A))
    println()
    show(io, mime, boundary(A))
    println()
    show(io, mime, padding(A))
end

# Iterate over the parent for `Conditional` padding, 2x faster.
Base.iterate(A::AbstractStencilArray{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Conditional}, args...) =
    iterate(parent(A), args...)
Base.parent(A::AbstractStencilArray) = A.parent
for f in (:getindex, :view, :dotview)
    @eval begin
        Base.@propagate_inbounds function Base.$f(A::AbstractStencilArray, I::Union{Colon,Int64,AbstractArray}...)
            @boundscheck checkbounds(A, I...)
            @inbounds Base.$f(parent(A), I...)
        end
        Base.@propagate_inbounds function Base.$f(A::AbstractStencilArray, i1::Int, I::Int...)
            @boundscheck checkbounds(A, i1, I...)
            @inbounds Base.$f(parent(A), i1, I...)
        end
    end
end
Base.@propagate_inbounds Base.setindex!(d::AbstractStencilArray, x, I::Int...) =
    setindex!(parent(d), x, I...)
Base.@propagate_inbounds Base.setindex!(d::AbstractStencilArray, x, I...) =
    setindex!(parent(d), x, I...)
Base.size(::AbstractStencilArray{S}) where S = tuple_contents(S)

Base.similar(A::AbstractStencilArray) = similar(parent(parent(A)), size(A))
Base.similar(A::AbstractStencilArray, ::Type{T}) where T = similar(parent(parent(A)), T, size(A))
Base.similar(A::AbstractStencilArray, I::Tuple{Int,Vararg{Int}}) = similar(parent(parent(A)), I)
Base.similar(A::AbstractStencilArray, ::Type{T}, I::Tuple{Int,Vararg{Int}}) where T =
    similar(parent(parent(A)), T, I)


const STENCILARRAY_KEYWORDS = """
- `boundary`: a [`BoundaryCondition`](@ref) like [`Wrap`](@ref).
- `padding`: [`Padding`](@ref) like [`Conditional`](@ref) or [`Halo{:in}`](@ref).
"""

"""
    StencilArray <: AbstractStencilArray

    StencilArray(A::AbstractArray, stencil::Stencil; kw...)

An array with a [`Stencil`](@ref) and a [`BoundaryCondition`](@ref), and [`Padding`](@ref).

For most uses a `StencilArray` works exactly the same as a regular array.

Except it can be indexed at any point with `stencil` to return a filled
`Stencil` object, or `neighbors` to return an `SVector` of neighbors.

## Arguments

- `A`: an AbstractArray
- `stencil`: a [`Stencil`](@ref).

## Keywords

$STENCILARRAY_KEYWORDS

## Example

```jldoctest
using Stencils, Statistics
sa = StencilArray((1:10) * (10:20)', Moore(1); boundary=Wrap())
sa .*= 2 # Broadcast works as usual
means = mapstencil(mean, sa) # mapstencil works
stencil(sa, 5, 6) # manually reading a stencil works too

# output

Moore{1, 2, 8, Int64}
█▀█
▀▀▀

with neighbors:
8-element StaticArraysCore.SVector{8, Int64} with indices SOneTo(8):
 112
 140
 168
 120
 180
 128
 160
 192
```
"""
struct StencilArray{S,R,T,N,A<:AbstractArray{T,N},H<:Union{Stencil{R,N},Layered{R,N}},BC,P} <: AbstractStencilArray{S,R,T,N,A,H,BC,P}
    parent::A
    stencil::H
    boundary::BC
    padding::P
    function StencilArray{S,R,T,N,A,H,BC,P}(parent::A, h::H, bc::BC, padding::P) where {S,R,T,N,A,H,BC,P}
        map(tuple_contents(S), _radii(Val{N}(), R)) do s, rs
            max(map(abs, rs)...) < s || throw(ArgumentError("stencil radius is larger than array axis $s"))
        end
        return new{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
    end
end
function StencilArray(parent::AbstractArray, hood::Union{Stencil{R},Layered{R}}, bc, padding) where R
    padded_parent = pad_array(padding, bc, hood, parent)
    S = Tuple{_size(padding, hood, padded_parent)...}
    StencilArray{S,R}(padded_parent, hood, bc, padding)
end
StencilArray{S}(parent::AbstractArray, hood::Union{Stencil{R},Layered{R}}, bc, padding) where {S,R} =
    StencilArray{S,R}(parent, hood, bc, padding)
StencilArray{S,R}(parent::A, h::H, bc::BC, padding::P) where {S,A<:AbstractArray{T,N},H<:Union{Stencil{R},Layered{R}},BC,P} where {R,T,N} =
    StencilArray{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
function StencilArray(parent::AbstractArray{<:Any,N}, stencil=Window{1,N}();
    boundary=Remove(zero(eltype(parent))),
    padding=Conditional(),
) where N
    StencilArray(parent, stencil, boundary, padding)
end

_size(::Conditional, stencil, parent) = size(parent)
_size(::Halo, ::Union{Stencil{R},Layered{R}}, parent) where R = size(parent) .- 2R

function Adapt.adapt_structure(to, A::StencilArray{S}) where S
    newparent = Adapt.adapt(to, parent(A))
    StencilArray{S}(newparent, stencil(A), boundary(A), padding(A))
end

ConstructionBase.constructorof(::Type{<:StencilArray{S}}) where S = StencilArray{S}

# Internals
"""
    AbstractSwitchingStencilArray

Abstract supertype for `AbstractStencilArray` that wrap two arrays that 
switch places with each broadcast.
"""
abstract type AbstractSwitchingStencilArray{S,R,T,N,A,H,BC,P} <: AbstractStencilArray{S,R,T,N,A,H,BC,P} end

Base.parent(d::AbstractSwitchingStencilArray) = source(d)

source(A::AbstractSwitchingStencilArray) = A.source
dest(A::AbstractSwitchingStencilArray) = A.dest
radius(d::AbstractStencilArray{<:Any,R}) where R = R
padval(d::AbstractStencilArray) = padval(boundary(d))


"""
    SwitchingStencilArray <: AbstractSwitchingStencilArray

An `AbstractArray` with a [`Stencil`](@ref), a [`BoundaryCondition`](@ref), [`Padding`](@ref),
and two array layers that are switched with each `broadcast_stencil` operation.

The use case for this operation is in simulations where stencil operations
are repeatedly run over the same data, or where a filter (such as a blur) needs
to be applied many times.

For most uses a `SwitchingStencilArray` works exactly the same as a
regular array - the `dest` array can be safely ignored.

However, when using `mapstencil!` you need to use the output, not the original
array. Switching does not happen in-place, but as a new returned array.

## Example

```jldoctest
using Stencils, Statistics

sa = SwitchingStencilArray(rand(10, 10), Moore(1); boundary=Wrap())
sa .*= 2 # Broadcast works as usual
mapstencil(mean, sa) # As does runing `mapstencils
hood = stencil(sa, 5, 10) # And retreiving a stencil
# But we can also run it in-place, here doing 10 iterations of mean blur:
# Note: if you dont assign new variable with `A =`, the array will
# not switch and will not be blurred.
let sa = sa
    for i in 1:10
        sa = mapstencil!(mean, sa)
    end
end
# output

```
"""
struct SwitchingStencilArray{S,R,T,N,A<:AbstractArray{T,N},H<:Union{Stencil{R,N},Layered{R,N}},BC,P} <: AbstractSwitchingStencilArray{S,R,T,N,A,H,BC,P}
    source::A
    dest::A
    stencil::H
    boundary::BC
    padding::P
    function SwitchingStencilArray{S,R,T,N,A,H,BC,P}(source::A, dest::A, h::H, bc::BC, padding::P) where {S,R,T,N,A,H,BC,P}
        map(tuple_contents(S), _radii(Val{N}(), R)) do s, rs
            max(map(abs, rs)...) < s || throw(ArgumentError("stencil radius is larger than array axis $s"))
        end
        return new{S,R,T,N,A,H,BC,P}(source, dest, h, bc, padding)
    end
end

# Construct from a single parent array
function SwitchingStencilArray(parent::AbstractArray{<:Any,N}, stencil=Window{1,N}();
    boundary=Remove(zero(eltype(parent))),
    padding=Conditional(),
) where N
    SwitchingStencilArray(parent, stencil, boundary, padding)
end
# Get S from the parent size
# Build the source and dest padded arrays
function SwitchingStencilArray(parent::AbstractArray, hood::StencilOrLayered, bc, padding)
    padded_source = pad_array(padding, bc, hood, parent)
    padded_dest = pad_array(padding, bc, hood, parent)
    S = Tuple{_size(padding, hood, padded_source)...}
    return SwitchingStencilArray{S}(padded_source, padded_dest, hood, bc, padding)
end
function SwitchingStencilArray{S}(
    source::A, dest::A, h::H, bc::BC, padding::P
) where {S,A<:AbstractArray{T,N},H<:Union{Stencil{R},Layered{R}},BC,P} where {R,T,N}
    SwitchingStencilArray{S,R,T,N,A,H,BC,P}(source, dest, h, bc, padding)
end

"""
    switch(A::SwitchingStencilArray)

Swap the source and dest of a `SwitchingStencilArray`.
"""
switch(A::SwitchingStencilArray{S}) where S =
    SwitchingStencilArray{S}(dest(A), source(A), stencil(A), boundary(A), padding(A))

Base.parent(A::SwitchingStencilArray) = A.source

function Adapt.adapt_structure(to, A::SwitchingStencilArray{S}) where S
    newsource = Adapt.adapt(to, A.source)
    newdest = Adapt.adapt(to, A.dest)
    SwitchingStencilArray{S}(newsource, newdest, stencil(A), boundary(A), padding(A))
end

ConstructionBase.constructorof(::Type{<:SwitchingStencilArray{S}}) where S = SwitchingStencilArray{S}
