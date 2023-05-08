
"""
    AbstractStencilArray <: StaticArray

Supertype for arrays with a [`Stencil`](@ref),
a [BoundaryCondition](@ref), and [`Padding`](@ref).
"""
abstract type AbstractStencilArray{S,R,T,N,A,H,BC,P} <: AbstractArray{T,N} end

boundary(A::AbstractStencilArray) = A.boundary
padding(A::AbstractStencilArray) = A.padding
Base.@propagate_inbounds stencil(A::AbstractStencilArray, I::Tuple) =
    update_stencil(A, CartesianIndex(I))
Base.@propagate_inbounds stencil(A::AbstractStencilArray, I::Union{CartesianIndex,Int}...) =
    stencil(A, CartesianIndex(to_indices(A, I)))
Base.@propagate_inbounds stencil(A::AbstractStencilArray, I::CartesianIndex) =
    update_stencil(A, I)
Base.@propagate_inbounds stencil(A::AbstractStencilArray) =
    A.stencil

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
function copy!(S::AbstractStencilArray{<:Any,R,T,1} where T, A::AbstractVector) where R
    pad_axes = map(ax -> ax .+ R, axes(A))
    copyto!(parent(source(S)), CartesianIndices(pad_axes), A, CartesianIndices(A))
    return A
end
function copy!(A::AbstractVector, S::AbstractStencilArray{<:Any,R,T,1} where T) where R
    pad_axes = map(ax -> ax .+ R, axes(A))
    copyto!(A, CartesianIndices(A), parent(source(S)), CartesianIndices(pad_axes))
    return A
end
function copy!(A::SparseArrays.AbstractCompressedVector, S::AbstractStencilArray{<:Any,R,T,1} where T) where R
    pad_axes = map(ax -> ax .+ R, axes(A))
    copyto!(A, CartesianIndices(A), parent(source(S)), CartesianIndices(pad_axes))
    return A
end

"""

    neighbors(hood::Stencil, A::AbstractArray, I) => SArrayt

Get a single stencil from an array, as a `Tuple`, checking bounds.
"""
@inline neighbors(A::AbstractStencilArray, I::NTuple{<:Any,Int}) = neighbors(A, I...)
@inline neighbors(A::AbstractStencilArray, I::Int...) = neighbors(A, CartesianIndex(I))
@inline function neighbors(A::AbstractStencilArray{<:Any,R,<:Any,N}, I::CartesianIndex) where {R,N}
    if A.padding isa Halo # Conditional has checks internally
        low = CartesianIndex(ntuple(_ -> -R, N))
        high = CartesianIndex(ntuple(_ -> R, N))
        checkbounds(parent(A), I + low)
        checkbounds(parent(A), I + high)
    end
    return unsafe_neighbors(A, I)
end

# function Base.show(io, mime::MIME"text/plain", A::AbstractStencilArray)
#     invoke(show, (AbstractArray,), A)
#     println()
#     show(io, mime, stencil(A))
#     println()
#     show(io, mime, boundary(A))
#     println()
#     show(io, mime, padding(A))
# end

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

"""
    StencilArray <: AbstractStencilArray

An array with a [`Stencil`](@ref) and a [BoundaryCondition](@ref), and [`Padding`](@ref).

For most uses a `StencilArray` works exactly the same as a regular array.

Except it can be indexed at any point with `stencil` to return a filled
`Stencil` object, or `neighbors` to return an `SVector` of neighbors.

## Example

```
using Stencils
A = StencilArray((1:10) * (10:20)'; stencil=Moore(2), boundary=Wrap())
A .*= 2 # Broadcast works as usual
hood = stencil(A, 5, 10)

# ouput
Moore{1, 2, 8, StaticArraysCore.SVector{8, Int64}}
█▀█
▀▀▀
with neighbors:
8-element StaticArraysCore.SVector{8, Int64} with indices SOneTo(8):
 144
 180
 216
 152
 228
 160
 200
 240
"""
struct StencilArray{S,R,T,N,A<:AbstractArray{T,N},H<:Stencil{R,N},BC,P} <: AbstractStencilArray{S,R,T,N,A,H,BC,P}
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
function StencilArray(parent::AbstractArray, hood::Stencil{R}, bc, padding) where R
    padded_parent = pad_array(padding, bc, hood, parent)
    S = Tuple{_size(padding, hood, padded_parent)...}
    StencilArray{S,R}(padded_parent, hood, bc, padding)
end
StencilArray{S}(parent::AbstractArray, hood::Stencil{R}, bc, padding) where {S,R} =
    StencilArray{S,R}(parent, hood, bc, padding)
StencilArray{S,R}(parent::A, h::H, bc::BC, padding::P) where {S,A<:AbstractArray{T,N},H<:Stencil{R},BC,P} where {R,T,N} =
    StencilArray{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
function StencilArray(parent::AbstractArray{<:Any,N}, stencil=Window{1,N}();
    boundary=Remove(zero(eltype(parent))),
    padding=Conditional(),
) where N
    StencilArray(parent, stencil, boundary, padding)
end

_size(::Conditional, ::Stencil, parent) = size(parent)
_size(::Halo, ::Stencil{R}, parent) where R = size(parent) .- 2R

function Adapt.adapt_structure(to, A::StencilArray{S}) where S
    newparent = Adapt.adapt(to, parent(A))
    StencilArray{S}(newparent, stencil(A), boundary(A), padding(A))
end

ConstructionBase.constructorof(::Type{<:StencilArray{S}}) where S = StencilArray{S}

# Stencil vector

struct LazyStencilVector{L,T,R,N,H,A<:AbstractStencilArray{<:Any,<:Any,T,N,<:Any,H}} <: StaticVector{L,T}
    parent::A
    center::CartesianIndex{N}
    LazyStencilVector(a::A, I::CartesianIndex) where {A<:AbstractStencilArray{S,R,T,N,<:Any,H}} where {S,R,T,N,H<:Stencil{R,N,L}} where {L} =
        new{L,T,R,N,H,A}(a, I)
end
LazyStencilVector(A, I::Tuple) = LazyStencilVector(A, CartesianIndex(I))
# S,R,T,N,A,H,BC,P

Base.parent(v::LazyStencilVector) = v.parent
stencil(v::LazyStencilVector) = stencil(parent(v))
center(v::LazyStencilVector) = v.center

Base.@propagate_inbounds function Base.getindex(v::LazyStencilVector, i::Int)
    neighbor_getindex(parent(v), indexat(stencil(v), center(v), i))
end

Base.size(v::LazyStencilVector) = (length(v),)
Base.length(::LazyStencilVector{Tuple{L}}) where L = L


# Internals

"""
    unsafe_readneighbors(hood::Stencil, A::AbstractArray, I) => SArray

Get a single stencil from an array, as a `Tuple`, without checking bounds.
"""
@inline unsafe_neighbors(A::AbstractStencilArray, I::CartesianIndex) =
    unsafe_neighbors(A, stencil(A), I)
@inline function unsafe_neighbors(A::AbstractStencilArray, hood::Stencil, I::CartesianIndex)
    map(indices(hood, I)) do P
        @inbounds neighbor_getindex(A, CartesianIndex(P))
    end
end

"""
    update_stencil(x, A::AbstractArray, I) => Stencil

Set the neighbors of a stencil to values from the array A around index `I`.
Bounds checks will reduce performance, aim to use `unsafe_setneighbors` directly.
"""
Base.@propagate_inbounds update_stencil(A::AbstractStencilArray, I::CartesianIndex) =
    update_stencil(A, stencil(A), I)
Base.@propagate_inbounds update_stencil(A::AbstractStencilArray, hood::Stencil, I::CartesianIndex) =
    setneighbors(stencil(A), neighbors(A, I))

unsafe_update_stencil(A::AbstractStencilArray, I::CartesianIndex) =
    unsafe_update_stencil(A, stencil(A), I)
unsafe_update_stencil(A::AbstractStencilArray, hood::Stencil, I::CartesianIndex) =
    setneighbors(hood, unsafe_neighbors(A, hood, I))

Base.@propagate_inbounds function neighbor_getindex(A::AbstractStencilArray, I::CartesianIndex)
    neighbor_getindex(A, boundary(A), padding(A), I)
end
# If `Halo` padded we can just use regular `getindex`
# on the parent array, which is an `OffsetArray`
Base.@propagate_inbounds function neighbor_getindex(A::AbstractStencilArray, ::BoundaryCondition, pad::Halo, I::CartesianIndex)
    @boundscheck checkbounds(parent(A), I)
    @inbounds parent(A)[I]
end
# `Conditional` needs handling. For Wrap we swap the side.
# This also means we don't need bounds checking as the
# stencil can't be larger than the array itself.
function nighbor_getindex(A::AbstractStencilArray{S}, ::Wrap, pad::Conditional, I::CartesianIndex) where S
    sz = tuple_contents(S)
    wrapped_inds = map(Tuple(I), sz) do i, s
        i < 1 ? i + s : (i > s ? i - s : i)
    end
    return @inbounds A[wrapped_inds...]
end
# For Remove we use padval if out of bounds
function neighbor_getindex(A::AbstractStencilArray, padding::Remove, pad::Conditional, I::CartesianIndex)
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


abstract type AbstractSwitchingStencilArray{S,R,T,N,A,H,BC,P} <: AbstractStencilArray{S,R,T,N,A,H,BC,P} end

Base.parent(d::AbstractSwitchingStencilArray) = source(d)

source(A::AbstractSwitchingStencilArray) = A.source
dest(A::AbstractSwitchingStencilArray) = A.dest
radius(d::AbstractStencilArray{<:Any,R}) where R = R
padval(d::AbstractStencilArray) = padval(boundary(d))


"""
    SwitchingStencilArray <: AbstractSwitchingStencilArray

An `AbstractArray` with a [`Stencil`](@ref), a [BoundaryCondition](@ref), [`Padding`](@ref),
and two array layers that are switched with each `broadcast_stencil` operation.

The use case for this operation is in simulations where stencil operations
are repeatedly run over the same data, or where a filter (such as a blur) needs
to be applied many times.

For most uses a `SwitchingStencilArray` works exactly the same as a
regular array - the `dest` array can be safely ignored.

## Example

```
using Stencils
A = SwitchingStencilArray((1:10) * (10:20)'; stencil=Moore(2), boundary=Wrap())
A .*= 2 # Broadcast works as usual
hood = stencil(A, 5, 10)

# ouput
Moore{1, 2, 8, StaticArraysCore.SVector{8, Int64}}
█▀█
▀▀▀
with neighbors:
8-element StaticArraysCore.SVector{8, Int64} with indices SOneTo(8):
 144
 180
 216
 152
 228
 160
 200
 240
"""
struct SwitchingStencilArray{S,R,T,N,A<:AbstractArray{T,N},H<:Stencil{R,N},BC,P} <: AbstractStencilArray{S,R,T,N,A,H,BC,P}
    source::A
    dest::A
    stencil::H
    boundary::BC
    padding::P
    function SwitchingStencilArray{S,R,T,N,A,H,BC,P}(source::A, dest::A, h::H, bc::BC, padding::P) where {S,R,T,N,A,H,BC,P}
        map(tuple_contents(S), _radii(Val{N}(), R)) do s, rs
            max(map(abs, rs)...) < s || throw(ArgumentError("stencil radius is larger than array axis $s"))
        end
        return new{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
    end
end
function SwitchingStencilArray(parent::AbstractArray, hood::Stencil{R}, bc, padding) where R
    padded_source = pad_array(padding, bc, hood, parent)
    padded_dest = pad_array(padding, bc, hood, parent)
    S = Tuple{_size(padding, hood, padded_source)...}
    SwitchingStencilArray{S,R}(padded_source, padded_dest, hood, bc, padding)
end
SwitchingStencilArray{S}(parent::AbstractArray, hood::Stencil{R}, bc, padding) where {S,R} =
    SwitchingStencilArray{S,R}(parent, hood, bc, padding)
SwitchingStencilArray{S,R}(parent::A, h::H, bc::BC, padding::P) where {S,A<:AbstractArray{T,N},H<:Stencil{R},BC,P} where {R,T,N} =
    SwitchingStencilArray{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
function SwitchingStencilArray(parent::AbstractArray{<:Any,N}, stencil=Window{1,N}();
    boundary=Remove(zero(eltype(parent))),
    padding=Conditional(),
) where N
    SwitchingStencilArray(parent, stencil, boundary, padding)
end

switch(A::SwitchingStencilArray) =
    SwitchingStencilArray(dest(A), source(A), stencil(A), boundary(A), padding(A))
