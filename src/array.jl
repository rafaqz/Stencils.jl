
"""
    AbstractStencilArray <: StaticArray

Supertype for arrays with a [`Stencil`](@ref),
a [`BoundaryCondition`](@ref), and [`Padding`](@ref).
"""
abstract type AbstractStencilArray{R,T,N,A,S,BC,P} <: AbstractArray{T,N} end

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
Base.@propagate_inbounds stencil(stencil::StencilOrLayered, A::AbstractStencilArray, I::CartesianIndex) =
    rebuild(stencil, neighbors(stencil, A, I), _center(stencil, A, I))
Base.@propagate_inbounds stencil(A::AbstractStencilArray) = A.stencil

@inline _center(::Stencil, A::AbstractStencilArray, I::CartesianIndex) = A[I]
@inline _center(stencil::Layered, A::AbstractStencilArray, I::CartesianIndex) = 
    map(l -> _center(l, A, I), stencil)

"""
    unsafe_stencil(x, A::AbstractArray, I) => Stencil

Update the neighbors of a stencil to values from the array A around index `I`,
without checking bounds of `I`. Bounds checking of neighbors still occurs, but
with the assumption that `I` is inbounds.
"""
@inline unsafe_stencil(A::AbstractStencilArray, I::CartesianIndex) =
    unsafe_stencil(stencil(A), A, I)
@inline unsafe_stencil(stencil::StencilOrLayered, A::AbstractStencilArray, I::CartesianIndex) =
    rebuild(stencil, unsafe_neighbors(stencil, A, I), _center(stencil, A, I))

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

    neighbors([stencil::Stencil,] A::AbstractStencilArray, I) => SArray

Get stencil neighbors from `A` around center `I` as an `SVector`.
"""
@inline neighbors(A::AbstractStencilArray, I::NTuple{<:Any,Int}) = neighbors(A, I...)
@inline neighbors(A::AbstractStencilArray, I::Int...) = neighbors(A, CartesianIndex(I))
@inline neighbors(A::AbstractStencilArray, I::CartesianIndex) = neighbors(stencil(A), A, I)
@inline function neighbors(stencil::StencilOrLayered, A::AbstractStencilArray, I::CartesianIndex)
    _checkbounds(padding(A), A, I)
    return @inbounds unsafe_neighbors(stencil, A, I)
end

function _checkbounds(::Halo, A::AbstractStencilArray{R,<:Any,N}, I) where {R,N}
    # Check the corners of the array
    checkbounds(parent(A), I)
    checkbounds(parent(A), I + CartesianIndex(ntuple(_ -> 2R, N)))
end
function _checkbounds(::Conditional, A, I)
    # Check the center, we handle the padding later
    checkbounds(parent(A), I)
end

"""
    unsafe_neighbors([stencil::Stencil,] A::AbstractStencilArray, I::CartesianIndex) => SArray

Get stencil neighbors from `A` around center `I` as an `SVector`, without checking bounds of `I`.
"""
@inline unsafe_neighbors(A::AbstractStencilArray, I::CartesianIndex) = unsafe_neighbors(stencil(A), A, I)
@inline unsafe_neighbors(stencil::StencilOrLayered, A::AbstractStencilArray, I::CartesianIndex) =
    unsafe_neighbors(stencil, padding(A), A, I)
@inline function unsafe_neighbors(
    stencil::Stencil, ::Padding, A::AbstractStencilArray{R}, I::CartesianIndex
) where R
    map(indices(stencil, I)) do N
        unsafe_getneighbor(A, N)
    end
end
@inline function unsafe_neighbors(
    stencil::Stencil, ::Conditional, A::AbstractStencilArray{R,<:Any,N}, I::CartesianIndex
) where {R,N}
    # If the stencil corners are in-bounds we don't need bounds checks later
    radii = CartesianIndex(ntuple(_ -> -R, Val{N}()))
    if checkbounds(Bool, A, I + radii) && checkbounds(Bool, A, I - radii)
        map(indices(stencil, I)) do N
            unsafe_getneighbor(A, N)
        end
    else
        map(indices(stencil, I)) do N
            getneighbor(A, N)
        end
    end
end
@inline function unsafe_neighbors(
    stencil::Layered, p::Padding, A::AbstractStencilArray, I::CartesianIndex
)
    map(l -> unsafe_neighbors(l, p, A, I), stencil)
end

"""
    getneighbor(A::AbstractStencilArray, I::CartesianIndex)

Get an array value from a stencil neighborhood.

This method handles boundary conditions.
"""
@inline function getneighbor(A::AbstractStencilArray, I::Tuple)
    getneighbor(A, boundary(A), padding(A), I)
end
# For Remove we use padval if out of bounds
@inline function getneighbor(A::AbstractStencilArray, boundary::Remove, ::Conditional, I::Tuple)
    checkbounds(Bool, A, I...) ? (@inbounds A[I...]) : boundary.padval
end
# Wrap and Reflect are always inbounds
@inline getneighbor(A::AbstractStencilArray, bounds::Union{Wrap,Reflect}, pad::Conditional, I::Tuple) =
    unsafe_getindex(A, pad, bounded_index(A, bounds, pad, I)...)

@inline function indices(A::AbstractStencilArray, I::Tuple)
     map(indices(stencil(A), I)) do J
         bounded_index(A, J)
     end
end

@inline bounded_index(A::AbstractStencilArray, I::Int...) = bounded_index(A, I)
@inline bounded_index(A::AbstractStencilArray, I::CartesianIndex) = bounded_index(A, Tuple(I))
@inline bounded_index(A::AbstractStencilArray, I::Tuple) =
    bounded_index(A, boundary(A), padding(A), I)
# Halo doesn't change the bounded_index
@inline bounded_index(A::AbstractStencilArray, boundary, padding::Halo, I::Tuple) = I
# We cant do much here - the caller needs a bounds check
@inline bounded_index(A::AbstractStencilArray, boundary::Remove, padding::Conditional, I::Tuple) = I
@inline function bounded_index(
    A::AbstractStencilArray, boundary::Reflect, padding::Conditional, I::Tuple
)
    map(I, size(A)) do i, s
        if i < 1
            2 - i
        elseif i > s
            2 * s - i
        else
            i
        end
    end
end
@inline function bounded_index(
    A::AbstractStencilArray, boundary::Wrap, padding::Conditional, I::Tuple
)
    map(I, size(A)) do i, s
        if i < 1
            i + s
        elseif i > s
            i - s
        else
            i
        end
    end
end

@inline function unsafe_getneighbor(A::AbstractStencilArray, I::Tuple)
    unsafe_getneighbor(A, boundary(A), padding(A), I)
end
@inline function unsafe_getneighbor(
    A::AbstractStencilArray, ::BoundaryCondition, pad::Padding, I::Tuple
)
    unsafe_getindex(A, pad, I...)
end

_sub_radius(I, R) = map(i -> i .- R, I)

# update_boundary!
# Reset or wrap boundary where required. This allows us to ignore
# bounds checks on stencils and still use a wraparound grid.
update_boundary!(As::Tuple) = map(update_boundary!, As)
update_boundary!(A::AbstractStencilArray) =
    update_boundary!(A, padding(A), boundary(A))
# Conditional sets boundary conditions on the fly
update_boundary!(A::AbstractStencilArray, ::Conditional, ::BoundaryCondition) = A
update_boundary!(A::AbstractStencilArray{R}, ::Halo, ::Use) where {R} = A
# Halo needs updating
@generated function update_boundary!(
    A::AbstractStencilArray{R,T,N}, ::Halo, bc::BoundaryCondition
) where {R,T,N}
    expr = Expr(:block)
    i = 1
    for _ in 1:N
        inds_expr1 = Expr(:tuple)
        inds_expr2 = Expr(:tuple)
        for n in 1:N
            if i == n
                push!(inds_expr1.args, :(Base.OneTo(1:R)))
                push!(inds_expr2.args, :(s[$n]+R+1:s[$n]+2R))
            else
                push!(inds_expr1.args, :(Base.OneTo(s[$n]+2R)))
                push!(inds_expr2.args, :(Base.OneTo(s[$n]+2R)))
            end
        end
        push!(expr.args, :(Isrc1 = $inds_expr1))
        push!(expr.args, :(Isrc2 = $inds_expr2))
        push!(expr.args, :(I1 = _sub_radius(Isrc1, R)))
        push!(expr.args, :(I2 = _sub_radius(Isrc2, R)))
        push!(expr.args, :(src[Isrc1...] .= halo_val.((A,), (bc,), CartesianIndices(I1))))
        push!(expr.args, :(src[Isrc2...] .= halo_val.((A,), (bc,), CartesianIndices(I2))))
        i += 1
    end
    return quote
        s = size(A)
        src = parent(A)
        $expr
        return after_update_boundary!(A)
    end
end

@inline halo_val(A, boundary::Remove, I::CartesianIndex) = padval(boundary)
@inline halo_val(A, boundary::Union{Wrap,Reflect}, I::CartesianIndex) = begin
    bI = bounded_index(A, boundary, Conditional(), Tuple(I))
    return A[bI...]
end

# Allow additional boundary updating behaviours
after_update_boundary!(A) = A

radii(x::Int, s::NTuple{N}) where N = ntuple(_ -> ntuple(_ -> x, Val{N}()), Val{N}())
radii(x::Tuple, s::Tuple) = x


# Base methods
function Base.copy!(S::AbstractStencilArray{R}, A::AbstractArray) where R
    pad_axes = add_halo(S, axes(S))
    copyto!(parent(S), CartesianIndices(pad_axes), A, CartesianIndices(A))
    return
end
function Base.copy!(A::AbstractArray, S::AbstractStencilArray{R}) where R
    pad_axes = add_halo(S, axes(S))
    copyto!(A, CartesianIndices(A), parent(S), CartesianIndices(pad_axes))
    return A
end
function Base.copy!(dst::AbstractStencilArray{RD}, src::AbstractStencilArray{RS}) where {RD,RS}
    dst_axes = add_halo(dst, axes(dst))
    src_axes = add_halo(src, axes(src))
    copyto!(
        parent(dst), CartesianIndices(dst_axes),
        parent(src), CartesianIndices(src_axes)
    )
    return dst
end

function Base.copyto!(dst::AbstractStencilArray, idst::CartesianIndices, src::AbstractStencilArray, isrc::CartesianIndices)
    dst_axes = add_halo(dst, idst)
    src_axes = add_halo(src, isrc)
    copyto!(parent(dst), dst_axes, parent(src), src_axes)
    return dst
end
function Base.copyto!(dst::AbstractArray, idst::CartesianIndices, src::AbstractStencilArray, isrc::CartesianIndices)
    src_axes = add_halo(src, isrc)
    copyto!(dst, idst, parent(src), src_axes)
    return dst
end
function Base.copyto!(dst::AbstractStencilArray, idst::CartesianIndices, src::AbstractArray, isrc::CartesianIndices)
    dst_axes = add_halo(dst, idst)
    copyto!(parent(dst), dst_axes, src, isrc)
    return dst
end
function Base.copyto!(dst::AbstractStencilArray, src::AbstractStencilArray)
    dst_axes = add_halo(dst, axes(dst))
    src_axes = add_halo(src, axes(src))
    copyto!(
        parent(dst), CartesianIndices(dst_axes),
        parent(src), CartesianIndices(src_axes),
    )
    return dst
end
function Base.copyto!(dst::AbstractArray, src::AbstractStencilArray)
    src_axes = add_halo(src, axes(src))
    copyto!(
        dst, CartesianIndices(dst),
        parent(src), CartesianIndices(src_axes),
    )
    return dst
end
function Base.copyto!(dst::AbstractStencilArray, src::AbstractArray)
    dst_axes = add_halo(dst, axes(dst))
    copyto!(
        parent(dst), CartesianIndices(dst_axes),
        src, CartesianIndices(src),
    )
    return dst
end

function Base.show(io::IO, mime::MIME"text/plain", A::AbstractStencilArray)
    summary(io, A)
    isempty(A) && return
    print(io, ":")
    Base.show_circular(io, A) && return

    println(io)
    Base.print_array(io, A)
    print(io, "\n\nstencil: ")
    show(io, mime, stencil(A))
    print(io, "boundary: ")
    show(io, mime, boundary(A))
    print(io, "\npadding: ")
    show(io, mime, padding(A))
end

# Iterate over the parent for `Conditional` padding, 2x faster.
Base.iterate(A::AbstractStencilArray{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Conditional}, args...) =
    iterate(parent(A), args...)
Base.parent(A::AbstractStencilArray) = A.parent
for f in (:getindex, :view, :dotview)
    unsafe_f = Symbol(string("unsafe_", f))
    @eval begin
        # Base.@propagate_inbounds function Base.$f(
        #     A::AbstractStencilArray{R}, I::Union{Colon,Int64,AbstractArray}...
        # ) where R
        #     @boundscheck checkbounds(A, I...)
        #     @inbounds Base.$f(parent(A), I...)
        # end
        Base.@propagate_inbounds function Base.$f(
            A::AbstractStencilArray{R}, i1::Int, Is::Int...
        ) where R
            @boundscheck checkbounds(A, i1, Is...)
            $unsafe_f(A, padding(A), i1, Is...)
        end
        $unsafe_f(A::AbstractStencilArray, I...) = $unsafe_f(A, padding(A), I...)
        function $unsafe_f(A::AbstractStencilArray{R}, ::Halo, I...) where R
            @inbounds Base.$f(parent(A), add_halo(A, I)...)
        end
        function $unsafe_f(A::AbstractStencilArray, ::Padding, I...)
            @inbounds Base.$f(parent(A), I...)
        end
    end
end
Base.@propagate_inbounds function Base.setindex!(A::AbstractStencilArray, x, I::Int...)
    @boundscheck checkbounds(A, I...)
    unsafe_setindex!(A, x, I...)
end
unsafe_setindex!(A::AbstractStencilArray, x, I...) = unsafe_setindex!(A, padding(A), x, I...)
function unsafe_setindex!(A::AbstractStencilArray{R}, ::Halo, x, I...) where R
    @inbounds setindex!(parent(A), x, add_halo(A, I)...)
end
function unsafe_setindex!(A::AbstractStencilArray, ::Padding, x, I...)
    @inbounds setindex!(parent(A), x, I...)
end

add_halo(A::AbstractStencilArray, I) = add_halo(padding(A), A, I)
add_halo(::Halo, A::AbstractStencilArray, I) = _add_halo(A, I)
add_halo(::Padding, A::AbstractStencilArray, I) = I

_add_halo(A::AbstractStencilArray, I::Tuple) = map(i -> _add_halo(A, i), I)
_add_halo(::AbstractStencilArray{R}, I::Integer) where R = I + R
_add_halo(::AbstractStencilArray{R}, I::AbstractUnitRange) where R = I .+ R
_add_halo(::AbstractStencilArray{R}, I::CartesianIndices{N}) where {R,N} =
    CartesianIndex(ntuple(_ -> R, Val(N))) .+ I
_add_halo(::AbstractStencilArray{R}, I::CartesianIndex{N}) where {R,N} =
    CartesianIndex(ntuple(_ -> R, Val(N))) + I

Base.size(A::AbstractStencilArray) = _size(padding(A), stencil(A), parent(A))

Base.similar(A::AbstractStencilArray) = similar(parent(A), size(A))
Base.similar(A::AbstractStencilArray, ::Type{T}) where T = similar(parent(A), T, size(A))
Base.similar(A::AbstractStencilArray, I::Tuple{Int,Vararg{Int}}) = similar(parent(A), I)
Base.similar(A::AbstractStencilArray, ::Type{T}, I::Tuple{Int,Vararg{Int}}) where T =
    similar(parent(A), T, I)


const STENCILARRAY_KEYWORDS = """
- `boundary`: a [`BoundaryCondition`](@ref) like [`Wrap`](@ref). `Remove()` is the default.
- `padding`: [`Padding`](@ref) like [`Conditional`](@ref) or [`Halo{:in}`](@ref). `Conditional()` is the default.
"""

"""
    StencilArray <: AbstractStencilArray

    StencilArray(A::AbstractArray, stencil::Stencil; kw...)

An array with a [`Stencil`](@ref) and a [`BoundaryCondition`](@ref), and [`Padding`](@ref).

For most uses a `StencilArray` works exactly the same as a regular array.

Except it can be indexed at any point with `stencil` to return a filled
`Stencil` object, or `neighbors` to return an `SVector` of neighbors.

## Arguments

- `A`: an `AbstractArray`
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
struct StencilArray{
    R,T,N,A<:AbstractArray{T,N},S<:Union{Stencil{R},Layered{R}},BC,P
} <: AbstractStencilArray{R,T,N,A,S,BC,P}
    parent::A
    stencil::S
    boundary::BC
    padding::P
    function StencilArray{R,T,N,A,S,BC,P}(
        parent::A, stencil::S, boundary::BC, padding::P
    ) where {R,T,N,A,S,BC,P}
        map(size(parent), _radii(Val{N}(), R)) do s, rs
            max(map(abs, rs)...) < s || throw(ArgumentError("stencil radius is larger than array axis $s"))
        end
        return new{R,T,N,A,S,BC,P}(parent, stencil, boundary, padding)
    end
end
function StencilArray(
    parent::A, stencil::S, boundary::BC, padding::P
) where {A<:AbstractArray{T,N},S<:Union{Stencil{R},Layered{R}},BC,P} where {R,T,N}
    StencilArray{R,T,N,A,S,BC,P}(parent, stencil, boundary, padding)
end
function StencilArray(parent::AbstractArray{<:Any,N}, stencil=Window{1,N}();
    boundary=Remove(zero(eltype(parent))),
    padding=Conditional(),
) where N
    padded_parent = pad_array(padding, boundary, stencil, parent)
    StencilArray(padded_parent, stencil, boundary, padding)
end

_size(::Conditional, stencil, parent) = size(parent)
function _size(::Halo, ::Union{Stencil{R},Layered{R}}, parent) where R 
    size(parent) .- 2R
end

#Allocates similar array for StencilArray. Assumes that the stencil, boundary and padding are immutables.
Base.similar(A::StencilArray) =
    StencilArray(similar(parent(A)), stencil(A), boundary(A), padding(A))
Base.similar(A::StencilArray, ::Type{T}) where T =
    StencilArray(similar(parent(A), T), stencil(A), boundary(A), padding(A))

function Base.copy(src::StencilArray)
    cp = similar(src)
    copyto!(parent(cp), parent(src)) #copy parent array to make sure that padding is also copied
    return cp
end

function Adapt.adapt_structure(to, A::StencilArray)
    newparent = Adapt.adapt(to, parent(A))
    newstencil = Adapt.adapt(to, stencil(A))
    StencilArray(newparent, newstencil, boundary(A), padding(A))
end

# Internals
"""
    AbstractSwitchingStencilArray

Abstract supertype for `AbstractStencilArray` that wrap two arrays that
switch places with each broadcast.
"""
abstract type AbstractSwitchingStencilArray{R,T,N,A,S,BC,P} <: AbstractStencilArray{R,T,N,A,S,BC,P} end

Base.parent(d::AbstractSwitchingStencilArray) = source(d)

source(A::AbstractSwitchingStencilArray) = A.source
source(A::AbstractStencilArray) = parent(A)

dest(A::AbstractSwitchingStencilArray) = A.dest
dest(A::AbstractStencilArray) = parent(A)

radius(d::AbstractStencilArray{R}) where R = R
padval(d::AbstractStencilArray) = padval(boundary(d))

"""
    SwitchingStencilArray <: AbstractSwitchingStencilArray

    SwitchingStencilArray(A::AbstractArray, stencil; 
        boundary=Remove(zero(eltype(parent))),
        padding=Conditional(),
    )

An `AbstractArray` with a [`Stencil`](@ref), a [`BoundaryCondition`](@ref), [`Padding`](@ref),
and two array layers that are switched with each `broadcast_stencil` operation.

The use case for this operation is in simulations where stencil operations
are repeatedly run over the same data, or where a filter (such as a blur) needs
to be applied many times.

For most uses a `SwitchingStencilArray` works exactly the same as a
regular array - the `dest` array can be safely ignored.

However, when using `mapstencil!` you need to use the output, not the original
array. Switching does not happen in-place, but as a new returned array.

## Arguments

- `A`: an `AbstractArray`
- `stencil`: a [`Stencil`](@ref).

## Keywords

$STENCILARRAY_KEYWORDS

## Example

```jldoctest
using Stencils, Statistics

sa = SwitchingStencilArray(rand(10, 10), Moore(1); boundary=Wrap())
sa .*= 2 # Broadcast works as usual
mapstencil(mean, sa) # As does runing `mapstencils
s = stencil(sa, 5, 10) # And retreiving a single stencil
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
struct SwitchingStencilArray{
    R,T,N,A<:AbstractArray{T,N},S<:Union{Stencil{R},Layered{R}},BC,P
} <: AbstractSwitchingStencilArray{R,T,N,A,S,BC,P}
    source::A
    dest::A
    stencil::S
    boundary::BC
    padding::P
    function SwitchingStencilArray{R,T,N,A,S,BC,P}(
        source::A, dest::A, stencil::S, boundary::BC, padding::P
    ) where {R,T,N,A,S,BC,P}
        size(source) == size(dest) || 
            throw(ArgumentError("source and dest arrays must be the same size, got $(size(source)) and $(size(dest))"))
        map(size(source), _radii(Val{N}(), R)) do s, rs
            max(map(abs, rs)...) < s || 
                throw(ArgumentError("stencil radius is larger than array axis $s"))
        end
        return new{R,T,N,A,S,BC,P}(source, dest, stencil, boundary, padding)
    end
end

# Construct from a single parent array
function SwitchingStencilArray(parent::AbstractArray{<:Any,N}, stencil=Window{1,N}();
    boundary=Remove(zero(eltype(parent))),
    padding=Conditional(),
) where N
    SwitchingStencilArray(parent, stencil, boundary, padding)
end
# Build the source and dest padded arrays
function SwitchingStencilArray(parent::AbstractArray, stencil::StencilOrLayered, bc, padding)
    padded_source = pad_array(padding, bc, stencil, parent)
    padded_dest = pad_array(padding, bc, stencil, parent)
    padded_dest = padded_source === padded_dest ? copy(padded_source) : padded_dest
    return SwitchingStencilArray(padded_source, padded_dest, stencil, bc, padding)
end
function SwitchingStencilArray(
    source::A, dest::A, stencil::S, boundary::BC, padding::P
) where {A<:AbstractArray{T,N},S<:Union{Stencil{R},Layered{R}},BC,P} where {R,T,N}
    SwitchingStencilArray{R,T,N,A,S,BC,P}(source, dest, stencil, boundary, padding)
end

"""
    switch(A::SwitchingStencilArray)

Swap the source and dest of a `SwitchingStencilArray`.
"""
switch(A::SwitchingStencilArray) =
    SwitchingStencilArray(dest(A), source(A), stencil(A), boundary(A), padding(A))

Base.parent(A::SwitchingStencilArray) = source(A)

Base.similar(src::SwitchingStencilArray) =
    SwitchingStencilArray(similar(source(src)), similar(dest(src)), stencil(src), boundary(src), padding(src))
Base.similar(src::SwitchingStencilArray, ::Type{T}) where {T} =
    SwitchingStencilArray(similar(source(src), T), similar(dest(src), T), stencil(src), boundary(src), padding(src))

function Base.copy(src::SwitchingStencilArray)
    cp = similar(src)
    copyto!(source(cp), source(src))
    copyto!(dest(cp), dest(src))
    return cp
end

function Adapt.adapt_structure(to, A::SwitchingStencilArray)
    newsource = Adapt.adapt(to, A.source)
    newdest = Adapt.adapt(to, A.dest)
    SwitchingStencilArray(newsource, newdest, stencil(A), boundary(A), padding(A))
end

