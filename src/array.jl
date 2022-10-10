
"""
    AbstractNeighborhoodArray <: StaticArray

Supertype for arrays with a [`Neighborhood`](@ref) and a [BoundaryCondition](@ref),
and [`Padding`](@ref), and fixed size like a `SizedArray` from StaticArrays.jl
"""
abstract type AbstractNeighborhoodArray{S,R,T,N} <: AbstractArray{T,N} end

boundary_condition(A::AbstractNeighborhoodArray) = A.boundary_condition
padding(A::AbstractNeighborhoodArray) = A.padding
neighborhood(A::AbstractNeighborhoodArray, I::CartesianIndex) = update_neighborhood(A, I)
neighborhood(A::AbstractNeighborhoodArray) = A.neighborhood

Base.iterate(A::AbstractNeighborhoodArray, args...) = iterate(inner_view(A), args...)
Base.parent(A::AbstractNeighborhoodArray) = A.parent
for f in (:getindex, :view, :dotview)
    @eval begin
        Base.@propagate_inbounds Base.$f(A::AbstractNeighborhoodArray, I::Union{Colon,Int64,AbstractArray}...) =
            Base.$f(parent(A), I...)
        Base.@propagate_inbounds Base.$f(A::AbstractNeighborhoodArray, i1::Int, I::Int...) = Base.$f(parent(A), i1, I...)
    end
end
Base.@propagate_inbounds Base.setindex!(d::AbstractNeighborhoodArray, x, I::Int...) =
    setindex!(parent(d), x, I...)
Base.@propagate_inbounds Base.setindex!(d::AbstractNeighborhoodArray, x, I...) =
    setindex!(parent(d), x, I...)
Base.size(::AbstractNeighborhoodArray{S}) where S = tuple_contents(S)

# inner_view(A::AbstractNeighborhoodArray{S,R}) where {S,R} = view(parent(A), axes(A))

"""
    NeighborhoodArray

An array with padding, known size, a [`Neighborhood`](@ref) and a [BoundaryCondition](@ref).
"""
struct NeighborhoodArray{S,R,T,N,A<:AbstractArray{T,N},H<:Neighborhood{R,N},BC,P} <: AbstractNeighborhoodArray{S,R,T,N}
    parent::A
    neighborhood::H
    boundary_condition::BC
    padding::P
    function NeighborhoodArray{S,R,T,N,A,H,BC,P}(parent::A, h::H, bc::BC, padding::P) where {S,R,T,N,A,H,BC,P}
        map(tuple_contents(S)) do s
            R < s || throw(ArgumentError("neighborhood radius R is larger than array axis $s"))
        end
        na = new{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
        update_boundary!(na)
        return na
    end
end
function NeighborhoodArray(parent::AbstractArray, hood::Neighborhood{R}, bc, padding) where R
    padded_parent = pad_array(padding, bc, hood, parent)
    S = Tuple{_size(padding, hood, padded_parent)...}
    NeighborhoodArray{S,R}(padded_parent, hood, bc, padding)
end
NeighborhoodArray{S}(parent::AbstractArray, hood::Neighborhood{R}, bc, padding) where {S,R} =
    NeighborhoodArray{S,R}(parent, hood, bc, padding)
NeighborhoodArray{S,R}(parent::A, h::H, bc::BC, padding::P) where {S,A<:AbstractArray{T,N},H<:Neighborhood{R},BC,P} where {R,T,N} =
    NeighborhoodArray{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
function NeighborhoodArray(parent;
    neighborhood=Window{1}(),
    boundary_condition=Remove(zero(eltype(parent))),
    padding=Conditional(),
)
    NeighborhoodArray(parent, neighborhood, boundary_condition, padding)
end

_size(::Conditional, ::Neighborhood, parent) = size(parent)
_size(::Halo, ::Neighborhood{R}, parent) where R = size(parent) .- 2R

function Adapt.adapt_structure(to, A::NeighborhoodArray{S}) where S
    newparent = Adapt.adapt(to, parent(A))
    NeighborhoodArray{S}(newparent, neighborhood(A), boundary_condition(A), padding(A))
end

ConstructionBase.constructorof(::Type{<:NeighborhoodArray{S}}) where S = NeighborhoodArray{S}

# Return a SizedArray with similar, instead of a StaticArray
Base.similar(A::AbstractNeighborhoodArray) = similar(parent(parent(A)), size(A))
Base.similar(A::AbstractNeighborhoodArray, ::Type{T}) where T = similar(parent(parent(A)), T, size(A))
Base.similar(A::AbstractNeighborhoodArray, I::Tuple{Int,Vararg{Int}}) = similar(parent(parent(A)), I)
Base.similar(A::AbstractNeighborhoodArray, ::Type{T}, I::Tuple{Int,Vararg{Int}}) where T =
    similar(parent(parent(A)), T, I)

"""
    neighbors(hood::Neighborhood, A::AbstractArray, I) => SArray

Get a single neighborhood from an array, as a `Tuple`, checking bounds.
"""
@inline function neighbors(A::AbstractNeighborhoodArray{<:Any,R,<:Any,N}, I::CartesianIndex) where {R,N}
    # Unpadded has checks internally
    if A.padding isa Halo
        low = CartesianIndex(ntuple(_ -> -R, N))
        high = CartesianIndex(ntuple(_ -> R, N))
        checkbounds(parent(A), I + low)
        checkbounds(parent(A), I + high)
    end
    return unsafe_neighbors(A, I)
end

"""
    unsafe_readneighbors(hood::Neighborhood, A::AbstractArray, I) => SArray

Get a single neighborhood from an array, as a `Tuple`, without checking bounds.
"""
@inline function unsafe_neighbors(A::AbstractNeighborhoodArray, I::CartesianIndex)
    map(indices(neighborhood(A), I)) do P
        neighbor_getindex(A, P...)
    end
end

"""
    updateneighbors(x, A::AbstractArray, I) => Neighborhood

Set the neighbors of a neighborhood to values from the array A around index `I`.
Bounds checks will reduce performance, aim to use `unsafe_setneighbors` directly.
"""
@inline function update_neighborhood(A::AbstractNeighborhoodArray, I::CartesianIndex)
    setneighbors(neighborhood(A), neighbors(A, I))
end

"""
    unsafe_updateneighbors(x, A::AbstractArray, I) => Neighborhood

Set the neighbors of a neighborhood to values from the array A around index `I`.

No bounds checks occur, ensure that A has padding of at least the neighborhood radius.
"""
@inline function unsafe_update_neighborhood(A::AbstractNeighborhoodArray, I::CartesianIndex)
    setneighbors(neighorhood(A), unsafe_neighbors(A, I))
end

neighbor_getindex(A::AbstractNeighborhoodArray, I::Int...) =
    neighbor_getindex(A, boundary_condition(A), padding(A), I...)
# If Halo padded we can just use regular `getindex`
# on the parent array, which is an `OffsetArray`
neighbor_getindex(A::AbstractNeighborhoodArray, ::BoundaryCondition, pad::Halo, I::Int...) = @inbounds parent(A)[I...]
# Unpadded needs handling. For Wrap we swap the side:
function neighbor_getindex(A::AbstractNeighborhoodArray{S}, ::Wrap, pad::Conditional, I::Int...) where S
    sz = tuple_contents(S)
    wrapped_inds = map(I, sz) do i, s
        i < 1 ? i + s : (i > s ? i - s : i)
    end
    return @inbounds A[wrapped_inds...]
end
# For Remove we use padval if out of bounds
function neighbor_getindex(A::AbstractNeighborhoodArray, x::Remove, pad::Conditional, I::Int...)
    return checkbounds(Bool, A, I...) ? (@inbounds A[I...]) : x.padval 
end

# update_boundary!
# Reset or wrap boundary where required. This allows us to ignore 
# bounds checks on neighborhoods and still use a wraparound grid.
update_boundary!(As::Tuple) = map(update_boundary!, As)
update_boundary!(A::NeighborhoodArray) =
    update_boundary!(A, padding(A), boundary_condition(A))
# Conditional sets boundary conditions on the fly
update_boundary!(A::AbstractNeighborhoodArray, ::Conditional, ::BoundaryCondition) = A
# Halo needs updating
function update_boundary!(A::AbstractNeighborhoodArray{S,R}, ::Halo, bc::Remove) where {S<:Tuple{L},R} where {L}
    src = parent(A)
    @inbounds src[1-R:0] .= Ref(padval(bc))
    @inbounds src[L+1:L+R] .= Ref(padval(bc))
    return A
end
function update_boundary!(A::AbstractNeighborhoodArray{S,R}, ::Halo, bc::Remove) where {S<:Tuple{Y,X},R} where {Y,X}
    src = parent(A)
    # X
    @inbounds src[axes(src, 1), 1-R:0] .= Ref(padval(bc))
    @inbounds src[axes(src, 1), X+1:X+R] .= Ref(padval(bc))
    # Y
    @inbounds src[1-R:0, axes(src, 2)] .= Ref(padval(bc))
    @inbounds src[Y+1:Y+R, axes(src, 2)] .= Ref(padval(bc))
    return A
end
function update_boundary!(A::AbstractNeighborhoodArray{S,R}, ::Halo, bc::Remove) where {S<:Tuple{Z,Y,X},R} where {Z,Y,X}
    src = parent(A)
    # X
    @inbounds src[axes(src, 1), axes(src, 2), 1-R:0] .= Ref(padval(bc))
    @inbounds src[axes(src, 1), axes(src, 2), X+1:X+R] .= Ref(padval(bc))
    # Y             
    @inbounds src[axes(src, 1), 1-R:0, axes(src, 3)] .= Ref(padval(bc))
    @inbounds src[axes(src, 1), Y+1:Y+R, axes(src, 3)] .= Ref(padval(bc))
    # Z             
    @inbounds src[1-R:0, axes(src, 2), axes(src, 3)] .= Ref(padval(bc))
    @inbounds src[Z+1:Z+R, axes(src, 2), axes(src, 3)] .= Ref(padval(bc))
    return A
end
function update_boundary!(A::AbstractNeighborhoodArray{S,R}, ::Halo, ::Wrap) where {S<:Tuple{L},R} where {L}
    src = parent(A)
    startpad = 1-R:0
    endpad = L+1:L+R
    startvals = 1:R
    endvals = L-R+1:L
    @assert length(startpad) == length(endvals) == R
    @assert length(endpad) == length(startvals) == R
    @inbounds copyto!(src, CartesianIndices((startpad,)), src, CartesianIndices((endvals,)))
    @inbounds copyto!(src, CartesianIndices((endpad,)), src, CartesianIndices((startvals,)))
    return A
end
function update_boundary!(A::AbstractNeighborhoodArray{S,R}, ::Halo, ::Wrap) where {S<:Tuple{Y,X},R} where {Y,X}
    src = parent(A)
    n_xs, n_ys = X, Y
    startpad_x = startpad_y = 1-R:0
    endpad_x = n_xs+1:n_xs+R
    endpad_y = n_ys+1:n_ys+R
    start_x = start_y = 1:R
    end_x = n_xs+1-R:n_xs
    end_y = n_ys+1-R:n_ys
    xs = 1-R:n_xs+R
    ys = 1-R:n_ys+R

    @assert length(startpad_x) == length(start_x) == R
    @assert length(endpad_x) == length(end_x) == R
    @assert length(startpad_y) == length(start_y) == R
    @assert length(endpad_y) == length(end_y) == R
    @assert map(length, (xs, ys)) === size(src)

    CI = CartesianIndices
    # Sides ---
    @inbounds copyto!(src, CI((xs, startpad_y)), src, CI((xs, end_y)))
    @inbounds copyto!(src, CI((xs, endpad_y)), src, CI((xs, start_y)))
    @inbounds copyto!(src, CI((startpad_x, ys)), src, CI((end_x, ys)))
    @inbounds copyto!(src, CI((endpad_x, ys)), src, CI((start_x, ys)))

    # Corners ---
    @inbounds copyto!(src, CI((startpad_x, startpad_y)), src, CI((end_x, end_y)))
    @inbounds copyto!(src, CI((startpad_x, endpad_y)), src, CI((end_x, start_y)))
    @inbounds copyto!(src, CI((endpad_x, startpad_y)), src, CI((start_x, end_y)))
    @inbounds copyto!(src, CI((endpad_x, endpad_y)), src, CI((start_x, start_y)))

    return after_update_boundary!(A)
end
function update_boundary!(A::AbstractNeighborhoodArray{S,R}, ::Halo, ::Wrap) where {S<:Tuple{Z,Y,X},R} where {Z,Y,X}
    src = parent(A)
    n_xs, n_ys, n_zs = X, Y, Z
    startpad_x = startpad_y = startpad_z = 1-R:0
    endpad_x = n_xs+1:n_xs+R
    endpad_y = n_ys+1:n_ys+R
    endpad_z = n_ys+1:n_zs+R
    start_x = start_y = start_z = 1:R
    end_x = n_xs+1-R:n_xs
    end_y = n_ys+1-R:n_ys
    end_z = n_zs+1-R:n_zs
    xs = 1-R:n_xs+R
    ys = 1-R:n_ys+R
    zs = 1-R:n_zs+R

    @assert length(startpad_x) == length(start_x) == R
    @assert length(endpad_x) == length(end_x) == R
    @assert length(startpad_y) == length(start_y) == R
    @assert length(endpad_y) == length(end_y) == R
    @assert map(length, (xs, ys, zs)) === size(src)

    CI = CartesianIndices
    # Sides ---
    # X
    @inbounds copyto!(src, CI((startpad_x, ys, zs)), src, CI((end_x, ys, zs)))
    @inbounds copyto!(src, CI((endpad_x, ys, zs)), src, CI((start_x, ys, zs)))
    # Y
    @inbounds copyto!(src, CI((xs, startpad_y, zs)), src, CI((xs, end_y, zs)))
    @inbounds copyto!(src, CI((xs, endpad_y, zs)), src, CI((xs, start_y, zs)))
    # Z
    @inbounds copyto!(src, CI((xs, ys, startpad_z)), src, CI((xs, ys, end_z)))
    @inbounds copyto!(src, CI((xs, ys, endpad_z)), src, CI((xs, ys, start_z)))

    # Corners ---
    @inbounds copyto!(src, CI((startpad_x, startpad_y, startpad_z)), src, CI((end_x, end_y, end_z)))
    @inbounds copyto!(src, CI((startpad_x, startpad_y, endpad_z)), src, CI((end_x, end_y, start_z)))
    @inbounds copyto!(src, CI((startpad_x, endpad_y, startpad_z)), src, CI((end_x, start_y, end_z)))
    @inbounds copyto!(src, CI((startpad_x, endpad_y, endpad_y)), src, CI((end_x, start_y, start_z)))
    @inbounds copyto!(src, CI((endpad_x, endpad_y, endpad_z)), src, CI((start_x, start_y, start_z)))
    @inbounds copyto!(src, CI((endpad_x, startpad_y, endpad_z)), src, CI((end_x, start_y, start_z)))
    @inbounds copyto!(src, CI((endpad_x, endpad_y, startpad_z)), src, CI((start_x, start_y, end_z)))
    @inbounds copyto!(src, CI((endpad_x, startpad_y, startpad_z)), src, CI((start_x, end_y, end_z)))
    return after_update_boundary!(A)
end

# Allow additional boundary updating behaviours
after_update_boundary!(A) = A


radii(x::Int, s::NTuple{N}) where N = ntuple(_ -> ntuple(_ -> x, Val{N}()), Val{N}())
radii(x::Tuple, s::Tuple) = x
