"""
    BoundaryCondition

Abstract supertype for flags that specify the boundary conditions used in the simulation,
used in [`inbounds`](@ref) and to update [`NeighborhoodRule`](@ref) grid padding.
These determine what happens when a neighborhood or jump extends outside of the grid.
"""
abstract type BoundaryCondition end

"""
    Wrap <: BoundaryCondition

    Wrap()

[`BoundaryCondition`](@ref) flag to wrap cordinates that boundary boundaries back to the
opposite side of the grid.

Specifiy with:

```julia
ruleset = Ruleset(rule; boundary=Wrap())
# or
output = sim!(output, rule; boundary=Wrap())
```
"""
struct Wrap <: BoundaryCondition end

"""
    Remove <: BoundaryCondition

    Remove()

[`BoundaryCondition`](@ref) flag that specifies to assign `padval` to cells that overflow 
grid boundaries. `padval` defaults to `zero(eltype(grid))` but can be assigned as a keyword
argument to an [`Output`](@ref).

Specifiy with:

```julia
ruleset = Ruleset(rule; boundary=Remove())
# or
output = sim!(output, rule; boundary=Remove())
```
"""
struct Remove{PV} <: BoundaryCondition
    padval::PV
end

# See interface docs
# @inline inbounds(data::Union{GridData,AbstractSimData}, I::Tuple) = inbounds(data, I...)
# @inline inbounds(data::Union{GridData,AbstractSimData}, I...) = 
#     _inbounds(boundary_condition(data), gridsize(data), I...)

# @inline function _inbounds(boundary::BoundaryCondition, size::Tuple, i1, i2)
#     a, inbounds_a = _inbounds(boundary, size[1], i1)
#     b, inbounds_b = _inbounds(boundary, size[2], i2)
#     (a, b), inbounds_a & inbounds_b
# end
# @inline _inbounds(::Remove, size::Number, i::Number) = i, _isinbounds(size, i)
# @inline function _inbounds(::Wrap, size::Number, i::Number)
#     if i < oneunit(i)
#         size + rem(i, size), true
#     elseif i > size
#         rem(i, size), true
#     else
#         i, true
#     end
# end

# # See interface docs
# @inline isinbounds(data::Union{AbstractSimData}, I::Tuple) = isinbounds(data, I...)
# @inline isinbounds(data::Union{AbstractSimData}, I...) = _isinbounds(gridsize(data), I...)
# @inline isinbounds(data::AbstractNeighborhoodArray, I...) = _isinbounds(gridsize(data), I...)

# @inline _isinbounds(size::Tuple, I...) = all(map(_isinbounds, size, I))
# @inline _isinbounds(size, i) = i >= one(i) && i <= size


radii(x::Int, s::NTuple{N}) where N = ntuple(_ -> ntuple(_ -> x, Val{N}()), Val{N}())
radii(x::Tuple, s::Tuple) = x

# updateboundary
# Reset or wrap boundary where required. This allows us to ignore 
# bounds checks on neighborhoods and still use a wraparound grid.
updateboundary!(As::Tuple) = map(updateboundary!, As)
function updateboundary!(A::NeighborhoodArray{<:Any,R}) where R
    R === 0 || R === () && return A
    return updateboundary!(A, boundary_condition(A))
end
function updateboundary!(g::AbstractNeighborhoodArray{S,R}, ::Remove) where {S<:Tuple{L},R} where {L}
    src = parent(A)
    @inbounds src[1:R] .= Ref(padval(g))
    @inbounds src[L+R+1:L+2R] .= Ref(padval(g))
    return g
end
function updateboundary!(g::AbstractNeighborhoodArray{S,R}, ::Remove) where {S<:Tuple{Y,X},R} where {Y,X}
    src = parent(A)
    # Left
    @inbounds src[1:Y, 1:R] .= Ref(padval(A))
    # Right
    @inbounds src[1:Y, X+R+1:X+2R] .= Ref(padval(A))
    # Top middle
    @inbounds src[1:R, R+1:X+R] .= Ref(padval(A))
    # Bottom middle
    @inbounds src[Y+R+1:Y+2R, R+1:X+R] .= Ref(padval(A))
    return A
end
function updateboundary!(g::AbstractNeighborhoodArray{S,R}, ::Wrap) where {S<:Tuple{L},R} where {L}
    src = parent(A)
    startpad = 1:R
    endpad = L+R+1:L+2R
    startvals = R+1:2R+1
    endvals = L:L+R
    @inbounds copyto!(src, CartesianIndices((startpad,)), src, CartesianIndices((endvals,)))
    @inbounds copyto!(src, CartesianIndices((endpad,)), src, CartesianIndices((startvals,)))
    return A
end
function updateboundary!(g::AbstractNeighborhoodArray{S,R}, ::Wrap) where {S<:Tuple{Y,X},R} where {Y,X}
    src = parent(A)
    nrows, ncols = gridsize(A)
    startpadrow = startpadcol = 1:R
    endpadrow = nrows+R+1:nrows+2R
    endpadcol = ncols+R+1:ncols+2R
    startrow = startcol = 1+R:2R
    endrow = nrows+1:nrows+R
    endcol = ncols+1:ncols+R
    rows = 1+R:nrows+R
    cols = 1+R:ncols+R

    # Sides ---
    # Left
    @inbounds copyto!(src, CartesianIndices((rows, startpadcol)),
                      src, CartesianIndices((rows, endcol)))
    # Right
    @inbounds copyto!(src, CartesianIndices((rows, endpadcol)),
                      src, CartesianIndices((rows, startcol)))
    # Top
    @inbounds copyto!(src, CartesianIndices((startpadrow, cols)),
                      src, CartesianIndices((endrow, cols)))
    # Bottom
    @inbounds copyto!(src, CartesianIndices((endpadrow, cols)),
                      src, CartesianIndices((startrow, cols)))

    # Corners ---
    # Top Left
    @inbounds copyto!(src, CartesianIndices((startpadrow, startpadcol)),
                      src, CartesianIndices((endrow, endcol)))
    # Top Right
    @inbounds copyto!(src, CartesianIndices((startpadrow, endpadcol)),
                      src, CartesianIndices((endrow, startcol)))
    # Botom Left
    @inbounds copyto!(src, CartesianIndices((endpadrow, startpadcol)),
                      src, CartesianIndices((startrow, endcol)))
    # Botom Right
    @inbounds copyto!(src, CartesianIndices((endpadrow, endpadcol)),
                      src, CartesianIndices((startrow, startcol)))

    _wrapopt!(A)
    return A
end

_wrapopt!(g) = _wrapopt!(g, opt(g))
_wrapopt!(g, opt) = g


# Reading windows without padding
# struct Unpadded{S,V}
#     padval::V
# end
# Unpadded{S}(padval::V) where {S,V} = Unadded{S,V}(padval)
# struct Padded{S,K,V}
#     padval::V
# end
# Padded{S,K}(padval::V) where {S,K,V} = Padded{S,K,V}(padval)

# @generated function unsafe_readneighbors(
#     hood::Neighborhood{R,N}, boundary_condition::Union{Remove,Wrap}, padinfo::Padded{S,K,V}, A::AbstractArray{T,N}, I::NTuple{N,Int}
# ) where {T,R,N,V,S,K}
#     inner_size = tuple_contents(S)
#     known_inds = tuple_contents(K)
#     # Generate expressions for each offset - either return the padval or index into A
#     neighbor_exprs = map(offsets(hood)) do O
#         ind_expr = Expr(:tuple)
#         for (n, i, s, o) in zip(ntuple(identity, N), known_inds, inner_size, O)
#             if i isa Colon
#                 # Use the runtime index
#                 return push!(expr, (i isa Colon ? :(I[$n]) : i))
#             end
#             if i + o < 1 
#                 if boundary_condition <: Remove
#                     # Short cut out of the loop and just return the padval
#                     return :(padinfo.padval)  
#                 else # Wrap
#                     # Use an index from thaae other side of this axis
#                     push!(expr, i + o + s)
#                 end
#             elseif i + o > s
#                 if boundary_condition <: Remove
#                     # Short cut out of the loop and just return the padvala
#                     return :(padinfo.padval)  
#                 else # Wrap
#                     # Use an index from the other side of this axis
#                     push!(expr, i + o - s)
#                 end
#             end
#         end 
#         return :(A[$(ind_expr)...])
#     end

#     return neighbor_exprs
# end

# function columns(hood::Neighborhood{R}, A) where {R}
    # unsafe_readneighbors(
# end

