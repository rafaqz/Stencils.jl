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

padval(bc::Remove) = bc.padval

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

