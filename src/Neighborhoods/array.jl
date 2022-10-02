"""
"""
abstract type AbstractNeighborhoodArray{S,R,T,N} <: StaticArray{S,T,N} end

"""
    NeighborhoodArray

An 
"""
struct NeighborhoodArray{S,R,T,N,A<:AbstractArray{T,N},BC,PV,H}
    parent::A
    boundary_condition::BC
    padval::PV
    neighborhood::H
end
NeighborhoodArray(parent::AbstractArray, hood::Neighborhood, bc, pv) =
    NeighborhoodArray{Tuple{size(parent)...}}(parent, hood, bc, pv)
NeighborhoodArray{S}(parent::AbstractArray, h::Neighborhood{R}) where {S,R} =
    NeighborhoodArray{S,R}(parent, h, bc, pv)
NeighborhoodArray{S,R}(parent::A, h::H, bc::BC, pv::PV) where {S,R,A<:AbstractArray{T,N},H,BC,PV} where {T,N} =
    NeighborhoodArray{S,R,T,N,A,H,BC,PV}(parent, h, bc, pv)

boundary_condition(A::NeighborhoodArray) = A.boundary_condition
neighbors(A::NeighborhoodArray, I...) = unsafe_readneighbors(neighborhood(A), A, I...)
neighborhood(A::NeighborhoodArray, I...) = unsafe_updateneighbors(neighborhood(A), A, I...)
neighborhood(A::NeighborhoodArray) = A.neighborhood

Base.parent(A::AbstractNeighborhoodArray) = A.parent
for f in (:getindex, :view, :dotview)
    @eval Base.@propagate_inbounds Base.$f(d::AbstractNeighborhoodArray, I...) =
        $f(parent(d), I...)
end
Base.@propagate_inbounds Base.setindex!(d::AbstractNeighborhoodArray, x, I...) =
    setindex!(parent(d), x, I...)

# Return a SizedArray with similar, instead of a StaticArray
Base.similar(A::AbstractNeighborhoodArray) = similar(_unpad_view(A))
Base.similar(A::AbstractNeighborhoodArray, ::Type{T}) where T = similar(_unpad_view(A), T)
Base.similar(A::AbstractNeighborhoodArray, I::Tuple{Int,Vararg{Int}}) = similar(_unpad_view(A), I)
Base.similar(A::AbstractNeighborhoodArray, ::Type{T}, I::Tuple{Int,Vararg{Int}}) where T =
    similar(_unpad_view(A), T, I)

_unpad_view(A::AbstractNeighborhoodArray) = view(parent(A), axes(A)...)
_unpad_view(A, N::AbstractNeighborhoodArray) = view(A, axes(d)...)

