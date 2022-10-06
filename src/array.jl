abstract type Padding end

struct Padded{X} <: Padding end
struct Unpadded <: Padding end

"""
"""
abstract type AbstractNeighborhoodArray{S,R,T,N} <: StaticArray{S,T,N} end

"""
    NeighborhoodArray

An array with padding, known size, a [`Neighborhood`](@ref) and a [BoundaryCondition](@ref). 
"""
struct NeighborhoodArray{S,R,T,N,A<:AbstractArray{T,N},H,BC,P} <: AbstractNeighborhoodArray{S,R,T,N}
    parent::A
    neighborhood::H
    boundary_condition::BC
    padding::P
    function NeighborhoodArray{S,R,T,N,A,H,BC,P}(parent::A, h::H, bc::BC, padding::P) where {S,R,T,N,A,H,BC,P}
        map(tuple_contents(S)) do s
            R < s || throw(ArgumentError("neighborhood radius R is larger than array axis $s"))
        end
        new{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
    end
end
function NeighborhoodArray(parent::AbstractArray, hood::Neighborhood{R}, bc, padding) where R
    padded = _pad_array(parent, hood, padding)
    S = Tuple{_size(padded, hood, padding)...}
    NeighborhoodArray{S,R}(padded, hood, bc, padding)
end
NeighborhoodArray{S}(parent::AbstractArray, hood::Neighborhood{R}, bc, padding) where {S,R} =
    NeighborhoodArray{S,R}(parent, hood, bc, padding)
NeighborhoodArray{S,R}(parent::A, h::H, bc::BC, padding::P) where {S,R,A<:AbstractArray{T,N},H,BC,P} where {T,N} =
    NeighborhoodArray{S,R,T,N,A,H,BC,P}(parent, h, bc, padding)
function NeighborhoodArray(parent;
    neighborhood=Window{1}(),
    boundary_condition=Remove(zero(eltype(parent))),
    padding=Unpadded(),
)
    NeighborhoodArray(parent, neighborhood, boundary_condition, padding)
end

_size(A, hood::Neighborhood, ::Unpadded) = size(A)
_size(A, ::Neighborhood{R}, ::Padded) where R = size(A) .- 2R

_pad_array(parent, hood, padding::Unpadded) = parent
function _pad_array(parent, hood::NeighborhoodArray{R}, padding::Padded{:in}) where R
    OffsetArray(parent, offset_axes(parent, hood))
end
function _pad_array(parent, hood::NeighborhoodArray{R}, padding::Padded{:out}) where R
    outer_array(padding, hood)
end

boundary_condition(A::NeighborhoodArray) = A.boundary_condition
padding(A::NeighborhoodArray) = A.padding
neighbors(A::NeighborhoodArray, I...) = unsafe_readneighbors(neighborhood(A), A, I...)
neighborhood(A::NeighborhoodArray, I...) = unsafe_updateneighbors(neighborhood(A), A, I...)
neighborhood(A::NeighborhoodArray) = A.neighborhood

Base.parent(A::AbstractNeighborhoodArray) = A.parent
for f in (:getindex, :view, :dotview)
    @eval begin
        Base.@propagate_inbounds Base.$f(A::NeighborhoodArray, I::Union{Colon,Int64,AbstractArray,SOneTo,StaticArray{<:Tuple,Int64}}...) =
            Base.$f(parent(A), I...)
        Base.@propagate_inbounds Base.$f(A::NeighborhoodArray, i1::Int, I::Int...) = Base.$f(parent(A), i1, I...)
    end
end
Base.@propagate_inbounds Base.setindex!(d::AbstractNeighborhoodArray, x, I...) =
    setindex!(parent(d), x, I...)

# Return a SizedArray with similar, instead of a StaticArray
# Base.similar(A::AbstractNeighborhoodArray) = similar(_unpad_view(A))
# Base.similar(A::AbstractNeighborhoodArray, ::Type{T}) where T = similar(_unpad_view(A), T)
# Base.similar(A::AbstractNeighborhoodArray, I::Tuple{Int,Vararg{Int}}) = similar(_unpad_view(A), I)
# Base.similar(A::AbstractNeighborhoodArray, ::Type{T}, I::Tuple{Int,Vararg{Int}}) where T =
#     similar(_unpad_view(A), T, I)
