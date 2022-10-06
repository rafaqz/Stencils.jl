const RADIUSDOC = """
    `radius` can be a `Neighborhood`, an `Int`, or a tuple of tuples,
    e.g. for 2d it could be: `((1, 2), (2, 1))::Tuple{Tuple{Int,Int},Tuple{Int,Int}}`.
    """

"""
    outer_axes(A, hood::Neighborhood{R})
    outer_axes(A, radius::Int)

Add padding to axes of array `A`, returning a `Tuple` of `UnitRange`.
$RADIUSDOC
"""
function outer_axes(A, rs::Tuple)
    map(axes(A), rs) do axis, r
        firstindex(axis) - r[1]:lastindex(axis) + r[2]
    end
end

"""
    inner_axes(A, radius)

Remove padding of `radius` from axes of `A`, returning a `Tuple` of `UnitRange`.
$RADIUSDOC
"""
function inner_axes(A, rs::Tuple)
    map(axes(A), rs) do axis, r
        (first(axis) + r[1]):(last(axis) - r[2])
    end
end


function offset_axes(A, rs::Tuple)
    map(axes(A), rs) do axis, r
        (first(axis) - r[1]):(last(axis) - r[2])
    end
end

"""
    outer_array(A, radius; [padval])

Add padding of `radius` to array `A`, redurning a new array.

$RADIUSDOC

`padval` defaults to `zero(eltype(A))`.
"""
function outer_array(A, radius; padval=zero(eltype(A)))
    _outer_array(A, radius, padval)
end

# Handle either specific pad radius for each edge or single Int radius
function _outer_array(A::AbstractArray, r::Int, padval)
    _outer_array(A, _radii(A, r), padval)
end
function _outer_array(A::AbstractArray{T}, rs::Tuple, padval) where T
    paddedaxes = outer_axes(A, rs)
    T1 = promote_type(T, typeof(padval))
    paddedparent = similar(A, T1, length.(paddedaxes)...)
    paddedparent .= Ref(padval)
    padded = OffsetArray(paddedparent, paddedaxes)
    inner_view(paddedparent, rs) .= A
    return padded
end

"""
    inner_array(A, radius)

Remove padding of `radius` from array `A`, returning a new array.

$RADIUSDOC
"""
function inner_array(A::OffsetArray, rs::Tuple) 
    _checkpad(A, rs)
    return inner_array(parent(A), rs)
end
inner_array(A::AbstractArray, rs::Tuple) = A[inner_axes(A, rs)...]

"""
    inner_view(A, radius)

Remove padding of `radius` from array `A`, returning a view of `A`.

$RADIUSDOC
"""
function inner_view(A::OffsetArray, rs::Tuple) 
    _checkpad(A, rs)
    return inner_view(parent(A), rs)
end
inner_view(A::AbstractArray, rs::Tuple) = view(A, inner_axes(A, rs)...)

# Handle a Neighborhood or Int for radius in all (un)pad methods
for f in (:outer_axes, :inner_axes, :outer_array, :inner_array, :inner_view)
    @eval begin
        $f(A, hood::Neighborhood{R}; kw...) where R = $f(A, R; kw...)
        $f(A, radius::Int; kw...) = $f(A, _radii(A, radius); kw...)
    end
end

function _checkpad(A, rs)
    o_pad = map(a -> -(first(a) - 1), axes(A)) 
    r_pad = map(first, rs)
    o_pad == r_pad || throw(ArgumentError("OffsetArray padding $opad does not match radii padding $r_pad"))
    return nothing
end

_radii(A::AbstractArray{<:Any,N}, r) where N = ntuple(_ -> (r, r), N)


neighbor_getindex(A::NeighborhoodArray, I...) =
    neighbor_getindex(A, boundaary_condition(A), padding(A), I...)
# If Padded we can just use regular `getindex`
# on the parent array, which is an `OffsetArray`
neighbor_getindex(A::AbstractNeighborhoodArray, ::BoundaryCondition, ::Padded, I::Int...) = parent(A)[I...]
# Unpadded needs handling. For Wrap we swap the side:
function neighbor_getindex(A::AbstractNeighborhoodArray{S}, ::Wrap, ::Unpadded, I::Int...) where S
    sz = tuple_contents(S)
    wrapped_inds = map(I, sz) do i, s
        i < 1 ? i + s : (i > s ? i - s : i)
    end
    return @inbounds A[wrapped_inds...]
end
# For Remove we use padval if out of bounds
function neighbor_getindex(A::AbstractNeighborhoodArray, x::Remove, ::Unpadded, I::Int...)
    return checkbounds(Bool, A, I...) ? x.padval : (@inbounds A[I...])
end
