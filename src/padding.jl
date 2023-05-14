"""
    Padding

Abstract supertype for padding modes, e.g. 
[`Conditional`](@ref) and [`Halo`](@ref).
"""
abstract type Padding end

"""
    Halo{X} <: Padding

Padding that uses an in-memory halo around the array so that parts of a stencil
that go off the edge of the array can index directly into it without a bounds check
or any conditional. This has the benefit of possibly better performance during window
broadcasts, but some downsides.

In `:out` mode, a whole new array is alocated, larger than the original.
This may not be worth doing unless you are using it multiple times. with `:in`
mode, the outside edge of the array is used as padding. This may be more accurate 
as there are no boundary effects from using a padding value.:w

# Example
```julia
halo_in = Halo(:in)
halo_out = Halo(:out)
```
"""
struct Halo{X} <: Padding end
Halo() = Halo{:out}() # Default to :out
Halo(k::Symbol) = Halo{k}()

"""
    Conditional <: Padding

Padding that doesn't change the array size, but checks `getindex` for out-of-bounds
indexing, and inserts `padval` with [`Remove`](@ref) or values from the other side
of the array with [`Wrap`](@ref).
"""
struct Conditional <: Padding end


const RADIUSDOC = """
    `radius` can be a `Stencil`, an `Int`, or a tuple of tuples,
    e.g. for 2d it could be: `((1, 2), (2, 1))::Tuple{Tuple{Int,Int},Tuple{Int,Int}}`.
    """

"""
    outer_axes(A, hood::Stencil{R})
    outer_axes(A, radius::Int)

Add padding to axes of array `A`, returning a `Tuple` of `UnitRange`.
$RADIUSDOC
"""
function outer_axes(A, r)
    map(axes(A), _radii(A, r)) do axis, r
        firstindex(axis) - r[1]:lastindex(axis) + r[2]
    end
end

"""
    inner_axes(A, radius)

Remove padding of `radius` from axes of `A`, returning a `Tuple` of `UnitRange`.
$RADIUSDOC
"""
function inner_axes(A, r)
    ax = map(axes(A), _radii(A, r)) do axis, r
        (first(axis) + r[1]):(last(axis) - r[2])
    end
    return ax
end

function offset_axes(A, r)
    map(axes(A), _radii(A, r)) do axis, r
        (first(axis) - r[1]):(last(axis) - r[2])
    end
end

"""
    inner_view(A, radius)

Remove padding of `radius` from array `A`, returning a view of `A`.

$RADIUSDOC
"""
function inner_view(A, r) 
    rs = _radii(A, r)
    return inner_view(parent(A), rs)
end
# Unwrap the OffsetArray
inner_view(A::OffsetArray, rs::Tuple) = inner_view(parent(A), rs)
inner_view(A::AbstractArray, rs::Tuple) = view(A, inner_axes(A, rs)...)

"""
    outer_array(A, radius; [padval])

Add padding of `radius` to array `A`, redurning a new array.

$RADIUSDOC

`padval` defaults to `zero(eltype(A))`.
"""
pad_array(::Conditional, ::BoundaryCondition, hood::Stencil, parent::AbstractArray) = parent
function pad_array(::Halo{:out}, bc::BoundaryCondition, hood::Stencil, parent::AbstractArray{T}) where T
    rs = _radii(parent, hood)
    padded_axes = outer_axes(parent, rs)
    padded_parent = similar(parent, T, length.(padded_axes))
    inner_view(padded_parent, rs) .= parent
    padded_offset = OffsetArray(padded_parent, padded_axes)
    return padded_offset
end
function pad_array(::Halo{:in}, bc::BoundaryCondition, hood::Stencil, parent::AbstractArray)
    return OffsetArray(parent, offset_axes(parent, hood))
end
