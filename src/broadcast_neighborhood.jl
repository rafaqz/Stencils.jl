"""
    broadcast_neighborhood(f, hood::Neighborhood, args...)

Simple neighborhood application, where `f` is passed
each neighborhood in `A`, returning a new array.

The result is smaller than `A` on all sides, by the neighborhood radius.
"""
function broadcast_neighborhood(f, source::AbstractNeighborhoodArray{<:Any,<:Any,T,N}, args::AbstractArray...) where {T,N}
    _checksizes((source, args...))
    # Get the type of the neighborhood
    emptyhood = SVector(ntuple(_ -> zero(T), length(neighborhood(source))))
    H = typeof(setneighbors(neighborhood(source), emptyhood))
    # Use nasty broadcast mechanism `_return_type` to get the new eltype
    T_return = Base._return_type(f, Tuple{H})#,map(eltype, args)...})
    dest = similar(parent(source), T_return, size(source))
    broadcast_neighborhood!(f, dest, source, args...)
end
# broadcast_neighborhood(f, A::AbstractArray, args::AbstractArray...; kw...) =
#     broadcast_neighborhood(f, NeighborhoodArray(A; kw...), args...; kw...)

kernel_setup() = KernelAbstractions.CPU(), 1

"""
    broadcast_neighborhood!(f, hood::Neighborhood{R}, dest, sources...)

Simple neighborhood broadcast where `f` is passed each neighborhood
of `src` (except padding), writing the result of `f` to `dest`.

`dest` must either be smaller than `src` by the neighborhood radius on all
sides, or be the same size, in which case it is assumed to also be padded.
"""
function broadcast_neighborhood!(f, dest, source::NeighborhoodArray, args::AbstractArray...)
    _checksizes((dest, source, args...))
    update_boundary!(source)
    device = KernelAbstractions.get_device(parent(source))
    n = device isa GPU ? 64 : 4
    broadcast_kernel!(device, n)(f, dest, source, args...; ndrange=size(dest)) |> wait
    return dest
end

@kernel function broadcast_kernel!(f, dest, source)#, args...)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(neighborhood(source, I))#, _maybe_arg_getindex(I, args...)...)
    nothing
end

_maybe_arg_getindex(I, arg1, args...) = (arg1[I], _maybe_arg_getindex(I, args...)...)
_maybe_arg_getindex(I) = ()

function _checksizes(sources::Tuple)
    s1 = first(sources)
    map(sources) do s
        size(s) === size(s1) || throw(ArgumentError("Source array sizes must match. Found: $(size(s1)) and $(size(s))"))
    end
    return nothing
end

# function applyneighborhood(f, hood, sources::Tuple, I)
#     hoods = map(s -> unsafe_updateneighbors(hood, s, I), sources)
#     vals = map(s -> s[I], sources)
#     f(hoods)
# end
# function applyneighborhood(f, hood, source::AbstractArray, I, xs...)
#     f(unsafe_updateneighbors(hood, source, I))
# end
