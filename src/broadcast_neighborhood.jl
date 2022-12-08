"""
    broadcast_neighborhood(f, hood::Neighborhood, args...)

Simple neighborhood application, where `f` is passed
each neighborhood in `A`, returning a new array.

The result is smaller than `A` on all sides, by the neighborhood radius.
"""
function broadcast_neighborhood(f, source::AbstractNeighborhoodArray{<:Any,<:Any,T,N}, args::AbstractArray...) where {T,N}
    _checksizes((source, args...))
    # Get the type of the neighborhood
    bc = boundary_condition(source)
    T1 = bc isa Remove ? promote_type(T, typeof(padval(bc))) : T
    L = length(neighborhood(source))
    emptyneighbors = SVector{L,T}(ntuple(_ -> zero(T), L))
    H = typeof(setneighbors(neighborhood(source), emptyneighbors))
    # Use nasty broadcast mechanism `_return_type` to get the new eltype
    T_return = Base._return_type(f, Tuple{H,map(eltype, args)...})
    dest = similar(parent(source), T_return, size(source))
    broadcast_neighborhood!(f, dest, source, args...)
end
broadcast_neighborhood(f, hood::Neighborhood, A::AbstractArray, args::AbstractArray...; kw...) =
    broadcast_neighborhood(f, NeighborhoodArray(A, hood; kw...), args...)

_emptyhood(x) = _emptyhood(x, neighborhood(x))
function _emptyhood(x::AbstractArray{T}, hood::Neighborhood{<:Any,<:Any,L}) where {T,L} 
    z = zero(T)
    return SVector(ntuple(_ -> z, Val{L}()))
end

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
    # update_boundary!(source)
    device = KernelAbstractions.get_device(parent(source))
    n = device isa GPU ? 64 : 4
    if length(args) == 0
        kernel! = broadcast_kerneln0!(device, n)
    elseif length(args) == 1
        kernel! = broadcast_kerneln1!(device, n)
    elseif length(args) == 2
        kernel! = broadcast_kerneln2!(device, n)
    elseif length(args) == 3
        kernel! = broadcast_kerneln3!(device, n)
    elseif length(args) == 4
        kernel! = broadcast_kerneln4!(device, n)
    end
    kernel!(f, dest, source, args...; ndrange=size(dest)) |> wait
    return dest
end
@kernel function broadcast_kerneln0!(f, dest, source)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(neighborhood(source, I))
    nothing
end
@kernel function broadcast_kerneln1!(f, dest, source, a1)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(neighborhood(source, I), a1[I])
    nothing
end
@kernel function broadcast_kerneln2!(f, dest, source, a1, a2)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(neighborhood(source, I), a1[I], a2[I])
    nothing
end
@kernel function broadcast_kerneln3!(f, dest, source, a1, a2, a3)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(neighborhood(source, I), a1[I], a2[I], a3[I])
    nothing
end
@kernel function broadcast_kerneln4!(f, dest, source, a1, a2, a3, a4)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(neighborhood(source, I), a1[I], a2[I], a3[I], a4[I])
    nothing
end

Base.@propagate_inbounds _arg_getindex(I, arg1, args...) = (arg1[I], _arg_getindex(I, args...)...)
Base.@propagate_inbounds _arg_getindex(I) = ()

function _checksizes(sources::Tuple)
    s1 = first(sources)
    map(sources) do s
        size(s) === size(s1) || throw(ArgumentError("Source array sizes must match. Found: $(size(s1)) and $(size(s))"))
    end
    return nothing
end

function applyneighborhood(f, hood, sources::Tuple, I)
    hoods = map(s -> neighborhood(s, I), sources)
    vals = map(s -> s[I], sources)
    f(hoods)
end
function applyneighborhood(f, hood, source::AbstractArray, I)
    f(neighborhood(source, I))
end
