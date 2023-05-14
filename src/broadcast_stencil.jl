"""
    broadcast_stencil(f, hood::Stencil, args...)

Simple stencil application, where `f` is passed
each stencil in `A`, returning a new array.

The result is smaller than `A` on all sides, by the stencil radius.
"""
function broadcast_stencil(f, source::AbstractStencilArray{<:Any,<:Any,T,N}, args::AbstractArray...) where {T,N}
    _checksizes((source, args...))
    # Get the type of the stencil
    bc = boundary(source)
    T1 = bc isa Remove ? promote_type(T, typeof(padval(bc))) : T
    L = length(stencil(source))
    emptyneighbors = SVector{L,T}(ntuple(_ -> zero(T), L))
    H = typeof(setneighbors(stencil(source), emptyneighbors))
    # Use nasty broadcast mechanism `_return_type` to get the new eltype
    T_return = Base._return_type(f, Tuple{H,map(eltype, args)...})
    dest = similar(parent(source), T_return, size(source))
    broadcast_stencil!(f, dest, source, args...)
end
broadcast_stencil(f, hood::Stencil, A::AbstractArray, args::AbstractArray...; kw...) =
    broadcast_stencil(f, StencilArray(A, hood; kw...), args...)

_emptyhood(x) = _emptyhood(x, stencil(x))
function _emptyhood(x::AbstractArray{T}, hood::Stencil{<:Any,<:Any,L}) where {T,L} 
    z = zero(T)
    return SVector(ntuple(_ -> z, Val{L}()))
end

kernel_setup() = KernelAbstractions.CPU(), 1

"""
    broadcast_stencil!(f, dest, source::StencilArray, args...)
    broadcast_stencil!(f, A::SwitchingStencilArray, args...)

Stencil broadcast where `f` is passed each stencil
of `src`, writing the result of `f` to `dest`.

For SwitchingStencilArray the internal source and dest arrays are used,
returning a switched version of the array.

`dest` must either be smaller than `src` by the stencil radius on all
sides, or be the same size, in which case it is assumed to also be padded.
"""
function broadcast_stencil!(f, A::SwitchingStencilArray, args::AbstractArray...)
    broadcast_stencil!(f, dest(A), A, args...)
    return switch(A)
end
function broadcast_stencil!(f, A::SwitchingStencilArray, B::AbstractStencilArray, args::AbstractArray...)
    broadcast_stencil!(f, dest(A), A, B, args...)
    return switch(A)
end
function broadcast_stencil!(f, dest, source::AbstractStencilArray, args::AbstractArray...)
    _checksizes((dest, source, args...))
    update_boundary!(source)
    device = KernelAbstractions.get_device(parent(source))
    n = device isa GPU ? 64 : 4
    # This is awful but the KernelAbstractions kernel 
    # doesn't seem to be type stable without it.
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

# TODO remove these if we can make this type stable with a single method
@kernel function broadcast_kerneln0!(f, dest, source)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I))
    nothing
end
@kernel function broadcast_kerneln1!(f, dest, source, a1)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), a1[I])
    nothing
end
@kernel function broadcast_kerneln2!(f, dest, source, a1, a2)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), a1[I], a2[I])
    nothing
end
@kernel function broadcast_kerneln3!(f, dest, source, a1, a2, a3)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), a1[I], a2[I], a3[I])
    nothing
end
@kernel function broadcast_kerneln4!(f, dest, source, a1, a2, a3, a4)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), a1[I], a2[I], a3[I], a4[I])
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

function applystencil(f, hood, sources::Tuple, I)
    hoods = map(s -> stencil(s, I), sources)
    vals = map(s -> s[I], sources)
    f(hoods)
end
function applystencil(f, hood, source::AbstractArray, I)
    f(stencil(source, I))
end
