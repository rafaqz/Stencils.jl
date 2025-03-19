"""
    mapstencil(f, A::StencilArray, args::AbstractArray...)
    mapstencil(f, stencil::Stencil, A::AbstractArray, args::AbstractArray...; kw...)

Stencil mapping where `f` is passed a [`Stencil`](@ref) centered at each
index in `A`, followed by the values from `args` at each stencil center.

## Keywords

$STENCILARRAY_KEYWORDS 

The result is returned as a new array.
"""
function mapstencil(
    f, source::AbstractStencilArray{<:Any,T,N}, args::AbstractArray...
) where {T,N}
    _checksizes((source, args...))
    T_return = _return_type(f, source, args...)
    dest = similar(parent(source), T_return, size(source))
    return mapstencil!(f, dest, source, args...)
end
function mapstencil(
    f, hood::StencilOrLayered, A::AbstractArray, args::AbstractArray...; 
    kw...
)
    sa = StencilArray(A, hood; kw...)
    T_return = _return_type(f, sa, args...)
    dest = if padding(sa) isa Halo{:in}
        # We need to shrink the dest array size
        I = map(axes(A)) do ax
            first(ax) + radius(hood):last(ax) - radius(hood)
        end
        # Take a view shrunk by the radius in case the array has specific axes
        similar(view(A, I...), T_return)
    else
        similar(A, T_return)
    end
    return mapstencil!(f, dest, sa, args...)
end

function _return_type(f, A::AbstractStencilArray{<:Any,T}, args...) where T
    st = stencil(A)
    bc = boundary(A)
    T1 = bc isa Remove ? promote_type(T, typeof(padval(bc))) : T
    emptyneighbors = _zero_values(T1, st)
    emptycenters = _zero_center(T1, st)
    H = typeof(rebuild(st, emptyneighbors, emptycenters))
    # Use nasty broadcast mechanism `_return_type` to get the new eltype
    return Base._return_type(f, Tuple{H,map(eltype, args)...})
end

_zero_values(::Type{T}, ::Stencil{<:Any,<:Any,L}) where {T,L} = SVector{L,T}(ntuple(_ -> zero(T), L))
_zero_center(::Type{T}, ::Stencil) where T = zero(T)

kernel_setup() = KernelAbstractions.CPU(; static=true), 64

"""
    mapstencil!(f, dest::AbstractArray, source::StencilArray, args::AbstractArray...)
    mapstencil!(f, A::SwitchingStencilArray, args::AbstractArray...)

Stencil mapping where `f` is passed a stencil centered at each index
in `src`, followed by the values from `args` at each stencil center.
The result of `f` is written to `dest`.

For SwitchingStencilArray the internal source and dest arrays are used,
returning a switched version of the array.

`dest` must either be smaller than `src` by the stencil radius on all
sides, or be the same size, in which case it is assumed to also be padded.
"""
function mapstencil!(f, A::SwitchingStencilArray, args::AbstractArray...)
    pd = padding(A) isa Halo ? Halo{:in}() : padding(A)
    src = StencilArray(source(A), stencil(A), boundary(A), pd)
    dst = StencilArray(dest(A), stencil(A), boundary(A), pd)
    mapstencil!(f, dst, src, args...)
    return switch(A)
end
function mapstencil!(f, A::SwitchingStencilArray, B::AbstractStencilArray, args::AbstractArray...)
    dst = StencilArray(dest(A), stencil(A), boundary(A), padding(A))
    mapstencil!(f, dst, A, B, args...)
    return switch(A)
end
function mapstencil!(
    f, dest, source::AbstractStencilArray, args::AbstractArray...
)
    _checksizes((dest, source, args...))
    update_boundary!(source)

    backend = KernelAbstractions.get_backend(parent(source))
    workgroups = 64 # 64 seems like a sweet spot for both CPU and GPU ?
    sz = size(source)

    # This is awful but the KernelAbstractions kernel 
    # doesn't seem to be type stable with splatted args.
    if length(args) == 0
        # We use a static kernel size. We have the size
        # in the type so we may as well use it.
        kernel! = mapstencil_kerneln0!(backend, workgroups, sz)
    elseif length(args) == 1
        kernel! = mapstencil_kerneln1!(backend, workgroups, sz)
    elseif length(args) == 2
        kernel! = mapstencil_kerneln2!(backend, workgroups, sz)
    elseif length(args) == 3
        kernel! = mapstencil_kerneln3!(backend, workgroups, sz)
    elseif length(args) == 4
        kernel! = mapstencil_kerneln4!(backend, workgroups, sz)
    end

    kernel!(f, dest, source, args...)

    KernelAbstractions.synchronize(backend)

    return dest
end

# TODO remove these if we can make this type stable with a single method
@kernel function mapstencil_kerneln0!(f, dest, source)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I))
    nothing
end
@kernel function mapstencil_kerneln1!(f, dest, source, a1)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), a1[I])
    nothing
end
@kernel function mapstencil_kerneln2!(f, dest, source, a1, a2)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), a1[I], a2[I])
    nothing
end
@kernel function mapstencil_kerneln3!(f, dest, source, a1, a2, a3)
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), a1[I], a2[I], a3[I])
    nothing
end
@kernel function mapstencil_kerneln4!(f, dest, source, a1, a2, a3, a4)
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

# TODO These methods have unused variables and don't seem to be used anywhere
function applystencil(f, hood, sources::Tuple, I)
    hoods = map(s -> stencil(s, I), sources)
    vals = map(s -> s[I], sources)
    f(hoods)
end
function applystencil(f, hood, source::AbstractArray, I)
    f(stencil(source, I))
end
