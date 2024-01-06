abstract type AbstractKernelStencil{R,N,L,T,H} <: Stencil{R,N,L,T} end

neighbors(hood::AbstractKernelStencil) = neighbors(stencil(hood))
offsets(::Type{<:AbstractKernelStencil{<:Any,<:Any,<:Any,<:Any,H}}) where H = offsets(H)
positions(hood::AbstractKernelStencil, I::Tuple) = positions(stencil(hood), I)

"""
    kernel(hood::AbstractKernelStencil) => iterable

Returns the kernel object, an array or iterable matching the length
of the stencil.
"""
function kernel end
kernel(hood::AbstractKernelStencil) = hood.kernel

"""
    stencil(x) -> Stencil

Returns a stencil object.
"""
function stencil end
stencil(hood::AbstractKernelStencil) = hood.stencil

"""
    kernelproduct(hood::AbstractKernelStencil)
    kernelproduct(hood::Stencil, kernel)

Take the vector dot produce of the stencil and the kernel,
without recursion into the values of either. Essentially `Base.dot`
without recursive calls on the contents, as these are rarely what is
intended.
"""
function kernelproduct(hood::AbstractKernelStencil)
    kernelproduct(stencil(hood), kernel(hood))
end
function kernelproduct(hood::Stencil{<:Any,<:Any,L}, kernel) where L
    sum = zero(first(hood))
    @simd for i in 1:L
        @inbounds sum += hood[i] * kernel[i]
    end
    return sum
end
function kernelproduct(hood::Window{<:Any,<:Any,L}, kernel) where L
    sum = zero(first(hood))
    @simd for i in 1:L
        @inbounds sum += neighbors(hood)[i] * kernel[i]
    end
    return sum
end

"""
    Kernel <: AbstractKernelStencil

    Kernel(stencil::Stencil, kernel::AbstractArray)
    Kernel(f::Function, stencil::Stencil)

Wrap any other stencil object, and includes a kernel array of
the same length and positions as the stencil. A function of
the stencil and kernel, like [`kernelproduct`](@ref) can be used in 
[`mapstencil`](@ref).

A function `f` may be passed as the first argument,
and a kernel array will be calculated with `map(f, distances(stencil))`.
"""
struct Kernel{R,N,L,T,F,H,K} <: AbstractKernelStencil{R,N,L,T,H}
    f::F
    stencil::H
    kernel::K
end
function Kernel{R,N,L}(f::F, hood::H, kernel) where {R,N,L,F,H<:Stencil{R,N,L}}
    k1 = if F <: Function 
        map(f, distances(hood))
    else
        kernel
    end
    length(hood) == length(kernel) || _kernel_length_error(hood, kernel)
    return Kernel{R,N,L,eltype(k1),F,H,typeof(k1)}(f, hood, k1)
end
function Kernel(f::Union{Function,Nothing}, hood::H, kernel=map(_ -> nothing, hood)) where {H<:Stencil{R,N,L}} where {R,N,L}
    Kernel{R,N,L}(f, hood, kernel)
end
Kernel(hood::Stencil, kernel::AbstractArray) = Kernel(nothing, hood, kernel)
Kernel(A::AbstractArray) = Kernel(Window{first(size(A)) ÷ 2,ndims(A)}(), A)
Kernel(A::StaticArray) = Kernel(Window{first(size(A)) ÷ 2,ndims(A)}(), A)

function _kernel_length_error(hood, kernel)
    throw(ArgumentError("Stencil length $(length(hood)) does not match kernel length $(length(kernel))"))
end

# We *dont* want `rebuild` to trigger kernel function rebuilds
# so we can update the neighborhood with no runtime cost
function rebuild(n::Kernel{R,N,L}, neighbors) where {R,N,L}
    hood = rebuild(stencil(n), neighbors)
    return Kernel{R,N,L,eltype(hood),typeof(n.f),typeof(hood),typeof(kernel(n))}(n.f, hood, kernel(n))
end

# We *do* want ConstructionBase to trigger kernel function rebuilds
# So e.g. ModelParameters.jl updates will actually do something
# and we can optimise parameters stored in a functor or anonymous function
ConstructionBase.constructorof(::Type{<:Kernel}) = Kernel

# In case the kernel is an Array, we need to `adapt` it for GPUs
function Adapt.adapt_structure(to, s::Kernel{R,N,L,T,F,H,K}) where {R,N,L,T,F,H,K}
    newstencil = Adapt.adapt(to, stencil(s))
    newkernel = Adapt.adapt(to, s.kernel)
    return Kernel{R,N,L,T,F,typeof(newstencil),typeof(newkernel)}(s.f, newstencil, newkernel)
end
