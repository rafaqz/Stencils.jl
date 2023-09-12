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

    Kernel(stencil, kernel)

Wrap any other stencil object, and includes a kernel of
the same length and positions as the stencil.
"""
struct Kernel{R,N,L,T,H,K} <: AbstractKernelStencil{R,N,L,T,H}
    stencil::H
    kernel::K
end
Kernel(A::AbstractArray) = Kernel(Window(A), A)
Kernel(A::StaticArray) = Kernel(Window(A), A)
function Kernel(hood::H, kernel::K) where {H<:Stencil{R,N,L,T},K} where {R,N,L,T}
    length(hood) == length(kernel) || _kernel_length_error(hood, kernel)
    Kernel{R,N,L,T,H,K}(hood, kernel)
end
function Kernel{R,N,L,T}(hood::H, kernel::K) where {R,N,L,H<:Stencil{R,N,L,T},K} where T
    Kernel{R,N,L,T,H,K}(hood, kernel)
end

function _kernel_length_error(hood, kernel)
    throw(ArgumentError("Stencil length $(length(hood)) does not match kernel length $(length(kernel))"))
end

function rebuild(n::Kernel{R,N,L,<:Any,<:Any,K}, neighbors) where {R,N,L,K}
    hood = rebuild(stencil(n), neighbors)
    return Kernel{R,N,L,eltype(hood),typeof(hood),K}(hood, kernel(n))
end

