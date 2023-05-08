"""
    AbstractKernelStencil <: Stencil

Abstract supertype for kernel stencils.

These can wrap any other stencil object, and include a kernel of
the same length and positions as the stencil.
"""
abstract type AbstractKernelStencil{R,N,L,H} <: Stencil{R,N,L} end

neighbors(hood::AbstractKernelStencil) = neighbors(stencil(hood))
offsets(::Type{<:AbstractKernelStencil{<:Any,<:Any,<:Any,H}}) where H = offsets(H)
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
struct Kernel{R,N,L,H,K} <: AbstractKernelStencil{R,N,L,H}
    stencil::H
    kernel::K
end
Kernel(A::AbstractMatrix) = Kernel(Window(A), A)
function Kernel(hood::H, kernel::K) where {H<:Stencil{R,N,L},K} where {R,N,L}
    length(hood) == length(kernel) || _kernel_length_error(hood, kernel)
    Kernel{R,N,L,H,K}(hood, kernel)
end
function Kernel{R,N,L}(hood::H, kernel::K) where {R,N,L,H<:Stencil{R,N,L},K}
    Kernel{R,N,L,H,K}(hood, kernel)
end

function _kernel_length_error(hood, kernel)
    throw(ArgumentError("Stencil length $(length(hood)) does not match kernel length $(length(kernel))"))
end

function setneighbors(n::Kernel{R,N,L,<:Any,K}, _neighbors) where {R,N,L,K}
    hood = setneighbors(stencil(n), _neighbors)
    return Kernel{R,N,L,typeof(hood),K}(hood, kernel(n))
end

