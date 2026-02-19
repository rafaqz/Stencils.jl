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
    f::F, source::AbstractStencilArray{<:Any,T,N}, args::AbstractArray...
) where {F,T,N}
    _checksizes((source, args...))
    T_return = _return_type(f, source, args...)
    dest = similar(parent(source), T_return, size(source))
    return mapstencil!(f, dest, source, args...)
end
function mapstencil(
    f::F, hood::StencilOrLayered, A::AbstractArray, args::AbstractArray...; 
    kw...
) where F
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

function _return_type(f::F, A::AbstractStencilArray{<:Any,T}, args...) where {F,T}
    arg_types = map(_arg_return_type, (A, args...))
    return Base._return_type(f, Tuple{arg_types...})
end

# For StencilArrays in args, return the stencil type; otherwise return eltype
function _arg_return_type(A::AbstractStencilArray{<:Any,T}) where T
    st = stencil(A)
    bc = boundary(A)
    T1 = bc isa Remove ? promote_type(T, typeof(padval(bc))) : T
    emptyneighbors = _zero_values(T1, st)
    emptycenter = _zero_center(T1, st)

    return typeof(rebuild(st, emptyneighbors, emptycenter))
end
_arg_return_type(A::AbstractArray) = eltype(A)

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
function mapstencil!(f::F, A::SwitchingStencilArray, args::AbstractArray...) where F
    pd = padding(A) isa Halo ? Halo{:in}() : padding(A)
    src = StencilArray(source(A), stencil(A), boundary(A), pd)
    dst = StencilArray(dest(A), stencil(A), boundary(A), pd)
    mapstencil!(f, dst, src, args...)
    return switch(A)
end
function mapstencil!(f::F, A::SwitchingStencilArray, B::AbstractStencilArray, args::AbstractArray...) where F
    dst = StencilArray(dest(A), stencil(A), boundary(A), padding(A))
    mapstencil!(f, dst, A, B, args...)
    return switch(A)
end
function mapstencil!(
    f::F, dest, source::AbstractStencilArray, args::AbstractArray...
) where F
    _checksizes((dest, source, args...))
    update_boundary!(source)
    # Update boundaries for any StencilArray args
    foreach(a -> a isa AbstractStencilArray && update_boundary!(a), args)

    backend = KernelAbstractions.get_backend(parent(source))
    kernel! = mapstencil_kernel!(backend)
    kernel!(f, dest, source, args; ndrange=size(dest))
    KernelAbstractions.synchronize(backend)

    return dest
end

@kernel function mapstencil_kernel!(f::F, dest, source, args) where F
    I = @index(Global, Cartesian)
    @inbounds dest[I] = f(stencil(source, I), _getarg.(args, (I,))...)
    nothing
end

# Get stencil for AbstractStencilArray, otherwise just index
@inline _getarg(a::AbstractStencilArray, I) = stencil(a, I)
@inline _getarg(a::AbstractArray, I) = a[I]

Base.@propagate_inbounds _arg_getindex(I, arg1, args...) = (arg1[I], _arg_getindex(I, args...)...)
Base.@propagate_inbounds _arg_getindex(I) = ()

function _checksizes(sources::Tuple)
    s1 = first(sources)
    map(sources) do s
        size(s) === size(s1) || throw(ArgumentError("Source array sizes must match. Found: $(size(s1)) and $(size(s))"))
    end
    return nothing
end

"""
    scatterstencil!(f, op, dest, source::AbstractStencilArray, args...)

Outward stencil operation where each cell scatters values to its neighbors.

Unlike `mapstencil!` which gathers from neighbors to compute a single output value,
`scatterstencil!` computes values that are scattered (written) to neighbor cells.

# Arguments
- `f`: Function `f(stencil, args...) -> SVector` returning values for each neighbor position
- `op`: Reduction operation (e.g., `+`, `max`) applied when multiple cells write to same location
- `dest`: Destination array (modified in place)
- `source`: Source `AbstractStencilArray`
- `args...`: Additional arrays passed to `f`

# Thread Safety
Lock-free by processing non-adjacent columns in separate passes.

# Example
```julia
# Scatter water to downhill neighbors based on elevation difference
sa = StencilArray(water, Moore(1))
dem_sa = StencilArray(dem, Moore(1))

scatterstencil!(+, dest, sa, dem_sa) do water_hood, dem_hood
    c_water = center(water_hood)
    c_dem = center(dem_hood)
    # Return outflow to each neighbor
    map(neighbors(dem_hood)) do n_dem
        Δz = c_dem - n_dem
        Δz > 0 ? c_water * Δz * 0.1 : zero(c_water)
    end
end
```
"""
function scatterstencil!(
    f::F, op::OP, dest::AbstractArray, source::AbstractStencilArray, args::AbstractArray...
) where {F,OP}
    _checksizes((dest, source, args...))
    update_boundary!(source)
    foreach(a -> a isa AbstractStencilArray && update_boundary!(a), args)

    _scatterstencil_cpu!(f, op, dest, source, args)
    return dest
end

# CPU implementation - lock-free by processing non-adjacent columns
# 2D version
function _scatterstencil_cpu!(
    f::F, op::OP, dest::AbstractArray{T,2}, source::AbstractStencilArray{R,T,2}, args::Tuple
) where {F,OP,T,R}
    ny, nx = size(source)
    stride = 2R + 1  # columns this far apart can't overlap
    bc = boundary(source)

    # Process in passes: columns 1, 1+stride, 1+2*stride, ... then 2, 2+stride, ...
    for offset in 1:stride
        Threads.@threads :static for j in offset:stride:nx
            for i in 1:ny
                hood = stencil(source, i, j)
                arg_vals = map(a -> _getarg(a, CartesianIndex(i, j)), args)
                scatter_vals = f(hood, arg_vals...)

                offs = offsets(hood)
                for (k, val) in enumerate(scatter_vals)
                    ni, nj = i + offs[k][1], j + offs[k][2]
                    _scatter_to!(op, dest, ni, nj, ny, nx, val, bc)
                end
            end
        end
    end
    return dest
end

# Handle different boundary conditions for scatter target
@inline function _scatter_to!(op, dest, ni, nj, ny, nx, val, ::Wrap)
    # Wrap indices to valid range
    ni = mod1(ni, ny)
    nj = mod1(nj, nx)
    @inbounds dest[ni, nj] = op(dest[ni, nj], val)
end

@inline function _scatter_to!(op, dest, ni, nj, ny, nx, val, ::Remove)
    # Skip out-of-bounds
    if 1 <= ni <= ny && 1 <= nj <= nx
        @inbounds dest[ni, nj] = op(dest[ni, nj], val)
    end
end

@inline function _scatter_to!(op, dest, ni, nj, ny, nx, val, ::Reflect)
    # Reflect indices
    ni = _reflect_index(ni, ny)
    nj = _reflect_index(nj, nx)
    @inbounds dest[ni, nj] = op(dest[ni, nj], val)
end

@inline function _reflect_index(i, n)
    if i < 1
        return 2 - i
    elseif i > n
        return 2n - i
    else
        return i
    end
end

@inline function _scatter_to!(op, dest, ni, nj, ny, nx, val, ::Use)
    # Skip out-of-bounds (same as Remove for scatter)
    if 1 <= ni <= ny && 1 <= nj <= nx
        @inbounds dest[ni, nj] = op(dest[ni, nj], val)
    end
end

# SwitchingStencilArray version
function scatterstencil!(f::F, op::OP, A::SwitchingStencilArray, args::AbstractArray...) where {F,OP}
    pd = padding(A) isa Halo ? Halo{:in}() : padding(A)
    src = StencilArray(source(A), stencil(A), boundary(A), pd)
    dst_arr = dest(A)
    # Clear destination before accumulating
    fill!(dst_arr, zero(eltype(dst_arr)))
    scatterstencil!(f, op, dst_arr, src, args...)
    return switch(A)
end
