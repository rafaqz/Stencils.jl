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

# Disambiguating method when SwitchingStencilArray is passed with an explicit source
function scatterstencil!(
    f::F, op::OP, A::SwitchingStencilArray, source::AbstractStencilArray, args::AbstractArray...
) where {F,OP}
    dst_arr = dest(A)
    fill!(dst_arr, zero(eltype(dst_arr)))
    scatterstencil!(f, op, dst_arr, source, args...)
    return switch(A)
end

# SwitchingStencilArray version (uses internal source/dest arrays)
function scatterstencil!(f::F, op::OP, A::SwitchingStencilArray, args::AbstractArray...) where {F,OP}
    pd = padding(A) isa Halo ? Halo{:in}() : padding(A)
    src = StencilArray(source(A), stencil(A), boundary(A), pd)
    dst_arr = dest(A)
    # Clear destination before accumulating
    fill!(dst_arr, zero(eltype(dst_arr)))
    scatterstencil!(f, op, dst_arr, src, args...)
    return switch(A)
end
