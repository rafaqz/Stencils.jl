@stencil Moore """
Moore stencils define the stencil as all cells within a horizontal or
vertical distance of the central cell. The central cell is omitted.
"""
@generated function offsets(::Type{<:Moore{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    L = (2R + 1)^N - 1
    # First half
    for I in CartesianIndices(ntuple(_-> -R:R, N))[1:L÷2]
        push!(offsets_expr.args, :($(Tuple(I))))
    end
    # Skip the middle position
    # Second half
    for I in CartesianIndices(ntuple(_-> -R:R, N))[L÷2+2:L+1]
        push!(offsets_expr.args, :($(Tuple(I))))
    end
    return :(SVector($offsets_expr))
end
