_moore_length(R, N) = (2R + 1)^N - 1

@stencil Moore _moore_length """
Moore stencils define the stencil as all cells within a horizontal or
vertical distance of the central cell. The central cell is omitted.
"""

@generated function offsets(::Type{<:Moore{R,N,L}}) where {R,N,L}
    offsets_expr = Expr(:tuple)
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
