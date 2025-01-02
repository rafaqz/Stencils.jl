@stencil VonNeumann """
Diamond-shaped neighborhood (in 2 dimensions), without the central cell
In 1 dimension it is identical to [`Moore`](@ref).
"""
@generated function offsets(::Type{<:VonNeumann{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    rngs = ntuple(_ -> -R:R, N)
    for I in CartesianIndices(rngs)
        manhatten_distance = sum(map(abs, Tuple(I)))
        if manhatten_distance in 1:R
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end
