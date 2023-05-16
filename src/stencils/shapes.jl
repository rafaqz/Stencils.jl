@stencil Cross "A cross-shaped neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"
@generated function offsets(::Type{<:Cross{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        if sum(map(==(0), Tuple(I))) >= (N-1)
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil AngledCross "A neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"
@generated function offsets(::Type{<:AngledCross{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        t = Tuple(I)
        matches = map(Base.tail(t)) do x
            abs(x) == abs(first(t))
        end
        if sum(matches) == (N - 1)
           push!(offsets_expr.args, t)
        end
    end
    return :(SVector($offsets_expr))
end

@stencil ForwardSlash "A neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"
@generated function offsets(::Type{<:ForwardSlash{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        t = Tuple(I)
        matches = map(Base.tail(t)) do x
            x == -first(t)
        end
        if sum(matches) == (N - 1)
           push!(offsets_expr.args, t)
        end
    end
    return :(SVector($offsets_expr))
end

@stencil BackSlash "A neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"
@generated function offsets(::Type{<:BackSlash{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        t = Tuple(I)
        matches = map(Base.tail(t)) do x
            x == first(t)
        end
        if sum(matches) == (N - 1)
           push!(offsets_expr.args, t)
        end
    end
    return :(SVector($offsets_expr))
end

@stencil Circle "A circular stencil"
@generated function offsets(::Type{<:Circle{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        # If the center of the pixel is inside the radius
        if sqrt(sum(map(x -> x^2, Tuple(I)))) < R - 0.5
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil Vertical "A vertical bar"
@generated function offsets(::Type{<:Vertical{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        t = Tuple(I)
        if length(t) > 1 && I.I[2] == 0 || length(t) == 1 && t[1] == 0
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil Horizontal "A horizontal bar"
@generated function offsets(::Type{<:Horizontal{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        t = Tuple(I)
        if length(t) > 1 && I.I[1] == 0
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end
