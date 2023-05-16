
_cross_length(R, N) = 2R * N + 1

@stencil Cross _cross_length "A cross-shaped neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"

# The central cell is included
@generated function offsets(::Type{<:Cross{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        if sum(map(==(0), Tuple(I))) >= (N-1)
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end


@stencil AngledCross _cross_length "A neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"

# The central cell is included
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


_slash_length(R, N) = 2R + 1

@stencil ForwardSlash _slash_length "A neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"

# The central cell is included
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

@stencil BackSlash _slash_length "A neighboorhood where offsets of zero on at least N-1 axes are included in the neighborhoods"

# The central cell is included
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

_incircle(I, R) = sqrt(sum(map(x -> x^2, Tuple(I)))) < R - 0.5

function _circle_length(R, N)
    count(CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))) do I
        _incircle(I, R)
    end
end

@stencil Circle _circle_length "A circular stencil"

# The central cell is included
@generated function offsets(::Type{<:Circle{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        # If the center of the pixel is inside the radius
        if _incircle(I, R)
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end


_bar_length(R, N) = 2R + 1

@stencil Vertical _bar_length "A circular stencil"

# The central cell is included
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

@stencil Horizontal _bar_length "A circular stencil"


# The central cell is included
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
