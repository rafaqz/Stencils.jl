@stencil Cross "A cross-shaped neighboorhood where positions with offsets of `0` on least `N-1` axes are included"
@generated function offsets(::Type{<:Cross{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        if sum(map(==(0), Tuple(I))) >= (N-1)
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil AngledCross "A neighboorhood where all diagonals are included, for `2:N` dimensions"
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

@stencil ForwardSlash "A neighboorhood where only 'forward' diagonals are included. Contains `2R+1` neighbors for `2:N` dimensions"
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

@stencil BackSlash "A neighboorhood along the 'backwards' diagonal. Contains `2R+1` neighbors for for `2:N` dimensions"
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

@stencil Circle "A circular or spherical stencil"
@generated function offsets(::Type{<:Circle{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> OffsetArrays.IdOffsetRange(-R:R), N))
        # If the center of the pixel is inside the radius
        if sqrt(sum(map(x -> x^2, Tuple(I)))) < R + 0.5
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil Vertical "A vertical bar or plane"
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

@stencil Horizontal "A horizontal bar or plane"
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

@stencil Diamond "A diamond or regular octahedron"
@generated function offsets(::Type{<:Diamond{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    rngs = ntuple(_ -> -R:R, N)
    for I in CartesianIndices(rngs)
        manhatten_distance = sum(map(abs, Tuple(I)))
        if manhatten_distance in 0:R
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end
