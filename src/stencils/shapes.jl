@stencil Cross "A cross-shaped neighboorhood where positions with offsets of `0` on least `N-1` axes are included"
@generated function offsets(::Type{<:Cross{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
        if sum(map(==(0), Tuple(I))) >= (N-1)
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil AngledCross "A neighboorhood where all diagonals are included, for `2:N` dimensions"
@generated function offsets(::Type{<:AngledCross{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
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
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
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
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
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
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
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
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
        t = Tuple(I)
        if length(t) > 1 && t[2] == 0 || length(t) == 1 && t[1] == 0
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil Horizontal "A horizontal bar or plane"
@generated function offsets(::Type{<:Horizontal{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
        t = Tuple(I)
        if length(t) > 1 && t[1] == 0
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

# Custom macroexpand of the `@stencil` because of two radii
"""
Annulus <: Stencil

Annulus(; outer_radius=2, inner_radius=1, ndims=2)
Annulus(outer_radius, inner_radius, ndims)
Annulus{RO,RI,N}()

A donut or hollowed spherical stencil
"""
struct Annulus{RO,RI,N,L,T} <: Stencil{RO,N,L,T}  # Inner radius is ignored for Stencil type
    neighbors::SVector{L,T}
    Annulus{RO,RI,N,L,T}(neighbors::StaticVector{L,T}) where {RO,RI,N,L,T} = new{RO,RI,N,L,T}(neighbors)
end
Annulus{RO,RI,N,L}(neighbors::StaticVector{L,T}) where {RO,RI,N,L,T} = Annulus{RO,RI,N,L,T}(neighbors)
Annulus{RO,RI,N,L}() where {RO,RI,N,L} = Annulus{RO,RI,N,L}(SVector(ntuple(_ -> nothing, L)))
function Annulus{RO,RI,N}(args::StaticVector...) where {RO,RI,N}
    L = length(offsets(Annulus{RO,RI,N}))
    Annulus{RO,RI,N,L}(args...)
end
Annulus{RO}(args::StaticVector...) where {RO} = Annulus{RO,RO - 1,2}(args...)
Annulus{RO,RI}(args::StaticVector...) where {RO,RI} = Annulus{RO,RI,2}(args...)
Annulus(args::StaticVector...; outer_radius=2, inner_radius=1, ndims=2) = Annulus{outer_radius,inner_radius,ndims}(args...)
Annulus(outer_radius::Int, inner_radius::Int=outer_radius - 1, ndims::Int=2) = Annulus{outer_radius,inner_radius,ndims}()

@inline Stencils.rebuild(n::Annulus{RO,RI,N,L}, neighbors) where {RO,RI,N,L} = Annulus{RO,RI,N,L}(neighbors)

@generated function offsets(::Type{<:Annulus{RO,RI,N}}) where {RO,RI,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> -RO:RO, N))
        # If the center of the pixel is inside the radius
        dist = sqrt(sum(map(x -> x^2, Tuple(I))))
        if dist < RO + 0.5 && dist >= RI + 0.5
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil Cardinal "Cardinal (as in N,S,W,E compass directions) stencil"
@generated function offsets(::Type{<:Cardinal{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
        CI = abs.(Tuple(I))
        if sum(CI) == R && maximum(CI) == R
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end

@stencil Ordinal "Ordinal (as in NE,SE,SW,NW wind directions) stencil"
@generated function offsets(::Type{<:Ordinal{R,N}}) where {R,N}
    offsets_expr = Expr(:tuple)
    for I in CartesianIndices(ntuple(_ -> -R:R, N))
        CI = abs.(Tuple(I))
        if sum(CI) == R * N && maximum(CI) == R
            push!(offsets_expr.args, Tuple(I))
        end
    end
    return :(SVector($offsets_expr))
end
