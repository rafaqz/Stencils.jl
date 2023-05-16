_window_length(R, N) = (2R + 1)^N

@stencil Window _window_length "A neighboorhood of radius R that includes the central cell."

# The central cell is included
@generated function offsets(::Type{<:Window{R,N}}) where {R,N}
    D = 2R + 1
    offsets = ntuple(i -> (rem(i - 1, D) - R, (i - 1) ÷ D - R), D^N)
    return :(SVector($offsets))
end
