@stencil Window "A neighboorhood of radius R that includes the central cell."

Window(A::AbstractArray) = Window{(size(A, 1) - 1) ÷ 2,ndims(A)}()

# The central cell is included
@generated function offsets(::Type{<:Window{R,N}}) where {R,N}
    D = 2R + 1
    offsets = ntuple(i -> (rem(i - 1, D) - R, (i - 1) ÷ D - R), D^N)
    return :(SVector($offsets))
end
