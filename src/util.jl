#
# util.jl
#
# utility functions for kriging
#

function dist(a::Union(Number, Array), b::Union(Number, Array))
    # cartesian distance between points a,b
    return sqrt( sum((a - b).^2) )
end

function augment_lm(A::Array)
    # augment *A* by adding one more row and column for a lagrange multiplier
    A = vcat(A, ones(eltype(A), 1, size(A,2)))
    A = hcat(A, ones(eltype(A), size(A,1), 1))
    A[end, end] = 0.0
    return A
end

