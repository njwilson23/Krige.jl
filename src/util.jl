#
# util.jl
#
# utility functions for kriging
#

# cartesian distance between points a,b
function dist(a::Union(Number, Array), b::Union(Number, Array))
    return sqrt( sum((a - b).^2) )
end

# geostatistical (not pearson!) semivariance of a population
function semivar(a::Array)
    n = length(a)
    if n == 0
        g = nan(1.0)
    else
        g = (a' * a)[1] / (2n)
    end
    return g
end

# geostatistical (not pearson!) covariance between two vectors
function cov_geo(a::Array, b::Array)
    if size(a) == size(b)
        error("incompatible matrices")
    end
    n = size(a)[1]
    c = (a - mean(a))' * (b - mean(b)) / n
    return c
end

# augment *A* by adding one more row and column for a lagrange multiplier
function augment_lm(A::Array)
    n, m = size(A)
    Am = ones(n+1, n+1)
    Am[1:n,1:n] = A
    Am[n+1,n+1] = 0.0
    return Am
end

