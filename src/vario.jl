#
# vario.jl
#
# calculate an experimental variogram
#

function dist(a, b)
    # cartesian distance between points a,b
    return sqrt( sum((a - b).^2) )
end

function compute_distances(A)
    # calculate a compact distance matrix for vectors in *A*
    n = size(A,1)
    distmat = zeros(sum(1:n-1))
    cnt = 1
    for i = 1:n
        for j = i+1:n
            distmat[cnt] = dist(A[i,:], A[j,:])
            cnt = cnt + 1
        end
    end
    return distmat
end

function zdiffmat(Z)
    # return a compact matrix of z-differences
    n = length(Z)
    zmat = zeros(sum(1:n-1))
    cnt = 1
    for i = 1:n
        for j = i+1:n
            zmat[cnt] = Z[j] - Z[i]
            cnt = cnt + 1
        end
    end
    return zmat
end

function expvario(X, Z, interval::Number, maxdist::Number)
    # calculate an experimental variogram for scalar data *Z* at locations *X*

    distmat = compute_distances(X)
    zmat = zdiffmat(Z)
    lags = interval:interval:maxdist

    G = zeros(length(lags))

    for i=1:length(lags)

        minh = lags[i] - interval
        maxh = lags[i]
        zbin = zmat[minh .< distmat .<= maxh]
        var_ = (a) -> length(a) > 0 ? var(a) : nan(1.0)
        G[i] = 0.5 * var_(zbin)

    end

    return cat(2, lags, G)
end

