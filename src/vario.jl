#
# vario.jl
#
# calculate an experimental variogram
#

# calculate a compact distance matrix for vectors in *A*
function compute_distances(A)
    n = size(A,1)
    distmat = zeros(sum(1:n-1))
    cnt = 1
    for i = 1:n
        for j = i+1:n
            distmat[cnt] = norm(A[i,:] - A[j,:])
            cnt = cnt + 1
        end
    end
    return distmat
end

# return a compact matrix of z-differences
function zdiffmat(Z)
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

# calculate an experimental variogram for scalar data *Z* at locations *X*
function est_variogram(X, Z, interval::Number, maxdist::Number)

    distmat = compute_distances(X)
    zmat = zdiffmat(Z)
    lags = interval:interval:maxdist

    G = zeros(length(lags))
    for i=1:length(lags)
        minh = lags[i] - interval
        maxh = lags[i]
        zbin = zmat[minh .< distmat .<= maxh]
        G[i] = semivar(zbin)
    end

    return cat(2, lags, G)
end

