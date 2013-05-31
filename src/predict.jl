#
# predict.jl
#
# kriging prediction of geographical data
#

module Predict
export buildcovmet, ordinary_krig
#include("util.jl")
#include("model.jl")
using Util
using Model


function buildcovmat(M, X::Array{Float64,2})
    # assemble matrix of modelled variance
    n = length(X, 1)
    K = zeros(Float64, n, n)

    for i = 1:n
        for j = i:n
            K[i,j] = Model.evaluate(M, Util.dist(X[i,:], X[j,:]))
        end
    end

    K = K + triu(K,1)'
    return K
end

function buildcovmat(M, X::Array{Float64,1})
    # assemble matrix of modelled variance
    n = length(X)
    K = zeros(Float64, n, n)

    for i = 1:n
        for j = i:n
            K[i,j] = Model.evaluate(M, Util.dist(X[i], X[j]))
        end
    end

    K = K + triu(K,1)'
    return K
end

function idx_near(X::Array, pt, radius)
    n = length(X)

end

function ordinary_krig(M, Xs, Zs, Xp, sampleradius=100.0)
    # use model variogram *M* to predict values (Zp) at prediction locations
    # *Xp* given samples (*Xs*, *Zs*)

    if ndims(Xp) == 1
        n = length(Xp)
    elseif ndims(Xp) == 2
        n = length(Xp, 1)
    else
        error("prediction coordinates must have ndim == 1|2")
    end

    Zp = zeros(Float64, n)

    # calculate variance matrix
    K = buildcovmat(M, Zs)

    # iterate through all prediction points
    for i = 1:length(n)

        xp = Xp[i,:]

        # select a subset of the samples
        isel = zeros(Int16, n)
        cnt = 1
        for ismp = 1:length(Xs)  # fix this - needs to handle 2d arrays
            if Util.dist(Xs[ismp,:], xp) < sampleradius
                isel[cnt] = ismp
                cnt = cnt +1
            end
        end
        isel = isel[1:cnt-1]

        # build kriging system from the selection
        Km = Util.augment_lm(K[isel,isel])
        k = [evaluate(M, dist(Xs[i_,:], xp)) for i_=isel]
        append!(k, [1.0])

        # make prediction
        W = inv(Km) * k
        pred = W[1:end-1]' * Zs[isel]
        print(pred[1], ' ')
        Zp[i] = pred[1]
    end

    println()

    return Zp
end




end

