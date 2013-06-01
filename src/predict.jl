#
# predict.jl
#
# kriging prediction of geographical data
#

function buildcovmat(M::Variogram_like, X::Array{Float64})
    # assemble matrix of modelled variance
    n = size(X)[1]
    K = zeros(Float64, n, n)
    maxvar = sill(M)

    for i = 1:n
        for j = i:n
            #K[i,j] = maxvar - evaluate(M, dist(X[i,:], X[j,:]))
            K[i,j] = - evaluate(M, dist(X[i,:], X[j,:]))
        end
    end

    K = K + triu(K,1)'
    return K
end

# use model variogram *M* to predict values (Zp) at prediction locations
# *Xp* given samples (*Xs*, *Zs*)
function ordinary_krig(M::Variogram_like, Xs::Array, Zs::Array, Xp::Array, sampleradius=100.0)
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
    for i = 1:n

        xp = Xp[i,:]

        # select a subset of the samples
        isel = zeros(Int16, n)
        cnt = 1
        for ismp = 1:size(Xs)[1]
            if dist(Xs[ismp,:], xp) < sampleradius
                isel[cnt] = ismp
                cnt = cnt +1
            end
        end
        isel = isel[1:cnt-1]

        # build kriging system from the selection
        Km = augment_lm(K[isel,isel])
        maxvar = sill(M)
        k = [maxvar-evaluate(M, dist(Xs[i_,:], xp)) for i_=isel]
        append!(k, [1.0])

        # calculate condition number
        kappa = norm(Km) * norm(inv(Km))
        if kappa > 1e3
            println("Warning - high condition number: ", kappa)
        end

        # make prediction
        W = inv(Km) * k
        pred = W[1:end-1]' * Zs[isel]
        Zp[i] = pred[1]
    end

    return Zp
end

