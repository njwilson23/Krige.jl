#
# predict.jl
#
# kriging prediction of geographical data
#

# assemble matrix of modelled variance
function buildcovmat(M::Variogram_like, X::Array{Float64})
    n = size(X)[1]
    K = zeros(Float64, n, n)
    maxvar = sill(M)

    for i = 1:n
        for j = i:n
            K[i,j] = maxvar - evaluate(M, dist(X[i,:], X[j,:]))
        end
    end

    K = K + triu(K,1)'
    return K
end

# use model variogram *M* to predict values (Zp) at prediction locations
# *Xp* given samples (*Xs*, *Zs*)
function ordinary_krig(M::Variogram_like, Xs::Array, Zs::Array, Xp::Array, sampleradius=100.0)
    n = size(Xp, 1)
    Zp = zeros(Float64, n)
    maxvar = sill(M)

    # calculate variance matrix
    K = buildcovmat(M, Xs)
    nu = mean(Zs)

    for i = 1:n
        xp = Xp[i,:]

        # select a subset of the samples
        isel = zeros(Int16, size(Xs)[1])
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
        k = convert(Array{Float64}, [maxvar-evaluate(M, dist(Xs[i_,:], xp)) for i_=isel])
        append!(k, [1.0])

        # make prediction
        L = Km \ k
        pred = L[1:end-1]' * (Zs[isel] .- nu) .+ nu
        Zp[i] = pred[1]
    end

    return Zp
end

