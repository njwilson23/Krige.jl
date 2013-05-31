#
# unit tests for Krige.jl
#


#include("src/vario.jl")
#include("src/model.jl")
#include("src/predict.jl")

using Krige
using Base.Test

######### Variogram estimation ##########
#include("src/util.jl")
#@test dist([1.0,1.0], [2.0,3.0]) == sqrt(5)

let V = [0.0 0.0; 1.0 2.0; 2.0 1.0]
    @test Krige.compute_distances(V) == [sqrt(5), sqrt(5), sqrt(2)]
end

let V = [5.0, 4.0, 1.0]
    @test Krige.zdiffmat(V) == [-1.0, -4.0, -3.0]
end

######### Variogram modelling ##########
let h=[0:10:100]
    g = 3.0 * (1.0 - exp( -(h / 50.0).^2))
    m = GaussianVariogram(1.0, 20.0)
    m_fitted = fit!(m, g, h)
    @test_approx_eq_eps g evaluate(m_fitted, h) 1e-3
end

let a = SphericalVariogram(1.,2.),
    b = SphericalVariogram(3.,4.),
    c = NuggetVariogram(0.5,0.0)

    @test Krige.getp(a) == [1., 2.]
    @test Krige.getp(b) == [3., 4.]

    V = CompositeVariogram([a,b,c])
    @test Krige.getp(V) == [1., 2., 3., 4., 0.5, 0.0]
end

######### Prediction ##########
let X=[0.0:10.0]

    # create fitting data
    mdata = GaussianVariogram(2.0, 6.0)
    Z = evaluate(mdata, X)

    # extract a subset for training
    ii = [1,3,4,7,8,10]
    Xs = X[ii]
    Zs = Z[ii]

    g = est_variogram(Xs, Zs, 1.0, 9.0)
    m = GaussianVariogram(1.0, 5.0)
    m = Krige.fit!(m, g[:,2], g[:,1]) + NuggetVariogram(0.01, 0.0)
    println(Krige.getp(m))

    # make an estimate
    Zp = ordinary_krig(m, Xs, Zs, X)
    println("Residuals:")
    println(Zp - Z)

end


