#
# unit tests for Krige.jl
#


#include("src/vario.jl")
#include("src/model.jl")
#include("src/predict.jl")

using Krige
using Base.Test
using Winston

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

function pvariogram_fit(m::Variogram_like, g::ExperimentalVariogram)
    p = FramedPlot()
    add(p, Points(g.lags, g.g))
    add(p, Curve(g.lags, evaluate(m, g.lags)))
    return p
end

let h=[0:10:100]
    g = 3.0 * (1.0 - exp( -(h / 50.0).^2))
    m = GaussianVariogram(1.0, 20.0)
    m_fitted = fit!(m, ExperimentalVariogram(h,g))
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

let x = [0.0 : 100.0]
    z = .25x + 8.0 * rand(length(x))
    vg = est_variogram(x, z, 10.0, 100.0)
    m = fit!(GaussianVariogram(1.0, 5.0), vg)
    #p = pvariogram_fit(m, vg[:,1], vg[:,2])
    #file(p, "variogram-fit-gaussian.png")
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

    g = est_variogram(Xs, Zs, 2.0, 9.0)
    m = GaussianVariogram(2.0, 6.0)# + NuggetVariogram(0.01, 0.0)
    #m = Krige.fit!(GaussianVariogram(1.0, 5.0), g)# + NuggetVariogram(0.01, 0.0)
    println("Variogram parameters:")
    println(Krige.getp(m))
    #p = pvariogram_fit(m, g[:,1], g[:,2])
    #Winston.file(p, "variogram-fit.png")

    # make an estimate
    Zp = ordinary_krig(m, Xs, Zs, X)
    println("Residuals:")
    println(Zp - Z)

    include("src/plot.jl")
    prediction_performance1d(Xs, Zs, X, Zp, "prediction-result.png")

end


