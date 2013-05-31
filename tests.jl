#
# unit tests for Krige.jl
#


include("src/util.jl")
include("src/vario.jl")
include("src/model.jl")
include("src/predict.jl")

using Base.Test

######### Variogram estimation ##########
@test Util.dist([1.0,1.0], [2.0,3.0]) == sqrt(5)

let V = [0.0 0.0; 1.0 2.0; 2.0 1.0]
    @test Vario.compute_distances(V) == [sqrt(5), sqrt(5), sqrt(2)]
end

let V = [5.0, 4.0, 1.0]
    @test Vario.zdiffmat(V) == [-1.0, -4.0, -3.0]
end

######### Variogram modelling ##########
let h=[0:10:100]
    g = 3.0 * (1.0 - exp( -(h / 50.0).^2))
    m = Model.GaussianVariogram(1.0, 20.0)
    m_fitted = Model.fit!(m, [1.0, 20.0], g, h)
    @test_approx_eq_eps g Model.evaluate(m_fitted, h) 1e-3
end


######### Prediction ##########
let X=[0.0:10.0]

    Z=cos(X ./ pi)

    ii = [1,3,4,7,8,10]
    Xs = X[ii]
    Zs = Z[ii]

    g = Vario.expvario(Xs, Zs, 1.0, 10.0)
    m = Model.GaussianVariogram(1.0, 5.0)
    m = Model.fit!(m, [1.0, 5.0], g[:,2], g[:,1])
    Zp = Predict.ordinary_krig(m, Xs, Zs, X)
    println(Zp)

end


