
include("src/vario.jl")
include("src/model.jl")

using Base.Test

@test Vario.dist([1.0,1.0], [2.0,3.0]) == sqrt(5)

let V = [0.0 0.0; 1.0 2.0; 2.0 1.0]
    @test Vario.compute_distances(V) == [sqrt(5), sqrt(5), sqrt(2)]
end

let V = [5.0, 4.0, 1.0]
    @test Vario.zdiffmat(V) == [-1.0, -4.0, -3.0]
end

# Simple Gaussian variogram fitting
let h=[0:10:100]
    g = 3.0 * (1.0 - exp( -(h / 50.0).^2))
    m = Model.GaussianVariogram(1.0, 20.0)
    m_fitted = Model.fit!(m, [1.0, 20.0], g, h)
    @test_approx_eq_eps g Model.evaluate(m_fitted, h) 1e-3
end


