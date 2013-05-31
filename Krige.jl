#
# Krige.jl
#
# geostatistical functions for
#   - variogram estimation
#   - variogram modelling
#   - spatial prediction
#

module Krige
using Optim

export compute_distances, zdiffmat, expvario
export GaussianVariogram, SphericalVariogram, LogVariogram, LinearVariogram,
       NuggetVariogram, CompositeVariogram, Variogram_like, fit!, evaluate
export buildcovmet, ordinary_krig

include("src/util.jl")
include("src/vario.jl")
include("src/model.jl")
include("src/predict.jl")

end
