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
    
    export GaussianVariogram, SphericalVariogram, LogVariogram, LinearVariogram,
           NuggetVariogram, CompositeVariogram, Variogram_like,
           est_variogram,
           evaluate,
           fit!,
           ordinary_krig
    
    include("src/util.jl")
    include("src/vario.jl")
    include("src/model.jl")
    include("src/predict.jl")

end
