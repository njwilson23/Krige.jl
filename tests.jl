
include("vario.jl")

using Base.Test

@test dist([1.0,1.0], [2.0,3.0]) == sqrt(5)

V = [0.0 0.0;
     1.0 2.0;
     2.0 1.0]
@test compute_distances(V) == [sqrt(5), sqrt(5), sqrt(2)]

V = [5.0, 4.0, 1.0]
@test zdiffmat(V) == [-1.0, -4.0, -3.0]

