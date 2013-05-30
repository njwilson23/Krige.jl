#
# model.jl
#
# Define variogram model prototypes. The type hierarchy looks like:
# - ModelVariogram
#  - GaussianVariogram
#  - SphericalVeriogram
#  - NuggetVariogram
#
# Functions:
#
# varat(M::ModelVariogram, h:Vector)
# evaluate the model variogram at lags *h*
#
module Model

using Optim

# type declarations

abstract ModelVariogram

immutable type GaussianVariogram <: ModelVariogram
    coef::Real
    sill::Real
end

immutable type SphericalVariogram <: ModelVariogram
    coef::Real
    sill::Real
end

immutable type LogVariogram <: ModelVariogram
    coef::Real
    sill::Real
end

immutable type LinearVariogram <: ModelVariogram
    coef::Real
    sill::Real
end

immutable type NuggetVariogram <: ModelVariogram
    coef::Real
    sill::Real      # Should be zero?
end

immutable type CompositeVariogram
    vs::Array{ModelVariogram}
end

Variogram_like = Union(ModelVariogram, CompositeVariogram)

# method definitions

*(coef::Real, m::ModelVariogram) = typeof(m)(m.coef * coef, m.sill)
*(m::ModelVariogram, coef::Real) = typeof(m)(m.coef * coef, m.sill)
*(coef::Real, c::CompositeVariogram) = CompositeVariogram([coef*v for v=c.vs])
*(c::CompositeVariogram, coef::Real) = CompositeVariogram([coef*v for v=c.vs])

+(m::ModelVariogram, n::ModelVariogram) = CompositeVariogram([m, n])
+(m::ModelVariogram, n::CompositeVariogram) = push(n.vs, m)
+(m::CompositeVariogram, n::ModelVariogram) = push(n.vs, m)
+(m::CompositeVariogram, n::CompositeVariogram) = vcat(m, n)

evaluate(m::GaussianVariogram, h) = m.coef * (1.0 - exp( -(h ./ m.sill).^2 ))
evaluate(m::SphericalVariogram, h) = m.coef * 
                                    [h_<m.sill ?
                                     1.5 * h_/m.sill - 0.5 * (h_/m.sill)^3 :
                                     1.0 for h_=h]
evaluate(m::LogVariogram, h) = m.coef * [h_ == 0.0 ? 0.0 : log(h_+m.sill) for h_=h]
evaluate(m::LinearVariogram, h) = m.coef * [h_<m.sill ? h_/m.sill : 1.0 for h_=h]
evaluate(m::NuggetVariogram, h) = m.coef * (h .> m.sill)
evaluate(c::CompositeVariogram, h) = sum([evaluate(v,h) for v=c.vs])

function taketwo(A)
    assert(length(A) % 2 == 0)
    res = zeros(Any, int(length(A)/2))
    for i=1:length(A)/2
        res[i] = [A[2*(i-1)+1], A[2*i]]
    end
    return res
end

tune!(m::ModelVariogram, p) = typeof(m)(p[1], p[2])
tune!(m::CompositeVariogram, p) =
    CompositeVariogram([p_(p_[1][1],p_[2]) for p_=zip(m.vs, taketwo(p))])

function fit!(M::Variogram_like, p, g, h)
    # Fit variogram model *M* with parameters *p* to an experimental variogram
    # *g* at lags *h*.
    f_obj = (p) -> sum((evaluate(tune!(M, p), h) - g).^2)
    res = Optim.optimize(f_obj, p)#, method=:nelder_mead)
    tune!(M, res.minimum)
end

end
