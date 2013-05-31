#
# model.jl
#
# define variogram model prototypes
#

abstract ModelVariogram

immutable type GaussianVariogram <: ModelVariogram
    sill::Real
    rng::Real
end

immutable type SphericalVariogram <: ModelVariogram
    sill::Real
    rng::Real
end

immutable type LogVariogram <: ModelVariogram
    sill::Real
    rng::Real
end

immutable type LinearVariogram <: ModelVariogram
    sill::Real
    rng::Real
end

immutable type NuggetVariogram <: ModelVariogram
    sill::Real
    rng::Real      # Should be zero?
end

immutable type CompositeVariogram
    vs::Array{ModelVariogram}
end

Variogram_like = Union(ModelVariogram, CompositeVariogram)


*(sill::Real, m::ModelVariogram) = typeof(m)(m.sill * sill, m.rng)
*(m::ModelVariogram, sill::Real) = typeof(m)(m.sill * sill, m.rng)
*(sill::Real, c::CompositeVariogram) = CompositeVariogram([sill*v for v=c.vs])
*(c::CompositeVariogram, sill::Real) = CompositeVariogram([sill*v for v=c.vs])

+(m::ModelVariogram, n::ModelVariogram) = CompositeVariogram([m, n])
+(m::ModelVariogram, n::CompositeVariogram) = push(n.vs, m)
+(m::CompositeVariogram, n::ModelVariogram) = push(n.vs, m)
+(m::CompositeVariogram, n::CompositeVariogram) = vcat(m, n)


evaluate(m::GaussianVariogram, h::Array) = m.sill * (1.0 - exp( -(h ./ m.rng).^2 ))
evaluate(m::GaussianVariogram, h::Number) = m.sill * (1.0 - exp( -(h ./ m.rng).^2 ))

evaluate(m::SphericalVariogram, h::Array) = 
    m.sill * [h_<m.rng ? 1.5 * h_/m.rng - 0.5 * (h_/m.rng)^3 : 1.0 for h_=h]
evaluate(m::SphericalVariogram, h::Number) =
    m.sill * (h<m.rng ? 1.5 * h/m.rng - 0.5 * (h/m.rng)^3 : 1.0)

evaluate(m::LogVariogram, h::Array) =
    m.sill * [h_ == 0.0 ? 0.0 : log(h_+m.rng) for h_=h]
evaluate(m::LogVariogram, h::Number) =
    m.sill * (h == 0.0 ? 0.0 : log(h+m.rng))

evaluate(m::LinearVariogram, h::Array) =
    m.sill * [h_<m.rng ? h_/m.rng : 1.0 for h_=h]
evaluate(m::LinearVariogram, h::Number) =
    m.sill * (h<m.rng ? h/m.rng : 1.0)

evaluate(m::NuggetVariogram, h) = m.sill * (h .> m.rng)

evaluate(c::CompositeVariogram, h) = sum([evaluate(v,h) for v=c.vs])

getp(m::ModelVariogram) = [m.sill, m.rng]
getp(c::CompositeVariogram) = reduce((a,b) -> vcat(a,b), [getp(a) for a=c.vs])

sill(m::ModelVariogram) = m.sill
sill(c::CompositeVariogram) = sum([v.sill for v=c.vs])

function taketwo(A)
    assert(length(A) % 2 == 0)
    res = zeros(Any, int(length(A)/2))
    for i=1:length(A)/2
        res[i] = [A[2*(i-1)+1], A[2*i]]
    end
    return res
end

tune(m::ModelVariogram, p) = typeof(m)(p[1], p[2])
tune(m::CompositeVariogram, p) =
    CompositeVariogram([p_(p_[1][1],p_[2]) for p_=zip(m.vs, taketwo(p))])

function fit!(M::Variogram_like, g, h)
    # Fit variogram model *M* with parameters *p* to an experimental variogram
    # *g* at lags *h*.
    f_obj = (p) -> sum((evaluate(tune(M, p), h) - g).^2)
    res = Optim.optimize(f_obj, getp(M), method=:nelder_mead)
    return tune(M, res.minimum)
end

