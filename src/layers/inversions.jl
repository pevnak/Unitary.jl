#define inversions of the most common functions
# λ * ifelse(x > 0, x/1, α * (exp(x) - 1))
function invselu(x::T) where {T<:Real}
  λ = NNlib.oftype(x, 1.0507009873554804934193349852946)
  α = NNlib.oftype(x, 1.6732632423543772848170429916717)
  x = x / λ
  ifelse(x > 0, x, log.(1 + x/α + eps(x)))
end

function invleakyrelu(x::T) where {T<:Real}
  ifelse(x > 0, x, 100 * x)
end


function invtanh(x::Real) 
	x = min(x, 0.9999f0)
	x = max(x, -0.9999f0)
	(log(1 + x) - log(1 - x)) / 2
end
invσ(x::Real) = log(x) - log(1 - x)

Base.inv(::typeof(identity)) = identity
Base.inv(::typeof(NNlib.selu)) = invselu
Base.inv(::typeof(tanh)) = invtanh
Base.inv(::typeof(invselu)) = NNlib.selu
Base.inv(::typeof(invtanh)) = tanh
Base.inv(::typeof(NNlib.σ)) = invσ
Base.inv(::typeof(invσ)) = NNlib.σ
Base.inv(::typeof(NNlib.leakyrelu)) = invleakyrelu
Base.inv(::typeof(invleakyrelu)) = NNlib.leakyrelu

#define inversion of a Chain
Base.inv(m::Chain) = Chain(inv.(m.layers[end:-1:1])...)


explicitgrad(::typeof(identity), x) = 1f0
explicitgrad(::typeof(tanh), x) = 1f0 - tanh(x)^2
explicitgrad(::typeof(NNlib.σ), x) = σ(x)*(1f0 - σ(x))
explicitgrad(::typeof(NNlib.leakyrelu), x) = ifelse(x > 0, NNlib.oftype(x, 1), NNlib.oftype(x, 0.01))
function explicitgrad(::typeof(NNlib.selu), x) 
  λ = oftype(x/1, 1.0507009873554804934193349852946)
  α = oftype(x/1, 1.6732632423543772848170429916717)
  λ * ifelse(x > 0, 1, α * exp(x))
end
