struct SVDDense{U, D, V, B, S}
	u::U
	d::D
	v::V
	b::B
	σ::S
end

Base.show(io::IO, m::SVDDense) = print(io, "SVDDense{$(length(m.d)), $(m.σ)}")

Flux.@treelike(SVDDense)


"""
	SVDDense(n, σ; indexes = :random)

	Dense layer with square weight matrix of dimension `n` parametrized in 
	SVD decomposition using `UnitaryButterfly`  parametrization of unitary matrix.
	
	`σ` --- an invertible and transfer function, cuurently implemented `selu` and `identity`
	indexes --- method of generating indexes of givens rotations (`:butterfly` for the correct generation; `:random` for randomly generated patterns)
"""
SVDDense(n, σ; indexes = :random) = SVDDense(UnitaryButterfly(n, indexes = indexes), 
			rand(n),
			UnitaryButterfly(n, indexes = indexes),
			zeros(n),
			σ)


(m::SVDDense)(x::AbstractMatVec) = m.σ.(m.u * (m.d .* (m.v * x)) .+ m.b)
function (m::SVDDense)(x::Tuple)
	x, logdet = x
	pre = m.u * (m.d .* (m.v * x)) .+ m.b
	g = explicitgrad.(m.σ, pre)
	(m.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ sum(log.(abs.(m.d) .+ 1f-6)))
end

struct InvertedSVDDense{U, D, V, B, S}
	u::U
	d::D
	v::V
	b::B
	σ::S
end
Flux.@treelike(InvertedSVDDense)

Base.inv(m::SVDDense) = InvertedSVDDense(inv(m.u), m.d, inv(m.v), m.b, inv(m.σ))
Base.inv(m::InvertedSVDDense) = SVDDense(inv(m.u), m.d, inv(m.v), m.b, inv(m.σ))

(m::InvertedSVDDense)(x::AbstractMatVec)  = m.v * ((m.u * (m.σ.(x) .- m.b)) ./ m.d)

#define inversions of the most common functions
function invselu(x::Real)
  λ = NNlib.oftype(x/1, 1.0507009873554804934193349852946)
  α = NNlib.oftype(x/1, 1.6732632423543772848170429916717)
  x = x / λ
  ifelse(x > 0, x/1, log.(1 + x/α))
end

invtanh(x::Real) = (log(1 + x) - log(1 - x)) / 2
invσ(x::Real) = log(x) - log(1 - x)

Base.inv(::typeof(identity)) = identity
Base.inv(::typeof(NNlib.selu)) = invselu
Base.inv(::typeof(tanh)) = invtanh
Base.inv(::typeof(invselu)) = NNlib.selu
Base.inv(::typeof(invtanh)) = tanh
Base.inv(::typeof(NNlib.σ)) = invσ
Base.inv(::typeof(invσ)) = NNlib.σ

#define inversion of a Chain
Base.inv(m::Chain) = Chain(inv.(m.layers[end:-1:1])...)


explicitgrad(::typeof(identity), x) = 1
explicitgrad(::typeof(tanh), x) = 1 - tanh(x)^2
explicitgrad(::typeof(NNlib.σ), x) = σ(x)*(1 - σ(x))
function explicitgrad(::typeof(NNlib.selu), x) 
  λ = oftype(x/1, 1.0507009873554804934193349852946)
  α = oftype(x/1, 1.6732632423543772848170429916717)
  λ * ifelse(x > 0, 1, α * exp(x))
end
