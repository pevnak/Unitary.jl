struct SVDDense{U, D, V, B, S}
	u::U
	d::D
	v::V
	b::B
	σ::S
end

Base.show(io::IO, m::SVDDense) = print(io, "SVDDense{$(size(m.d)), $(m.σ)}")

Flux.@functor SVDDense

"""
	SVDDense(n, σ; indexes = :random)

	Dense layer with square weight matrix of dimension `n` parametrized in 
	SVD decomposition using `UnitaryButterfly`  parametrization of unitary matrix.
	
	`σ` --- an invertible and transfer function, cuurently implemented `selu` and `identity`
	indexes --- method of generating indexes of givens rotations (`:butterfly` for the correct generation; `:random` for randomly generated patterns)
"""
function SVDDense(n::Int, σ, unitary = :householder; indexes = :random, maxn::Int = n)
	if unitary == :householder
		return(_svddense_householder(n, σ))
	elseif unitary == :butterfly
		return(_svddense_butterfly(n, σ, maxn = maxn, indexes = indexes))
	else 
		@error "unknown type of unitary matrix $unitary"
	end
end


using LinearAlgebra

_svddense_butterfly(n::Int, σ; indexes = :random, maxn::Int = n) = 
	SVDDense(InPlaceUnitaryButterfly(UnitaryButterfly(n, indexes = indexes, maxn = maxn)), 
			DiagonalRectangular(rand(Float32,n), n, n),
			InPlaceUnitaryButterfly(UnitaryButterfly(n, indexes = indexes, maxn = maxn)),
			zeros(Float32,n),
			σ)

_svddense_householder(n::Int, σ) = 
	SVDDense(UnitaryHouseholder(Float32, n), 
			DiagonalRectangular(rand(Float32,n), n, n),
			UnitaryHouseholder(Float32, n) ,
			zeros(Float32,n),
			σ)

(m::SVDDense)(x::AbstractMatVec) = m.σ.(m.u * (m.d * (m.v * x .+ m.b)))

function (m::SVDDense)(xx::Tuple)
	x, logdet = xx
	pre = m.u * (m.d * (m.v * x .+ m.b)) 
	g = explicitgrad.(m.σ, pre)
	(m.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(m.d))
end

struct InvertedSVDDense{U, D, V, B, S}
	u::U
	d::D
	v::V
	b::B
	σ::S
end
Flux.@functor InvertedSVDDense

Base.inv(m::SVDDense) = InvertedSVDDense(inv(m.u), inv(m.d), inv(m.v), m.b, inv(m.σ))
Base.inv(m::InvertedSVDDense) = SVDDense(inv(m.u), inv(m.d), inv(m.v), m.b, inv(m.σ))

(m::InvertedSVDDense)(x::AbstractMatVec)  = m.v * (m.d * (m.u * (m.σ.(x)))) .- m.b

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


invtanh(x::Real) = (log(1 + x) - log(1 - x)) / 2
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
