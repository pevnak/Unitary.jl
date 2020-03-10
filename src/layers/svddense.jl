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
function SVDDense(n::Int, σ, unitary = :householder)
	n == 1 && return(ScaleShift(1, σ))
	if unitary == :householder
		return(_svddense_householder(n, σ))
	elseif unitary == :butterfly || unitary == :givens
		return(_svddense_butterfly(n, σ))
	else 
		@error "unknown type of unitary matrix $unitary"
	end
end


using LinearAlgebra

_svddense_butterfly(n::Int, σ) = 
	SVDDense(Butterfly(n), 
			DiagonalRectangular(rand(Float32,n), n, n),
			Butterfly(n),
			0.01f0.*randn(Float32,n),
			σ)

_svddense_householder(n::Int, σ) = 
	SVDDense(UnitaryHouseholder(Float32, n), 
			DiagonalRectangular(rand(Float32,n), n, n),
			UnitaryHouseholder(Float32, n) ,
			0.01f0.*randn(Float32,n),
			σ)

(m::SVDDense)(x::AbstractMatVec) = m.σ.(m.u * (m.d * (m.v * x)) .+ m.b)

function (m::SVDDense)(xx::Tuple{A,B}) where {A,B}
	x, logdet = xx
	pre = m.u * (m.d * (m.v * x)) .+ m.b
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

(m::InvertedSVDDense)(x::AbstractMatVec)  = m.v * (m.d * (m.u * (m.σ.(x) .- m.b))) 
