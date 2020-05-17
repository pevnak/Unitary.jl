struct SVDTransform{U, D, V, B, S}
	u::U
	d::D
	v::V
	b::B
	σ::S
end

Base.show(io::IO, m::SVDTransform) = print(io, "SVDTransform{$(size(m.d)), $(m.σ)}")

Flux.@functor SVDTransform

"""
	SVDTransform(n, σ; indexes = :random)

	Transform layer with square weight matrix of dimension `n` parametrized in 
	SVD decomposition using `UnitaryUnitaryGivens`  parametrization of unitary matrix.
	
	`σ` --- an invertible and transfer function, cuurently implemented `selu` and `identity`
	indexes --- method of generating indexes of givens rotations (`:givens` for the correct generation; `:random` for randomly generated patterns)
"""
function SVDTransform(n::Int, σ, unitary = :householder)
	n == 1 && return(ScaleShift(1, σ))
	if unitary == :householder
		return(_svdtransform_householder(n, σ))
	elseif unitary == :givens
		return(_svdtransform_givens(n, σ))
	else 
		@error "unknown type of unitary matrix $unitary"
	end
end


using LinearAlgebra

_svdtransform_givens(n::Int, σ) = 
	SVDTransform(UnitaryGivens(n), 
			DiagonalRectangular(rand(Float32,n), n, n),
			UnitaryGivens(n),
			0.01f0.*randn(Float32,n),
			σ)

_svdtransform_householder(n::Int, σ) = 
	SVDTransform(UnitaryHouseholder(n), 
			DiagonalRectangular(rand(Float32,n), n, n),
			UnitaryHouseholder(n),
			0.01f0.*randn(Float32,n),
			σ)

(m::SVDTransform)(x::AbstractMatVec) = m.σ.(m.u * (m.d * (m.v * x)) .+ m.b)

function (m::SVDTransform)(xx::Tuple{A,B}) where {A,B}
	x, logdet = xx
	pre = m.u * (m.d * (m.v * x)) .+ m.b
	g = explicitgrad.(m.σ, pre)
	(m.σ.(pre), logdet .+ sum(log.(g), dims = 1) .+ _logabsdet(m.d))
end

struct InvertedSVDTransform{U, D, V, B, S}
	u::U
	d::D
	v::V
	b::B
	σ::S
end
Flux.@functor InvertedSVDTransform

Base.inv(m::SVDTransform) = InvertedSVDTransform(inv(m.u), inv(m.d), inv(m.v), m.b, inv(m.σ))
Base.inv(m::InvertedSVDTransform) = SVDTransform(inv(m.u), inv(m.d), inv(m.v), m.b, inv(m.σ))

(m::InvertedSVDTransform)(x::AbstractMatVec)  = m.v * (m.d * (m.u * (m.σ.(x) .- m.b))) 
