using LinearAlgebra

struct Svd{U, D, V}
	u::U
	d::D
	v::V
end

Base.show(io::IO, a::Svd) = print(io, "svd matrix of size $(size(a)) with $(typeof(a.u)))")

Flux.@functor Svd
"""
	svd(n, s)

	Square matrix in svd decomposition
	`n` --- dimension of matrix
	`s` --- parametrization for unitary matrices (:householder or :givens)
"""

function Svd(n::Int, s=:givens)
	n == 1 && return DiagonalRectangular(rand(Float32,n), n, n)
	if s == :householder
		return _svd_householder(n)
	elseif s == :givens
		return _svd_givens(n)
	else
		@error "Unknown unitary decomposition"
	end
end

_svd_householder(n::Int) = Svd(UnitaryHouseholder(n),
                               DiagonalRectangular(rand(Float32, n), n, n),
                               UnitaryHouseholder(n))
_svd_givens(n::Int) = Svd(UnitaryGivens(n),
                          DiagonalRectangular(rand(Float32, n), n, n),
                          UnitaryGivens(n))

*(a::Svd, x::AbstractMatVec) = a.u * (a.d * (a.v * x))

Base.inv(a::Svd) = Svd(inv(a.v), inv(a.d), inv(a.u))
Base.size(a::Svd) = size(a.d)
Matrix(a::Svd) = a.u * (a.d * Matrix(a.v))
transpose(a::Svd) = Svd(inv(a.v), a.d, inv(a.u))

 _logabsdet(a::Svd) = _logabsdet(a.d)
