using Zygote:@adjoint
using LinearAlgebra

struct lowup{T}
	m::Matrix{T}
	n::Int
end

struct inverted_lowup{T}
	m::Matrix{T}
	n::Int
end

Flux.@functor lowup
Flux.trainable(a::lowup) = (a.m,)
Flux.@functor inverted_lowup
Flux.trainable(a::inverted_lowup) = (a.m,)

#Constructors#
lowup(n::Int) = lowup(Float32, n)
lowup(T::DataType, n::Int) = lowup(rand(T, n, n), n)
function lowup(a::AbstractMatrix)
	@assert size(a, 1) == size(a, 2)
	lowup(Matrix(a), size(a, 1))
end


inverted_lowup(n::Int) = inverted_lowup(Float32, n)
inverted_lowup(T::DataType, n::Int) = inverted_lowup(rand(T, n, n), n)
function inverted_lowup(a::AbstractMatrix)
	(@assert size(a, 1) == size(a, 2))
	inverted_lowup(Matrix(a), size(a, 1))
end

import Base: size
size(a::lowup) = (a.n, a.n)
size(a::inverted_lowup) = (a.n, a.n)

function _logabsdet(a::lowup{T}) where {T<:Number}
	out = zero(T)
	for i = 1:a.n
		out += log(abs(a.m[i, i] + eps(T)))
	end
	out
end

function _logabsdet(a::inverted_lowup{T}) where {T<:Number}
	out = zero(T)
	for i = 1:a.n
		out += log(abs(a.m[i, i] + eps(T)))
	end
	out
end

#Inversion
function LinearAlgebra.inv(a::lowup)
	m = similar(a.m)
	UnitLowerTriangular(m) .= inv(UnitLowerTriangular(a.m))
	UpperTriangular(m) .= inv(UpperTriangular(a.m))
	inverted_lowup(m)
end

function LinearAlgebra.inv(a::inverted_lowup)
	m = similar(a.m)
	UnitLowerTriangular(m) .= inv(UnitLowerTriangular(a.m))
	UpperTriangular(m) .= inv(UpperTriangular(a.m))
	lowup(m)
end
