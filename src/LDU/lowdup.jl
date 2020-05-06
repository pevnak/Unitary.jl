using Zygote:@adjoint
using LinearAlgebra
using NNlib

struct lowdup{T}
	m::Matrix{T}
	n::Int
end

struct inverted_lowdup{T}
	m::Matrix{T}
	n::Int
end

Flux.@functor lowdup
Flux.trainable(a::lowdup) = (a.m,)
Flux.@functor inverted_lowdup
Flux.trainable(a::inverted_lowdup) = (a.m,)

diagsp(m::Matrix, n::Int = size(m, 1)) = Diagonal([softplus(m[i, i]) for i in 1:n])

import Base.getproperty
function Base.getproperty(a::Union{lowdup, inverted_lowdup}, s::Symbol)
	if s == :l
		return UnitLowerTriangular(a.m)
	elseif s == :u
		return UnitUpperTriangular(a.m)
	elseif s == :d
		diagsp(a.m, a.n)
	else
		return getfield(a, s)
	end
end
import Base.setproperty!
function Base.setproperty!(a::Union{lowdup, inverted_lowdup}, s::Symbol, x)
	if s == :l
		a.l .= x
	elseif s == :u
		a.u .= x
	else
		return setfield!(a, s, x)
	end
end

#Constructors#
lowdup(n::Int) = lowdup(Float32, n)
lowdup(T::DataType, n::Int) = lowdup(rand(T, n, n), n)
function lowdup(a::AbstractMatrix)
	@assert size(a, 1) == size(a, 2)
	lowdup(Matrix(a), size(a, 1))
end

inverted_lowdup(n::Int) = inverted_lowdup(Float32, n)
inverted_lowdup(T::DataType, n::Int) = inverted_lowdup(rand(T, n, n), n)
function inverted_lowdup(a::AbstractMatrix)
	(@assert size(a, 1) == size(a, 2))
	inverted_lowdup(Matrix(a), size(a, 1))
end

import Base.size
Base.size(a::lowdup) = (a.n, a.n)
Base.size(a::inverted_lowdup) = (a.n, a.n)

function _logabsdet(a::Union{lowdup{T}, inverted_lowdup{T}}) where {T<:Number}
	out = zero(T)
	for i = 1:a.n
		out += log(softplus(a.m[i, i]))
	end
	out
end

#Inversion

softplusinv(x::Real) = x + log1p(-exp(-x))

import LinearAlgebra.inv
function LinearAlgebra.inv(a::lowdup)
	out = inverted_lowdup(similar(a.m))
	out.l = inv(a.l)
	out.u = inv(a.u)
	for i = 1:a.n
		out.m[i, i] = softplusinv(1/softplus(a.m[i, i]))
	end
	out
end

function LinearAlgebra.inv(a::inverted_lowdup)
	out = lowdup(similar(a.m))
	out.l = inv(a.l)
	out.u = inv(a.u)
	for i = 1:a.n
		out.m[i, i] = softplusinv(1/softplus(a.m[i, i]))
	end
	out
end
