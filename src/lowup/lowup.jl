using Zygote:@adjoint
using LinearAlgebra

struct lowup{T}
	m::Matrix{T}
	n::Int
	invs::Bool
end

Flux.@functor lowup
Flux.trainable(a::lowup) = (a.m,)

#Constructors#
lowup(n::Int) = lowup(Float32, n, false)
lowup(n::Int, invs::Bool) = lowup(Float32, n, invs)
lowup(T::DataType, n::Int, invs::Bool) = lowup(rand(T, n, n), n, invs)
lowup(T::DataType, n::Int) = lowup(T, n, false)
lowup(a::AbstractMatrix) = lowup(Matrix(a), size(a, 1), false)
function lowup(a::AbstractMatrix, invs::Bool)
	@assert size(a, 1) == size(a, 2)
	lowup(Matrix(a), size(a, 1), invs)
end

#Basic functions
import Base: size
size(a::lowup) = a.n
Base.Matrix(a::lowup) = a.invs ?
			UpperTriangular(a.m) * UnitLowerTriangular(a.m) :
			UnitLowerTriangular(a.m) * UpperTriangular(a.m)
function _logabsdet(a::lowup{T}) where {T<:Number}
	out = zero(T)
	for i = 1:a.n
		out += log(abs(a.m[i, i] + eps(T)))
	end
	out
end

function LinearAlgebra.transpose(a::lowup)
	out = lowup(a.m', a.invs)
	if a.invs
		for i = 1:a.n
			for j = 1:i-1
				out.m[j, i] *= out.m[i, i]
				out.m[i, j] /= out.m[i, i]
			end
		end
	else
		for i = 1:a.n
			for j = i+1:a.n
				out.m[j, i] /= out.m[i, i]
				out.m[i, j] *= out.m[i, i]
			end
		end
	end
	out
end


#Inversion
function LinearAlgebra.inv(a::lowup)
	m = similar(a.m)
	UnitLowerTriangular(m) .= inv(UnitLowerTriangular(a.m))
	UpperTriangular(m) .= inv(UpperTriangular(a.m))
	lowup(m, !a.invs)
end

mulaxlu(m, x, invs) = invs ?
			UpperTriangular(m) * UnitLowerTriangular(m) * x :
			UnitLowerTriangular(m) * UpperTriangular(m) * x

mulxalu(m, x, invs) = invs ?
			x * UpperTriangular(m) * UnitLowerTriangular(m) :
			x * UnitLowerTriangular(m) * UpperTriangular(m)

import Base: *
function *(a::lowup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulaxlu(a.m, x, a.invs)
end
function *(x::AbstractMatVec, a::lowup)
	@assert size(x, 2) == a.n
	mulxalu(a.m, x, a.invs)
end

function ∇mulaxlu(Δ, m, invs, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	x = deepcopy(x)
	if invs
		Δloc = u'*Δ
		@inbounds for i = 1:a-1
			for j = i+1:a
				for k = 1:b
					∇m[j, i] += x[i, k] * Δloc[j, k]
				end
			end
		end
		Δloc = l'*Δloc
		x = l*x
		@inbounds for i = 1:a
			for j = 1:i
				for k = 1:b
					∇m[j, i] += x[i, k] * Δ[j, k]
				end
			end
		end
	else
		Δloc = l'*Δ
		@inbounds for i = 1:a
			for j = 1:i
				for k = 1:b
					∇m[j, i] += x[i, k] * Δloc[j, k]
				end
			end
		end
		Δloc = u'*Δloc
		x = u*x
		@inbounds for i = 1:a-1
			for j = i+1:a
				for k = 1:b
					∇m[j, i] += x[i, k] * Δ[j, k]
				end
			end
		end
	end
	(∇m, Δloc, nothing)
end

function ∇mulxalu(Δ, m, invs, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	x = deepcopy(x)
	if invs
		Δloc = Δ*l'
		@inbounds for i = 1:b
			for j = 1:i
				for k = 1:a
					∇m[j, i] += x[k, j] * Δloc[k, i]
				end
			end
		end
		Δloc = Δloc*u'
		x = x*u
		@inbounds for i = 1:b-1
			for j = i+1:b
				for k = 1:a
					∇m[j, i] += x[k, j] * Δ[k, i]
				end
			end
		end
	else
		Δloc = Δ*u'
		@inbounds for i = 1:b-1
			for j = i+1:b
				for k = 1:a
					∇m[j, i] += x[k, j] * Δloc[k, i]
				end
			end
		end
		Δloc = Δloc*l'
		x = x*l
		@inbounds for i = 1:b
			for j = 1:i
				for k = 1:a
					∇m[j, i] += x[k, j] * Δ[k, i]
				end
			end
		end
	end
	(∇m, Δloc, nothing)
end

@adjoint function mulaxlu(m, x, invs)
	o = mulaxlu(m, x, invs)
	return o, Δ -> ∇mulaxlu(Δ, m, invs, x)
end

@adjoint function mulxalu(m, x, invs)
	o = mulxalu(m, x, invs)
	return o, Δ -> ∇mulxalu(Δ, m, invs, x)
end
