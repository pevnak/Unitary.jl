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

function ∇mulaxlu(Δ, m, invs, o)
	∇m = zero(m)
	Δ = deepcopy(Matrix(Δ))
	a, b = size(o)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	if invs
		o = inv(u)*o
		@inbounds for i = 1:a
			for j = 1:i
				for k = 1:b
					∇m[j, i] += o[i, k] * Δ[j, k]
				end
			end
		end
		Δ = u'*Δ
		o = inv(l)*o
		@inbounds for i = 1:a-1
			for j = i+1:a
				for k = 1:b
					∇m[j, i] += o[i, k] * Δ[j, k]
				end
			end
		end
		Δ = l'*Δ
	else
		o = inv(l)*o
		@inbounds for i = 1:a-1
			for j = i+1:a
				for k = 1:b
					∇m[j, i] += o[i, k] * Δ[j, k]
				end
			end
		end
		Δ = l'*Δ
		o = inv(u)*o
		@inbounds for i = 1:a
			for j = 1:i
				for k = 1:b
					∇m[j, i] += o[i, k] * Δ[j, k]
				end
			end
		end
		Δ = u'*Δ
	end
	(∇m, Δ, nothing)
end

function ∇mulxalu(Δ, m, invs, o)
	∇m = zero(m)
	Δ = deepcopy(Matrix(Δ))
	a, b = size(o)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	if invs
		o = o*inv(l)
		@inbounds for i = 1:b-1
			for j = i+1:b
				for k = 1:a
					∇m[j, i] += o[k, j] * Δ[k, i]
				end
			end
		end
		Δ = Δ*l'
		o = o*inv(u)
		@inbounds for i = 1:b
			for j = 1:i
				for k = 1:a
					∇m[j, i] += o[k, j] * Δ[k, i]
				end
			end
		end
		Δ = Δ*u'
	else
		o = o*inv(u)
		@inbounds for i = 1:b
			for j = 1:i
				for k = 1:a
					∇m[j, i] += o[k, j] * Δ[k, i]
				end
			end
		end
		Δ = Δ*u'
		o = o*inv(l)
		@inbounds for i = 1:b-1
			for j = i+1:b
				for k = 1:a
					∇m[j, i] += o[k, j] * Δ[k, i]
				end
			end
		end
		Δ = Δ*l'
	end
	(∇m, Δ, nothing)
end

@adjoint function mulaxlu(m, x, invs)
	o = 	invs ?
		UpperTriangular(m) * UnitLowerTriangular(m) * x :
		UnitLowerTriangular(m) * UpperTriangular(m) * x
	return o, Δ -> ∇mulaxlu(Δ, m, invs, o)
end

@adjoint function mulxalu(m, x, invs)
	o = 	invs ?
		x * UpperTriangular(m) * UnitLowerTriangular(m) :
		x * UnitLowerTriangular(m) * UpperTriangular(m)
	return o, Δ -> ∇mulxalu(Δ, m, invs, o)
end
