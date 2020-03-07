using Zygote:@adjoint
using LinearAlgebra

struct lowup{T}
	m::Matrix{T}
	n::Int
end

Flux.@functor lowup
Flux.trainable(a::lowup) = (a.m,)

#Constructors#
lowup(n::Int) = lowup(Float32, n)
lowup(T::DataType, n::Int) = lowup(rand(T, n, n), n)
function lowup(a::AbstractMatrix)
	@assert size(a, 1) == size(a, 2)
	lowup(Matrix(a), size(a, 1))
end

#Basic functions
Base.Matrix(a::lowup) = UnitLowerTriangular(a.m) * UpperTriangular(a.m)
function LinearAlgebra.transpose(a::lowup)
	out = lowup(a.m')
	for i = 1:a.n
		for j = i+1:a.n
			out.m[j, i] /= out.m[i, i]
			out.m[i, j] *= out.m[i, i]
		end
	end
	out
end

mulax(m, x) = UnitLowerTriangular(m) * UpperTriangular(m) * x
mulxa(m, x) = x * UnitLowerTriangular(m) * UpperTriangular(m)

import Base: *
function *(a::lowup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulax(a.m, x)
end
function *(x::AbstractMatVec, a::lowup)
	@assert size(x, 2) == a.n
	mulxa(a.m, x)
end

function ∇mulax(Δ, m, x)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	∇x = (l*u)'*Δ
	∇m = zero(m)
	for i = 1:a
		for j = 1:i
			∇m[j, i] = dot(sum(Δ.*l[:, j], dims=1), x[i, :])
		end
	end
	N = u*x;
	for i = 1:a-1
		for j = i+1:a
			for k = 1:b
				∇m[j, i] += Δ[j, k] * N[i, k]
			end
		end
	end
	(∇m, ∇x)
end

function ∇mulxa(Δ, m, x)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	∇x = Δ*(l*u)'
	∇m = zero(m)
	N = x*l;
	for i = 1:b
		for j = 1:i
			for k = 1:a
				∇m[j, i] += Δ[k, i] * N[k, j]
			end
		end
	end
	for i = 1:b-1
		for j = i+1:b
			∇m[j, i] = dot(sum(Δ.*x[:, j], dims=1), u[i, :])
		end
	end
	(∇m, ∇x)
end


#with inverse

function ∇mulax_inv(Δ, m, o)
	∇m = zero(m)
	Δ = deepcopy(Matrix(Δ))
	a, b = size(o)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
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
	(∇m, Δ)
end

function ∇mulxa_inv(Δ, m, o)
	∇m = zero(m)
	Δ = deepcopy(Matrix(Δ))
	a, b = size(o)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
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
	(∇m, Δ)
end

#@adjoint function mulax(m, x)
#	return mulax(m, x), Δ -> ∇mulax(Δ, m, x)
#end
#
#@adjoint function mulxa(m, x)
#	return mulxa(m, x), Δ -> ∇mulxa(Δ, m, x)
#end

@adjoint function mulax(m, x)
	o = UnitLowerTriangular(m) * UpperTriangular(m) * x
	return o, Δ -> ∇mulax_inv(Δ, m, o)
end

@adjoint function mulxa(m, x)
	o = x * UnitLowerTriangular(m) * UpperTriangular(m)
	return o, Δ -> ∇mulxa_inv(Δ, m, o)
end
