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

#Basic functions
import Base: size
size(a::lowup) = (a.n, a.n)
size(a::inverted_lowup) = (a.n, a.n)
Base.Matrix(a::lowup) = Matrixlu(a.m)
Matrixlu(m) = UnitLowerTriangular(m) * UpperTriangular(m)
Base.Matrix(a::inverted_lowup) = Matrixilu(a.m)
Matrixilu(m) = UpperTriangular(m) * UnitLowerTriangular(m)

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

function LinearAlgebra.transpose(a::lowup)
	m = deepcopy(transpose(a.m))
	for i = 1:a.n-1
		for j = i+1:a.n
			m[j, i] /= m[i, i]
			m[i, j] *= m[i, i]
		end
	end
	lowup(m)
end
function LinearAlgebra.transpose(a::inverted_lowup)
	m = deepcopy(transpose(a.m))
	for i = 2:a.n
		for j = 1:i-1
			m[j, i] *= m[i, i]
			m[i, j] /= m[i, i]
		end
	end
	inverted_lowup(m)
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

mulaxlu(m, x) = UnitLowerTriangular(m) * UpperTriangular(m) * x
mulaxilu(m, x) = UpperTriangular(m) * UnitLowerTriangular(m) * x

mulxalu(m, x) = x * UnitLowerTriangular(m) * UpperTriangular(m)
mulxailu(m, x) = x * UpperTriangular(m) * UnitLowerTriangular(m)

import Base: *
function *(a::lowup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulaxlu(a.m, x)
end
function *(x::AbstractMatVec, a::lowup)
	@assert size(x, 2) == a.n
	mulxalu(a.m, x)
end
function *(a::inverted_lowup, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulaxilu(a.m, x)
end
function *(x::AbstractMatVec, a::inverted_lowup)
	@assert size(x, 2) == a.n
	mulxailu(a.m, x)
end

function ∇mulaxlu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	x = deepcopy(x)
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
	(∇m, Δloc)
end

function ∇mulaxilu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	x = deepcopy(x)
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
	(∇m, Δloc)
end

function ∇mulxalu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	x = deepcopy(x)
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
	(∇m, Δloc)
end

function ∇mulxailu(Δ, m, x)
	∇m = zero(m)
	a, b = size(x)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	x = deepcopy(x)
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
	(∇m, Δloc)
end

@adjoint function mulaxlu(m, x)
	o = mulaxlu(m, x)
	return o, Δ -> ∇mulaxlu(Δ, m, x)
end

@adjoint function mulxalu(m, x)
	o = mulxalu(m, x)
	return o, Δ -> ∇mulxalu(Δ, m, x)
end

@adjoint function mulaxilu(m, x)
	o = mulaxilu(m, x)
	return o, Δ -> ∇mulaxilu(Δ, m, x)
end

@adjoint function mulxailu(m, x)
	o = mulxailu(m, x)
	return o, Δ -> ∇mulxailu(Δ, m, x)
end

function ∇Matrixlu(Δ, m)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	(UnitLowerTriangular(Δ*u') + UpperTriangular(l'*Δ) - I, )
end

function ∇Matrixilu(Δ, m)
	l = UnitLowerTriangular(m)
	u = UpperTriangular(m)
	(UnitLowerTriangular(u'*Δ) + UpperTriangular(Δ*l') - I, )
end

@adjoint function Matrixlu(m)
	return Matrixlu(m), Δ -> ∇Matrixlu(Δ, m)
end

@adjoint function Matrixilu(m)
	return Matrixilu(m), Δ -> ∇Matrixilu(Δ, m)
end
