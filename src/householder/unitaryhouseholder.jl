using Zygote:@adjoint
using LinearAlgebra

struct UnitaryHouseholder{T}
	Y::Matrix{T}
	transposed::Bool
	n::Int
end

Flux.@functor UnitaryHouseholder
Flux.trainable(m::UnitaryHouseholder) = (m.Y,)

#Constructors#
UnitaryHouseholder(n::Int) = UnitaryHouseholder(Float64, n)

function UnitaryHouseholder(T::DataType, n::Int)
	Y = Matrix(LowerTriangular(rand(T, n, n)))
	UnitaryHouseholder(Y, false, n)
end

function UnitaryHouseholder(Y::AbstractMatrix)
	@assert size(Y, 1) == size(Y, 2)
	UnitaryHouseholder(Matrix(LowerTriangular(Y)), false, size(Y, 1))
end

#Basic functions
Base.size(a::UnitaryHouseholder) = (a.n, a.n)
Base.size(a::UnitaryHouseholder, i::Int) = a.n

Base.eltype(a::UnitaryHouseholder{T}) where {T} = T
LinearAlgebra.transpose(a::UnitaryHouseholder) = UnitaryHouseholder(a.Y, !a.transposed, a.n)
Base.inv(a::UnitaryHouseholder) = LinearAlgebra.transpose(a)
Base.show(io::IO, ::MIME"text/plain", a::UnitaryHouseholder) = print(io, "$(a.n)x$(a.n) UnitaryHouseholder")

@inline HH_t(Y::AbstractMatrix, i::Int) = 2 / sum((@view Y[:, i]).^2)
@inline HH_t_LT(Y::AbstractMatrix, i::Int) = 2 / sum((@view Y[i:end, i]).^2)
@inline HH_t_UT(Y::AbstractMatrix, i::Int) = 2 / sum((@view Y[1:i, i]).^2)
@inline HH_t(y) = 2 / dot(y, y)
@inline HH_t(y, n) = 2 / dot(y[n:end], y[n:end])

function mulax(Y, x, transposed, n)
	out = deepcopy(x)
	mulax!(Y, out, transposed, n)
	out
end

function mulxa(Y, x, transposed, n)
	out = deepcopy(x)
	mulxa!(Y, out, transposed, n)
	out
end

function mulax!(y, x, n)
	# multiply by single reflection defined by y
	# n - first nonzero component of y
	t = HH_t(y, n)
	@inbounds for j = 1:size(x, 2)
		tmp = t * dot(y[n:end], x[n:end, j])
		for k = n:size(x, 1)
			x[k, j] -= tmp * y[k]
		end
	end
end

function mulax!(Y, x, transposed, n)
	ran = transposed ? (1:n) : (n:-1:1)
	@inbounds for i = ran
		mulax!(Y[:, i], x, i)
	end
end

function mulxa!(y, x, n)
	t = HH_t(y[1:end], n)
	@inbounds for j = 1:size(x, 1)
		tmp = t * dot(y[n:end], x[j, n:end])
		for k = n:size(x, 2)
			x[j, k] -= tmp * y[k]
		end
	end
end

function mulxa!(Y, x, transposed, n)
	ran = transposed ? (n:-1:1) : (1:n)
	@inbounds for i = ran
		mulxa!(Y[:, i], x, i)
	end
end

function Base.Matrix(a::UnitaryHouseholder)
	out = Matrix{eltype(a)}(I, a.n, a.n)
	mulax!(a.Y, out, a.transposed, a.n)
	out
end

import Base: *
function *(a::UnitaryHouseholder, x) #AbstractMatVec)
	@assert size(x, 1) == a.n
	mulax(a.Y, x, a.transposed, a.n)
end

function *(x::AbstractMatrix, a::UnitaryHouseholder) #AbstractMatVec
	@assert size(x, 2) == a.n
	mulxa(a.Y, x, a.transposed, a.n)
end

function pdiff_t(y, b::Int)
	- HH_t(y)^2 * y[b]
end

function pdiff_reflect(y, b::Int)
	t = HH_t(y)
	ty = t * y
	out = y[b] * ty * ty'
	@inbounds for i in 1:length(y)
		out[i, b] -= ty[i]
		out[b, i] -= ty[i]
	end
	out
end

function grad_mulax(Δ, Y, transposed, o)
	∇Y = zero(Y)
	Δ = deepcopy(Matrix(Δ))
	n = size(Y, 2)
	ran = transposed ? (n:-1:1) : (1:n)
	@inbounds for k = ran
		mulax!(Y[:, k], o, k)
		for i = k:n
			∇Y[i, k] = sum(Δ.*(pdiff_reflect(Y[:, k], i)*o))
		end
		mulax!(Y[:, k], Δ, k)
	end
	(∇Y, Δ, nothing, nothing)
end

function grad_mulxa(Δ, Y, transposed, o)
	∇Y = zero(Y)
	Δ = deepcopy(Matrix(Δ))
	n = size(Y, 2)
	ran = transposed ? (1:n) : (n:-1:1)
	@inbounds for k = ran
		mulxa!(Y[:, k], o, k)
		for i = k:n
			∇Y[i, k] = sum(Δ.*(o*pdiff_reflect(Y[:, k], i)))
		end
		mulxa!(Y[:, k], Δ, k)
	end
	(∇Y, Δ, nothing, nothing)
end

@adjoint function mulax(Y::AbstractMatrix, x::AbstractMatVec, transposed::Bool, n::Int)
	o = mulax(Y, x, transposed, n)
	return o, Δ -> grad_mulax(Δ, Y, transposed, o)
end

@adjoint function mulxa(Y::AbstractMatrix, x::AbstractMatVec, transposed::Bool, n::Int)
	o = mulxa(Y, x, transposed, n)
	return o, Δ -> grad_mulxa(Δ, Y, transposed, o)
end
