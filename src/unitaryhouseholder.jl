struct UnitaryHouseholder{M<:AbstractMatrix}
	Y::M
	transposed::Bool
	n::Int
end

Flux.children(u::UnitaryHouseholder) = ([u.Y], )
Flux.mapchildren(f, u::UnitaryHouseholder) = f.(Flux.children((u, )))

HH_t(Y::AbstractMatrix, i::Int) = 2 / (Y[:, i]' * Y[:, i])
HH_t(y::Vector) = 2 / (y' * y)

function T_matrix(Y::AbstractMatrix)
	n = size(Y, 1)
	@assert size(Y, 2) <= n
	T = UpperTriangular(Array{eltype(Y)}(undef, n, n))
	T[1, 1] = HH_t(Y, 1)
	@inbounds for i = 2:n
		T[i, i] = HH_t(Y, i)
		T[1:i-1, i] = -T[i, i] * T[1:i-1, 1:i-1] * Y[:, 1:i-1]' * Y[:, i]
	end
	T
end

Zygote.@nograd T_matrix

#Constructors#
UnitaryHouseholder(n::Int) = UnitaryHouseholder(Float64, n)

function UnitaryHouseholder(T::DataType, n::Int)
	Y = LowerTriangular(rand(T, n, n))
	UnitaryHouseholder(Y, false, n)
end

function UnitaryHouseholder(Y::AbstractMatrix)
	@assert size(Y, 1) == size(Y, 2)
	Y = LowerTriangular(Y)
	UnitaryHouseholder(Y, false, size(Y, 1))
end


#Basic functions
Base.size(a::UnitaryHouseholder) = (a.n, a.n)
Base.size(a::UnitaryHouseholder, i::Int) = a.n

Base.eltype(a::UnitaryHouseholder{T}) where {T} = T
LinearAlgebra.transpose(a::UnitaryHouseholder) = UnitaryHouseholder(a.Y, !a.transposed, a.n)
Base.inv(a::UnitaryHouseholder) = LinearAlgebra.transpose(a)
Base.show(io::IO, a::UnitaryHouseholder) = print(io, "$(a.n)x$(a.n) UnitaryHouseholder")

function Base.Matrix(a::UnitaryHouseholder)
	T = (a.transposed ? T_matrix(a.Y)' : T_matrix(a.Y))
	I - a.Y*T*(a.Y)'
end

function mulax(Y, x, transposed)
	T = (transposed ? T_matrix(Y)' : T_matrix(Y))
	x - Y * T * Y' * x
end

function mulax(Y, x, transposed, T)
	x - Y * T * Y' * x
end

function mulxa(x, Y, transposed)
	T = (transposed ? T_matrix(Y)' : T_matrix(Y))
	x - x * Y * T * Y'
end

function mulxa(x, Y, transposed, T)
	x - x * Y * T' * Y'
end

function *(a::UnitaryHouseholder, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulax(a.Y, x, a.transposed)
end

function *(x::AbstractMatVec, a::UnitaryHouseholder)
	@assert size(x, 2) == a.n
	mulxa(x, a.Y, a.transposed)
end

function pdiff_t(y::Vector, b::Int)
	- HH_t(y)^2 * y[b]
end

function pdiff_reflect(y::Vector, b::Int)
	out = - pdiff_t(y, b) * y *y'
	t = HH_t(y)
	@inbounds out[:, b] -= t*y
	@inbounds out[b, :] -= t*y
	out
end

function pdiff(Y::AbstractMatrix, T::AbstractMatrix, transposed::Bool, a::Int, b::Int)
# a - vector index
# b - vector element index
	n = size(Y, 1)
	@assert 1 <= a <= b <= n
	out = Array{eltype(Y), 2}(undef, n, n)
	@inbounds leading = (a==1 ? I : I - Y[:, 1:(a-1)] * T[1:(a-1), 1:(a-1)] * Y[:, 1:(a-1)]')
	@inbounds tailing = (n==a ? I : I - Y[:, (a+1):n] * T[(a+1):n, (a+1):n] * Y[:, (a+1):n]')
	if transposed
		leading, tailing = tailing, leading
	end
	@inbounds leading * pdiff_reflect(Y[:, a], b) * tailing
end

function diff_U(Y::AbstractMatrix, T::AbstractMatrix, transposed::Bool, δY)
# Y and δY must be lower triangular
	b, a = size(Y)
	δU = zeros(eltype(Y), b, b)
	for i = 1:(a==b ? a-1 : a)
		#pdiff with regards to the last element is constant zero matrix
		@inbounds leading = (i==1 ? I : I - Y[:, 1:(i-1)] * T[1:(i-1), 1:(i-1)] * Y[:, 1:(i-1)]')
		@inbounds tailing = (a==i ? I : I - Y[:, (i+1):a] * T[(i+1):a, (i+1):a] * Y[:, (i+1):a]')
		if transposed
			leading, tailing = tailing, leading
		end
		@inbounds for j = i:b
			δU[:, i:b] += (leading * pdiff_reflect(Y[:, i], j) * tailing)[:, i:b] * δY[j, i]
		end
	end
	δU
end


function grad_mul_Y(Y::AbstractMatrix, T::AbstractMatrix, transposed::Bool, x::AbstractMatVec, Δ)
# Y must be lower triangular
	b, a = size(Y)
	∇mul = zeros(eltype(Y), b, b)
	for i = 1:(a==b ? a-1 : a)
		#pdiff with regards to the last element is constant zero matrix
		@inbounds leading = (i==1 ? I : I - Y[:, 1:(i-1)] * T[1:(i-1), 1:(i-1)] * Y[:, 1:(i-1)]')
		@inbounds tailing = (a==i ? I : I - Y[:, (i+1):a] * T[(i+1):a, (i+1):a] * Y[:, (i+1):a]')
		if transposed
			leading, tailing = tailing, leading
		end
		@inbounds for j = i:b
			∇mul[j, i] = sum(Δ.*(leading * pdiff_reflect(Y[:, i], j) * tailing * x))
		end
	end
	∇mul
end


function grad_mul_x(Y::AbstractMatrix, T::AbstractMatrix, x::AbstractMatVec, Δ)
# Y must be lower triangular
	U = I - Y*T*Y'
	b = size(x, 1)
	a = ndims(x)==1 ? 1 : size(x, 2)
	∇mul = zeros(eltype(Y), b, a)
	for i = 1:a
		for j = 1:b
			@inbounds ∇mul[j, i] = sum(Δ[:, i].*U[:, j])
		end
	end
	∇mul
end


using Zygote:@adjoint


@adjoint function mulax(Y::AbstractMatrix, x::AbstractMatVec, transposed::Bool)
	T = (transposed ? T_matrix(Y)' : T_matrix(Y))
	return mulax(Y, x, transposed, T), Δ -> (grad_mul_Y(Y, T, transposed, x, Δ), grad_mul_x(Y, T, x, Δ), nothing)
end


#@adjoint function UnitaryHouseholder(Y, T, transposed, n)
#	return UnitaryHouseholder(Y, T, transposed, n), Δ -> (Δ.Y, Δ.T, Δ.transposed, Δ.n)
#end
