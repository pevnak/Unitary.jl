using Zygote:@adjoint

include("yth.jl")
struct UnitaryHouseholder{T}
	Y::YUH{T}
	transposed::Bool
	n::Int
end

Flux.@functor UnitaryHouseholder
Flux.trainable(m::UnitaryHouseholder) = (m.Y,)

#Constructors#
UnitaryHouseholder(n::Int) = UnitaryHouseholder(Float64, n)

function UnitaryHouseholder(T::DataType, n::Int)
	Y = YUH(rand(T, n, n))
	UnitaryHouseholder(Y, false, n)
end

function UnitaryHouseholder(Y::AbstractMatrix)
	@assert size(Y, 1) == size(Y, 2)
	Y = YUH(Y)
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
	a.transposed ? a.Y.U' : a.Y.U
end

function mulax(Y, x, transposed)
	transposed ? Y.U'*x : Y.U*x
end

function mulxa(x, Y, transposed)
	transposed ? x*Y.U' : x*Y.U
end

function *(a::UnitaryHouseholder, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	mulax(a.Y, x, a.transposed)
end

function *(x::AbstractMatVec, a::UnitaryHouseholder)
	@assert size(x, 2) == a.n
	mulxa(x, a.Y, a.transposed)
end

function pdiff_t(y, b::Int)
	- HH_t(y)^2 * y[b]
end

# function pdiff_reflect(y, b::Int)
# 	out = - pdiff_t(y, b) * y * y'
# 	t = HH_t(y)
# 	@inbounds out[:, b] -= t*y
# 	@inbounds out[b, :] -= t*y
# 	out
# end

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
	@inbounds for i = 1:(a==b ? a-1 : a)
		#pdiff with regards to the last element is constant zero matrix
		leading = (i==1 ? I : I - Y[:, 1:(i-1)] * T[1:(i-1), 1:(i-1)] * Y[:, 1:(i-1)]')
		tailing = (a==i ? I : I - Y[:, (i+1):a] * T[(i+1):a, (i+1):a] * Y[:, (i+1):a]')
		if transposed
			leading, tailing = tailing, leading
		end
		for j = i:b
			δU[:, i:b] += (leading * pdiff_reflect(Y[:, i], j) * tailing)[:, i:b] * δY[j, i]
		end
	end
	δU
end


function grad_mul_Y(Y::AbstractMatrix, T::AbstractMatrix, transposed::Bool, x::AbstractMatVec, Δ)
# Y must be lower triangular
# T must be lower triangular if transposed is true
	b, a = size(Y)
	@assert a==b
	n = a
	∇mul = zeros(eltype(Y), n, n)
	Tail = @views I - Y[:, 2:n] * T[2:n, 2:n] * Y[:, 2:n]'
	tmp = Array{eltype(Y), 2}(undef, n, n)
	if transposed
		@inbounds for j = 1:n
			∇mul[j, 1] = sum(Δ.*(Tail * pdiff_reflect(Y[:, 1], j) * x))
		end
		@inbounds for i = 2:(n-1)
			#pdiff with regards to the last element is constant zero matrix
			Lead = @views I - Y[:, 1:(i-1)] * T[1:(i-1), 1:(i-1)] * Y[:, 1:(i-1)]'
			Tail = @views I - Y[(i+1):n, (i+1):n] * T[(i+1):n, (i+1):n] * Y[(i+1):n, (i+1):n]'
			for j = i:n
				P = @views pdiff_reflect(Y[i:n, i], j-i+1)
				tmp[i, :] = @views P[1, 1] * Lead[i, :]' + P[1, 2:end]' * Lead[(i+1):n, :]
				tmp[(i+1):n, :] = @views Tail * (P[2:end, 1] * Lead[i, :]' + P[2:end, 2:end] * Lead[(i+1):n, :])
				∇mul[j, i] = @views sum(Δ[i:end, :] .* (tmp[i:end, :] * x))
			end
		end
	else
		@inbounds for j = 1:n
			∇mul[j, 1] = sum(Δ.*(pdiff_reflect(Y[:, 1], j) * Tail * x))
		end
		@inbounds for i = 2:(n-1)
			#pdiff with regards to the last element is constant zero matrix
			Lead = @views I - Y[:, 1:(i-1)] * T[1:(i-1), 1:(i-1)] * Y[:, 1:(i-1)]'
			Tail = @views I - Y[(i+1):n, (i+1):n] * T[(i+1):n, (i+1):n] * Y[(i+1):n, (i+1):n]'
			for j = i:n
				P = @views pdiff_reflect(Y[i:n, i], j-i+1)
				tmp[:, i] = @views Lead[:, i] * P[1, 1] + Lead[:, (i+1):end] * P[2:end, 1]
				tmp[:, (i+1):n] = @views (Lead[:, i] * P[1, 2:end]' + Lead[:, (i+1):end] * P[2:end, 2:end]) * Tail
				∇mul[j, i] = @views sum(Δ .*( tmp[:, i:end] * x[i:end, :]))
			end
		end
	end
	∇mul
end


function grad_mul_x(U::AbstractMatrix, x::AbstractMatVec, Δ)
	b = size(x, 1)
	a = ndims(x)==1 ? 1 : size(x, 2)
	∇mul = zeros(eltype(U), b, a)
	@inbounds for i = 1:a
		for j = 1:b
			∇mul[j, i] = @views sum( Δ[:, i] .* U[:, j])
		end
	end
	∇mul
end



@adjoint function mulax(b::YUH, x::AbstractMatVec, transposed::Bool)
	T = transposed ? T_matrix(b.Y)' : T_matrix(b.Y)
	return b.U*x, Δ -> (grad_mul_Y(b.Y, T, transposed, x, Δ), grad_mul_x(b.U, x, Δ), nothing)
end


#@adjoint function UnitaryHouseholder(Y, T, transposed, n)
#	return UnitaryHouseholder(Y, T, transposed, n), Δ -> (Δ.Y, Δ.T, Δ.transposed, Δ.n)
#end
