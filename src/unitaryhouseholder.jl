struct UnitaryHouseholder{T<:Real}
	Y::AbstractMatrix{T}
	T::AbstractMatrix{T}
	transposed::Bool
	n::Int
end


HH_t(Y::AbstractMatrix, i::Int) = 2 / (Y[:, i]' * Y[:, i])


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


function UnitaryHouseholder(n::Int)
	Y = LowerTriangular(rand(n, n))
	UnitaryHouseholder(Y, T_matrix(Y), false, n)
end

Base.size(a::UnitaryHouseholder) = (a.n, a.n)
Base.size(a::UnitaryHouseholder, i::Int) = a.n

Base.eltype(a::UnitaryHouseholder{T}) where {T} = T
LinearAlgebra.transpose(a::UnitaryHouseholder) = UnitaryHouseholder(a.Y, a.T', !a.transposed, a.n)
Base.inv(a::UnitaryHouseholder) = LinearAlgebra.transpose(a)
Base.show(io::IO, a::UnitaryHouseholder) = print(io, "$(a.n)x$(a.n) UnitaryHouseholder")



Base.Matrix(a::UnitaryHouseholder) = I - a.Y*a.T*(a.Y)'


function *(a::UnitaryHouseholder, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	x - a.Y * a.T * a.Y' * x
end

function *(x::AbstractMatVec, a::UnitaryHouseholder)
	@assert size(x, 2) == a.n
	x - x * a.Y * a.T * a.Y'
end

function pdiff_t(y::Vector, b::Int)
	- HH_t(y)^2 * y[b]
end

function pdiff_reflect(y::Vector, b::Int)
	out = - pdiff_t(y, b) * y *y'
	t = HH_t(y)
	out[:, b] -= t*y
	out[b, :] -= t*y
	out
end

function pdiff(Y::AbstractMatrix, T::AbstractMatrix, transposed::Bool, a::Int, b::Int)
	n = size(Y, 1)
	@assert 1 <= a <= b <= n
	out = Array{eltype(Y), 2}(undef, n, n)
	leading = (a==1 ? I : Y[:, 1:(a-1)] * T[1:(a-1), 1:(a-1)]) * Y[:, 1:(a-1)]'
	tailing = (d==a ? I : Y[:, (a+1):n] * T[(a+1):n, (a+1):n]) * Y[:, (a+1):n]'
	if transposed
		leading, tailing = tailing, leading
	end
	leading * pdiff_reflect(Y[:, a], b) * tailing
end

function diff_U(Y::AbstractMatrix, T::AbstractMatrix, transposed::Bool, δY::AbstractMatrix)
# Y and δY must be lower triangular
	b, a = size(Y)
	δU = zeros(eltype(Y), b, b)
	for i = 1:(a==b ? a-1 : a)
		#pdiff with regards to the last element is constant zero matrix
		leading = (i==1 ? I : U_matrix(Y[:, 1:(i-1)], T[1:(i-1), 1:(i-1)]))
		tailing = (i==a ? I : U_matrix(Y[:, (i+1):a], T[(i+1):a, (i+1):a]))
		if transposed
			leading, tailing = tailing, leading
		end
		for j = i:b
			display((i, j))
			δU[:, i:b] += (leading * pdiff_reflect(Y[:, i], j) * tailing)[:, i:b] * δY[j, i]
		end
	end
	δU
end
