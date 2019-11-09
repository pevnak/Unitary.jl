struct UnitaryHouseholder{T<:Real}
	Y::AbstractMatrix{T}
	transposed::Bool
	n::Int
end

UnitaryHouseholder(n::Int) = UnitaryHouseholder(LowerTriangular(rand(n, n)), false, n) 

Base.size(a::UnitaryHouseholder) = (a.n, a.n)
Base.size(a::UnitaryHouseholder, i::Int) = a.n

Base.eltype(a::UnitaryHouseholder{T}) where {T} = T
LinearAlgebra.transpose(a::UnitaryHouseholder) = UnitaryHouseholder(a.Y, !a.transposed, a.n)
Base.inv(a::UnitaryHouseholder) = LinearAlgebra.transpose(a)
Base.show(io::IO, a::UnitaryHouseholder) = print(io, "$(a.n)x$(a.n) UnitaryHouseholder")


HH_t(Y::AbstractMatrix, i::Int) = 2 / (Y[:, i]' * Y[:, i])

function T_matrix(a::UnitaryHouseholder)
	@assert size(a.Y, 2) <= a.n
	T = UpperTriangular(Array{eltype(a.Y)}(undef, a.n, a.n))
	T[1, 1] = HH_t(a.Y, 1)
	@inbounds for i = 2:a.n
		T[i, i] = HH_t(a.Y, i)
		T[1:i-1, i] = -T[i, i] * T[1:i-1, 1:i-1] * a.Y[:, 1:i-1]' * a.Y[:, i]
	end
	a.transposed ? T' : T
end

Base.Matrix(a::UnitaryHouseholder) = I - a.Y*T_matrix(a)*(a.Y)'


function *(a::UnitaryHouseholder, x::AbstractMatVec)
	@assert size(x, 1) == a.n
	Base.Matrix(a) * x
end

function *(x::AbstractMatVec, a::UnitaryHouseholder)
	@assert size(x, 2) == a.n
	x * Base.Matrix(a)
end
