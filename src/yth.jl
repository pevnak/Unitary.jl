struct YUH{F} <: AbstractMatrix{F}
	Y::Matrix{F}
	U::Matrix{F}
end


Base.size(a::YUH) = size(a.Y)
Base.size(a::YUH, i) = size(a.Y, i)

Base.show(io, a::YUH) = pritnln(io, "Unitary matrix of size $(size(a.Y))")

function YUH(x::Matrix)
	@assert size(x,1) == size(x,2)
	a = YUH(Matrix(LowerTriangular(x)), similar(x))
	updateu!(a)
	a
end


Base.getindex(a::YUH{F}, i, j) where {F} = (i < j) ? zero(F) : a.Y[i,j]

function Base.copyto!(a::YUH, b::Matrix)
	for j in 1:size(a.Y, 2)
		for i in 1:size(a.Y, 1)
			i < j && continue
			a.Y[i,j] = b[i, j] 
		end
	end
	updateu!(a)
end

function Base.copyto!(a::YUH, bc::Base.Broadcast.Broadcasted)
	is, js = axes(bc)
	for j in js 
		for i in is 
			i < j && continue
			a.Y[i,j] = bc[i,j]
		end
	end
	updateu!(a)
end

@inline HH_t(Y::AbstractMatrix, i::Int) = 2 / sum((@view Y[:, i]).^2)
@inline HH_t(Y::LowerTriangular, i::Int) = 2 / sum(Y[j,i]^2 for j in i:size(Y,2))
@inline HH_t(y) = 2 / dot(y, y)

function T_matrix(Y::AbstractMatrix)
	n = size(Y, 1)
	@assert size(Y, 2) <= n
	T = zeros(eltype(Y), n, n)
	T[1, 1] = HH_t(Y, 1)
	@inbounds for i = 2:n
		T[i, i] = HH_t(Y, i)
		T[1:i-1, i] = @views -T[i, i] * T[1:i-1, 1:i-1] * Y[:, 1:i-1]' * Y[:, i]
	end
	T
end

function updateu!(a::YUH)
	T = T_matrix(a.Y)
	a.U .= I - a.Y*T*a.Y'
end
