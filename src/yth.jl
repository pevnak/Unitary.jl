struct YTH{F} <: AbstractMatrix{F}
	Y::LowerTriangular{F, Matrix{F}}
	T::Matrix{F}
end


Base.size(a::YTH) = size(a.Y)
Base.size(a::YTH, i) = size(a.Y, i)

Base.show(io, a::YTH) = pritnln(io, "Unitary matrix of size $(size(a.Y))")

function YTH(x::Matrix)
	@assert size(x,1) == size(x,2)
	a = YTH(LowerTriangular(x), similar(x))
	updatet!(a)
	a
end


Base.getindex(a::YTH{F}, i, j) where {F} = (i < j) ? zero(F) : a.Y[i,j]

function Base.copyto!(a::YTH, b::Matrix)
	for i in 1:size(a.Y, 1)
		for j in 1:size(a.Y, 2)
			i < j && continue
			a.Y[i,j] = b[i, j] 
		end
	end
	updatet!(a)
end

function Base.copyto!(a::YTH, bc::Base.Broadcast.Broadcasted)
	is, js = axes(bc)
	for j in js 
		for i in is 
			i < j && continue
			a.Y[i,j] = bc[i,j]
		end
	end
	updatet!(a)
end

function updatet!(a::YTH)
	Y = a.Y 
	T = a.T
	n = size(Y, 1)
	@assert size(Y, 2) <= n
	T .= 0
	T[1, 1] = HH_t(Y, 1)
	@inbounds for i = 2:n
		T[i, i] = HH_t(Y, i)
		T[1:i-1, i] = -T[i, i] * T[1:i-1, 1:i-1] * Y[:, 1:i-1]' * Y[:, i]
	end
end

HH_t(Y::AbstractMatrix, i::Int) = 2 / sum((@view Y[:, i]).^2)
HH_t(Y::LowerTriangular, i::Int) = 2 / sum(Y[j,i]^2 for j in i:size(Y,2))
HH_t(y::Vector) = 2 / dot(y, y)

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
