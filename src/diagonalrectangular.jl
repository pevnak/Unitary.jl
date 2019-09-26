"""
	DiagonalRectangular{T}

	A rectangular matrix of size (n,m) with a non-zero elements 
	on diagonal only. For example

	```
	[1 0 0
	 0 1 0]
	```

	or 

	```
	[ 1 0
	  0 1
	  0 0]
"""	
struct DiagonalRectangular{T<:Number} 
	d::Vector{T}
	n::Int 
	m::Int
end

function DiagonalRectangular(x::T, n, m) where {T<:Number} 
	DiagonalRectangular(fill(x, min(n,m)), n, m)
end

DiagonalRectangular(x::T, n) where {T<:Number} = DiagonalRectangular(x, n, n)


function *(a::DiagonalRectangular{T}, x::TransposedMatVec) where {T}
	@assert a.m == size(x,1)
	if a.m == a.n 
		return(a.d .* x)
	elseif a.n < a.m
		return(a.d .* x[1:a.n, :])
	else
		o = zeros(T, a.n, size(x,2)) 
		o[1:a.m, :] .= a.d .* x
		return(o)
	end
end

function *(x::TransposedMatVec, a::DiagonalRectangular{T}) where {T}
	@assert a.n == size(x,2)
	if a.m == a.n 
		return(transpose(a.d) .* x)
	elseif a.m < a.n
		return(transpose(a.d) .* x[:, 1:a.m])
	else
		o = zeros(T, size(x,1), a.m)
		o[:, 1:a.n] .= transpose(a.d) .* x
		return(o)
	end
end

transpose(a::DiagonalRectangular) = DiagonalRectangular(a.d, a.m, a.n)
Base.inv(a::DiagonalRectangular) = DiagonalRectangular(1 ./ a.d, a.m, a.n)
LinearAlgebra.logabsdet(a::DiagonalRectangular{T}) where {T} = sum(log.(a.d .+ eps(T)))
Base.size(a::DiagonalRectangular) = (a.n, a.m)
function Base.size(a::DiagonalRectangular, i::Int)
	if i == 1 
		return(a.n)
	elseif a == 2
		return(a.m)
	else
		@error "DiagonalRectangular has only two dimensions"
	end
end

function Base.Matrix(a::DiagonalRectangular{T}) where {T}
	x = zeros(T, a.n, a.m)
	for i in 1:min(a.m, a.n)
		x[i,i] = a.d[i]
	end
	x
end
