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


function *(a::DiagonalRectangular, x::TransposedMatVec)
	@assert a.m == size(x,1)
	if a.m == a.n 
		return( a.d .* x)
	else
		return(a.d .* x[1:a.n, :])
	end
end

function *(x::TransposedMatVec, a::DiagonalRectangular)
	@assert a.n == size(x,2)
	if a.m == a.n 
		return(transpose(a.d) .* x)
	else
		return(transpose(a.d) .* x[:, 1:a.m])
	end
end

transpose(a::DiagonalRectangular) = DiagonalRectangular(a.d, a.m, a.n)