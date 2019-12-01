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

Flux.@functor DiagonalRectangular

function DiagonalRectangular(x::T, n, m) where {T<:Number} 
	DiagonalRectangular(fill(x, min(n,m)), n, m)
end

DiagonalRectangular(x::T, n) where {T<:Number} = DiagonalRectangular(x, n, n)


*(a::DiagonalRectangular, x::AbstractMatrix) = diagmul(a.d, a.n, a.m, x)
*(a::DiagonalRectangular, x::AbstractVector) = diagmul(a.d, a.n, a.m, x)

function diagmul(d::Vector{T}, n::Int, m::Int, x) where {T}	
	@assert m == size(x,1)
	if m == n 
		return(d .* x)
	elseif n < m
		return(d .* x[1:n, :])
	else
		o = zeros(T, n, size(x,2)) 
		o[1:m, :] .= d .* x
		return(o)
	end
end

function ∇diagmul(Δ, d::Vector{T}, n::Int, m::Int, a) where {T}	
	if m == n 
		return(sum(Δ .* a, dims = 2)[:])
	elseif m < n
		return(sum(Δ[1:m, :] .* a, dims = 2))
	else
		return(sum(Δ .* a[1:n, :], dims = 2)[:])
	end
end

@adjoint function diagmul(d::Vector, n::Int, m::Int, a)
	return diagmul(d, n, m, a) , Δ -> (∇diagmul(Δ, d, n, m, a), nothing, nothing, diagmul(d, m, n, Δ))
end

*(x::AbstractVector, a::DiagonalRectangular) = diagmul(x, a.d, a.n, a.m)
*(x::AbstractMatrix, a::DiagonalRectangular) = diagmul(x, a.d, a.n, a.m)
function diagmul(x, d::Vector{T}, n::Int, m::Int) where {T}
	@assert n == size(x,2)
	if m == n 
		return(transpose(d) .* x)
	elseif m < n
		return(transpose(d) .* x[:, 1:m])
	else
		o = zeros(T, size(x,1), m)
		o[:, 1:n] .= transpose(d) .* x
		return(o)
	end
end

function ∇diagmul(Δ, a, d::Vector{T}, n::Int, m::Int) where {T}
	if m == n 
		return(sum(Δ .* a, dims = 1)[:])
	elseif m < n
		return(sum(Δ .* a[:, 1:m], dims = 1)[:])
	else
		return(sum(Δ[:,1:n] .* a, dims = 1)[:])
	end
end

@adjoint function diagmul(a, d::Vector, n::Int, m::Int)
	return diagmul(a, d, n, m) , Δ -> (diagmul(Δ, d, m, n),  ∇diagmul(Δ, a, d, n, m), nothing, nothing)
end

_logabsdet(a::DiagonalRectangular{T}) where {T<:Number} = sum(log.(abs.(a.d) .+ eps(T)))

Base.transpose(a::DiagonalRectangular) = DiagonalRectangular(a.d, a.m, a.n)
Base.inv(a::DiagonalRectangular) = DiagonalRectangular(1 ./ a.d, a.m, a.n)
LinearAlgebra.logabsdet(a::DiagonalRectangular) = _logabsdet(a.d)
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
