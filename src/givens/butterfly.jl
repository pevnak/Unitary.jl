struct Givens{T} 
	θs::Vector{T}
	idxs::Vector{Tuple{Int,Int}}
	n::Int
end

function Givens(n) 
	idxs = [(i,j) for i in 1:n for j in i+1:n];
	θs = rand(Float32,length(idxs))*2*π;
	Givens(θs, idxs, n)
end

struct TransposedGivens{B<:Givens} 
	parent::B
end

Flux.@functor Givens
Flux.@functor TransposedGivens

function Base.Matrix(a::Givens{T}) where {T}
	a * T.(Matrix(I(a.n))) 
end

Base.Matrix(a::TransposedGivens) = transpose(Matrix(a.parent))


Base.size(a::Givens) = (a.n,a.n)
Base.size(a::Givens, i::Int) = a.n
Base.size(a::TransposedGivens,i...) = size(a.parent)

Base.eltype(a::Givens{T}) where {T} = T
Base.eltype(a::TransposedGivens) = eltype(a.parent)
LinearAlgebra.transpose(a::Givens) = TransposedGivens(a)
LinearAlgebra.transpose(a::TransposedGivens) = a.parent
Base.inv(a::Givens) = transpose(a)
Base.inv(a::TransposedGivens) = transpose(a)
Base.show(io::IO, a::Givens) = print(io, "Givens ",a.θs)
Base.show(io::IO, a::TransposedGivens) = print(io, "Givensᵀ ",a.parent.θ)
Base.zero(a::Givens) = Givens(zero(a.θs), a.i, a.j, a.n)
Base.zero(a::TransposedGivens) = TransposedGivens(zero(a.parent))


Zygote.@adjoint function LinearAlgebra.transpose(x::TransposedGivens)
  return(transpose(x), Δ -> (TransposedGivens(Δ.θ),))
end

Zygote.@adjoint function LinearAlgebra.transpose(x::Givens)
  return transpose(x), Δ -> (Givens(Δ.parent.θs, x.idxs, x.n),)
end



*(a::Givens, x::TransposedMatVec) = (@assert a.n == size(x,1); _mulax(a.θs, a.idxs, x, 1))
*(x::TransposedMatVec, a::Givens) = (@assert a.n == size(x,2); _mulxa(x, a.θs, a.idxs, 1))
*(a::TransposedGivens, x::TransposedMatVec) = (@assert a.parent.n == size(x,1); _mulax(a.parent.θs, a.parent.idxs, x, -1))
*(x::TransposedMatVec, a::TransposedGivens) = (@assert a.parent.n == size(x,2); _mulxa(x, a.parent.θs, a.parent.idxs, -1))

"""
	_mulax(θ::Vector, x::MatVec)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
_mulax(θs, idxs, x, t) = _mulax!(deepcopy(x), θs, idxs, x, t)

@inline function _mulax!(o, cosθ::Number, sinθ::Number, i::Int, j::Int, x, t::Int)
	@inbounds for c in 1:size(x, 2)
		xi, xj = x[i,c], x[j,c]
		o[i, c] =  cosθ * xi - t*sinθ * xj
		o[j, c] =  t*sinθ * xi + cosθ * xj
	end
end

function _mulax!(o, θs, idxs, x, t)
	order = t == 1 ? (1:length(idxs)) : (length(idxs):-1:1)
	@inbounds for k in order
		sinθ, cosθ  = sincos(θs[k])
		_mulax!(o, cosθ, sinθ, idxs[k][1], idxs[k][2], o, t)
	end
	o
end

"""
	_∇mulax(θ, Δ, x)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
function _∇mulax(Δ, θs, idxs, o, t)
	∇θ = similar(θs)
	Δ = deepcopy(Matrix(Δ))
	order = t == 1 ? (length(idxs):-1:1) : (1:length(idxs))
	@inbounds for k in order
		sinθ, cosθ  = sincos(θs[k])
		i,j = idxs[k][1], idxs[k][2]
		_mulax!(o, cosθ, sinθ, i, j, o, -t) #compute the input
		∇θ[k] = _∇mulax(Δ, cosθ, sinθ, i, j, o, t) #compute the gradient
		_mulax!(Δ, cosθ, sinθ, i, j, Δ, -t) # computer the gradient of input
	end
	(∇θ, nothing, Δ, nothing)
end

@inline function _∇mulax(Δ, cosθ::Number, sinθ::Number, i::Int, j::Int, x, t::Int)
	Δθ = zero(cosθ)
	@inbounds for c in 1:size(Δ,2)
		Δθ += Δ[i,c] * (- sinθ * x[i,c] - t*cosθ * x[j,c])
		Δθ += Δ[j,c] * (  t*cosθ * x[i,c] - sinθ * x[j,c])
	end
	return(Δθ)
end

_mulxa(x, θs, idxs, t) = _mulxa!(deepcopy(x), x, θs, idxs, t)

function _mulxa!(o, x, θs, idxs, t)
	order = t == 1 ? (length(idxs):-1:1) : (1:length(idxs))
	@inbounds for k in order
		sinθ, cosθ  = sincos(θs[k])
		_mulxa!(o, o, cosθ, sinθ, idxs[k][1], idxs[k][2], t)
	end
	o
end

@inline function _mulxa!(o, x, cosθ::Number, sinθ::Number, i::Int, j::Int, t::Int)
	@inbounds for c in 1:size(x, 1)
		xi, xj = x[c, i], x[c, j]	#this makes it safe to rewrite the inpur
		o[c, i] =    cosθ * xi + t*sinθ * xj
		o[c, j] =  - t*sinθ * xi + cosθ * xj
	end
end

function _∇mulxa(Δ, o, θs, idxs, t)
	∇θ = similar(θs)
	Δ = deepcopy(Matrix(Δ))
	order = t == 1 ? (1:length(idxs)) : (length(idxs):-1:1)
	@inbounds for k in order
		sinθ, cosθ  = sincos(θs[k])
		i,j = idxs[k][1], idxs[k][2]
		_mulxa!(o, o, cosθ, sinθ, i, j, -t) #compute the input
		∇θ[k] = _∇mulxa(Δ, o, cosθ, sinθ, i, j, t) #compute the gradient
		_mulxa!(Δ, Δ, cosθ, sinθ, i, j, -t) # computer the gradient of input
	end
	(Δ, ∇θ, nothing, nothing)
end

@inline function _∇mulxa(Δ, x, cosθ::Number, sinθ::Number, i::Int, j::Int, t::Int)
	Δθ = zero(cosθ)
	@inbounds for c in 1:size(Δ, 1)
		Δθ +=  Δ[c, i] * (-sinθ * x[c, i] + t*cosθ * x[c, j])
		Δθ +=  Δ[c, j] * (-t*cosθ * x[c, i] - sinθ * x[c, j])
	end
	return(Δθ)
end


Zygote.@adjoint function _mulax(θs, idxs, x, t)
	o = _mulax(θs, idxs, x, t)
	return(o, Δ -> _∇mulax(Δ, θs, idxs, o, t))
end

Zygote.@adjoint function _mulxa(x, θs, idxs, t)
	o = _mulxa(x, θs, idxs, t)
	return(o, Δ -> _∇mulxa(Δ, o, θs, idxs, t))
end
