struct Butterfly{T} 
	θs::Vector{T}
	idxs::Vector{Tuple{Int,Int}}
	n::Int
end

function Butterfly(n) 
	idxs = [(i,j) for i in 1:n for j in i+1:n];
	θs = 2π*rand(Float32,length(idxs));
	Butterfly(θs, idxs, n)
end

struct TransposedButterfly{B<:Butterfly} 
	parent::B
end

Flux.@functor Butterfly
Flux.@functor TransposedButterfly

function Base.Matrix(a::Butterfly{T}) where {T}
	a * T.(Matrix(I(a.n))) 
end

Base.Matrix(a::TransposedButterfly) = transpose(Matrix(a.parent))


Base.size(a::Butterfly) = (a.n,a.n)
Base.size(a::Butterfly, i::Int) = a.n
Base.size(a::TransposedButterfly,i...) = size(a.parent)

Base.eltype(a::Butterfly{T}) where {T} = T
Base.eltype(a::TransposedButterfly) = eltype(a.parent)
LinearAlgebra.transpose(a::Butterfly) = TransposedButterfly(a)
LinearAlgebra.transpose(a::TransposedButterfly) = a.parent
Base.inv(a::Butterfly) = transpose(a)
Base.inv(a::TransposedButterfly) = transpose(a)
Base.show(io::IO, a::Butterfly) = print(io, "Butterfly ",a.θs)
Base.show(io::IO, a::TransposedButterfly) = print(io, "Butterflyᵀ ",a.parent.θ)
Base.zero(a::Butterfly) = Butterfly(zero(a.θs), a.i, a.j, a.n)
Base.zero(a::TransposedButterfly) = TransposedButterfly(zero(a.parent))


Zygote.@adjoint function LinearAlgebra.transpose(x::TransposedButterfly)
  return(transpose(x), Δ -> (TransposedButterfly(Δ.θ),))
end

Zygote.@adjoint function LinearAlgebra.transpose(x::Butterfly)
  return transpose(x), Δ -> (Butterfly(Δ.parent.θs, x.idxs, x.n),)
end



*(a::Butterfly, x::TransposedMatVec) = (@assert a.n == size(x,1); _mulax(a.θs, a.idxs, x, 1))
*(x::TransposedMatVec, a::Butterfly) = (@assert a.n == size(x,2); _mulxa(x, a.θs, a.idxs, 1))
*(a::TransposedButterfly, x::TransposedMatVec) = (@assert a.parent.n == size(x,1); _mulax(a.parent.θs, a.parent.idxs, x, -1))
*(x::TransposedMatVec, a::TransposedButterfly) = (@assert a.parent.n == size(x,2); _mulxa(x, a.parent.θs, a.parent.idxs, -1))

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
		cosθ, sinθ = cos(θs[k]), sin(θs[k])
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
		cosθ, sinθ = cos(θs[k]), sin(θs[k])
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
		cosθ, sinθ = cos(θs[k]), sin(θs[k])
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
		cosθ, sinθ = cos(θs[k]), sin(θs[k])
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
