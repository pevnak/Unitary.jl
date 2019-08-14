
struct Butterfly{N, T<:Number} 
	θ::Vector{T}
	i::NTuple{N,Int}
	j::NTuple{N,Int}	
	n::Int
	function Butterfly(θ::Vector{T}, i::Vector, j::Vector, n) where {T<:Number}
		@assert length(θ) == length(i) == length(j)
		@assert maximum(i) <= n  
		@assert maximum(j) <= n  
		@assert isempty(intersect(i,j))
		N = length(θ)
		new{N,T}(θ, tuple(i...), tuple(j...), n)
	end
end

Butterfly(i, j, n) = Butterfly(2π*rand(length(i)), i, j, n)

struct TransposedButterfly{B<:Butterfly} 
	parent::B
end

Flux.@treelike(Butterfly)
Flux.@treelike(TransposedButterfly)


function Base.Matrix(a::Butterfly{N, T}) where {N, T}
	A = zeros(T, a.n, a.n)
	for k = 1:N
		θ, i, j = a.θ[k], a.i[k], a.j[k]
		A[i,i], A[i,j] =  cos(θ), -sin(θ);
		A[j,i], A[j,j] =  sin(θ),  cos(θ);		
	end
	A
end

Base.Matrix(a::TransposedButterfly) = transpose(Matrix(a.parent))


Base.size(a::Butterfly) = (a.n,a.n)
Base.size(a::Butterfly, i::Int) = a.n
Base.size(a::TransposedButterfly,i...) = size(a.parent)

Base.eltype(a::Butterfly{N,T}) where {N,T} = T
Base.eltype(a::TransposedButterfly) = eltype(a.parent)
LinearAlgebra.transpose(a::Butterfly) = TransposedButterfly(a)
LinearAlgebra.transpose(a::TransposedButterfly) = a.parent
Base.inv(a::Butterfly) = transpose(a)
Base.inv(a::TransposedButterfly) = transpose(a)
Base.show(io::IO, a::Butterfly) = print(io, "Butterfly ",a.θ)
Base.show(io::IO, a::TransposedButterfly) = print(io, "Butterflyᵀ ",a.parent.θ)
Base.zero(a::Butterfly) = Butterfly(zero(a.θ), a.i, a.j, a.n)
Base.zero(a::TransposedButterfly) = TransposedButterfly(zero(a.parent))


# Zygote.@adjoint function LinearAlgebra.transpose(x::TransposedButterfly)
#   return transpose(x), Δ -> (TransposedButterfly(Δ.θ),)
# end

# Zygote.@adjoint function LinearAlgebra.transpose(x::Butterfly)
#   return transpose(x), Δ -> (Butterfly(Δ.θ),)
# end



*(a::Butterfly, x::TransposedMatVec) = (@assert a.n == size(x,1); _mulax(a.θ, a.i, a.j, x, 1))
*(x::TransposedMatVec, a::Butterfly) = (@assert a.n == size(x,2); _mulxa(x, a.θ, a.i, a.j, 1))
*(a::TransposedButterfly, x::TransposedMatVec) = (@assert a.parent.n == size(x,1); _mulax(a.parent.θ, a.parent.i, a.parent.j, x, -1))
*(x::TransposedMatVec, a::TransposedButterfly) = (@assert a.parent.n == size(x,2); _mulxa(x, a.parent.θ, a.parent.i, a.parent.j, -1))

# # @adjoint Butterfly(θ) = Butterfly(θ), Δ -> (Butterfly(Δ),)
# # @adjoint TransposedButterfly(θ) = TransposedButterfly(θ), Δ -> (TransposedButterfly(Δ),)

"""
	_mulax(θ::Vector, x::MatVec)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
_mulax(θs, is, js, x, t) = _mulax!(deepcopy(x), θs, is, js, x, t)
function _mulax!(o, θs, is, js, x, t)
	@assert size(o) == size(x)
	cosθs, sinθs = cos.(θs), sin.(θs)
	for c in 1:size(x, 2)
		@inbounds for k = 1:length(is)
			sinθ, cosθ, i, j = sinθs[k], cosθs[k], is[k], js[k]	
			xi, xj = x[i,c], x[j,c]
			o[i, c] =  cosθ * xi - t*sinθ * xj
			o[j, c] =  t*sinθ * xi + cosθ * xj
		end
	end
	o
end

"""
	_∇mulax(θ, Δ, x)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
function _∇mulax(Δ, θs, is, js, x, t)
	∇θ = similar(θs)
	cosθs, sinθs = cos.(θs), sin.(θs)
	fill!(∇θ, 0)
	for c in 1:size(x, 2)
		@inbounds for k = 1:length(is)
			sinθ, cosθ, i, j = sinθs[k], cosθs[k], is[k], js[k]	
			∇θ[k] +=  Δ[i,c] * (- sinθ * x[i,c] - t*cosθ * x[j,c])
			∇θ[k] +=  Δ[j,c] * (  t*cosθ * x[i,c] - sinθ * x[j,c])
		end
	end
	∇θ
end

_mulxa(x, θs, is, js, t) = _mulxa!(deepcopy(x), x, θs, is, js, t)
function _mulxa!(o, x, θs, is, js, t)
	@assert size(o) == size(x)
	cosθs, sinθs = cos.(θs), sin.(θs)
	for c in 1:size(x, 1)
		@inbounds for k = 1:length(is)
			sinθ, cosθ, i, j = sinθs[k], cosθs[k], is[k], js[k]	
			xi, xj = x[c, i], x[c, j]
			o[c, i] =    cosθ * xi + t*sinθ * xj
			o[c, j] =  - t*sinθ * xi + cosθ * xj
		end
	end
	o
end

function _∇mulxa(Δ, x, θs, is, js, t)
	∇θ = similar(θs)
	cosθs, sinθs = cos.(θs), sin.(θs)
	fill!(∇θ, 0)
	for c in 1:size(x, 1)
		@inbounds for k = 1:length(is)
			sinθ, cosθ, i, j = sinθs[k], cosθs[k], is[k], js[k]	
			∇θ[k] +=  Δ[c, i] * (-sinθ * x[c, i] + t*cosθ * x[c, j])
			∇θ[k] +=  Δ[c, j] * (-t*cosθ * x[c, i] - sinθ * x[c, j])
		end
	end
	∇θ
end

@adjoint function _mulax(θs, is, js, x, t)
	return _mulax(θs, is, js, x, t) , Δ -> (_∇mulax(Δ, θs, is, js, x, t), nothing, nothing, _mulax(θs, is, js, Δ, -t))
end

@adjoint function _mulxa(x, θs, is, js, t)
	return _mulxa(x, θs, is, js, t) , Δ -> (_mulxa(Δ, θs, is, js, -t), _∇mulxa(Δ, x, θs, is, js, t), nothing, nothing, nothing)
end
