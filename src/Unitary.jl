module Unitary

using Flux, LinearAlgebra
import Base.*

struct UnitaryMatrix{T}
	θ::T	
end

Flux.@treelike(UnitaryMatrix)


"""
	Matrix(a::UnitaryMatrix{T})

	create a dense matrix from the unitary
"""
function Base.Matrix(a::UnitaryMatrix{T}) where {T<:Vector}
	θ = a.θ[1]
	[cos(θ)  (- sin(θ)); sin(θ) cos(θ)]
end

#TODO: do this properly
Base.size(a::UnitaryMatrix) = (2,2)
Base.size(a::UnitaryMatrix, i::Int) = (i, i)
Base.length(a::UnitaryMatrix) = 4

LinearAlgebra.transpose(a::UnitaryMatrix) = LinearAlgebra.Transpose(a)

function *(a::UnitaryMatrix{T}, x) where {T<:Vector}
	θ = a.θ[1]
	_mulax!(similar(x), a, x, sin(θ), cos(θ))
end

function *(a::Transpose{X,A}, x::AbstractArray{T,2}) where {X, A<:UnitaryMatrix, T}
	θ = a.parent.θ[1]
	_mulax!(similar(x), a, x, - sin(θ), cos(θ))
end
	

function _mulax!(o, a, x, sinθ, cosθ)
	@assert size(x, 1) == 2
	@inbounds for i in 1:size(x, 2)
		o[1, i] =  cosθ * x[1,i] - sinθ * x[2,i]
		o[2, i] =  sinθ * x[1,i] + cosθ * x[2,i]
	end
	o
end

function *(x, a::UnitaryMatrix{T}) where {T<:Vector}
	θ = a.θ[1]
	_mulxa!(similar(x), a, x, sin(θ), cos(θ))
end

function _mulxa!(o, a, x, sinθ, cosθ)
	@assert size(x, 2) == 2
	for i in 1:size(x, 1)
		o[i, 1] =    cosθ * x[i, 1] + sinθ * x[i, 2]
		o[i, 2] =  - sinθ * x[i, 1] + cosθ * x[i, 2]
	end
	o
end

function *(x::AbstractArray{T,2}, a::Transpose{X, A}) where {X, A<:UnitaryMatrix, T}
	θ = a.parent.θ[1]
	_mulxa!(similar(x), a, x, - sin(θ), cos(θ))
end



*(a::UnitaryMatrix{T}, b::AbstractMatrix) where {T<: TrackedArray} = Flux.Tracker.track(mul, a, b)
*(a::AbstractMatrix, b::UnitaryMatrix{T}) where {T<: TrackedArray} = Flux.Tracker.track(mul, a, b)
# Flux.Tracker.@grad function mul(a::Flux.Tracker.TrackedMatrix, b::NGramMatrix)
#   return mul(Flux.data(a),b) , Δ -> (multrans(Δ, b),nothing)
# end

# UnitaryMatrix{T} where {T<: TrackedArray}

end # module
