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
Flux.data(a::UnitaryMatrix) = UnitaryMatrix(Flux.data(a.θ))

LinearAlgebra.transpose(a::UnitaryMatrix) = LinearAlgebra.Transpose(a)

*(a::UnitaryMatrix{T}, x) where {T<:Vector} = _mulax(x, a.θ)
_mulax(x, θ) = _mulax(x , sin(θ[1]), cos(θ[1]))

*(a::Transpose{X,A}, x::AbstractArray{T,2}) where {X, A<:UnitaryMatrix, T} = _mulatx(x, a.parent.θ)
_mulatx(x, θ) = _mulax(x , - sin(θ[1]), cos(θ[1]))

function _mulax(x, sinθ, cosθ)
	@assert size(x, 1) == 2
	o = similar(x)
	@inbounds for i in 1:size(x, 2)
		o[1, i] =  cosθ * x[1,i] - sinθ * x[2,i]
		o[2, i] =  sinθ * x[1,i] + cosθ * x[2,i]
	end
	o
end

_∇mulax(θ, Δ, x) = _∇mulax(sin(θ[1]), cos(θ[1]), Δ, x)
function _∇mulax(sinθ, cosθ, Δ, x::AbstractArray{T,2}) where {P<:TrackedArray, T}
	∇θ = similar(Flux.data(Δ), 1)
	fill!(∇θ, 0)
	for i in 1:size(x, 2)
		∇θ[1] +=  Δ[1,i] * (- sinθ * x[1,i] - cosθ * x[2,i])
		∇θ[1] +=  Δ[2,i] * (  cosθ * x[1,i] - sinθ * x[2,i])
	end
	∇θ
end

*(x, a::UnitaryMatrix{T}) where {T<:Vector} = _mulxa(x, a.θ)
_mulxa(x, θ) = _mulxa(x, sin(θ[1]), cos(θ[1]))

*(x::AbstractArray{T,2}, a::Transpose{X, A}) where {X, A<:UnitaryMatrix, T} = _mulxat(x, a.parent.θ)
_mulxat(x, θ) = _mulxa(x, - sin(θ[1]), cos(θ[1]))

function _mulxa(x, sinθ, cosθ)
	@assert size(x, 2) == 2
	o = similar(x)
	for i in 1:size(x, 1)
		o[i, 1] =    cosθ * x[i, 1] + sinθ * x[i, 2]
		o[i, 2] =  - sinθ * x[i, 1] + cosθ * x[i, 2]
	end
	o
end

# *(a::UnitaryMatrix{T}, b::AbstractMatrix) where {T<: TrackedArray} = Flux.Tracker.track(*, a, b)
# *(a::AbstractMatrix, b::UnitaryMatrix{T}) where {T<: TrackedArray} = Flux.Tracker.track(*, a, b)
# Flux.Tracker.@grad function *(a::UnitaryMatrix{P}, b) where {P<:TrackedArray}
#   return Flux.data(a) * b , Δ -> (Δ * transpose(b), transpose(a) * Δ)
# end

# @grad a::AbstractMatrix * b::AbstractVecOrMat =
#   data(a)*data(b), Δ -> (Δ * transpose(b), transpose(a) * Δ)

# UnitaryMatrix{T} where {T<: TrackedArray}

end # module
