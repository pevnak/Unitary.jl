module Unitary

using Flux, LinearAlgebra
import Base.*

struct UnitaryMatrix{T}
	θ::T	
end

Flux.param(a::UnitaryMatrix) = UnitaryMatrix(param(a.θ))
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

*(a::UnitaryMatrix, x) = _mulax(a.θ, x)
_mulax(θ, x) = _mulax((sin(θ[1]), cos(θ[1])), x)

*(a::Transpose{X,A}, x::AbstractArray{T,2}) where {X, A<:UnitaryMatrix, T} = _mulatx(a.parent.θ, x)
_mulatx(θ, x) = _mulax((- sin(θ[1]), cos(θ[1])), x)

function _mulax(sincosθ::Tuple, x)
	sinθ, cosθ = sincosθ
	@assert size(x, 1) == 2
	o = similar(x)
	for i in 1:size(x, 2)
		o[1, i] =  cosθ * x[1,i] - sinθ * x[2,i]
		o[2, i] =  sinθ * x[1,i] + cosθ * x[2,i]
	end
	o
end

_∇mulax(θ, Δ, x) = _∇mulax(Δ, (sin(θ[1]), cos(θ[1])), x)
function _∇mulax(Δ, sincosθ::Tuple, x::AbstractArray{T,2}) where {P<:TrackedArray, T}
	sinθ, cosθ = sincosθ
	∇θ = similar(Δ, 1)
	fill!(∇θ, 0)
	for i in 1:size(x, 2)
		∇θ[1] +=  Δ[1,i] * (- sinθ * x[1,i] - cosθ * x[2,i])
		∇θ[1] +=  Δ[2,i] * (  cosθ * x[1,i] - sinθ * x[2,i])
	end
	∇θ
end

*(x, a::UnitaryMatrix) = _mulxa(x, a.θ)
_mulxa(x, θ) = _mulxa(x, (sin(θ[1]), cos(θ[1])))

*(x::AbstractArray{T,2}, a::Transpose{X, A}) where {X, A<:UnitaryMatrix, T} = _mulxat(x, a.parent.θ)
_mulxat(x, θ) = _mulxa(x, (- sin(θ[1]), cos(θ[1])))

function _mulxa(x, sincosθ::Tuple)
	sinθ, cosθ = sincosθ 
	@assert size(x, 2) == 2
	o = similar(x)
	for i in 1:size(x, 1)
		o[i, 1] =    cosθ * x[i, 1] + sinθ * x[i, 2]
		o[i, 2] =  - sinθ * x[i, 1] + cosθ * x[i, 2]
	end
	o
end

_mulax(a::TrackedArray, x::AbstractMatrix) = Flux.Tracker.track(_mulax, a, x)
_mulatx(a::TrackedArray, x::AbstractMatrix) = Flux.Tracker.track(_mulatx, a, x)
_mulxa(x::AbstractMatrix, a::TrackedArray) = Flux.Tracker.track(_mulxa, x, a)

Flux.Tracker.@grad function _mulax(θ::TrackedArray, x)
	return _mulax(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulax(Flux.data(θ), Flux.data(Δ), Flux.data(x)), _mulatx(Flux.data(θ), Flux.data(Δ)))
end
Flux.Tracker.@grad function _mulatx(θ::TrackedArray, x)
  return _mulatx(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulax(Flux.data(θ), Flux.data(Δ), Flux.data(x)), (println("here");_mulax(Flux.data(θ), Flux.data(Δ))))
end
# (Δ * transpose(b), transpose(a) * Δ)


end # module
