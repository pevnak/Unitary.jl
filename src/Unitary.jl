module Unitary

using Flux, LinearAlgebra
import Base.*

struct UnitaryMatrix{T}
	θ::T	
end

const TransposedUnitaryMatrix = Transpose{X,A} where {X, A<:UnitaryMatrix}

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
Flux.data(a::TransposedUnitaryMatrix) = transpose(UnitaryMatrix(Flux.data(a.parent.θ)))

LinearAlgebra.transpose(a::UnitaryMatrix) = LinearAlgebra.Transpose(a)

*(a::UnitaryMatrix, x) = _mulax(a.θ, x)
_mulax(θ, x) = _mulax((sin(θ[1]), cos(θ[1])), x)

*(a::TransposedUnitaryMatrix, x::AbstractMatrix) = _mulatx(a.parent.θ, x)
*(a::TransposedUnitaryMatrix, x::TrackedMatrix) = _mulatx(a.parent.θ, x)
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
_∇mulatx(θ, Δ, x) = _∇mulax(Δ, (sin(θ[1]), - cos(θ[1])), x)
function _∇mulax(Δ, sincosθ::Tuple, x::AbstractMatrix)
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

*(x::AbstractMatrix, a::TransposedUnitaryMatrix) = _mulxat(x, a.parent.θ)
*(x::TrackedMatrix, a::TransposedUnitaryMatrix) = _mulxat(x, a.parent.θ)
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

_∇mulxa(θ, Δ, x) = _∇mulxa(Δ, x, (sin(θ[1]), cos(θ[1])))
_∇mulxat(θ, Δ, x) = _∇mulxa(Δ, x, (sin(θ[1]), - cos(θ[1])))
function _∇mulxa(Δ, x::AbstractMatrix, sincosθ::Tuple)
	sinθ, cosθ = sincosθ
	∇θ = similar(Δ, 1)
	fill!(∇θ, 0)
	for i in 1:size(x, 1)
		∇θ[1] +=  Δ[i, 1] * (-sinθ * x[i, 1] + cosθ * x[i, 2])
		∇θ[1] +=  Δ[i, 2] * (-cosθ * x[i, 1] - sinθ * x[i, 2])
	end
	∇θ
end


_mulax(a::TrackedArray, x::AbstractMatrix) = Flux.Tracker.track(_mulax, a, x)
_mulatx(a::TrackedArray, x::AbstractMatrix) = Flux.Tracker.track(_mulatx, a, x)
_mulxa(x::AbstractMatrix, a::TrackedArray) = Flux.Tracker.track(_mulxa, x, a)
_mulxat(x::AbstractMatrix, a::TrackedArray) = Flux.Tracker.track(_mulxat, x, a)

Flux.Tracker.@grad function _mulax(θ::TrackedArray, x)
	return _mulax(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulax(Flux.data(θ), Flux.data(Δ), Flux.data(x)), _mulatx(Flux.data(θ), Flux.data(Δ)))
end
Flux.Tracker.@grad function _mulatx(θ::TrackedArray, x)
  return _mulatx(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulatx(Flux.data(θ), Flux.data(Δ), Flux.data(x)), _mulax(Flux.data(θ), Flux.data(Δ)))
end

Flux.Tracker.@grad function _mulxa(x, θ::TrackedArray)
	return _mulxa(Flux.data(x), Flux.data(θ),) , Δ -> (_mulxat(Flux.data(Δ), Flux.data(θ)), _∇mulxa(Flux.data(θ), Flux.data(Δ), Flux.data(x)))
end
Flux.Tracker.@grad function _mulxat(x, θ::TrackedArray)
  return _mulxat(Flux.data(x), Flux.data(θ)) , Δ -> (_mulxa(Flux.data(Δ), Flux.data(θ)), _∇mulxat(Flux.data(θ), Flux.data(Δ), Flux.data(x)))
end
# (Δ * transpose(b), transpose(a) * Δ)


end # module
