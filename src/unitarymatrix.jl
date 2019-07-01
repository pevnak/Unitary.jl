
struct UnitaryMatrix{T}
	θ::T	
end

const TransposedUnitaryMatrix = Transpose{X,A} where {X, A<:UnitaryMatrix}
const TransposedVector = Transpose{X,A} where {X, A<: AbstractVector}


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


Base.size(a::UnitaryMatrix) = (2,2)
Base.size(a::UnitaryMatrix, i::Int) = (i, i)
Base.eltype(a::UnitaryMatrix) = eltype(a.θ)
Base.length(a::UnitaryMatrix) = 4
Flux.data(a::UnitaryMatrix) = UnitaryMatrix(Flux.data(a.θ))
Flux.data(a::TransposedUnitaryMatrix) = transpose(UnitaryMatrix(Flux.data(a.parent.θ)))
LinearAlgebra.transpose(a::UnitaryMatrix) = LinearAlgebra.Transpose(a)


*(a::UnitaryMatrix, x) = _mulax(a.θ, x)
*(x, a::UnitaryMatrix) = _mulxa(x, a.θ)
for t in [:TransposedMatVec, :TrackedMatrix, :TrackedVector]
	@eval *(a::TransposedUnitaryMatrix, x::$t) = _mulatx(a.parent.θ, x)
	@eval *(x::$t, a::TransposedUnitaryMatrix) = _mulxat(x, a.parent.θ)
end
# *(a::TransposedUnitaryMatrix, x::TrackedMatrix) = _mulatx(a.parent.θ, x)
# *(a::TransposedUnitaryMatrix, x::TrackedVector) = _mulatx(a.parent.θ, x)

# *(x::TransposedMatVec, a::TransposedUnitaryMatrix) = _mulxat(x, a.parent.θ)
# *(x::TrackedMatVec, a::TransposedUnitaryMatrix) = _mulxat(x, a.parent.θ)



"""
	_mulax(θ::Vector, x::MatVec)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
_mulax(θ::Vector, x::TransposedMatVec) = _mulax((sin(θ[1]), cos(θ[1])), x)
function _mulax(sincosθ::Tuple, x::TransposedMatVec)
	sinθ, cosθ = sincosθ
	@assert size(x, 1) == 2
	o = similar(x)
	for i in 1:size(x, 2)
		o[1, i] =  cosθ * x[1,i] - sinθ * x[2,i]
		o[2, i] =  sinθ * x[1,i] + cosθ * x[2,i]
	end
	o
end

"""
	_∇mulax(θ, Δ, x)

	multiply Unitary matrix defined by a rotation angle `θ` by a Matrix x
"""
_∇mulax(θ, Δ, x) = _∇mulax(Δ, (sin(θ[1]), cos(θ[1])), x)
function _∇mulax(Δ, sincosθ::Tuple, x::MatVec)
	sinθ, cosθ = sincosθ
	∇θ = similar(x, 1)
	fill!(∇θ, 0)
	for i in 1:size(x, 2)
		∇θ[1] +=  Δ[1,i] * (- sinθ * x[1,i] - cosθ * x[2,i])
		∇θ[1] +=  Δ[2,i] * (  cosθ * x[1,i] - sinθ * x[2,i])
	end
	∇θ
end
_mulatx(θ::Vector, x::TransposedMatVec) = _mulax((- sin(θ[1]), cos(θ[1])), x)
_∇mulatx(θ::Vector, Δ::TransposedMatVec, x::TransposedMatVec) = _∇mulax(Δ, (sin(θ[1]), - cos(θ[1])), x)


_mulxa(x::TransposedMatVec, θ::Vector) = _mulxa(x, (sin(θ[1]), cos(θ[1])))
_mulxat(x::TransposedMatVec, θ::Vector) = _mulxa(x, (- sin(θ[1]), cos(θ[1])))
function _mulxa(x::TransposedMatVec, sincosθ::Tuple)
	sinθ, cosθ = sincosθ 
	@assert size(x, 2) == 2
	o = similar(x)
	for i in 1:size(x, 1)
		o[i, 1] =    cosθ * x[i, 1] + sinθ * x[i, 2]
		o[i, 2] =  - sinθ * x[i, 1] + cosθ * x[i, 2]
	end
	o
end

_∇mulxa(θ::Vector, Δ::TransposedMatVec, x::TransposedMatVec) = _∇mulxa(Δ, x, (sin(θ[1]), cos(θ[1])))
_∇mulxat(θ::Vector, Δ::TransposedMatVec, x::TransposedMatVec) = _∇mulxa(Δ, x, (sin(θ[1]), - cos(θ[1])))
function _∇mulxa(Δ::TransposedMatVec, x::TransposedMatVec, sincosθ::Tuple)
	sinθ, cosθ = sincosθ
	∇θ = similar(x, 1)
	fill!(∇θ, 0)
	for i in 1:size(x, 1)
		∇θ[1] +=  Δ[i, 1] * (-sinθ * x[i, 1] + cosθ * x[i, 2])
		∇θ[1] +=  Δ[i, 2] * (-cosθ * x[i, 1] - sinθ * x[i, 2])
	end
	∇θ
end


_mulax(a::Vector, x::TrackedMatVec) = Flux.Tracker.track(_mulax, a, x)
_mulax(a::TrackedVector, x::TrackedMatVec) = Flux.Tracker.track(_mulax, a, x)
_mulax(a::TrackedVector, x::TransposedMatVec) = Flux.Tracker.track(_mulax, a, x)

_mulatx(a::TrackedVector, x::TransposedMatVec) = Flux.Tracker.track(_mulatx, a, x)
_mulatx(a::Vector, x::TrackedMatVec) = Flux.Tracker.track(_mulatx, a, x)
_mulatx(a::TrackedVector, x::TrackedMatVec) = Flux.Tracker.track(_mulatx, a, x)

_mulxa(x::TransposedMatVec, a::TrackedVector) = Flux.Tracker.track(_mulxa, x, a)
_mulxa(x::TrackedMatVec, a::TrackedVector) = Flux.Tracker.track(_mulxa, x, a)
_mulxa(x::TrackedMatVec, a::Vector) = Flux.Tracker.track(_mulxa, x, a)

_mulxat(x::TransposedMatVec, a::TrackedVector) = Flux.Tracker.track(_mulxat, x, a)
_mulxat(x::TrackedMatVec, a::TrackedVector) = Flux.Tracker.track(_mulxat, x, a)
_mulxat(x::TrackedMatVec, a::Vector) = Flux.Tracker.track(_mulxat, x, a)

Flux.Tracker.@grad function _mulax(θ::TrackedVector, x::AbstractArray)
	return _mulax(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulax(Flux.data(θ), Flux.data(Δ), Flux.data(x)), _mulatx(Flux.data(θ), Flux.data(Δ)))
end
Flux.Tracker.@grad function _mulax(θ::Vector, x::TrackedMatVec)
	return _mulax(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulax(Flux.data(θ), Flux.data(Δ), Flux.data(x)), _mulatx(Flux.data(θ), Flux.data(Δ)))
end
Flux.Tracker.@grad function _mulatx(θ::TrackedVector, x::TransposedMatVec)
  return _mulatx(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulatx(Flux.data(θ), Flux.data(Δ), Flux.data(x)), _mulax(Flux.data(θ), Flux.data(Δ)))
end

Flux.Tracker.@grad function _mulatx(θ::TrackedVector, x::TrackedMatVec)
  return _mulatx(Flux.data(θ), Flux.data(x)) , Δ -> (_∇mulatx(Flux.data(θ), Flux.data(Δ), Flux.data(x)), _mulax(Flux.data(θ), Flux.data(Δ)))
end

Flux.Tracker.@grad function _mulxa(x, θ::TrackedVector)
	return _mulxa(Flux.data(x), Flux.data(θ),) , Δ -> (_mulxat(Flux.data(Δ), Flux.data(θ)), _∇mulxa(Flux.data(θ), Flux.data(Δ), Flux.data(x)))
end
Flux.Tracker.@grad function _mulxat(x, θ::TrackedVector)
  return _mulxat(Flux.data(x), Flux.data(θ)) , Δ -> (_mulxa(Flux.data(Δ), Flux.data(θ)), _∇mulxat(Flux.data(θ), Flux.data(Δ), Flux.data(x)))
end
# (Δ * transpose(b), transpose(a) * Δ)
